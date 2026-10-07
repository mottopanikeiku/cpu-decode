"""Layer-resident Qwen2 FP32 oracle on the original BF16 snapshot.

Decoder/RMSNorm/rotary implementations are unmodified installed Transformers.
A window is evaluated layer-major, preserving the existing 256-token priming
batch followed by single-token cached calls. No full model is constructed.
Safetensors contexts end after each tensor/chunk is widened; only one decoder
layer's FP32 weights and KV cache are retained. Vocabulary output is written
with bounded chunk buffers, not accumulated in RAM or a writable mmap.

Limits: CPU only, FP32 weights/activations/KV and FP32 raw logits; the source
values are exactly BF16-to-FP32 widened, not recovered pre-BF16 training weights.
Vocabulary chunk rows are restricted to 1..4096 (default 1024 in quality_v2).
The protected corpus supplies 512 inputs/window, with 256 scored positions.
INFERENCE from the pinned 1.5B shapes: one layer has 46,797,824 FP32 values
(187,191,296 bytes); at most one BF16 tensor is mapped while loading it (largest
27,525,120 bytes). A 1024-row FP32 vocabulary chunk is 6,291,456 bytes; each
512-token hidden buffer is 3,145,728 bytes; one layer's full KV is 1,048,576
bytes. These are tensor payloads, NOT a measured process-memory bound: Python,
libraries, attention/MLP temporaries, BLAS workspaces and allocator retention
add memory. Execution must use an external 2000M cap; peak RSS is unmeasured.
"""
from pathlib import Path


def widened_tensor(path: Path, name: str, rows: tuple[int, int] | None = None):
    import torch
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as source:
        original = source.get_tensor(name) if rows is None else source.get_slice(name)[rows[0]:rows[1]]
        if original.dtype != torch.bfloat16:
            raise ValueError(f"Expected original BF16 tensor: {name}")
        widened = original.to(dtype=torch.float32)
        del original
    return widened


def load_layer(path: Path, config, index: int):
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    # Meta construction avoids an initialized FP32 duplicate during loading.
    with torch.device("meta"):
        layer = Qwen2DecoderLayer(config, index)
    prefix = f"model.layers.{index}."
    state = {name: widened_tensor(path, prefix + name) for name in layer.state_dict()}
    layer.load_state_dict(state, strict=True, assign=True)
    del state
    return layer.eval()


def gather_embeddings(path: Path, config, tokens: list[int], chunk_rows: int):
    import torch

    ids = torch.tensor(tokens, dtype=torch.long)
    if not tokens or min(tokens) < 0 or max(tokens) >= config.vocab_size:
        raise ValueError("Token IDs outside pinned vocabulary")
    hidden = torch.empty((1, len(tokens), config.hidden_size), dtype=torch.float32)
    for chunk in sorted({token // chunk_rows for token in tokens}):
        start = chunk * chunk_rows
        end = min(start + chunk_rows, config.vocab_size)
        weight = widened_tensor(path, "model.embed_tokens.weight", (start, end))
        selected = (ids >= start) & (ids < end)
        hidden[0, selected] = weight[ids[selected] - start]
        del weight
    return hidden


def window_hidden(model: Path, config, tokens: list[int], priming: int, chunk_rows: int):
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding

    if not 0 < priming <= len(tokens):
        raise ValueError("Invalid priming length")
    path = model / "model.safetensors"
    hidden = gather_embeddings(path, config, tokens, chunk_rows)
    rotary = Qwen2RotaryEmbedding(config).eval()
    all_positions = torch.arange(len(tokens), dtype=torch.long)
    embeddings = rotary(hidden, all_positions.unsqueeze(0))
    mask = torch.full((priming, priming), torch.finfo(torch.float32).min, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)[None, None]
    for index in range(config.num_hidden_layers):
        layer = load_layer(path, config, index)
        cache = DynamicCache()
        output = torch.empty_like(hidden)
        # Batch shape/reduction order matches quality_v2's original oracle.
        output[:, :priming] = layer(
            hidden[:, :priming], attention_mask=mask,
            position_ids=all_positions[:priming].unsqueeze(0), past_key_values=cache,
            use_cache=True, cache_position=all_positions[:priming],
            position_embeddings=tuple(value[:, :priming] for value in embeddings))
        for position in range(priming, len(tokens)):
            output[:, position:position + 1] = layer(
                hidden[:, position:position + 1], attention_mask=None,
                position_ids=all_positions[position:position + 1].unsqueeze(0),
                past_key_values=cache, use_cache=True,
                cache_position=all_positions[position:position + 1],
                position_embeddings=tuple(value[:, position:position + 1] for value in embeddings))
        del layer, cache, hidden
        hidden = output
        del output
    norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    norm.load_state_dict({"weight": widened_tensor(path, "model.norm.weight")}, assign=True)
    return norm(hidden)


def write_logits(model: Path, config, hidden, positions: list[int], output: Path, chunk_rows: int):
    import torch.nn.functional as functional

    if hidden.dtype.__str__() != "torch.float32":
        raise ValueError("Streamed oracle requires FP32 activations")
    if not positions or any(not 0 <= position < hidden.shape[1] for position in positions):
        raise ValueError("Invalid scored positions")
    # Use the tied embedding tensor, never an independent head copy.
    with output.open("xb") as stream:
        stream.truncate(len(positions) * config.vocab_size * 4)
        for start in range(0, config.vocab_size, chunk_rows):
            end = min(start + chunk_rows, config.vocab_size)
            weight = widened_tensor(model / "model.safetensors", "model.embed_tokens.weight", (start, end))
            for row, position in enumerate(positions):
                logits = functional.linear(hidden[:, position:position + 1], weight)[0, 0]
                stream.seek((row * config.vocab_size + start) * 4)
                stream.write(logits.numpy().astype("<f4", copy=False).tobytes())
                del logits
            del weight


class StreamedOracle:
    """Source verification belongs to the caller; this class never loads a model."""

    def __init__(self, model: Path, threads: int, chunk_rows: int):
        import torch
        from transformers import Qwen2Config

        if threads < 1 or not 1 <= chunk_rows <= 4096:
            raise ValueError("threads must be positive; chunk rows must be in [1,4096]")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.config = Qwen2Config.from_pretrained(str(model), local_files_only=True)
        self.config._attn_implementation = "eager"
        if (self.config.model_type != "qwen2" or not self.config.tie_word_embeddings
                or self.config.use_sliding_window or self.config.rope_scaling):
            raise ValueError("Expected pinned tied-head full-attention Qwen2 with default rotary")
        self.model = model
        self.chunk_rows = chunk_rows

    def write_window(self, tokens: list[int], positions: list[int], output: Path, priming: int):
        import torch

        with torch.inference_mode():
            hidden = window_hidden(self.model, self.config, tokens, priming, self.chunk_rows)
            write_logits(self.model, self.config, hidden, positions, output, self.chunk_rows)
            del hidden
