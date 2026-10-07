"""Tiny independent Transformers equivalence; no real model download required."""
import subprocess
import sys
from pathlib import Path

import pytest


def _check_streamed(tmp_path: Path, query_heads: int, kv_heads: int):
    import weakref

    import numpy as np
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from safetensors import safe_open
    import tools.streamed_oracle as streamed

    torch.manual_seed(123)
    config = Qwen2Config(vocab_size=67, hidden_size=query_heads * 4,
        intermediate_size=96, num_hidden_layers=3, num_attention_heads=query_heads,
        num_key_value_heads=kv_heads, tie_word_embeddings=True, rope_theta=1e6,
        rms_norm_eps=1e-6, attention_dropout=0.0)
    stored = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
    stored.save_pretrained(tmp_path)
    del stored
    oracle = streamed.StreamedOracle(tmp_path, 1, 7)
    # Independent, unmodified dense model: BF16 source widened once to FP32.
    dense = Qwen2ForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True).float().eval()
    original_load = streamed.load_layer
    previous = []
    layer_calls = []

    def tracked_load(path, layer_config, index):
        assert not previous or previous[-1]() is None, "Previous decoder layer remains resident"
        layer = original_load(path, layer_config, index)
        assert all(parameter.dtype == torch.float32 for parameter in layer.parameters())
        with safe_open(path, framework="pt", device="cpu") as source:
            for name, parameter in layer.named_parameters():
                torch.testing.assert_close(parameter, source.get_tensor(f"model.layers.{index}.{name}").float(), atol=0, rtol=0)
        layer_calls.append(index)
        previous.append(weakref.ref(layer))
        return layer

    streamed.load_layer = tracked_load
    tokens = [1, 6, 7, 66, 1, 19, 51, 2, 0, 33]
    positions = list(range(4, len(tokens)))
    with torch.inference_mode():
        hidden = streamed.window_hidden(tmp_path, oracle.config, tokens, 4, 7)
        assert previous[-1]() is None
        assert layer_calls == [0, 1, 2]
        result = dense.model(input_ids=torch.tensor([tokens[:4]]), use_cache=True, return_dict=True)
        cache = result.past_key_values
        expected_hidden = [result.last_hidden_state]
        expected_logits = []
        for position in positions:
            result = dense.model(input_ids=torch.tensor([[tokens[position]]]),
                past_key_values=cache, use_cache=True, return_dict=True)
            cache = result.past_key_values
            expected_hidden.append(result.last_hidden_state)
            expected_logits.append(dense.lm_head(result.last_hidden_state)[0, 0].numpy())
        torch.testing.assert_close(hidden, torch.cat(expected_hidden, dim=1), atol=2e-6, rtol=2e-5)
        for chunk in (1, 7, 32):
            output = tmp_path / f"logits-{chunk}.bin"
            streamed.write_logits(tmp_path, oracle.config, hidden, positions, output, chunk)
            actual = np.fromfile(output, dtype="<f4").reshape(len(positions), config.vocab_size)
            np.testing.assert_allclose(actual, expected_logits, atol=2e-6, rtol=2e-5)
            assert np.array_equal(actual.argmax(axis=1), np.asarray(expected_logits).argmax(axis=1))
            with pytest.raises(FileExistsError):
                streamed.write_logits(tmp_path, oracle.config, hidden, positions, output, chunk)
        with pytest.raises(ValueError, match="Token IDs"):
            streamed.gather_embeddings(tmp_path / "model.safetensors", config, [67], 7)
        output = tmp_path / "whole-window.bin"
        oracle.write_window(tokens, positions, output, 4)
        actual = np.fromfile(output, dtype="<f4").reshape(len(positions), config.vocab_size)
        np.testing.assert_allclose(actual, expected_logits, atol=2e-6, rtol=2e-5)
        assert previous[-1]() is None


@pytest.mark.parametrize("query_heads,kv_heads", [(2, 1), (12, 2)])
def test_layer_streaming_equals_unmodified_transformers(tmp_path, query_heads, kv_heads):
    # Isolate PyTorch's process-global thread configuration and model memory,
    # matching the repository's reference-test worker convention.
    code = "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['_check_streamed'](Path(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))"
    subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve()),
        str(tmp_path), str(query_heads), str(kv_heads)], check=True,
        cwd=Path(__file__).resolve().parents[1])
