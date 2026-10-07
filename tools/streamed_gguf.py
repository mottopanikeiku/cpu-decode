#!/usr/bin/env python3
"""Bounded BF16 GGUF conversion for the two verified Qwen2.5 snapshots.

Attribution: metadata, vocabulary, tensor names, BF16 encoding and GGUF writing
are the unmodified MIT-licensed llama.cpp implementations at
https://github.com/ggml-org/llama.cpp/tree/6c73b3e12dc501de35fe5f6979960d06921a2f6c
(conversion/base.py, conversion/qwen.py and
gguf-py/gguf/{lazy,quants,gguf_writer,gguf_reader}.py).
This adapter changes only tensor indexing/preparation: each upstream
LazyChunkedTensor callback opens a safetensors mapping, widens a row slice to
owned FP32 storage, and closes the mapping before upstream BF16 encoding.
Vectors are written as F32, exactly as in the upstream BF16 converter. No Q8
implementation lives here; prepare_llama invokes the pinned llama-quantize.

CLI: python tools/streamed_gguf.py --source "$LLAMA_SOURCE" --model "$MODEL"
     --model-id Qwen/Qwen2.5-0.5B-Instruct --outfile "$FRESH_BF16" --chunk-mib 4
The model-id defaults to the verified 1.5B S1 pin. Compare actual artifacts with:
     python tools/streamed_gguf.py --source "$LLAMA_SOURCE"
     --compare "$ORIGINAL_BF16" "$FRESH_BF16" --record "$COMPARISON_JSON"
Comparison hashes bounded reads using the pinned GGUF reader's offsets; it never
materializes tensor arrays. All metadata (including vocabulary), tensor names,
GGML types, shapes and raw payload hashes must match. Exit 1 means mismatch/error.
The caller verifies the source snapshot too and promotes only a successful
complete output. The helper refuses an existing destination; preparation may
replace only its dedicated failed streamed partial on retry.

INFERENCE: FP32 row payloads are at most 4 MiB by default (including 8960-wide
rows); BF16 source slices, upstream codec temporaries and allocator retention
add memory. No source mappings survive callbacks, no weight buffers accumulate
across tensors, and no writable mmap or 256 MiB spooled writer is used. Library and
tokenizer/metadata memory and actual peak RSS remain unmeasured; the external
2000M cap is still required.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib
import hashlib
import json
import logging
import struct
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from tools.download_model import MODEL_ID, S1_MODEL_ID, PINNED_MODELS, verify_snapshot
from tools.prepare_llama import LLAMA_COMMIT

DEFAULT_CHUNK_MIB = 4


@dataclass(frozen=True)
class TensorSpec:
    path: Path
    name: str
    shape: tuple[int, ...]


def validate_config(config: dict, model_id: str = S1_MODEL_ID) -> None:
    """Require the exact configuration in the selected verified source pin."""
    if model_id not in PINNED_MODELS:
        raise ValueError(f"Unsupported pinned model: {model_id}")
    expected = {
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "torch_dtype": "bfloat16", "hidden_size": 1536,
        "intermediate_size": 8960, "num_hidden_layers": 28,
        "num_attention_heads": 12, "num_key_value_heads": 2,
        "vocab_size": 151936, "tie_word_embeddings": True,
        "hidden_act": "silu", "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 32768,
        "use_sliding_window": False, "attention_dropout": 0.0,
        "bos_token_id": 151643, "eos_token_id": 151645,
        "initializer_range": 0.02, "max_window_layers": 21,
        "sliding_window": 32768, "transformers_version": "4.43.1",
        "use_cache": True,
    }
    if model_id == MODEL_ID:
        expected.update(hidden_size=896, intermediate_size=4864,
                        num_hidden_layers=24, num_attention_heads=14)
    for key in sorted(config.keys() | expected.keys()):
        if key not in expected or key not in config or config[key] != expected[key]:
            raise ValueError(f"Unsupported {model_id} configuration: {key}")


def row_ranges(shape: tuple[int, ...], chunk_bytes: int) -> list[tuple[int, int]]:
    """Split only on rows; every widened slice fits the requested FP32 budget."""
    if len(shape) not in (1, 2) or any(size <= 0 for size in shape):
        raise ValueError(f"Unsupported tensor shape: {shape}")
    width = shape[1] if len(shape) == 2 else 1
    if chunk_bytes < width * 4:
        raise ValueError("Chunk budget must hold at least one FP32 row")
    rows = chunk_bytes // (width * 4)
    return [(start, min(start + rows, shape[0])) for start in range(0, shape[0], rows)]


def index_safetensors(path: Path) -> dict[str, TensorSpec]:
    """Read only metadata, retaining no safetensors handles or mmap views."""
    from safetensors import safe_open

    specs = {}
    with safe_open(path, framework="pt", device="cpu") as source:
        for name in source.keys():
            sliced = source.get_slice(name)
            if sliced.get_dtype() != "BF16":
                raise ValueError(f"Expected original BF16 tensor: {name}")
            shape = tuple(sliced.get_shape())
            row_ranges(shape, DEFAULT_CHUNK_MIB * 1024 * 1024)
            specs[name] = TensorSpec(path, name, shape)
            del sliced
    if not specs:
        raise ValueError("No source tensors")
    return specs


def load_chunk(spec: TensorSpec, start: int, end: int):
    """Return an owned contiguous FP32 ndarray after closing the source mapping."""
    import torch
    from safetensors import safe_open

    if not 0 <= start < end <= spec.shape[0]:
        raise ValueError("Invalid tensor row range")
    with safe_open(spec.path, framework="pt", device="cpu") as source:
        sliced = source.get_slice(spec.name)
        if tuple(sliced.get_shape()) != spec.shape or sliced.get_dtype() != "BF16":
            raise ValueError(f"Changed BF16 tensor metadata: {spec.name}")
        original = sliced[start:end]
        if original.dtype != torch.bfloat16:
            raise ValueError(f"Expected original BF16 tensor: {spec.name}")
        widened = original.to(dtype=torch.float32)
        del original, sliced
    # NumPy retains the owned FP32 tensor, never the source mmap or its context.
    return widened.numpy()


def make_streamed_model_class(qwen2_model, gguf):
    """Subclass only indexing/preparation; keep upstream metadata/vocab/write."""
    class StreamedQwen2Model(qwen2_model):
        # TextModel requires the architecture in each subclass's own dictionary.
        model_arch = qwen2_model.model_arch

        def index_tensors(self, remote_hf_model_id=None):
            if remote_hf_model_id is not None:
                raise ValueError("Streamed GGUF requires a local pinned snapshot")
            return index_safetensors(self.dir_model / "model.safetensors")

        def prepare_tensors(self):
            if self.fuse_qkv or self.fuse_gate_up_exps or self.use_temp_file:
                raise ValueError("Streamed GGUF requires unfused, direct chunk writing")
            for name, spec in self.model_tensors.items():
                chunks = [partial(load_chunk, spec, start, end)
                          for start, end in row_ranges(spec.shape, self.chunk_bytes)]
                data = gguf.LazyChunkedTensor(chunks, spec.shape, "float32")
                bid = next((int(part) for part in name.split(".") if part.isdecimal()), None)
                # Qwen2ForCausalLM has an identity value transform with fusion off;
                # delegate the name mapping to its actual upstream implementation.
                for new_name, mapped in self.modify_tensors(data, name, bid):
                    if mapped is not data:
                        raise ValueError("Unsupported nonidentity Qwen2 tensor transform")
                    qtype = (gguf.GGMLQuantizationType.F32 if len(spec.shape) == 1
                             or new_name.endswith("_norm.weight")
                             else gguf.GGMLQuantizationType.BF16)
                    logging.getLogger("hf-to-gguf").info(
                        "%s BF16 -> %s, shape = %s (row-chunked)", new_name, qtype.name, spec.shape)
                    self.gguf_writer.add_tensor(new_name, data.quantize(qtype), raw_dtype=qtype)

    return StreamedQwen2Model


def load_upstream(source: Path):
    """Require the actual pinned checkout and its own gguf-py, not pip gguf."""
    source = source.resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != LLAMA_COMMIT:
        raise ValueError("Streamed GGUF requires the pinned llama.cpp commit")
    sys.path[:0] = [str(source / "gguf-py"), str(source)]
    gguf = importlib.import_module("gguf")
    qwen = importlib.import_module("conversion.qwen")
    for module, expected in ((gguf, source / "gguf-py" / "gguf"),
                             (qwen, source / "conversion")):
        if Path(module.__file__).resolve().parent != expected:
            raise ValueError("Converter module did not come from the pinned source")
    return qwen.Qwen2Model, gguf


def convert(source: Path, model: Path, outfile: Path, chunk_mib: int = DEFAULT_CHUNK_MIB,
            model_id: str = S1_MODEL_ID) -> None:
    if not 1 <= chunk_mib <= 16:
        raise ValueError("Chunk size must be between 1 and 16 MiB")
    if outfile.exists():
        raise FileExistsError(f"Refusing existing GGUF destination: {outfile}")
    config = json.loads((model / "config.json").read_text())
    validate_config(config, model_id)
    verify_snapshot(model, model_id=model_id)
    qwen2_model, gguf = load_upstream(source)
    import torch

    torch.set_num_threads(1)
    streamed_class = make_streamed_model_class(qwen2_model, gguf)
    converter = streamed_class(model, gguf.LlamaFileType.MOSTLY_BF16, outfile,
                               hparams=config, use_temp_file=False,
                               fuse_qkv=False, fuse_gate_up_exps=False)
    converter.chunk_bytes = chunk_mib * 1024 * 1024
    # Reserve exclusively before upstream opens the file for writing. Failed
    # outputs remain partial and are never promoted by prepare_llama.
    with outfile.open("xb"):
        pass
    try:
        converter.write()
    finally:
        converter.gguf_writer.close()
    # The actual upstream writer checks each tensor's byte count before close.
    # Also require a completed weights state and the expected real GGUF header.
    from gguf.gguf_writer import WriterState

    if converter.gguf_writer.state != WriterState.WEIGHTS:
        raise ValueError("GGUF writer did not complete all tensor payloads")
    expected_count = sum(len(shard) for shard in converter.gguf_writer.tensors)
    with outfile.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24:
        raise ValueError("Incomplete GGUF header")
    magic, version, count, _ = struct.unpack("<4sIQQ", header)
    if magic != b"GGUF" or version != gguf.GGUF_VERSION or count != expected_count:
        raise ValueError("Incomplete or inconsistent GGUF output")


HASH_CHUNK_BYTES = 1024 * 1024


def hash_region(stream, offset: int, length: int) -> str:
    """Hash exact raw bytes with fixed-size reads, never a full tensor copy."""
    if offset < 0 or length < 0:
        raise ValueError("Invalid GGUF byte range")
    stream.seek(offset)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        block = stream.read(min(remaining, HASH_CHUNK_BYTES))
        if not block:
            raise ValueError("Truncated GGUF byte range")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def artifact_hashes(path: Path, gguf) -> dict:
    """Inspect one real GGUF at a time, releasing its read-only mapping."""
    before = path.stat()
    reader = gguf.GGUFReader(path, mode="r")
    try:
        tensors = {}
        metadata = {}
        ranges = []
        with path.open("rb") as stream:
            whole_sha256 = hash_region(stream, 0, before.st_size)
            for name, field in reader.fields.items():
                length = sum(int(part.nbytes) for part in field.parts)
                if field.offset < 0 or field.offset + length > before.st_size:
                    raise ValueError(f"Invalid metadata byte range: {name}")
                metadata[name] = {
                    "types": [int(value) for value in field.types], "bytes": length,
                    "sha256": hash_region(stream, field.offset, length),
                }
            for tensor in reader.tensors:
                offset, length = int(tensor.data_offset), int(tensor.n_bytes)
                if length <= 0 or offset < reader.data_offset or offset + length > before.st_size:
                    raise ValueError(f"Invalid tensor byte range: {tensor.name}")
                ranges.append((offset, offset + length))
                tensors[tensor.name] = {
                    "type": int(tensor.tensor_type), "type_name": tensor.tensor_type.name,
                    "shape": [int(size) for size in tensor.shape], "bytes": length,
                    "sha256": hash_region(stream, offset, length),
                }
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("Overlapping GGUF tensor payloads")
        after = path.stat()
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("GGUF artifact changed during comparison")
        return {"bytes": before.st_size, "sha256": whole_sha256,
                "metadata": metadata, "tensors": tensors}
    finally:
        # Upstream exposes mmap-backed tensor views; none is read or retained.
        reader.fields.clear()
        reader.tensors.clear()
        reader.data._mmap.close()


def compare_gguf(source: Path, reference: Path, candidate: Path) -> dict:
    """Return observed hashes and mismatches, not a declared equivalence flag."""
    _, gguf = load_upstream(source)
    left = artifact_hashes(reference, gguf)
    right = artifact_hashes(candidate, gguf)
    mismatches = []
    for section in ("metadata", "tensors"):
        for name in sorted(left[section].keys() | right[section].keys()):
            if left[section].get(name) != right[section].get(name):
                mismatches.append({"section": section, "name": name,
                                   "reference": left[section].get(name),
                                   "candidate": right[section].get(name)})
    whole_equal = left["sha256"] == right["sha256"] and left["bytes"] == right["bytes"]
    return {
        "schema_version": 1, "llama_commit": LLAMA_COMMIT,
        "artifacts": {"$REFERENCE": left, "$CANDIDATE": right},
        "whole_file_equal": whole_equal, "tensor_metadata_equal": not mismatches,
        "equivalent": not mismatches,
        "basis": "whole-file-sha256" if whole_equal else "tensor-and-all-metadata-sha256",
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-id", choices=tuple(PINNED_MODELS), default=S1_MODEL_ID)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("REFERENCE", "CANDIDATE"))
    parser.add_argument("--record", type=Path, help="Fresh portable comparison JSON; never overwritten")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if args.compare is not None:
        if args.model is not None or args.outfile is not None or args.record is None:
            parser.error("--compare requires --record and cannot be combined with --model/--outfile")
        # Reserve before reading: do not overwrite records or either artifact.
        with args.record.open("x") as output:
            try:
                record = compare_gguf(args.source, *args.compare)
            except Exception as error:
                record = {"schema_version": 1, "required_llama_commit": LLAMA_COMMIT,
                          "equivalent": False, "error_type": type(error).__name__,
                          "error": "GGUF comparison could not complete; inspect stderr"}
                logging.error("GGUF comparison could not complete: %s", error)
            text = json.dumps(record, indent=2) + "\n"
            output.write(text)
        print(text, end="")
        if not record["equivalent"]:
            logging.error("GGUF comparison is false; inputs and JSON record retained")
            raise SystemExit(1)
    else:
        if args.model is None or args.outfile is None or args.record is not None:
            parser.error("conversion requires --model and --outfile; --record is comparison-only")
        convert(args.source, args.model, args.outfile, args.chunk_mib, model_id=args.model_id)


if __name__ == "__main__":
    main()
