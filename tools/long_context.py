#!/usr/bin/env python3
"""Long-context logit agreement with an FP32 Transformers oracle at 1024 and 4096 tokens.

The input is the first 4096 Qwen tokens of the pinned tinyshakespeare text.
The oracle (Transformers SDPA attention, FP32 arithmetic on the BF16 snapshot)
runs in a worker process that exits before any engine or llama.cpp reader
starts, and projects the tied LM head only at the evaluation windows. Every
candidate consumes the same token IDs in one fresh context and emits logits
only at those positions. Agreement is reported, never gated.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import gc
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

from tools.download_model import MODEL_ID, REVISION, file_hash, verify_snapshot
from tools.portable import ROOT, portable
from tools.prepare_llama import LLAMA_COMMIT, LLAMA_URL
from tools.reference import UNQUANTIZED, labeled_path, position_metrics, unique_labels, versions

TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
TEXT_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
TEXT_BYTES = 1115394
TOKEN_COUNT = 4096
DEFAULT_WINDOWS = "960-1023,4032-4095"
GGUF_TYPES = {"Q8_0": "MOSTLY_Q8_0", "Q4_0": "MOSTLY_Q4_0"}


def parse_windows(text: str, count: int) -> list[tuple[int, int]]:
    """Inclusive "A-B" windows, increasing, non-overlapping and inside [0, count)."""
    windows = []
    for item in text.split(","):
        first, separator, last = item.partition("-")
        if not separator or not first.isdigit() or not last.isdigit():
            raise ValueError(f"Window {item!r} is not an inclusive A-B range")
        first, last = int(first), int(last)
        if first > last or (windows and first <= windows[-1][1]) or last >= count:
            raise ValueError(f"Windows must be increasing, non-overlapping ranges inside {count} tokens: {text}")
        windows.append((first, last))
    return windows


def window_positions(windows: list[tuple[int, int]]) -> list[int]:
    return [position for first, last in windows for position in range(first, last + 1)]


def window_names(windows: list[tuple[int, int]]) -> list[str]:
    return [f"{first}-{last}" for first, last in windows]


def tokens_digest(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def fetch_text(raw_dir: Path) -> Path:
    """Download once into the cache and verify size and SHA-256 on every use."""
    path = raw_dir / "tinyshakespeare-input.txt"
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        with urllib.request.urlopen(TEXT_URL, timeout=60) as response, partial.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
        partial.replace(path)
    if path.stat().st_size != TEXT_BYTES or file_hash(path) != TEXT_SHA256:
        raise ValueError(f"Pinned long-context text checksum mismatch: {path}; move it aside to download again")
    return path


def oracle_worker(args) -> None:
    source_manifest = verify_snapshot(args.model)
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import AutoTokenizer

    from tools.reference import load_oracle

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    tokens = tokenizer.encode(args.text.read_text(encoding="utf-8"), add_special_tokens=False)
    if len(tokens) < TOKEN_COUNT:
        raise ValueError(f"Text has only {len(tokens)} tokens; {TOKEN_COUNT} are required")
    tokens = tokens[:TOKEN_COUNT]
    windows = parse_windows(args.windows, TOKEN_COUNT)
    positions = window_positions(windows)
    model = load_oracle(args.model, args.threads, "fp32", attention="sdpa")
    vocab = model.config.vocab_size
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    path = args.raw_dir / "oracle-logits.bin"
    with torch.inference_mode():
        # One causal pass over every token; no KV cache is kept.
        hidden = model.model(input_ids=torch.tensor([tokens], dtype=torch.long), use_cache=False, return_dict=True).last_hidden_state
        if hidden.dtype != torch.float32:
            raise ValueError("FP32 oracle unexpectedly produced non-FP32 activations")
        hidden = hidden[0, positions]
        logits = np.memmap(path, dtype="<f4", mode="w+", shape=(len(positions), vocab))
        for start in range(0, vocab, args.head_chunk):
            end = min(start + args.head_chunk, vocab)
            # Only this head slice is widened to FP32, as in reference.project_last.
            logits[:, start:end] = functional.linear(hidden, model.lm_head.weight[start:end].float()).numpy()
        logits.flush()
        del logits, hidden
    del model
    gc.collect()
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "verified_source": source_manifest, "weight_storage": "bfloat16",
                "arithmetic": "fp32", "attention": "Transformers sdpa, one causal pass over all tokens", "kv_cache": None,
                "head_chunk_rows": args.head_chunk, "threads": args.threads, "versions": versions(),
                "tokenizer": "Transformers AutoTokenizer on the whole text; no special tokens; first tokens kept",
                "text": {"url": TEXT_URL, "sha256": TEXT_SHA256, "bytes": TEXT_BYTES},
                "tokens": tokens, "windows": window_names(windows), "positions": positions,
                "logits": {"path": path.name, "shape": [len(positions), vocab], "dtype": "little-endian float32", "sha256": file_hash(path)}}
    (args.raw_dir / "oracle.json").write_text(json.dumps(metadata) + "\n")


def verify_gguf(manifest: dict, path: Path) -> tuple[str, dict]:
    """Return the preparation-manifest artifact type whose identity this file has."""
    if manifest["llama_commit"] != LLAMA_COMMIT or manifest["llama_repository"] != LLAMA_URL:
        raise ValueError("Artifact manifest does not name the pinned llama.cpp revision")
    source = manifest["source_model"]
    if (source["model_id"], source["revision"], source["stored_dtype"]) != (MODEL_ID, REVISION, "bfloat16"):
        raise ValueError("Artifact manifest does not name the pinned BF16 source model")
    digest = file_hash(path)
    for name, artifact in manifest["artifacts"].items():
        if name in GGUF_TYPES and artifact["sha256"] == digest and artifact["bytes"] == path.stat().st_size:
            return name, artifact
    raise ValueError(f"{path} matches no Q8_0/Q4_0 artifact in the preparation manifest")


def summarize(rows: list[dict]) -> dict:
    matches = sum(row["top1_match"] for row in rows)
    return {"positions": len(rows), "top1_matches": matches, "top1_agreement": matches / len(rows),
            "mean_kl_reference_candidate_nats": sum(row["kl_reference_candidate_nats"] for row in rows) / len(rows),
            "max_kl_reference_candidate_nats": max(row["kl_reference_candidate_nats"] for row in rows),
            "max_abs_error": max(row["max_abs_error"] for row in rows)}


def compare(args, oracle: dict, kind: str, label: str, command: list[str], prefix: Path) -> dict:
    import numpy as np

    subprocess.run(command, check=True)
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    shape = tuple(oracle["logits"]["shape"])
    if metadata["tokens"] != oracle["tokens"] or metadata["positions"] != oracle["positions"] or tuple(metadata["shape"]) != shape:
        raise ValueError(f"{label}: emitted positions/shape/tokens differ from the oracle's")
    candidate_path = prefix.with_suffix(".bin")
    if candidate_path.stat().st_size != shape[0] * shape[1] * 4:
        raise ValueError(f"{label}: logit file size does not match its shape")
    reference = np.memmap(args.raw_dir / oracle["logits"]["path"], dtype="<f4", mode="r", shape=shape)
    candidate = np.memmap(candidate_path, dtype="<f4", mode="r", shape=shape)
    rows = position_metrics(reference, candidate, atol=0.0, rtol=0.0)
    del reference, candidate
    if metadata["argmax"] != [row["candidate_top1"] for row in rows]:
        raise ValueError(f"{label}: argmax metadata disagrees with the written logits")
    for row, position in zip(rows, oracle["positions"], strict=True):
        # Agreement is reported without a tolerance gate.
        del row["allclose"]
        row["position"] = position
    windows = {}
    for name in oracle["windows"]:
        first, last = map(int, name.split("-"))
        windows[name] = summarize([row for row in rows if first <= row["position"] <= last])
    del metadata["tokens"], metadata["positions"]
    recorded = [("$TOKENS" if previous == "--tokens" else item) for previous, item in zip(["", *command], command)]
    return {"kind": kind, "label": label, "settings": metadata, "command": recorded,
            "logits": {"path": str(candidate_path), "sha256": file_hash(candidate_path)},
            "windows": windows, "overall": summarize(rows), "positions": rows}


def run(args) -> dict:
    windows = parse_windows(args.windows, TOKEN_COUNT)
    quant_models = unique_labels(args.quant_model, {UNQUANTIZED})
    ggufs = unique_labels(args.gguf, {UNQUANTIZED, *quant_models})
    manifest = json.loads(args.artifact_manifest.read_text()) if args.artifact_manifest else None
    if ggufs and not args.llama_reader:
        raise ValueError("--gguf requires --llama-reader")
    # Check every identity before the slow oracle pass.
    gguf_artifacts = {label: verify_gguf(manifest, path) if manifest else (None, None) for label, path in ggufs.items()}
    text = fetch_text(args.raw_dir)
    worker = [sys.executable, "-m", "tools.long_context", "--oracle-only", "--model", str(args.model.resolve()), "--text", str(text.resolve()),
              "--raw-dir", str(args.raw_dir.resolve()), "--threads", str(args.threads), "--windows", args.windows, "--head-chunk", str(args.head_chunk)]
    subprocess.run(worker, check=True, cwd=ROOT)
    oracle = json.loads((args.raw_dir / "oracle.json").read_text())
    if oracle["windows"] != window_names(windows) or len(oracle["tokens"]) != TOKEN_COUNT:
        raise ValueError("Oracle worker output does not match the requested windows")
    if file_hash(args.raw_dir / oracle["logits"]["path"]) != oracle["logits"]["sha256"]:
        raise ValueError("Oracle logits changed after the worker wrote them")
    token_text = ",".join(map(str, oracle["tokens"]))
    position_text = ",".join(window_names(windows))
    candidates = []
    engine = str(args.engine.resolve())
    for label, directory, kv in [(UNQUANTIZED, args.model, "f32"), *((label, directory, args.kv) for label, directory in quant_models.items())]:
        # The unquantized engine uses an FP32 KV cache, as in the short-prompt exactness gate.
        prefix = args.raw_dir / f"engine-{label}"
        command = [engine, "logits", "--model", str(directory.resolve()), "--tokens", token_text, "--positions", position_text,
                   "--threads", str(args.threads), "--kernel", args.kernel, "--kv", kv, "--output", str(prefix)]
        record = compare(args, oracle, "engine", label, command, prefix)
        if record["settings"]["kv_dtype"] != kv:
            raise ValueError(f"engine {label}: KV cache {record['settings']['kv_dtype']} differs from the requested {kv}")
        candidates.append(record)
    for label, path in ggufs.items():
        prefix = args.raw_dir / f"llama-{label}"
        command = [str(args.llama_reader.resolve()), "--model", str(path.resolve()), "--tokens", token_text, "--positions", position_text,
                   "--threads", str(args.threads), "--kv", args.kv, "--output", str(prefix)]
        record = compare(args, oracle, "llama.cpp", label, command, prefix)
        artifact_type, artifact = gguf_artifacts[label]
        if record["settings"]["kv_dtype"] != {"f16": "float16", "f32": "float32"}[args.kv]:
            raise ValueError(f"llama.cpp {label}: reader KV cache differs from the requested {args.kv}")
        if artifact_type is not None and record["settings"]["ftype"] != GGUF_TYPES[artifact_type]:
            raise ValueError(f"llama.cpp {label}: reader ftype {record['settings']['ftype']} differs from manifest {artifact_type}")
        record["artifact"] = {"path": str(path.resolve()), "type": artifact_type, "sha256": artifact["sha256"] if artifact else file_hash(path),
                              "tensor_types": artifact["tensor_types"] if artifact else None}
        candidates.append(record)
    # The IDs are kept in $RAW/oracle.json; the result records their digest.
    tokens = oracle.pop("tokens")
    result = {"comparison": "Full-vocabulary logits at the window positions after one fresh-context pass over the same token IDs; KL(reference || candidate) in nats",
              "quality_reporting_only": True,
              "text": {"url": TEXT_URL, "sha256": TEXT_SHA256, "bytes": TEXT_BYTES, "path": str(text.resolve())},
              "token_count": len(tokens), "tokens_sha256": tokens_digest(tokens),
              "tokens_sha256_input": "comma-joined decimal token IDs",
              "windows": window_names(windows), "threads": args.threads, "kernel_requested": args.kernel, "kv_requested": args.kv,
              "oracle": oracle,
              "artifact_manifest": {"path": str(args.artifact_manifest.resolve()), "sha256": file_hash(args.artifact_manifest)} if manifest else None,
              "candidates": candidates}
    locations = {args.model.resolve(): "$MODEL", args.engine.resolve(): "build/cpu-decode", args.raw_dir.resolve(): "$RAW", sys.executable: "$PYTHON"}
    for label, directory in quant_models.items():
        locations[directory.resolve()] = f"${label.upper()}"
    for label, path in ggufs.items():
        locations[path.resolve()] = f"$GGUF_{label.upper()}"
    if args.llama_reader:
        locations[args.llama_reader.resolve()] = "$LLAMA_LOGITS"
    if manifest:
        locations[args.artifact_manifest.resolve()] = "$ARTIFACT_MANIFEST"
    return portable(result, locations)


def table(result: dict) -> str:
    lines = [f"{'candidate':<24} {'window':<11} {'top1':>7} {'mean KL':>10} {'max KL':>10}"]
    for candidate in result["candidates"]:
        name = f"{candidate['kind']} {candidate['label']}"
        for window, summary in [*candidate["windows"].items(), ("overall", candidate["overall"])]:
            lines.append(f"{name:<24} {window:<11} {summary['top1_agreement']:>7.4f} {summary['mean_kl_reference_candidate_nats']:>10.3e} {summary['max_kl_reference_candidate_nats']:>10.3e}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned BF16 snapshot (oracle and unquantized engine)")
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--quant-model", type=labeled_path, action="append", default=[], metavar="LABEL=DIR")
    parser.add_argument("--llama-reader", type=Path, help="llama-logits built against the pinned llama.cpp")
    parser.add_argument("--gguf", type=labeled_path, action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--artifact-manifest", type=Path, help="Preparation manifest; every --gguf must match one of its artifacts")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--kernel", choices=("auto", "scalar", "avx512", "neon"), default="auto")
    parser.add_argument("--kv", choices=("f16", "f32"), default="f16", help="KV cache of quantized engines and llama.cpp; the unquantized engine uses f32")
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--head-chunk", type=int, default=1024)
    parser.add_argument("--raw-dir", type=Path, default=Path("external/long-context"))
    parser.add_argument("--output", type=Path, default=Path("results/long-context.json"))
    parser.add_argument("--text", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--oracle-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or args.head_chunk < 1:
        parser.error("threads and head-chunk must be positive")
    try:
        parse_windows(args.windows, TOKEN_COUNT)
    except ValueError as error:
        parser.error(str(error))
    if args.oracle_only:
        if args.text is None:
            parser.error("oracle worker requires --text")
        oracle_worker(args)
        return
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(table(result))


if __name__ == "__main__":
    main()
