#!/usr/bin/env python3
"""Check real checkpoint logits bitwise across thread counts, not throughput.

Run under an external memory limit, using a fresh raw directory and report path.
Each blocking native logits process exits before the next checkpoint is loaded.
Raw logits are streamed to disk and hashed without loading the full output.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import subprocess
from pathlib import Path

from tools.corpus_v2 import digest_json, load_manifest, protect_destination, window_alignment, write_json
from tools.download_model import file_hash
from tools.portable import ROOT, portable
from tools.quality_v2 import model_identity, native_settings, preflight_native_model

THREADS = (1, 2, 4, 6, 12)
KERNELS = ("simd512x4", "vnni16")
KV_DTYPES = ("f16", "f32")
POSITIONS = 129
SUFFIXES = (".stdout.txt", ".stderr.txt", ".bin", ".json")


def parse_cpu_order(value: str) -> list[int]:
    items = value.split(",")
    if any(not item.isascii() or not item.isdecimal() for item in items):
        raise argparse.ArgumentTypeError("CPU order must contain comma-separated nonnegative integers")
    cpus = [int(item) for item in items]
    if len(cpus) < max(THREADS) or len(set(cpus)) != len(cpus):
        raise argparse.ArgumentTypeError("CPU order must contain at least 12 distinct CPUs")
    return cpus


def artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}


def file_stamp(path: Path) -> tuple:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def validate_actual(actual: dict, requested: dict, inputs: list[int], vocab: int,
                    model: Path, binary: Path) -> dict:
    settings = {**native_settings(actual), "rope": actual["rope"]}
    for key, expected in requested.items():
        if settings.get(key) != expected:
            raise ValueError(f"Native {key} differs from request")
    activation = ("int16", 64) if requested["kernel"] == "vnni16" else ("float32", 0)
    if (actual["activation_dtype"], actual["activation_group_size"]) != activation:
        raise ValueError("Native activation metadata differs from requested kernel")
    if actual["model"] != str(model):
        raise ValueError("Native model path differs from request")
    if actual["tokens"] != inputs or actual["prompt_tokens"] != inputs:
        raise ValueError("Native input tokens differ from corpus prefix")
    if actual["logit_positions"] != list(range(POSITIONS)):
        raise ValueError("Native logit positions differ from full 129-position prefix")
    if actual["shape"] != [POSITIONS, vocab] or actual["vocab_size"] != vocab:
        raise ValueError("Native shape/vocabulary differs from checkpoint config")
    if actual["dtype"] != "float32" or actual["layout"] != "row-major little-endian":
        raise ValueError("Native logits dtype/layout differs from float32 binary contract")
    if actual["kv_capacity"] != POSITIONS:
        raise ValueError("Native KV capacity differs from input prefix")
    if binary.stat().st_size != POSITIONS * vocab * 4:
        raise ValueError("Native logits byte count differs from 129 x vocab x 4")
    return settings


def check_threads(args) -> dict:
    engine, model, corpus_path = args.engine.resolve(), args.model.resolve(), args.corpus.resolve()
    output, raw_dir = protect_destination(args.output), protect_destination(args.raw_dir)
    if args.output.is_symlink() or output.exists():
        raise ValueError("Report destination already exists; use a fresh immutable output")
    if raw_dir.is_relative_to(output):
        raise ValueError("Report destination cannot be an ancestor of the raw directory")
    if args.raw_dir.is_symlink() or raw_dir.exists():
        raise ValueError("Raw directory already exists; never overwrite prior raw outputs")
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise ValueError("Engine must be an executable ELF file")
    with engine.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("Engine must be an executable ELF file")
    cpus = args.cpu_order
    if (not isinstance(cpus, list) or len(cpus) < max(THREADS)
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus)):
        raise ValueError("CPU order must contain at least 12 distinct nonnegative CPU IDs")
    allowed = sorted(os.sched_getaffinity(0))
    if any(cpu not in allowed for cpu in cpus):
        raise ValueError("CPU order contains CPUs outside caller affinity")
    corpus = load_manifest(corpus_path)
    window = next(window for window in corpus["windows"] if window["split"] == "calibration")
    inputs = window_alignment(window)[0][:POSITIONS]
    config = json.loads((model / "config.json").read_text())
    vocab = config["vocab_size"]
    if type(vocab) is not int or vocab <= 0 or len(inputs) != POSITIONS or any(token >= vocab for token in inputs):
        raise ValueError("Invalid checkpoint vocabulary or corpus prefix")
    identity = model_identity(model, "native")
    preflight = preflight_native_model(model, identity, None)
    if preflight != {"group_size": 64, "scale_dtype": "f16", "weight_dtype": "int8"}:
        raise ValueError("Thread check requires the real g64f16 int8 checkpoint")
    cells = [(kernel, kv, thread) for kernel in KERNELS for kv in KV_DTYPES for thread in THREADS]
    prefixes = [raw_dir / f"{kernel}-{kv}-t{thread}" for kernel, kv, thread in cells]
    for prefix in prefixes:
        for suffix in SUFFIXES:
            path = prefix.with_suffix(suffix)
            protect_destination(path)
            if path == output or path.exists() or path.is_symlink():
                raise ValueError("Raw artifact conflicts with report or existing output")
    input_paths = [engine, model / "model.safetensors", model / "config.json", corpus_path]
    if output in input_paths or any(path.is_relative_to(raw_dir) for path in input_paths):
        raise ValueError("Output paths conflict with input artifacts")
    pinned = {"engine": artifact(engine), "model": identity, "corpus": artifact(corpus_path)}
    stamps = {path: file_stamp(path) for path in input_paths}
    source_files = [ROOT / "CMakeLists.txt", *sorted((ROOT / "src").glob("*.cpp")),
                    *sorted((ROOT / "include").glob("*.hpp"))]
    tool_files = sorted((ROOT / "tools").glob("*.py"))
    source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in source_files}
    tool_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in tool_files}
    locations = {engine: "$ENGINE", model: "$MODEL", corpus_path: "$CORPUS",
                 raw_dir: "$RAW", output: "$OUTPUT"}
    report = {"schema": "checkpoint-thread-invariance-v1", "status": "failure", "bitwise_invariant": False,
        "scope": "Within each kernel/KV pair across thread counts; not a quality or throughput measurement",
        "identity": pinned,
        "source_identity": {"files": source_hashes, "sha256": digest_json(source_hashes),
                            "relation_to_elf": "Current source identity only; not proof of ELF build provenance"},
        "tool_identity": {"entrypoint": "tools/check_threads.py", "files": tool_hashes, "sha256": digest_json(tool_hashes)},
        "workload": {"window_id": window["id"], "split": "calibration", "window_tokens_sha256": window["tokens_sha256"],
            "input_tokens": inputs, "input_tokens_sha256": digest_json(inputs), "logits_start": 0,
            "logit_positions": list(range(POSITIONS)), "shape": [POSITIONS, vocab], "dtype": "float32",
            "expected_bytes": POSITIONS * vocab * 4},
        "matrix": {"kernels": list(KERNELS), "kv_dtypes": list(KV_DTYPES), "threads": list(THREADS),
                   "cpu_order": cpus, "allowed_cpu_ids": allowed, "expected_processes": len(cells)},
        "runs": [], "pairs": [], "errors": []}
    raw_dir.mkdir(parents=True, exist_ok=False)
    baselines = {}
    for (kernel, kv, thread), prefix in zip(cells, prefixes, strict=True):
        requested = {**preflight, "kernel": kernel, "kv_dtype": kv, "threads": thread,
                     "cpu_set": cpus[:thread], "affinity": "strict", "scheduler": "pool",
                     "attention": "blocked", "rope": "cached"}
        command = [str(engine), "logits", "--model", str(model), "--tokens", ",".join(map(str, inputs)),
                   "--output", str(prefix), "--logits-start", "0", "--kernel", kernel, "--kv", kv,
                   "--threads", str(thread), "--cpu-set", ",".join(map(str, cpus[:thread])),
                   "--affinity", "strict", "--scheduler", "pool", "--attention", "blocked", "--rope", "cached"]
        record = {"id": prefix.name, "requested": requested, "command": command, "returncode": None,
                  "actual_metadata": None, "actual_settings": None, "validated": False,
                  "bitwise_equal_t1": None, "baseline_sha256": None, "artifacts": {}, "errors": []}
        stdout, stderr = prefix.with_suffix(".stdout.txt"), prefix.with_suffix(".stderr.txt")
        try:
            if any(file_stamp(path) != stamp for path, stamp in stamps.items()) or file_hash(engine) != pinned["engine"]["sha256"]:
                raise ValueError("Input artifact changed; native process not started")
            with stdout.open("x") as out, stderr.open("x") as err:
                # Blocking run waits for process exit, including nonzero exits.
                run = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=False)
            record["returncode"] = run.returncode
            if any(file_stamp(path) != stamp for path, stamp in stamps.items()) or file_hash(engine) != pinned["engine"]["sha256"]:
                raise ValueError("Input artifact changed during native process")
            if run.returncode != 0:
                raise ValueError(f"Native subprocess exited {run.returncode}")
            actual = json.loads(prefix.with_suffix(".json").read_text())
            record["actual_metadata"] = actual
            record["actual_settings"] = validate_actual(actual, requested, inputs, vocab, model, prefix.with_suffix(".bin"))
            record["validated"] = True
            record["number_logits"] = POSITIONS
            record["vocab"] = vocab
            record["byte_count"] = prefix.with_suffix(".bin").stat().st_size
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record["errors"].append(str(exc))
        for suffix in SUFFIXES:
            path = prefix.with_suffix(suffix)
            record["artifacts"][suffix] = artifact(path) if path.is_file() else None
        pair = (kernel, kv)
        if record["validated"]:
            digest = record["artifacts"][".bin"]["sha256"]
            if thread == 1:
                baselines[pair] = digest
            record["baseline_sha256"] = baselines.get(pair)
            record["bitwise_equal_t1"] = digest == baselines[pair] if pair in baselines else None
            if record["bitwise_equal_t1"] is not True:
                record["errors"].append("Whole-bin SHA256 differs from validated t1 baseline or baseline is unavailable")
        report["runs"].append(record)
    try:
        if model_identity(model, "native") != identity or file_hash(corpus_path) != pinned["corpus"]["sha256"]:
            report["errors"].append("Model/config/corpus identity changed during thread check")
    except (OSError, ValueError) as exc:
        report["errors"].append(f"Unable to recheck final input identity: {exc}")
    for kernel in KERNELS:
        for kv in KV_DTYPES:
            runs = [record for record in report["runs"] if record["requested"]["kernel"] == kernel and record["requested"]["kv_dtype"] == kv]
            report["pairs"].append({"kernel": kernel, "kv_dtype": kv, "baseline_threads": 1,
                "baseline_sha256": baselines.get((kernel, kv)),
                "bitwise_invariant": all(record["validated"] and record["bitwise_equal_t1"] is True and not record["errors"] for record in runs)})
    report["processes_completed"] = sum(record["returncode"] is not None for record in report["runs"])
    report["bitwise_invariant"] = not report["errors"] and all(pair["bitwise_invariant"] for pair in report["pairs"])
    report["status"] = "pass" if report["bitwise_invariant"] else "failure"
    report = portable(report, locations)
    write_json(output, report, exclusive=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-order", type=parse_cpu_order, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "results/v2/corpus.json")
    args = parser.parse_args(argv)
    try:
        report = check_threads(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"check_threads: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "bitwise_invariant": report["bitwise_invariant"],
                      "processes_completed": report["processes_completed"]}))
    return 0 if report["bitwise_invariant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
