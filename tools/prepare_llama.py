#!/usr/bin/env python3
"""Prepare a pinned native CPU llama.cpp baseline from the same BF16 snapshot.

This script builds and converts artifacts; it does not measure performance.
The downloaded clone is owned by this script and shared read-only by consumers.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from tools.download_model import REVISION, file_hash, verify_snapshot
from tools.portable import portable

# Resolved from GitHub's commits/master API on 2026-10-06, not a moving branch.
LLAMA_COMMIT = "6c73b3e12dc501de35fe5f6979960d06921a2f6c"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_CACHE = Path("external")
QUANTIZED = ["Q8_0", "Q4_0"]
# Read with the pinned clone's own gguf-py so the stored type names match llama.cpp's.
TENSOR_TYPES = """
import collections, json, sys
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
tensors = GGUFReader(sys.argv[2]).tensors
counts = collections.Counter(t.tensor_type.name for t in tensors)
print(json.dumps({"counts": dict(sorted(counts.items())), "token_embd.weight": next(t.tensor_type.name for t in tensors if t.name == "token_embd.weight")}))
"""


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(map(str, command)), file=sys.stderr, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def tensor_types(source: Path, path: Path) -> dict:
    result = subprocess.run([sys.executable, "-c", TENSOR_TYPES, str(source / "gguf-py"), str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def prepare(model: Path, cache: Path, jobs: int, output: Path) -> dict:
    if not 1 <= jobs <= 4:
        raise ValueError("Build/quantization jobs must be between 1 and 4")
    model = model.resolve()
    source_manifest = verify_snapshot(model)
    root = cache.resolve() / "llama.cpp" / LLAMA_COMMIT
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = root / "source"
        if not source.exists():
            run(["git", "init", str(source)])
            run(["git", "remote", "add", "origin", LLAMA_URL], source)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True)
        if commit.returncode or commit.stdout.strip() != LLAMA_COMMIT:
            # Only this freshly downloaded cache clone is changed, never a user's checkout.
            run(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], source)
            run(["git", "checkout", "--detach", LLAMA_COMMIT], source)
        build = root / "build"
        run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_EXAMPLES=ON", "-DLLAMA_BUILD_TOOLS=ON"])
        # The shared llama library is linked by this repository's llama-logits reader.
        run(["cmake", "--build", str(build), "--config", "Release", "--target", "llama", "llama-bench", "llama-quantize", "-j", str(jobs)])
        bf16 = root / f"qwen-{REVISION}-bf16.gguf"
        quantized = [(root / f"qwen-{REVISION}-{dtype.lower()}.gguf", dtype) for dtype in QUANTIZED]
        artifacts = [(bf16, "BF16"), *quantized]
        previous_path = root / "preparation.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        for path, dtype in artifacts:
            old = previous.get("artifacts", {}).get(dtype)
            if path.exists() and (old is None or file_hash(path) != old["sha256"]):
                raise ValueError(f"Unverified existing artifact: {path}; move it aside before preparing")
        if not bf16.exists():
            temporary = bf16.with_suffix(".partial.gguf")
            run([sys.executable, str(source / "convert_hf_to_gguf.py"), str(model), "--outtype", "bf16", "--use-temp-file", "--outfile", str(temporary)])
            temporary.replace(bf16)
        for path, dtype in quantized:
            if not path.exists():
                temporary = path.with_suffix(".partial.gguf")
                # Default llama-quantize tensor mix for the type; the recorded tensor types show any exceptions.
                run([str(build / "bin" / "llama-quantize"), str(bf16), str(temporary), dtype, str(jobs)])
                temporary.replace(path)
        manifest = {
            "llama_repository": LLAMA_URL,
            "llama_commit": LLAMA_COMMIT,
            "source_model": source_manifest,
            "build": {"type": "Release", "native_cpu": True, "gpu": False, "jobs": jobs},
            "converter_python": sys.executable,
            "bench_binary": str(build / "bin" / "llama-bench"),
            "quantization": {"Q8_0": "llama.cpp Q8_0: int8 weights in blocks of 32 with one F16 scale per block",
                             "Q4_0": "llama.cpp Q4_0 default mix: 4-bit weights in blocks of 32 with one F16 scale per block; see tensor_types for tensors llama-quantize keeps at another type"},
            "artifacts": {dtype: {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size,
                                  "tensor_types": tensor_types(source, path)} for path, dtype in artifacts},
        }
        manifest = portable(manifest, {cache.resolve(): "$CACHE", model.resolve(): "$MODEL", sys.executable: "$PYTHON"})
        previous_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/llama-preparation.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.model, args.cache, args.jobs, args.output), indent=2))


if __name__ == "__main__":
    main()
