#!/usr/bin/env python3
"""Prepare a pinned native CPU llama.cpp baseline from the same BF16 snapshot.

The downloaded clone is owned by this script and shared read-only by consumers.
Use --reuse-build to convert with an existing pinned build without rebuilding it.
The explicit 1.5B pin uses tools/streamed_gguf.py for bounded BF16 conversion;
Q8_0 still uses the pinned upstream llama-quantize executable.
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

from tools.download_model import MODEL_ID, S1_MODEL_ID, PINNED_MODELS, file_hash, verify_snapshot
from tools.portable import portable

# Resolved from GitHub's commits/master API on 2026-10-06, not a moving branch.
LLAMA_COMMIT = "6c73b3e12dc501de35fe5f6979960d06921a2f6c"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_CACHE = Path("external")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(map(str, command)), file=sys.stderr, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare(model: Path, cache: Path, jobs: int, output: Path, model_id: str = MODEL_ID, reuse_build: bool = False, quantize_buffer_mib: int | None = None) -> dict:
    if not 1 <= jobs <= 4:
        raise ValueError("Build/quantization jobs must be between 1 and 4")
    if quantize_buffer_mib is None and model_id != MODEL_ID:
        quantize_buffer_mib = 128
    if quantize_buffer_mib is not None and quantize_buffer_mib < 1:
        raise ValueError("Quantization buffer must be a positive MiB count")
    model = model.resolve()
    source_manifest = verify_snapshot(model, model_id=model_id)
    revision = source_manifest["revision"]
    root = cache.resolve() / "llama.cpp" / LLAMA_COMMIT
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = root / "source"
        if not source.exists() and reuse_build:
            raise ValueError("Pinned upstream source is missing; --reuse-build never fetches or builds")
        if not source.exists():
            run(["git", "init", str(source)])
            run(["git", "remote", "add", "origin", LLAMA_URL], source)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True)
        if commit.returncode or commit.stdout.strip() != LLAMA_COMMIT:
            if reuse_build:
                raise ValueError("Existing upstream source does not match the pinned llama.cpp commit")
            # Only this freshly downloaded cache clone is changed, never a user's checkout.
            run(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], source)
            run(["git", "checkout", "--detach", LLAMA_COMMIT], source)
        build = root / "build"
        if reuse_build:
            for name in ("llama-bench", "llama-quantize"):
                binary = build / "bin" / name
                if not binary.is_file() or not os.access(binary, os.X_OK):
                    raise ValueError(f"Missing existing upstream executable: {name}")
        else:
            run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_EXAMPLES=ON", "-DLLAMA_BUILD_TOOLS=ON"])
            run(["cmake", "--build", str(build), "--config", "Release", "--target", "llama-bench", "llama-quantize", "-j", str(jobs)])
        bf16 = root / f"qwen-{revision}-bf16.gguf"
        q8 = root / f"qwen-{revision}-q8_0.gguf"
        previous_path = root / f"preparation-{revision}.json"
        legacy_to_migrate = None
        legacy_path = root / "preparation.json"
        if not previous_path.exists() and (bf16.exists() or q8.exists()) and legacy_path.exists():
            previous = json.loads(legacy_path.read_text())
            legacy_to_migrate = legacy_path
        else:
            previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        if previous and (previous.get("source_model") != source_manifest or previous.get("llama_commit") != LLAMA_COMMIT):
            raise ValueError("Existing preparation manifest names a different source model or upstream commit")
        for path, dtype in [(bf16, "BF16"), (q8, "Q8_0")]:
            old = previous.get("artifacts", {}).get(dtype)
            if path.exists() and (old is None or file_hash(path) != old["sha256"]):
                raise ValueError(f"Unverified existing artifact: {path}; move it aside before preparing")
        # Retire the old name only after source, commit and existing bytes pass.
        if legacy_to_migrate is not None:
            legacy_to_migrate.replace(previous_path)
        if not bf16.exists():
            if model_id == S1_MODEL_ID:
                # This distinct path belongs to streamed attempts only. Never
                # touch the old converter's partial or any completed artifact.
                temporary = bf16.with_suffix(".streamed.partial.gguf")
                temporary.unlink(missing_ok=True)
                run([sys.executable, str(Path(__file__).with_name("streamed_gguf.py")),
                     "--source", str(source), "--model", str(model), "--model-id", model_id,
                     "--outfile", str(temporary), "--chunk-mib", "4"])
            else:
                temporary = bf16.with_suffix(".partial.gguf")
                run([sys.executable, str(source / "convert_hf_to_gguf.py"), str(model), "--outtype", "bf16", "--use-temp-file", "--outfile", str(temporary)])
            if model_id == S1_MODEL_ID:
                # Record the completed BF16 payload before promotion, so a Q8
                # failure never strands a real GGUF as an unverified artifact.
                checkpoint = {
                    "source_model": source_manifest, "llama_commit": LLAMA_COMMIT,
                    "artifacts": {
                        **previous.get("artifacts", {}),
                        "BF16": {"path": str(bf16), "sha256": file_hash(temporary),
                                 "bytes": temporary.stat().st_size},
                    },
                }
                checkpoint = portable(checkpoint, {cache.resolve(): "$CACHE", model: "$MODEL"})
                previous_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
            temporary.replace(bf16)
        if not q8.exists():
            temporary = q8.with_suffix(".partial.gguf")
            command = [str(build / "bin" / "llama-quantize")]
            if quantize_buffer_mib is not None:
                command += ["--max-buffer-size", str(quantize_buffer_mib)]
            run(command + [str(bf16), str(temporary), "Q8_0", str(jobs)])
            temporary.replace(q8)
        manifest = {
            "llama_repository": LLAMA_URL,
            "llama_commit": LLAMA_COMMIT,
            "source_model": source_manifest,
            "build": {"type": "Release", "native_cpu": True, "gpu": False, "jobs": jobs, "reused": reuse_build},
            "upstream_binaries": {name: {"sha256": file_hash(build / "bin" / name)} for name in ("llama-bench", "llama-quantize")},
            "converter_python": sys.executable,
            "bench_binary": str(build / "bin" / "llama-bench"),
            "quantization": "llama.cpp Q8_0, blocks of 32 weights; not the engine's per-output-channel int8 scheme",
            "quantization_buffer_mib": quantize_buffer_mib,
            "artifacts": {dtype: {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size} for path, dtype in [(bf16, "BF16"), (q8, "Q8_0")]},
        }
        manifest = portable(manifest, {cache.resolve(): "$CACHE", model.resolve(): "$MODEL", sys.executable: "$PYTHON"})
        previous_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=tuple(PINNED_MODELS), default=MODEL_ID)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reuse-build", action="store_true", help="Require the existing pinned source and binaries; never fetch, configure or build")
    parser.add_argument("--quantize-buffer-mib", type=int, help="Upstream --max-buffer-size in MiB; defaults to upstream's original buffer for 0.5B and 128 for 1.5B")
    parser.add_argument("--output", type=Path, help="Defaults to results/llama-preparation.json for 0.5B or results/v2/s1/llama-preparation.json for 1.5B")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("results/llama-preparation.json" if args.model_id == MODEL_ID else "results/v2/s1/llama-preparation.json")
    print(json.dumps(prepare(args.model, args.cache, args.jobs, args.output, model_id=args.model_id, reuse_build=args.reuse_build, quantize_buffer_mib=args.quantize_buffer_mib), indent=2))


if __name__ == "__main__":
    main()
