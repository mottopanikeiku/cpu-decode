"""Run one timing slice and record portable commands plus individual repeats."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from tools.portable import portable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENS = ",".join(map(str, json.loads((ROOT / "configs" / "prompts.json").read_text())["benchmark"]["seed_token_ids"]))


def execute(command: list[str], destination: Path, locations: dict) -> dict | list:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    destination.with_suffix(".stderr.txt").write_text(portable(completed.stderr, locations))
    if completed.returncode:
        failure = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        destination.with_suffix(".failure.json").write_text(json.dumps(portable(failure, locations), indent=2) + "\n")
        completed.check_returncode()
    data = portable(json.loads(completed.stdout), locations)
    destination.write_text(json.dumps(data, indent=2) + "\n")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["bandwidth", "engine", "llama", "eager"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--llama", type=Path)
    parser.add_argument("--threads", default="1,2,4,6,12")
    parser.add_argument("--contexts", default="128,1024,4096")
    parser.add_argument("--kernels")
    parser.add_argument("--rope", choices=["cached", "direct"], default="cached")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "measurements")
    args = parser.parse_args()
    if args.kernels is None:
        args.kernels = "simd512" if args.stage == "bandwidth" else "simd512x4"
    if args.stage != "bandwidth" and args.model is None:
        parser.error("--model is required for this stage")
    if args.stage == "llama" and args.llama is None:
        parser.error("--llama is required")
    threads = [int(x) for x in args.threads.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    kernels = args.kernels.split(",")
    args.output.mkdir(parents=True, exist_ok=True)
    locations = {}
    if args.model is not None:
        locations[args.model.resolve()] = "$MODEL"
    if args.llama is not None:
        locations[args.llama.resolve()] = "$LLAMA_BENCH"
    environment = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "compiler": subprocess.check_output(["g++", "--version"], text=True),
        "python": sys.version,
        "libraries": {name: version(name) for name in ["torch", "transformers", "numpy", "tokenizers"]},
        "command": sys.argv,
        "environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY"]},
    }
    records = []
    for thread in threads:
        for context in ([0] if args.stage == "bandwidth" else contexts):
            for kernel in (kernels if args.stage in {"bandwidth", "engine"} else ["baseline"]):
                name = f"{args.stage}-t{thread}-c{context}-{kernel}"
                path = args.output / f"{name}.json"
                if args.stage == "bandwidth":
                    command = [str(ROOT / "build" / "read-bandwidth"), "--threads", str(thread),
                               "--kernel", kernel, "--repeats", str(args.repeats)]
                elif args.stage == "engine":
                    command = [str(ROOT / "build" / "cpu-decode"), "bench", "--model", str(args.model),
                               "--threads", str(thread), "--context", str(context), "--kernel", kernel,
                               "--steps", str(args.steps), "--repeats", str(args.repeats), "--tokens", args.tokens]
                    command += ["--rope", args.rope]
                elif args.stage == "llama":
                    command = [str(args.llama), "-m", str(args.model), "-p", "0", "-n", str(args.steps),
                               "-d", str(context), "-t", str(thread), "-r", str(args.repeats),
                               "-ngl", "0", "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-o", "json"]
                else:
                    command = [sys.executable, str(ROOT / "tools" / "benchmark_eager.py"),
                               "--model", str(args.model), "--threads", str(thread), "--context", str(context),
                               "--steps", str(args.steps), "--repeats", str(args.repeats), "--tokens", args.tokens]
                data = execute(command, path, locations)
                records.append({"name": name, "command": command, "file": str(path.relative_to(ROOT)), "data": data})
                print(name, flush=True)
    environment_name = f"environment-{args.stage}-t{args.threads}-c{args.contexts}-k{args.kernels}-rope{args.rope}.json"
    (args.output / environment_name).write_text(
        json.dumps(portable({"environment": environment, "records": [{k: v for k, v in x.items() if k != "data"} for x in records]}, locations), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
