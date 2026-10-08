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
BANDWIDTH_KERNELS = {"x86_64": "simd256,simd512", "aarch64": "neon"}
OMP_KEYS = ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY"]
# The pinned llama.cpp casts an F32 K/V cache to F16 whenever flash attention is on
# (src/llama-graph.cpp:2727-2733), so F32 KV runs must disable it to stay F32.
LLAMA_FLASH_ATTENTION = {"f16": "auto", "f32": "0"}


def execute(command: list[str], destination: Path, locations: dict, env: dict[str, str]) -> dict | list:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env)
    destination.with_suffix(".stderr.txt").write_text(portable(completed.stderr, locations))
    if completed.returncode:
        failure = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        destination.with_suffix(".failure.json").write_text(json.dumps(portable(failure, locations), indent=2) + "\n")
        completed.check_returncode()
    data = portable(json.loads(completed.stdout), locations)
    destination.write_text(json.dumps(data, indent=2) + "\n")
    return data


def fast_places(listing: str) -> tuple[str, dict]:
    """One OpenMP place per physical core with the highest MAXMHZ, holding all its SMT siblings."""
    lines = [line.split() for line in listing.splitlines() if line.strip()]
    if not lines or lines[0] != ["CPU", "CORE", "MAXMHZ"]:
        raise ValueError(f"Unexpected `lscpu -e=CPU,CORE,MAXMHZ` header: {lines[0] if lines else 'empty output'}")
    rows = lines[1:]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("`lscpu -e=CPU,CORE,MAXMHZ` returned malformed rows")
    cores: dict[int, list[int]] = {}
    speeds: dict[int, set[str]] = {}
    for cpu, core, mhz in rows:
        if core == "-":
            continue  # offline CPU
        cores.setdefault(int(core), []).append(int(cpu))
        speeds.setdefault(int(core), set()).add(mhz)
    if not cores:
        raise ValueError("lscpu reported no online cores")
    reported = {mhz for values in speeds.values() for mhz in values}
    if reported == {"-"}:
        # Frequency limits are not exposed (common in VMs): no core can be ranked
        # above another, so every physical core counts as fast. Recorded explicitly.
        selected = sorted(cores)
        maximum = None
    elif "-" in reported:
        raise ValueError("lscpu reports MAXMHZ for only some CPUs; cannot select the fastest cores (use --places cores)")
    else:
        if any(len(values) != 1 for values in speeds.values()):
            raise ValueError("SMT siblings of one core report different MAXMHZ values")
        top = {core: float(next(iter(values))) for core, values in speeds.items()}
        maximum = max(top.values())
        selected = sorted(core for core, mhz in top.items() if mhz == maximum)
    places = ",".join("{" + ",".join(map(str, sorted(cores[core]))) + "}" for core in selected)
    return places, {"max_mhz": maximum, "selected_cores": selected, "maxmhz_available": maximum is not None}


def pinning(mode: str) -> tuple[dict[str, str], dict]:
    env = dict(os.environ)
    details: dict = {}
    if mode == "cores":
        env.update(OMP_PROC_BIND="close", OMP_PLACES="cores")
    elif mode == "fast":
        listing = subprocess.check_output(["lscpu", "-e=CPU,CORE,MAXMHZ"], text=True)
        places, details = fast_places(listing)
        details["lscpu_cpu_core_maxmhz"] = listing
        env.update(OMP_PROC_BIND="close", OMP_PLACES=places)
    return env, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["bandwidth", "engine", "llama"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--llama", type=Path, help="llama-bench executable (llama stage)")
    parser.add_argument("--threads", default="1,2,4,6,12")
    parser.add_argument("--contexts", default="128,1024,4096")
    parser.add_argument("--kernels", help="bandwidth: simd256,simd512 (x86-64) or neon (aarch64); engine: auto")
    parser.add_argument("--kv", choices=["f16", "f32"], default="f16")
    parser.add_argument("--weights", choices=["hugepage", "mmap"], help="engine weight memory (default hugepage)")
    parser.add_argument("--fuse", choices=["on", "off"], help="engine projection fusion (default on)")
    parser.add_argument("--label", help="file label; engine default {model dir name}-{kv}, required for llama")
    parser.add_argument("--places", choices=["cores", "fast", "none"], default="cores", help="OpenMP thread pinning for every stage")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "measurements")
    args = parser.parse_args()
    if args.stage == "bandwidth":
        for name in ["model", "llama", "weights", "fuse", "label"]:
            if getattr(args, name) is not None:
                parser.error(f"--{name} does not apply to the bandwidth stage")
        if args.kernels is None:
            machine = platform.machine()
            if machine not in BANDWIDTH_KERNELS:
                parser.error(f"no default bandwidth kernels for {machine}; pass --kernels")
            args.kernels = BANDWIDTH_KERNELS[machine]
    else:
        if args.model is None:
            parser.error("--model is required for this stage")
    if args.stage == "engine":
        if args.llama is not None:
            parser.error("--llama applies only to the llama stage")
        args.kernels = args.kernels or "auto"
        args.weights = args.weights or "hugepage"
        args.fuse = args.fuse or "on"
        args.label = args.label or f"{args.model.resolve().name}-{args.kv}"
    if args.stage == "llama":
        for name in ["kernels", "weights", "fuse"]:
            if getattr(args, name) is not None:
                parser.error(f"--{name} does not apply to the llama stage")
        if args.llama is None:
            parser.error("--llama is required")
        if args.label is None:
            parser.error("--label is required for the llama stage (e.g. q8_0-f16)")
    if min(args.steps, args.repeats) < 1:
        parser.error("--steps and --repeats must be positive")
    threads = [int(x) for x in args.threads.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    kernels = args.kernels.split(",") if args.kernels else [None]
    args.output.mkdir(parents=True, exist_ok=True)
    locations = {args.output.resolve(): "$OUTPUT"}
    if args.model is not None:
        locations[args.model.resolve()] = "$MODEL"
    if args.llama is not None:
        locations[args.llama.resolve()] = "$LLAMA_BENCH"
    env, places = pinning(args.places)
    environment = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "cpu_list": subprocess.check_output(["lscpu", "-e"], text=True),
        "compiler": subprocess.check_output(["g++", "--version"], text=True),
        "python": sys.version,
        "libraries": {name: version(name) for name in ["torch", "transformers", "numpy", "tokenizers"]},
        "command": sys.argv,
        "places": args.places,
        "places_details": places,
        "environment": {key: env.get(key) for key in OMP_KEYS},
        "label": args.label,
        "kv": args.kv if args.stage != "bandwidth" else None,
        "weights": args.weights,
        "fuse": args.fuse,
    }
    records = []
    for thread in threads:
        for context in ([0] if args.stage == "bandwidth" else contexts):
            for kernel in kernels:
                if args.stage == "bandwidth":
                    name = f"bandwidth-t{thread}-c0-{kernel}"
                    command = [str(ROOT / "build" / "read-bandwidth"), "--threads", str(thread),
                               "--kernel", kernel, "--repeats", str(args.repeats)]
                elif args.stage == "engine":
                    name = f"engine-{args.label}-t{thread}-c{context}" + (f"-{kernel}" if len(kernels) > 1 else "")
                    command = [str(ROOT / "build" / "cpu-decode"), "bench", "--model", str(args.model),
                               "--threads", str(thread), "--context", str(context), "--kernel", kernel,
                               "--kv", args.kv, "--weights", args.weights, "--fuse", args.fuse,
                               "--steps", str(args.steps), "--repeats", str(args.repeats), "--tokens", args.tokens]
                else:
                    name = f"llama-{args.label}-t{thread}-c{context}"
                    command = [str(args.llama), "-m", str(args.model), "-p", "0", "-n", str(args.steps),
                               "-d", str(context), "-t", str(thread), "-r", str(args.repeats),
                               "-ngl", "0", "-ctk", args.kv, "-ctv", args.kv, "-fa", LLAMA_FLASH_ATTENTION[args.kv], "-o", "json"]
                path = args.output / f"{name}.json"
                execute(command, path, locations, env)
                records.append({"name": name, "command": command, "file": str(path)})
                print(name, flush=True)
    parts = [args.stage] + ([args.label] if args.label else []) + [f"t{args.threads}"]
    if args.stage != "bandwidth":
        parts.append(f"c{args.contexts}")
    if args.stage == "engine":
        parts += [f"k{args.kernels}", args.weights, f"fuse{args.fuse}"]
    elif args.stage == "bandwidth":
        parts.append(f"k{args.kernels}")
    parts.append(f"places{args.places}")
    (args.output / f"environment-{'-'.join(parts)}.json").write_text(
        json.dumps(portable({"environment": environment, "records": records}, locations), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
