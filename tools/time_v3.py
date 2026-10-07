"""Run later on an idle machine: matched plain/lookup greedy decoding timings.

No performance observations are shipped with the feature. This script measures
prefill and decode separately in the native process after an untimed warmup.
"""
import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path

from tools.download_model import file_hash
from tools.kv_quality_v3 import save


def spread(values):
    if not values or any(value <= 0 for value in values):
        raise ValueError("Expected positive timing samples")
    return {"median_seconds": statistics.median(values), "min_seconds": min(values),
            "max_seconds": max(values), "samples_seconds": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--inputs", type=Path, default=Path("results/v3/lookup-inputs.json"))
    parser.add_argument("--output", type=Path, default=Path("results/v3/timing.json"))
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kv", choices=("f16", "f32", "i8", "i8-centered"), default="f16")
    parser.add_argument("--kernel", default="vnni16")
    parser.add_argument("--lookup", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 2 or args.steps < 2 or args.repeats < 2:
        parser.error("use 1..2 threads, at least two steps and at least two repeats")
    cases = json.loads(args.inputs.read_text())["prompts"]
    if not 0 <= args.case < len(cases):
        parser.error("case index outside input suite")
    case = cases[args.case]
    # Both decode conditions use the same batched prefill. A third condition
    # isolates prefill batching with lookup disabled.
    conditions = (("single-prefill", 0, 1), ("batched-prefill", 0, 8), ("lookup", args.lookup, 8))
    result = {"case": case["id"], "category": case["category"], "inputs_sha256": file_hash(args.inputs),
              "binary_sha256": file_hash(args.engine), "weights_sha256": file_hash(args.model / "model.safetensors"),
              "platform": platform.platform(), "cpu": platform.processor(), "threads": args.threads,
              "scope": "Model load excluded. Prefill and decode timed separately. First token from prefill; final generated token not forwarded. One full-generation warmup per process. Requires exclusive quiet-machine access.",
              "conditions": {}}
    expected = None
    # Alternate condition order across fresh processes to expose order effects.
    for repeat in range(args.repeats):
        order = conditions if repeat % 2 == 0 else tuple(reversed(conditions))
        for label, lookup, prefill in order:
            command = [str(args.engine.resolve()), "time-generate", "--model", str(args.model.resolve()),
                       "--tokens", ",".join(map(str, case["tokens"])), "--steps", str(args.steps),
                       "--repeats", "1", "--threads", str(args.threads), "--kernel", args.kernel,
                       "--kv", args.kv, "--lookup", str(lookup), "--ngram", "4", "--prefill-batch", str(prefill)]
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            data = json.loads(completed.stdout)
            sample = data["samples"][0]
            if expected is None:
                expected = sample["generated_tokens"]
            if sample["generated_tokens"] != expected:
                raise ValueError("Timed paths produced different greedy tokens")
            item = result["conditions"].setdefault(label, {"lookup": lookup, "prefill_batch": prefill,
                                                         "samples": [], "kv_cache_bytes": data["kv_cache_bytes"]})
            item["samples"].append(sample)
    for item in result["conditions"].values():
        item["prefill"] = spread([sample["prefill_seconds"] for sample in item["samples"]])
        item["decode"] = spread([sample["decode_seconds"] for sample in item["samples"]])
    save(args.output, result)


if __name__ == "__main__":
    main()
