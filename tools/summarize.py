"""Build all reported tables from committed raw JSON measurements."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def spread(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("empty measurement samples")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def load(path: Path) -> dict | list:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/measurements"))
    parser.add_argument("--ablations", type=Path, default=Path("results/ablations"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--kernel", default="simd512x4")
    args = parser.parse_args()
    bandwidth = {}
    bandwidth_rows = []
    for path in sorted(args.input.glob("bandwidth-t*-c0-*.json")):
        raw = load(path)
        stats = spread([x["GB_per_s"] for x in raw["samples"]])
        row = {"threads": raw["threads"], "kernel": raw["kernel"], **stats, "file": str(path)}
        bandwidth_rows.append(row)
        if raw["threads"] not in bandwidth or stats["median"] > bandwidth[raw["threads"]]["median"]:
            bandwidth[raw["threads"]] = row
    results = []
    for path in sorted(args.input.glob(f"engine-t*-c*-{args.kernel}.json")):
        raw = load(path)
        thread, context = raw["threads"], raw["context"]
        rate = spread([x["tokens_per_second"] for x in raw["samples"]])
        bytes_per_token = {key: statistics.mean(x["bytes_per_token"][key] for x in raw["samples"])
                           for key in raw["samples"][0]["bytes_per_token"]}
        bw = bandwidth[thread]
        ceiling = bw["median"] * 1e9 / bytes_per_token["total_min"]
        llama_path = args.input / f"llama-t{thread}-c{context}-baseline.json"
        eager_path = args.input / f"eager-t{thread}-c{context}-baseline.json"
        if not llama_path.exists() or not eager_path.exists():
            raise FileNotFoundError(f"Missing matched baseline for threads={thread}, context={context}")
        llama = load(llama_path)
        matches = [r for r in llama if r["n_threads"] == thread and r["n_depth"] == context and r["n_gen"] == raw["steps"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one matching llama test in {llama_path}")
        llama_rate = spread(matches[0]["samples_ts"])
        eager = load(eager_path)
        expected = {"threads": thread, "context": context, "steps": raw["steps"], "seed_token_ids": raw["prompt_tokens"]}
        if any(eager.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Eager settings do not match engine inputs in {eager_path}")
        eager_rate = spread([r["tokens_per_second"] for r in eager["samples"]])
        operation_medians = {key: statistics.median(s["operation_seconds"][key] / raw["steps"] for s in raw["samples"])
                             for key in raw["samples"][0]["operation_seconds"]}
        results.append({"threads": thread, "context": context, "kernel": raw["kernel"],
                        "engine_tps": rate, "llama_tps": llama_rate, "eager_tps": eager_rate,
                        "read_GB_per_s": bw["median"], "bandwidth_kernel": bw["kernel"],
                        "bandwidth_ceiling_tps": ceiling, "percent_of_ceiling": 100*rate["median"]/ceiling,
                        "engine_over_llama": rate["median"]/llama_rate["median"],
                        "bytes_per_token": bytes_per_token, "operation_seconds_per_token": operation_medians,
                        "files": [str(path), str(llama_path), str(eager_path), bw["file"]]})
    results.sort(key=lambda r: (r["threads"], r["context"]))
    ablations = []
    for path in sorted(args.ablations.rglob("engine-t*-c*-*.json")):
        raw = load(path)
        ablations.append({"threads": raw["threads"], "context": raw["context"], "kernel": raw["kernel"],
                          "weight_dtype": raw["weight_dtype"], "rope": raw["rope"],
                          "tokens_per_second": spread([s["tokens_per_second"] for s in raw["samples"]]), "file": str(path)})
    effects = []
    for thread, context in sorted({(r["threads"], r["context"]) for r in ablations}):
        variants = {(r["weight_dtype"], r["kernel"], r["rope"]): r for r in ablations
                    if r["threads"] == thread and r["context"] == context}
        pairs = [
            ("Int8 weights", ("bf16", "scalar", "cached"), ("int8", "scalar", "cached")),
            ("SIMD256", ("int8", "scalar", "cached"), ("int8", "simd256", "cached")),
            ("SIMD512 instead of SIMD256", ("int8", "simd256", "cached"), ("int8", "simd512", "cached")),
            ("Four SIMD512 accumulators", ("int8", "simd512", "cached"), ("int8", "simd512x4", "cached")),
            ("Cached RoPE", ("int8", "simd512x4", "direct"), ("int8", "simd512x4", "cached")),
        ]
        for change, before, after in pairs:
            if before in variants and after in variants:
                a, b = variants[before], variants[after]
                effects.append({"change": change, "threads": thread, "context": context,
                                "before": a, "after": b, "ratio": b["tokens_per_second"]["median"] / a["tokens_per_second"]["median"]})
    summary = {"statistic": "median; spread is min/max over repeated runs",
               "bandwidth_selection": "highest median of measured read SIMD widths at each thread count",
               "bandwidth": bandwidth_rows, "results": results, "ablations": ablations, "optimization_effects": effects}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["threads", "context", "engine_tps", "engine_min", "engine_max", "percent_of_ceiling", "llama_tps", "eager_tps", "engine_over_llama"])
        for r in results:
            writer.writerow([r["threads"], r["context"], r["engine_tps"]["median"], r["engine_tps"]["min"], r["engine_tps"]["max"],
                             r["percent_of_ceiling"], r["llama_tps"]["median"], r["eager_tps"]["median"], r["engine_over_llama"]])
    lines = ["# Generated result tables", "", "Source: `tools/summarize.py`. Rates are median tokens/s; engine spread is min–max.", "",
             "| Threads | Context | Engine (range) | % read ceiling | llama.cpp Q8_0 | PyTorch eager | Engine / llama |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        rate = r["engine_tps"]
        lines.append(f"| {r['threads']} | {r['context']} | {rate['median']:.2f} ({rate['min']:.2f}–{rate['max']:.2f}) | {r['percent_of_ceiling']:.1f}% | {r['llama_tps']['median']:.2f} | {r['eager_tps']['median']:.2f} | {r['engine_over_llama']:.2f}× |")
    lines += ["", "## Ablation variants", "", "| Threads | Context | Weights | Kernel | RoPE | Median tokens/s |",
              "|---:|---:|---|---|---|---:|"]
    for r in ablations:
        lines.append(f"| {r['threads']} | {r['context']} | {r['weight_dtype']} | {r['kernel']} | {r['rope']} | {r['tokens_per_second']['median']:.2f} |")
    lines += ["", "## Paired optimization effects", "", "| Change | Before tokens/s | After tokens/s | Ratio |",
              "|---|---:|---:|---:|"]
    for r in effects:
        lines.append(f"| {r['change']} | {r['before']['tokens_per_second']['median']:.2f} | {r['after']['tokens_per_second']['median']:.2f} | {r['ratio']:.2f}× |")
    (args.output / "tables.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
