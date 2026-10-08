"""Build all reported tables from committed raw JSON measurements."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

# Engine label -> llama.cpp label measured with the same KV dtype; the first is the primary pair.
PAIRS = [("q8-f16", "q8_0-f16"), ("q4h8-f16", "q4_0-f16"), ("q8-f32", "q8_0-f32"), ("q4-f16", "q4_0-f16")]
# Ablation fields, each with values ordered from "before" to "after" (the engine's optimized choice last).
EFFECT_FIELDS = {
    "fused_projections": [False, True],
    "weight_memory": ["mmap", "hugepage"],
    "kv_dtype": ["f32", "f16"],
    "weight_format": ["q8", "q4"],
    "head_format": ["q8", "q4"],
    "kernel": ["scalar", "neon", "avx512"],
}
# llama-bench flash_attn per KV dtype: -1 = auto for F16; 0 = disabled for F32, because the
# pinned llama.cpp casts an F32 K/V cache to F16 whenever flash attention is on.
LLAMA_FLASH_ATTENTION = {"f16": -1, "f32": 0}


def spread(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("empty measurement samples")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def load(path: Path) -> dict | list:
    return json.loads(path.read_text())


def labelled(directory: Path, prefix: str, label: str) -> dict[tuple[int, int], Path]:
    pattern = re.compile(rf"{re.escape(prefix)}-{re.escape(label)}-t(\d+)-c(\d+)\.json")
    found = {}
    for path in sorted(directory.glob(f"{prefix}-{label}-t*-c*.json")):
        match = pattern.fullmatch(path.name)
        if match:
            found[int(match[1]), int(match[2])] = path
    return found


def engine_record(path: Path, threads: int, context: int) -> dict:
    raw = load(path)
    if raw["threads"] != threads or raw["context"] != context:
        raise ValueError(f"{path} records threads={raw['threads']}, context={raw['context']}, not its file name")
    if raw.get("warmup_steps") != 1 or len(raw["samples"]) != raw["repeats"]:
        raise ValueError(f"Expected one warmup step and {raw['repeats']} samples in {path}")
    return raw


def llama_rate(path: Path, engine: dict, label: str) -> dict[str, float]:
    tests = load(path)
    matches = [t for t in tests if t["n_threads"] == engine["threads"] and t["n_depth"] == engine["context"] and t["n_gen"] == engine["steps"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one llama.cpp test with n_threads={engine['threads']}, n_depth={engine['context']}, n_gen={engine['steps']} in {path}")
    test = matches[0]
    kv = (test.get("type_k"), test.get("type_v"))
    if kv != (engine["kv_dtype"], engine["kv_dtype"]):
        raise ValueError(f"llama.cpp KV types {kv[0]}/{kv[1]} do not match engine kv_dtype {engine['kv_dtype']} in {path}")
    if test.get("n_gpu_layers") != 0:
        raise ValueError(f"Expected CPU-only llama.cpp run (n_gpu_layers 0) in {path}")
    if test.get("flash_attn") != LLAMA_FLASH_ATTENTION[engine["kv_dtype"]]:
        raise ValueError(f"Expected llama.cpp flash_attn {LLAMA_FLASH_ATTENTION[engine['kv_dtype']]} for {engine['kv_dtype']} KV, found {test.get('flash_attn')} in {path}")
    quant = label.split("-")[0].upper()
    if quant not in test.get("model_type", ""):
        raise ValueError(f"llama.cpp model_type {test.get('model_type')!r} does not contain {quant} in {path}")
    if len(test["samples_ts"]) != engine["repeats"]:
        raise ValueError(f"llama.cpp repetitions do not match engine repeats in {path}")
    return spread(test["samples_ts"])


def tokens_per_second(raw: dict) -> dict[str, float]:
    return spread([s["tokens_per_second"] for s in raw["samples"]])


def matched(engine_files: dict, llama_files: dict, engine_label: str, llama_label: str) -> list[dict]:
    rows = []
    for (thread, context), path in sorted(engine_files.items()):
        if (thread, context) not in llama_files:
            continue
        raw = engine_record(path, thread, context)
        engine, llama = tokens_per_second(raw), llama_rate(llama_files[thread, context], raw, llama_label)
        rows.append({"engine_label": engine_label, "llama_label": llama_label, "threads": thread, "context": context,
                     "engine_tps": engine, "llama_tps": llama, "engine_over_llama": engine["median"] / llama["median"],
                     "files": [str(path), str(llama_files[thread, context])]})
    return rows


def effect_key(field: str, value) -> tuple:
    order = EFFECT_FIELDS[field]
    return (order.index(value), "") if value in order else (len(order), str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/measurements"))
    parser.add_argument("--ablations", type=Path, default=Path("results/ablations"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--engine-label", default="q8-f16")
    parser.add_argument("--llama-label", default="q8_0-f16")
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
    engine_files = labelled(args.input, "engine", args.engine_label)
    llama_files = labelled(args.input, "llama", args.llama_label)
    if not engine_files:
        raise FileNotFoundError(f"No engine-{args.engine_label}-t*-c*.json in {args.input}")
    results = []
    for (thread, context), path in sorted(engine_files.items()):
        raw = engine_record(path, thread, context)
        if (thread, context) not in llama_files:
            raise FileNotFoundError(f"Missing llama-{args.llama_label}-t{thread}-c{context}.json for the primary pair")
        if thread not in bandwidth:
            raise FileNotFoundError(f"Missing bandwidth measurement for threads={thread}")
        rate = tokens_per_second(raw)
        llama = llama_rate(llama_files[thread, context], raw, args.llama_label)
        bytes_per_token = {key: statistics.mean(x["bytes_per_token"][key] for x in raw["samples"])
                           for key in raw["samples"][0]["bytes_per_token"]}
        bw = bandwidth[thread]
        ceiling = bw["median"] * 1e9 / bytes_per_token["total_min"]
        operation_medians = {key: statistics.median(s["operation_seconds"][key] / raw["steps"] for s in raw["samples"])
                             for key in raw["samples"][0]["operation_seconds"]}
        results.append({"threads": thread, "context": context, "engine_label": args.engine_label, "llama_label": args.llama_label,
                        "kernel": raw["kernel"], "kv_dtype": raw["kv_dtype"], "steps": raw["steps"],
                        "engine_tps": rate, "llama_tps": llama,
                        "read_GB_per_s": bw["median"], "bandwidth_kernel": bw["kernel"],
                        "bandwidth_ceiling_tps": ceiling, "percent_of_ceiling": 100 * rate["median"] / ceiling,
                        "engine_over_llama": rate["median"] / llama["median"],
                        "bytes_per_token": bytes_per_token, "operation_seconds_per_token": operation_medians,
                        "files": [str(path), str(llama_files[thread, context]), bw["file"]]})
    pairs = []
    for engine_label, llama_label in PAIRS:
        pairs += matched(labelled(args.input, "engine", engine_label), labelled(args.input, "llama", llama_label), engine_label, llama_label)
    ablations = []
    for path in sorted(args.ablations.rglob("engine-*.json")):
        raw = load(path)
        if raw.get("warmup_steps") != 1 or len(raw["samples"]) != raw["repeats"]:
            raise ValueError(f"Expected one warmup step and {raw['repeats']} samples in {path}")
        ablations.append({"threads": raw["threads"], "context": raw["context"], "steps": raw["steps"],
                          **{field: raw[field] for field in EFFECT_FIELDS},
                          "tokens_per_second": tokens_per_second(raw), "file": str(path)})
    ablations.sort(key=lambda r: (r["threads"], r["context"], *(effect_key(f, r[f]) for f in EFFECT_FIELDS), r["file"]))
    effects = []
    for thread, context in sorted({(r["threads"], r["context"]) for r in ablations}):
        group = [r for r in ablations if r["threads"] == thread and r["context"] == context]
        seen = {}
        for r in group:
            key = (r["steps"], *(r[f] for f in EFFECT_FIELDS))
            if key in seen:
                raise ValueError(f"Duplicate ablation variant: {seen[key]} and {r['file']}")
            seen[key] = r["file"]
        for field in EFFECT_FIELDS:
            for a in group:
                for b in group:
                    others = [f for f in EFFECT_FIELDS if f != field]
                    if a["steps"] != b["steps"] or any(a[f] != b[f] for f in others):
                        continue
                    if effect_key(field, a[field]) >= effect_key(field, b[field]):
                        continue
                    effects.append({"field": field, "change": f"{field}: {a[field]} → {b[field]}", "threads": thread, "context": context,
                                    "variant": {f: a[f] for f in others}, "before": a, "after": b,
                                    "ratio": b["tokens_per_second"]["median"] / a["tokens_per_second"]["median"]})
    effects.sort(key=lambda e: (e["threads"], e["context"], list(EFFECT_FIELDS).index(e["field"]),
                                effect_key(e["field"], e["before"][e["field"]]), effect_key(e["field"], e["after"][e["field"]]),
                                e["before"]["file"], e["after"]["file"]))
    summary = {"statistic": "median; spread is min/max over repeated runs",
               "bandwidth_selection": "highest median of measured read kernels at each thread count",
               "primary_pair": {"engine": args.engine_label, "llama": args.llama_label},
               "bandwidth": bandwidth_rows, "results": results, "pairs": pairs, "ablations": ablations, "ablation_effects": effects}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["threads", "context", "engine_label", "engine_tps", "engine_min", "engine_max", "bandwidth_ceiling_tps",
                         "percent_of_ceiling", "llama_label", "llama_tps", "engine_over_llama"])
        for r in results:
            writer.writerow([r["threads"], r["context"], r["engine_label"], r["engine_tps"]["median"], r["engine_tps"]["min"], r["engine_tps"]["max"],
                             r["bandwidth_ceiling_tps"], r["percent_of_ceiling"], r["llama_label"], r["llama_tps"]["median"], r["engine_over_llama"]])
    lines = ["# Generated result tables", "",
             f"Source: `tools/summarize.py`. Rates are median tokens/s; engine spread is min–max. Engine `{args.engine_label}` vs llama.cpp `{args.llama_label}`.", "",
             "| Threads | Context | Engine (range) | % read ceiling | llama.cpp | Engine / llama |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        rate = r["engine_tps"]
        lines.append(f"| {r['threads']} | {r['context']} | {rate['median']:.2f} ({rate['min']:.2f}–{rate['max']:.2f}) | {r['percent_of_ceiling']:.1f}% | {r['llama_tps']['median']:.2f} | {r['engine_over_llama']:.2f}× |")
    lines += ["", "## Matched pairs", "",
              "llama.cpp runs F16 KV with flash attention auto and F32 KV with flash attention disabled: the pinned llama.cpp casts an F32 K/V cache to F16 whenever flash attention is on.", "",
              "| Engine | llama.cpp | Threads | Context | Engine tokens/s | llama.cpp tokens/s | Engine / llama |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"| {r['engine_label']} | {r['llama_label']} | {r['threads']} | {r['context']} | {r['engine_tps']['median']:.2f} | {r['llama_tps']['median']:.2f} | {r['engine_over_llama']:.2f}× |")
    lines += ["", "## Ablation variants", "", "| Threads | Context | Weights | Head | Kernel | KV | Weight memory | Fused | Median tokens/s |",
              "|---:|---:|---|---|---|---|---|---|---:|"]
    for r in ablations:
        lines.append(f"| {r['threads']} | {r['context']} | {r['weight_format']} | {r['head_format']} | {r['kernel']} | {r['kv_dtype']} | {r['weight_memory']} | {'on' if r['fused_projections'] else 'off'} | {r['tokens_per_second']['median']:.2f} |")
    lines += ["", "## Paired ablation effects", "", "Each row changes exactly one field; ratio is after / before.", "",
              "| Threads | Context | Change | Before tokens/s | After tokens/s | Ratio |",
              "|---:|---:|---|---:|---:|---:|"]
    for r in effects:
        lines.append(f"| {r['threads']} | {r['context']} | {r['change']} | {r['before']['tokens_per_second']['median']:.2f} | {r['after']['tokens_per_second']['median']:.2f} | {r['ratio']:.2f}× |")
    (args.output / "tables.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: summary[key] for key in ["primary_pair", "results", "pairs"]}, indent=2))


if __name__ == "__main__":
    main()
