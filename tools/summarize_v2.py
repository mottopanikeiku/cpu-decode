"""Summarize measured v2 windows, rejecting unmatched settings and weaker baselines."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from tools.measure_v2 import ABLATION_CELLS, ABLATION_LABELS, CONTEXTS, ROOT, THREADS, candidates, cpu_mask, digest
from tools.portable import portable

BYTE_COMPONENTS = ["matrix_weights", "scales", "norm_bias", "embedding", "kv_read_min", "kv_write"]


def spread(values: list[float]) -> dict:
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("samples must be nonempty finite positive observations")
    median = statistics.median(values)
    percentage = 100 * (max(values) - min(values)) / median
    return {"median": median, "min": min(values), "max": max(values), "samples": len(values),
            "spread_percent": percentage, "noisy_over_5_percent": percentage > 5}


def flag(command: list[str], name: str) -> str:
    if command.count(name) != 1 or command.index(name) + 1 >= len(command):
        raise ValueError(f"missing or duplicate flag {name}")
    return command[command.index(name) + 1]


def require(actual: dict, expected: dict, description: str) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"{description} mismatch: {key} ({actual.get(key)!r} != {value!r})")


def profile_bytes(samples: list[dict], geometry: dict, kv_dtype: str, context: int, steps: int) -> dict:
    """Use observed format-specific weights/scales; independently check KV geometry."""
    cache_token = geometry["layers"] * geometry["kv_heads"] * geometry["head_dim"] * 2 * {"f16": 2, "f32": 4}[kv_dtype]
    expected_read = cache_token * (context + (steps + 1) / 2)
    for sample in samples:
        counts = sample["bytes_per_token"]
        if any(not math.isfinite(counts[k]) or counts[k] < 0 for k in BYTE_COMPONENTS):
            raise ValueError("invalid profile byte counts")
        if not math.isclose(counts["total_min"], sum(counts[k] for k in BYTE_COMPONENTS), rel_tol=1e-9):
            raise ValueError("profile total_min double counts or omits byte components")
        if counts["kv_write"] != cache_token or counts["kv_read_min"] != expected_read:
            raise ValueError("profile KV bytes do not match selected dtype and growing context")
        if counts["total_min"] <= 0:
            raise ValueError("profile byte bound must be positive")
    keys = samples[0]["bytes_per_token"].keys()
    if any(s["bytes_per_token"].keys() != keys for s in samples):
        raise ValueError("profile byte fields changed between samples")
    return {key: statistics.mean(s["bytes_per_token"][key] for s in samples) for key in keys}


def native_samples(invocation: dict, window: dict, protocol: dict, settings: dict) -> list[dict]:
    raw = invocation["data"]
    expected = {"threads": window["threads"], "context": window["context"], "steps": protocol["steps"],
                "repeats": protocol["repeats"], "warmup_steps": 1, "cpu_set": window["cpu_set"],
                "prompt_tokens": [int(x) for x in protocol["tokens"].split(",")],
                **{k: v for k, v in settings.items() if k != "kernel"}}
    require(raw, expected, "native settings")
    if settings["kernel"] != "auto" and raw.get("kernel") != settings["kernel"]:
        raise ValueError("native kernel differs from fixed setting")
    command = invocation["command"]
    for key, value in {"--threads": window["threads"], "--context": window["context"], "--steps": protocol["steps"],
                       "--repeats": protocol["repeats"], "--tokens": protocol["tokens"], "--kernel": settings["kernel"],
                       "--kv": settings["kv_dtype"], "--attention": settings["attention"],
                       "--scheduler": settings["scheduler"], "--rope": settings["rope"],
                       "--cpu-set": ",".join(map(str, window["cpu_set"])), "--interactive": "1"}.items():
        if flag(command, key) != str(value):
            raise ValueError(f"native command mismatch: {key}")
    if invocation.get("request") != "run" or len(raw["samples"]) != protocol["repeats"]:
        raise ValueError("native request/repetition count mismatch")
    for sample in raw["samples"]:
        if len(sample["step_seconds"]) != protocol["steps"] or len(sample["generated_tokens"]) != protocol["steps"]:
            raise ValueError("native measured token count mismatch")
        if not math.isclose(sample["tokens_per_second"], protocol["steps"] / sample["seconds"], rel_tol=1e-6):
            raise ValueError("native rate/seconds mismatch")
    return raw["samples"]


def baseline_samples(invocation: dict, window: dict, protocol: dict, candidate: dict) -> tuple[list[float], dict]:
    if invocation["returncode"] != 0 or not isinstance(invocation["data"], list) or len(invocation["data"]) != 1:
        raise ValueError("baseline must have one successful CPU decode result")
    raw = invocation["data"][0]
    pinned = candidate["affinity"] == "pinned"
    expected = {"n_threads": window["threads"], "n_depth": window["context"], "n_gen": protocol["steps"],
                "n_prompt": 0, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 0,
                "flash_attn": {"on": 1, "off": 0, "auto": -1}[candidate["flash_attn"]],
                "cpu_strict": pinned, "poll": candidate["poll"], "repack": True, "backends": "CPU"}
    require(raw, expected, "baseline settings")
    expected_mask = cpu_mask(window["cpu_set"]) if pinned else "0x0"
    if int(raw["cpu_mask"], 16) != int(expected_mask, 16):
        raise ValueError("baseline affinity mask mismatch")
    if "Q8_0" not in raw["model_type"] or not protocol["llama_commit"].startswith(raw["build_commit"]) or len(raw["build_commit"]) < 7:
        raise ValueError("baseline is not the pinned Q8_0 build")
    command = invocation["command"]
    for key, value in {"-p": 0, "-n": protocol["steps"], "-d": window["context"], "-t": window["threads"],
                       "-r": protocol["repeats"], "-ngl": 0, "-ctk": "f16", "-ctv": "f16",
                       "-fa": candidate["flash_attn"], "--cpu-mask": expected_mask,
                       "--cpu-strict": int(pinned), "--poll": candidate["poll"], "--repack": 1, "-o": "json"}.items():
        if flag(command, key) != str(value):
            raise ValueError(f"baseline command mismatch: {key}")
    if "--verbose" not in command or "--no-warmup" in command:
        raise ValueError("baseline needs verbose logs and its one-step warmup")
    values = raw["samples_ts"]
    if len(values) != protocol["repeats"] or len(raw["samples_ns"]) != len(values):
        raise ValueError("baseline repetition count mismatch")
    for rate, ns in zip(values, raw["samples_ns"]):
        if ns <= 0 or not math.isclose(rate, protocol["steps"] * 1e9 / ns, rel_tol=1e-5):
            raise ValueError("baseline rate/nanoseconds mismatch")
    return values, {key: value for key, value in raw.items() if key not in ["samples_ts", "samples_ns"]}


def summarize_candidate(window: dict, protocol: dict, candidate: dict) -> dict:
    require(window, {"protocol_id": protocol["id"], "native_settings": protocol["native"],
                     "cpu_set": protocol["cpu_order"][:window["threads"]]}, "window settings")
    if window["environment"]["nice"] < 19 or not window["native_process"]["success"] or window["native_process"]["returncode"] != 0:
        raise ValueError("window priority or shared native process failure")
    runs = [r for r in window["invocations"] if r["candidate_id"] == candidate["id"]]
    expected_order = [(engine, round_id) for round_id in range(protocol["rounds"]) for engine in ["native", "llama"]]
    if [(r["engine"], r["round"]) for r in runs] != expected_order:
        raise ValueError("window is not complete A B A B interleaving")
    samples, llama_rates, configurations = [], [], []
    for run in runs:
        if not run["success"] or run.get("error"):
            raise ValueError("unsuccessful invocation cannot be a sample")
        if run["engine"] == "native":
            samples.extend(native_samples(run, window, protocol, protocol["native"]))
        else:
            values, config = baseline_samples(run, window, protocol, candidate)
            llama_rates.extend(values)
            configurations.append(config)
    # Runtime timestamps/statistics differ; settings used for dispatch must not.
    settings_keys = ["build_commit", "cpu_info", "backends", "model_type", "model_size", "n_threads", "cpu_mask",
                     "cpu_strict", "poll", "type_k", "type_v", "n_gpu_layers", "flash_attn", "repack", "n_depth", "n_gen"]
    require(configurations[1], {k: configurations[0][k] for k in settings_keys}, "baseline round configuration")
    native = spread([s["tokens_per_second"] for s in samples])
    baseline = spread(llama_rates)
    if native["samples"] != baseline["samples"]:
        raise ValueError("compared configurations have unequal sample counts")
    bytes_per_token = profile_bytes(samples, protocol["model_geometry"], protocol["native"]["kv_dtype"], window["context"], protocol["steps"])
    return {"candidate": candidate, "native_tps": native, "baseline_tps": baseline,
            "baseline_configurations": configurations, "invocations_per_engine": protocol["rounds"],
            "bytes_per_token": bytes_per_token, "observed_native_kernels": sorted({r["data"]["kernel"] for r in runs if r["engine"] == "native"}),
            "raw": [{k: r[k] for k in ["engine", "round", "command", "stdout_file", "stderr_file"]} for r in runs]}


def select_best(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no successful measured baseline configurations")
    return max(rows, key=lambda row: (row["baseline_tps"]["median"], row["candidate"]["id"]))


def summarize_bandwidth(raw: dict, protocol: dict) -> dict:
    require(raw, {"protocol_id": protocol["id"], "cpu_set": protocol["cpu_order"][:raw["threads"]]}, "bandwidth affinity")
    if raw["environment"]["nice"] < 19:
        raise ValueError("bandwidth needs nice 19")
    rows = []
    for run in raw["invocations"]:
        if not run["success"] or run["returncode"] != 0:
            raise ValueError("unsuccessful bandwidth invocation")
        data = run["data"]
        require(data, {"threads": raw["threads"], "kind": "read_bandwidth"}, "bandwidth settings")
        command = run["command"]
        if command[:2] != ["taskset", "-c"] or command[2] != ",".join(map(str, raw["cpu_set"])):
            raise ValueError("bandwidth must pin the same selected core set")
        overrides = run["environment_overrides"]
        require(overrides, {"OMP_PLACES": ",".join(f"{{{x}}}" for x in raw["cpu_set"]), "OMP_PROC_BIND": "true", "OMP_DYNAMIC": "false"}, "bandwidth OpenMP affinity")
        if len(data["samples"]) != protocol["repeats"] or data["array_bytes"] < 256 * 1024 * 1024:
            raise ValueError("bandwidth repetition count/working set mismatch")
        for sample in data["samples"]:
            if not math.isclose(sample["GB_per_s"], data["array_bytes"] * data["passes"] / sample["seconds"] / 1e9, rel_tol=1e-5):
                raise ValueError("bandwidth byte/rate mismatch")
        rows.append({"kernel": data["kernel"], "GB_per_s": spread([s["GB_per_s"] for s in data["samples"]]),
                     "command": command, "stdout_file": run["stdout_file"], "stderr_file": run["stderr_file"]})
    if {r["kernel"] for r in rows} != {"simd256", "simd512"} or len(rows) != 2:
        raise ValueError("bandwidth requires both measured SIMD widths")
    return {"threads": raw["threads"], "cpu_set": raw["cpu_set"], "candidates": rows,
            "winner": max(rows, key=lambda r: r["GB_per_s"]["median"])}


def summarize(directory: Path, allow_partial: bool = False, target_ratio: float = 1.0,
              target_ceiling_percent: float = 50.0) -> dict:
    protocol = json.loads((directory / "protocol.json").read_text())
    if digest({k: v for k, v in protocol.items() if k != "id"}) != protocol["id"]:
        raise ValueError("frozen protocol digest mismatch")
    if not allow_partial and (protocol["development"] or protocol["steps"] < 64 or protocol["repeats"] < 5 or protocol["rounds"] != 2):
        raise ValueError("final summary requires final sampling settings")
    if protocol["native"]["kv_dtype"] != "f16" or protocol["warmup_steps"] != 1:
        raise ValueError("comparison protocol requires F16 KV and one warmup step")
    polls = list(dict.fromkeys(c["poll"] for c in protocol["candidates"]))
    if protocol["candidates"] != candidates(polls):
        raise ValueError("baseline candidate set must cover on/off/auto and pinned/unpinned")
    expected_candidates = {c["id"]: c for c in protocol["candidates"]}
    groups, failures, bandwidth = {}, [], {}
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        if raw.get("schema") == "cpu-decode-v2-bandwidth":
            try:
                row = summarize_bandwidth(raw, protocol)
                if row["threads"] in bandwidth:
                    raise ValueError("duplicate bandwidth thread count")
                bandwidth[row["threads"]] = {"file": str(path), **row}
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                failures.append({"file": str(path), "error": str(exc)})
        if raw.get("schema") != "cpu-decode-v2-window":
            continue
        cell = (raw["threads"], raw["context"])
        group = groups.setdefault(cell, {})
        for candidate in raw["candidates"]:
            try:
                if candidate != expected_candidates.get(candidate["id"]):
                    raise ValueError("candidate not in fixed candidate set")
                if candidate["id"] in group:
                    raise ValueError("duplicate candidate measurement (do not cherry-pick reruns)")
                group[candidate["id"]] = {"file": str(path), **summarize_candidate(raw, protocol, candidate)}
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                failures.append({"file": str(path), "candidate": candidate["id"], "error": str(exc)})
    results = []
    for (thread, context), measured in sorted(groups.items()):
        missing = sorted(set(expected_candidates) - set(measured))
        if not measured:
            continue
        winner = select_best(list(measured.values()))
        bw = bandwidth.get(thread)
        ceiling = bw["winner"]["GB_per_s"]["median"] * 1e9 / winner["bytes_per_token"]["total_min"] if bw else None
        percentage = 100 * winner["native_tps"]["median"] / ceiling if ceiling else None
        ratio = winner["native_tps"]["median"] / winner["baseline_tps"]["median"]
        results.append({"threads": thread, "context": context, "cpu_set": protocol["cpu_order"][:thread],
                        "native_tps": winner["native_tps"], "best_baseline_tps": winner["baseline_tps"],
                        "winner": winner, "candidates": list(measured.values()), "missing_candidates": missing,
                        "native_samples_all_candidates": sum(r["native_tps"]["samples"] for r in measured.values()),
                        "native_over_best_baseline": ratio, "read_ceiling_tps": ceiling,
                        "read_GB_per_s": bw and bw["winner"]["GB_per_s"], "bandwidth_file": bw and bw["file"],
                        "percent_of_ceiling": percentage,
                        "targets": {"native_over_best_baseline": ratio >= target_ratio,
                                    "percent_of_ceiling": percentage is not None and percentage >= target_ceiling_percent},
                        "noisy_over_5_percent": winner["native_tps"]["noisy_over_5_percent"] or winner["baseline_tps"]["noisy_over_5_percent"] or bool(bw and bw["winner"]["GB_per_s"]["noisy_over_5_percent"])})
    expected_cells = {(t, c) for t in THREADS for c in CONTEXTS}
    cells = {(r["threads"], r["context"]) for r in results}
    complete = cells == expected_cells and all(not r["missing_candidates"] and r["read_ceiling_tps"] is not None for r in results) and not failures
    ablations = []
    for path in sorted(directory.glob("ablation-t*-c*.json")):
        raw = json.loads(path.read_text())
        if raw.get("schema") != "cpu-decode-v2-ablation":
            continue
        try:
            if (raw["threads"], raw["context"]) not in ABLATION_CELLS:
                raise ValueError("ablation outside fixed cells")
            require(raw, {"protocol_id": protocol["id"], "cpu_set": protocol["cpu_order"][:raw["threads"]]}, "ablation settings")
            if not raw["native_process"]["success"] or raw["native_process"]["returncode"] != 0 or raw["environment"]["nice"] < 19:
                raise ValueError("ablation process/priority failure")
            if [(r["engine"], r["round"]) for r in raw["invocations"]] != [("native", i) for i in range(protocol["rounds"])]:
                raise ValueError("ablation round count mismatch")
            samples = []
            for run in raw["invocations"]:
                if not run["success"]:
                    raise ValueError("unsuccessful ablation invocation")
                samples.extend(native_samples(run, raw, protocol, raw["native_settings"]))
            bytes_per_token = profile_bytes(samples, protocol["model_geometry"], raw["native_settings"]["kv_dtype"], raw["context"], protocol["steps"])
            ablations.append({"file": str(path), "label": raw["label"], "threads": raw["threads"], "context": raw["context"],
                              "settings": raw["native_settings"], "weights_sha256": raw["weights_sha256"],
                              "observed_group_size": raw["invocations"][0]["data"]["group_size"],
                              "observed_scale_dtype": raw["invocations"][0]["data"]["scale_dtype"],
                              "tokens_per_second": spread([s["tokens_per_second"] for s in samples]), "bytes_per_token": bytes_per_token,
                              "commands": [r["command"] for r in raw["invocations"]]})
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            failures.append({"file": str(path), "error": str(exc)})
    if failures:
        complete = False
    ladders = []
    for thread, context in ABLATION_CELLS:
        matched = [r for r in ablations if (r["threads"], r["context"]) == (thread, context)]
        ordered = sorted(matched, key=lambda r: (ABLATION_LABELS.index(r["label"]) if r["label"] in ABLATION_LABELS else len(ABLATION_LABELS), r["label"]))
        ladders.append({"threads": thread, "context": context, "rungs": ordered,
                        "missing_labels": [label for label in ABLATION_LABELS if label not in {r["label"] for r in matched}]})
    return {"schema": "cpu-decode-v2-summary", "protocol_id": protocol["id"], "development": protocol["development"],
            "complete_final_matrix": complete and not protocol["development"], "missing_cells": sorted(expected_cells - cells),
            "selection": protocol["selection"], "selection_bias": "winner selected and reported on the same samples; no independent holdout",
            "statistic": "median rates; min/max and 100*(max-min)/median spread; >5% disclosed, never discarded",
            "rates_kind": "MEAS", "read_ceiling_kind": "EXT",
            "thresholds": {"native_over_best_baseline": target_ratio, "percent_of_ceiling": target_ceiling_percent},
            "targets_all_cells": {key: complete and not protocol["development"] and all(r["targets"][key] for r in results)
                                  for key in ["native_over_best_baseline", "percent_of_ceiling"]},
            "bandwidth": list(bandwidth.values()), "results": results, "ablations": ablations, "failures": failures,
            "ablation_ladders": ladders,
            "complete_ablation_ladder": all(not ladder["missing_labels"] for ladder in ladders),
            "noisy_candidates": [{"threads": r["threads"], "context": r["context"], "candidate": c["candidate"]["id"]}
                                 for r in results for c in r["candidates"]
                                 if c["native_tps"]["noisy_over_5_percent"] or c["baseline_tps"]["noisy_over_5_percent"]],
            "noisy_cells": [[r["threads"], r["context"]] for r in results if r["noisy_over_5_percent"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/v2"))
    parser.add_argument("--output", type=Path, default=Path("results/v2/summary.json"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--target-ratio", type=float, default=1.0)
    parser.add_argument("--target-ceiling-percent", type=float, default=50.0)
    args = parser.parse_args()
    if not math.isfinite(args.target_ratio) or args.target_ratio <= 0 or not math.isfinite(args.target_ceiling_percent) or args.target_ceiling_percent <= 0:
        parser.error("target thresholds must be finite and positive")
    summary = summarize(args.input, args.allow_partial, args.target_ratio, args.target_ceiling_percent)
    result_root = ROOT / "results"
    if args.output.resolve().is_relative_to(result_root) and not args.output.resolve().is_relative_to(result_root / "v2"):
        parser.error("v2 summary must not overwrite v1 results")
    input_alias = str(args.input.resolve().relative_to(ROOT)) if args.input.resolve().is_relative_to(ROOT) else "$INPUT"
    summary = portable(summary, {args.input.resolve(): input_alias, Path.home(): "$HOME"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{len(summary['results'])} cells; complete={summary['complete_final_matrix']}; failures={len(summary['failures'])}; noisy={summary['noisy_cells']}")
    if not args.allow_partial and not summary["complete_final_matrix"]:
        raise SystemExit("incomplete/invalid final matrix: summary retained with visible failures")


if __name__ == "__main__":
    main()
