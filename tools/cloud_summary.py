"""Summarize cloud timings with paired ratios and an ABBA-block bootstrap.

Run with ``python -m tools.cloud_summary --input raw.json --output summary.json
--csv summary.csv``. Add ``--markdown`` to print a table; no document is edited.
The interval is a per-cell percentile interval, not a simultaneous six-cell test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np

from tools.portable import portable

THREADS = (1, 2, 4)
CONTEXTS = (128, 4096)
STEPS = 128
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261007
NATIVE_KERNELS = ("scalar", "simd256", "simd512", "simd512x4", "vnni", "vnni16")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive(value, label: str) -> float:
    require(type(value) in (int, float), f"invalid {label}: expected finite positive number")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    require(valid, f"invalid {label}: expected finite positive number")
    return float(value)


def integer(value, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"invalid {label}: expected integer >= {minimum}")
    return value


def settings(value, backend: str, threads: int, context: int) -> dict:
    label = f"{backend}_settings"
    require(isinstance(value, dict), f"invalid {label}")
    for key, expected in (("backend", backend), ("threads", threads),
                          ("context", context), ("steps", STEPS)):
        if key in value:
            require(type(value[key]) is type(expected) and value[key] == expected,
                    f"mismatched {label}.{key}")
    if "model" in value:
        require(isinstance(value["model"], str) and bool(value["model"]), f"invalid {label}.model")
    cpus = value.get("cpu_set")
    require(isinstance(cpus, list) and len(cpus) == threads and
            all(type(cpu) is int and cpu >= 0 for cpu in cpus) and len(set(cpus)) == threads,
            f"invalid {label}.cpu_set")
    if backend == "native":
        require(value.get("kernel") in NATIVE_KERNELS, f"invalid {label}.kernel: requires resolved kernel")
        if "poll" in value:
            require(value["poll"] is None, f"invalid {label}.poll: native polling is not applicable")
    else:
        require(value.get("flash") in ("on", "off", "auto"), f"invalid {label}.flash")
        require(type(value.get("poll")) is int and value["poll"] == 50, f"invalid {label}.poll")
    return value


def validate(raw: dict, *, allow_partial: bool = False) -> list[dict]:
    """Require complete ABBA cells; partial matrices need explicit permission."""
    require(isinstance(raw, dict), "raw result must be an object")
    require(isinstance(raw.get("design_sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", raw["design_sha256"]) is not None,
            "invalid design_sha256")
    for key in ("environment", "artifacts"):
        require(isinstance(raw.get(key), dict) and bool(raw[key]), f"missing/invalid {key}")
    cells = raw.get("cells")
    require(isinstance(cells, list) and (1 <= len(cells) <= 6 if allow_partial else len(cells) == 6),
            "requires one to six completed cells" if allow_partial else "requires exactly six cells")
    seen = set()
    for cell in cells:
        require(isinstance(cell, dict), "invalid cell")
        threads = integer(cell.get("threads"), "threads", 1)
        context = integer(cell.get("context"), "context", 1)
        key = (threads, context)
        require(threads in THREADS and context in CONTEXTS, f"unexpected cell {key}")
        require(key not in seen, f"duplicate cell {key}")
        seen.add(key)
        require(type(cell.get("steps")) is int and cell["steps"] == STEPS, "cell steps must be 128")
        native = settings(cell.get("native_settings"), "native", threads, context)
        llama = settings(cell.get("llama_settings"), "llama", threads, context)
        require(set(native["cpu_set"]) == set(llama["cpu_set"]), "mismatched backend CPU sets")
        pairs = cell.get("pairs")
        require(isinstance(pairs, list) and len(pairs) >= 16 and len(pairs) % 2 == 0,
                "requires at least sixteen pairs in complete ABBA blocks")
        for index, pair in enumerate(pairs):
            require(isinstance(pair, dict), "invalid pair")
            require(integer(pair.get("id"), "pair.id") == index, "pair ids must be consecutive and chronological")
            require(integer(pair.get("block"), "pair.block") == index // 2,
                    "block ids must group consecutive AB/BA pairs")
            require(pair.get("order") == ("AB" if index % 2 == 0 else "BA"),
                    "each ABBA block requires AB then BA order")
            for backend, config in (("native", native), ("llama", llama)):
                if f"{backend}_settings" in pair:
                    require(pair[f"{backend}_settings"] == config, f"mismatched pair {backend} settings")
                sample = pair.get(backend)
                require(isinstance(sample, dict), f"missing {backend} sample")
                positive(sample.get("seconds"), f"{backend}.seconds")
                tokens = sample.get("tokens")
                require(isinstance(tokens, list) and len(tokens) == STEPS and
                        all(type(token) is int and token >= 0 for token in tokens),
                        f"{backend}.tokens must be a list of 128 nonnegative consumed token IDs")
                integer(sample.get("next_token"), f"{backend}.next_token")
                if "settings" in sample:
                    require(sample["settings"] == config, f"mismatched {backend} sample settings")
    # Six unique allowed cells necessarily cover the full Cartesian product.
    return sorted(cells, key=lambda cell: (cell["threads"], cell["context"]))


def block_bootstrap_ci(block_ratios: np.ndarray) -> list[float]:
    """Draw whole ABBA blocks, retaining both paired ratios in each drawn block."""
    ratios = np.asarray(block_ratios, dtype=np.float64)
    require(ratios.ndim == 2 and ratios.shape[1] == 2 and ratios.shape[0] >= 8,
            "bootstrap requires at least eight two-pair ABBA blocks")
    require(bool(np.all(np.isfinite(ratios) & (ratios > 0))), "invalid bootstrap ratios")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    medians = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 1000):
        stop = min(start + 1000, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, len(ratios), size=(stop - start, len(ratios)))
        medians[start:stop] = np.median(ratios[indices].reshape(stop - start, -1), axis=1)
    interval = np.percentile(medians, [2.5, 97.5], method="linear")
    return [positive(float(value), "bootstrap interval") for value in interval]


def engine_stats(seconds: list[float]) -> dict:
    ordered = sorted(seconds)
    mid = len(ordered) // 2
    median = ordered[mid - 1] + (ordered[mid] - ordered[mid - 1]) / 2
    rates = sorted(STEPS / value for value in seconds)
    median_rate = rates[mid - 1] + (rates[mid] - rates[mid - 1]) / 2
    return {"samples": len(seconds),
            "seconds": {"median": median, "min": ordered[0], "max": ordered[-1]},
            "tokens_per_second": {"median": positive(median_rate, "median tokens/s"),
                                  "min": positive(STEPS / ordered[-1], "min tokens/s"),
                                  "max": positive(STEPS / ordered[0], "max tokens/s")}}


def summarize(raw: dict, *, allow_partial: bool = False) -> dict:
    cells = validate(raw, allow_partial=allow_partial)
    # Preserve cost, pilot selection, environment and artifact identities without
    # inventing a second provenance schema or inferring settings from filenames.
    summary = deepcopy({key: value for key, value in raw.items() if key != "cells"})
    summary.update(schema="cpu-decode-cloud-summary", statistics={
        "estimand": "median(llama_seconds/native_seconds) over chronological pairs",
        "ratio": "native_tokens_per_second/llama_tokens_per_second",
        "bootstrap_unit": "ABBA block (two paired observations)",
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95, "interval": "percentile", "percentile_method": "linear",
        "scope": "per-cell; no simultaneous or strongest-nine claim",
    }, cells=[])
    completed = {(cell["threads"], cell["context"]) for cell in cells}
    summary["matrix"] = {
        "planned_cells": 6, "completed_cells": len(cells), "complete": len(cells) == 6,
        "missing_cells": [{"threads": threads, "context": context} for threads in THREADS for context in CONTEXTS
                          if (threads, context) not in completed]}
    for cell in cells:
        pairs = cell["pairs"]
        ratios = [positive(pair["llama"]["seconds"] / pair["native"]["seconds"], "paired ratio")
                  for pair in pairs]
        ci = block_bootstrap_ci(np.asarray(ratios).reshape(-1, 2))
        median = positive(float(np.median(ratios)), "median paired ratio")
        summary["cells"].append({
            "threads": cell["threads"], "context": cell["context"], "steps": STEPS,
            "native_settings": deepcopy(cell["native_settings"]),
            "llama_settings": deepcopy(cell["llama_settings"]),
            "pairs": len(pairs), "blocks": len(pairs) // 2,
            "native": engine_stats([pair["native"]["seconds"] for pair in pairs]),
            "llama": engine_stats([pair["llama"]["seconds"] for pair in pairs]),
            "paired_ratios": [{"id": pair["id"], "block": pair["block"], "order": pair["order"],
                               "native_over_llama": ratio,
                               "native_tokens": deepcopy(pair["native"]["tokens"]),
                               "llama_tokens": deepcopy(pair["llama"]["tokens"]),
                               "native_next_token": pair["native"]["next_token"],
                               "llama_next_token": pair["llama"]["next_token"]}
                              for pair, ratio in zip(pairs, ratios)],
            "native_over_llama": {"median": median, "ci95": ci},
            "decision": "native-win" if ci[0] > 1 else "llama-win" if ci[1] < 1 else "inconclusive",
        })
    return summary


def csv_text(summary: dict) -> str:
    fields = ["threads", "context", "steps", "pairs", "blocks"]
    fields += [f"{backend}_{unit}_{stat}" for backend in ("native", "llama")
               for unit in ("seconds", "tokens_per_second") for stat in ("median", "min", "max")]
    fields += ["native_over_llama_median", "ci95_lower", "ci95_upper", "decision",
               "native_settings", "llama_settings"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for cell in summary["cells"]:
        row = {key: cell[key] for key in ("threads", "context", "steps", "pairs", "blocks", "decision")}
        for backend in ("native", "llama"):
            for unit in ("seconds", "tokens_per_second"):
                for stat in ("median", "min", "max"):
                    row[f"{backend}_{unit}_{stat}"] = cell[backend][unit][stat]
            row[f"{backend}_settings"] = json.dumps(cell[f"{backend}_settings"], sort_keys=True, allow_nan=False)
        row.update(native_over_llama_median=cell["native_over_llama"]["median"],
                   ci95_lower=cell["native_over_llama"]["ci95"][0],
                   ci95_upper=cell["native_over_llama"]["ci95"][1])
        writer.writerow(row)
    return stream.getvalue()


def markdown_table(summary: dict) -> str:
    lines = ["| Threads | Context | Native tokens/s | llama tokens/s | Paired ratio | 95% block CI | Decision |",
             "|---:|---:|---:|---:|---:|---:|---|"]
    if not summary["matrix"]["complete"]:
        lines = [f"Partial matrix: {summary['matrix']['completed_cells']}/6 cells completed.", "", *lines]
    for cell in summary["cells"]:
        ratio = cell["native_over_llama"]
        lines.append(f"| {cell['threads']} | {cell['context']} | "
                     f"{cell['native']['tokens_per_second']['median']:.3f} | "
                     f"{cell['llama']['tokens_per_second']['median']:.3f} | "
                     f"{ratio['median']:.4f} | [{ratio['ci95'][0]:.4f}, {ratio['ci95'][1]:.4f}] | "
                     f"{cell['decision']} |")
    return "\n".join(lines) + "\n"


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def summarize_file(input_path: Path, output_path: Path, csv_path: Path, *, allow_partial: bool = False) -> dict:
    paths = [path.resolve() for path in (input_path, output_path, csv_path)]
    require(len(set(paths)) == len(paths), "input and output paths must be distinct")
    source = input_path.read_bytes()
    raw = json.loads(source, parse_constant=reject_constant)
    summary = portable(summarize(raw, allow_partial=allow_partial))
    summary["raw_result"] = {"file": portable(str(input_path)), "sha256": hashlib.sha256(source).hexdigest()}
    # Serialize before writing either output so invalid metadata cannot leave a
    # partially published result. Match the repository's indented JSON convention.
    json_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    table = csv_text(summary)
    for path in (output_path, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_text)
    csv_path.write_text(table)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table to stdout")
    parser.add_argument("--allow-partial", action="store_true", help="explicitly publish only completed cells, labeled partial")
    args = parser.parse_args(argv)
    try:
        summary = summarize_file(args.input, args.output, args.csv, allow_partial=args.allow_partial)
    except (ValueError, OSError, UnicodeError, OverflowError, ZeroDivisionError) as exc:
        parser.error(str(exc))
    if args.markdown:
        print(markdown_table(summary), end="")


if __name__ == "__main__":
    main()
