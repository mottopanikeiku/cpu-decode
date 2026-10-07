"""Publish a complete fifteen-cell v2 matrix without changing partial results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from tools.figure_v2 import figure
from tools.measure_v2 import CONTEXTS, THREADS, candidates, check_quality_eligibility, digest

ROOT = Path(__file__).resolve().parents[1]
TARGET_KEYS = (
    "native_over_best_baseline",
    "short_best_percent_of_ceiling",
    "long_all_percent_of_ceiling",
)
BANNED = re.compile(
    r"\b(?:custody|receipts?|frozen gates?|fail closed|evidence finalizer|sole writer|"
    r"attempt chain|claim eligibility|research program|evidence-grade|flagship|"
    r"authenticated|canonical|control plane|comprehensive|robust|seamless|"
    r"cutting-edge|production-ready)\b", re.IGNORECASE,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, label: str, *, positive: bool = True) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {label}")
    require(value > 0 if positive else value >= 0, f"invalid {label}")
    return value


def close(actual, expected: float, label: str, *, positive: bool = True) -> None:
    number(actual, label, positive=positive)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"contradictory {label}: {actual!r} != {expected!r}")


def stats(value: dict, label: str, samples: int | None = None) -> None:
    lo, median, hi = [number(value[key], f"{label}.{key}") for key in ("min", "median", "max")]
    require(lo <= median <= hi, f"inconsistent range: {label}")
    require(type(value["samples"]) is int and value["samples"] > 0, f"invalid samples: {label}")
    if samples is not None:
        require(value["samples"] == samples, f"{label} requires {samples} samples")
    spread = 100 * (hi - lo) / median
    close(value["spread_percent"], spread, f"{label}.spread_percent", positive=False)
    require(value["noisy_over_5_percent"] is (spread > 5), f"contradictory noise: {label}")


def flag(command: list, name: str, expected: str) -> None:
    require(isinstance(command, list) and all(isinstance(x, str) for x in command), "invalid command")
    require(command.count(name) == 1, f"missing/duplicate sampling flag {name}")
    index = command.index(name)
    require(index + 1 < len(command) and command[index + 1] == expected,
            f"final publication requires {name} {expected}")


def cpu_set(value: list) -> set[int]:
    require(isinstance(value, list) and bool(value) and
            all(type(cpu) is int and cpu >= 0 for cpu in value) and len(set(value)) == len(value),
            "invalid CPU set")
    return set(value)


def validate(summary: dict) -> list[dict]:
    """Check actual rows, not just the producer's completeness booleans."""
    require(summary["schema"] == "cpu-decode-v2-summary", "wrong summary schema")
    for key in ("complete_final_matrix", "complete_requested_matrix", "full_fifteen_cell_target_matrix", "target_eligible"):
        require(summary[key] is True, f"{key} must be true")
    require(summary["development"] is False, "development summary is not final")
    require(summary["matrix_scope"] == "full-fifteen-cell", "subset is not final")
    require(summary["rates_kind"] == "MEAS" and summary["read_ceiling_kind"] == "EXT", "wrong rate kinds")
    for key in ("missing_cells", "missing_bandwidth_threads", "failures"):
        require(summary[key] == [], f"nonempty {key}")
    require(isinstance(summary["protocol_id"], str) and
            re.fullmatch(r"[0-9a-f]{64}", summary["protocol_id"]) is not None, "invalid protocol id")
    source = summary.get("source_model")
    require(source is None or source["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct", "wrong source model")
    require(summary["requested_threads"] == THREADS and summary["requested_contexts"] == CONTEXTS,
            "wrong requested dimensions")
    expected_cells = [(t, c) for t in THREADS for c in CONTEXTS]
    require(summary["expected_cells"] == [list(cell) for cell in expected_cells], "wrong expected cells")
    require(summary["expected_candidate_windows"] == 135 and summary["expected_bandwidth_windows"] == 5,
            "wrong window counts")
    rows = summary["results"]
    require(isinstance(rows, list) and len(rows) == 15, "requires all 15 cells")
    require(all(type(row["threads"]) is int and type(row["context"]) is int for row in rows), "invalid cell dimensions")
    cells = [(row["threads"], row["context"]) for row in rows]
    require(len(set(cells)) == 15 and set(cells) == set(expected_cells), "duplicate or missing cell")
    thresholds = summary["thresholds"]
    require(set(thresholds) == set(TARGET_KEYS), "wrong target thresholds")
    for key in TARGET_KEYS:
        number(thresholds[key], f"threshold {key}")
    fixed_candidates = None
    for row in rows:
        label = f't{row["threads"]}-c{row["context"]}'
        require(row["missing_candidates"] == [], f"missing candidates: {label}")
        measured = row["candidates"]
        require(isinstance(measured, list) and len(measured) == 9, f"requires all nine candidates: {label}")
        ids = [entry["candidate"]["id"] for entry in measured]
        require(len(set(ids)) == 9, f"duplicate candidate: {label}")
        polls = {entry["candidate"]["poll"] for entry in measured}
        require(len(polls) == 1 and all(type(p) is int for p in polls), "requires one poll value")
        expected = {c["id"]: c for c in candidates(list(polls))}
        require({entry["candidate"]["id"]: entry["candidate"] for entry in measured} == expected,
                f"incomplete candidate flags: {label}")
        if fixed_candidates is None:
            fixed_candidates = expected
        require(expected == fixed_candidates, "candidate set changes between cells")
        native_cpus = cpu_set(row["cpu_set"])
        for entry in measured:
            stats(entry["native_tps"], f"{label}/{entry['candidate']['id']}/native", 10)
            stats(entry["baseline_tps"], f"{label}/{entry['candidate']['id']}/Q8", 10)
            require(entry["invocations_per_engine"] == 2, "requires two interleaved rounds")
            raw = entry["raw"]
            require([(r["engine"], r["round"]) for r in raw] ==
                    [(engine, round_id) for round_id in range(2) for engine in ("native", "llama")],
                    "requires complete ABAB order")
            for invocation in raw:
                native = invocation["engine"] == "native"
                flag(invocation["command"], "--steps" if native else "-n", "64")
                flag(invocation["command"], "--repeats" if native else "-r", "5")
            matched = native_cpus == cpu_set(entry["baseline_process_cpu_set"])
            require(entry["comparison_core_sets_matched"] is matched, "contradictory candidate core match")
        winner = max(measured, key=lambda entry: (entry["baseline_tps"]["median"], entry["candidate"]["id"]))
        require(row["winner"] == winner, f"not the strongest Q8 candidate: {label}")
        require(row["native_tps"] == winner["native_tps"] and row["best_baseline_tps"] == winner["baseline_tps"],
                f"winning-arm rates disagree: {label}")
        require(row["native_samples_all_candidates"] == 90, "wrong total native sample count")
        require(row["winner_core_sets_matched"] is winner["comparison_core_sets_matched"] and
                row["winner_baseline_cpu_set"] == winner["baseline_process_cpu_set"], "winner CPU fields disagree")
        ratio = row["native_tps"]["median"] / row["best_baseline_tps"]["median"]
        close(row["native_over_best_baseline"], ratio, f"{label}.ratio")
        ceiling = number(row["read_ceiling_tps"], f"{label}.ceiling")
        percent = 100 * row["native_tps"]["median"] / ceiling
        close(row["percent_of_ceiling"], percent, f"{label}.percent_of_ceiling")
        stats(row["read_GB_per_s"], f"{label}/bandwidth", 5)
        noisy = any(value["noisy_over_5_percent"] for value in
                    (row["native_tps"], row["best_baseline_tps"], row["read_GB_per_s"]))
        require(row["noisy_over_5_percent"] is noisy, f"contradictory cell noise: {label}")
        targets = {
            TARGET_KEYS[0]: ratio >= thresholds[TARGET_KEYS[0]],
            TARGET_KEYS[1]: percent >= thresholds[TARGET_KEYS[1]] if row["context"] == 128 else None,
            TARGET_KEYS[2]: percent >= thresholds[TARGET_KEYS[2]] if row["context"] == 4096 else None,
        }
        require(set(row["targets"]) == set(targets) and
                all(row["targets"][key] is value for key, value in targets.items()), "contradictory cell targets")
    predicates = {
        TARGET_KEYS[0]: all(row["targets"][TARGET_KEYS[0]] for row in rows),
        TARGET_KEYS[1]: any(row["targets"][TARGET_KEYS[1]] for row in rows if row["context"] == 128),
        TARGET_KEYS[2]: all(row["targets"][TARGET_KEYS[2]] for row in rows if row["context"] == 4096),
    }
    require(set(summary["target_predicates"]) == set(predicates) and
            all(summary["target_predicates"][key] is value for key, value in predicates.items()),
            "contradictory aggregate targets")
    require(summary["numeric_targets_met"] is all(predicates.values()), "contradictory target disposition")
    return sorted(rows, key=lambda row: (row["threads"], row["context"]))


def outcomes(summary: dict, rows: list[dict], identity: dict) -> dict:
    cells = []
    counts = {"native": 0, "Q8": 0, "tie": 0}
    for row in rows:
        native, q8 = row["native_tps"]["median"], row["best_baseline_tps"]["median"]
        outcome = "native" if native > q8 else "Q8" if q8 > native else "tie"
        counts[outcome] += 1
        cells.append({
            "threads": row["threads"], "context": row["context"],
            "native_tps": row["native_tps"], "best_q8_tps": row["best_baseline_tps"],
            "native_over_best_q8": native / q8, "outcome": outcome,
            "winner": row["winner"]["candidate"], "native_cpu_set": row["cpu_set"],
            "winner_core_sets_matched": row["winner_core_sets_matched"],
            "winner_baseline_cpu_set": row["winner_baseline_cpu_set"],
            "read_ceiling_tps": row["read_ceiling_tps"], "percent_of_ceiling": row["percent_of_ceiling"],
            "targets": row["targets"], "noisy_over_5_percent": row["noisy_over_5_percent"],
        })
    return {"schema": "cpu-decode-v2-cell-outcomes", "input": identity,
            "rates_kind": "MEAS", "read_ceiling_kind": "EXT", "cell_count": len(cells),
            "candidate_count_per_cell": 9, "samples_per_winning_arm": 10,
            "counts": counts, "noisy_cell_count": sum(row["noisy_over_5_percent"] for row in cells),
            "thresholds": summary["thresholds"], "target_predicates": summary["target_predicates"],
            "numeric_targets_met": summary["numeric_targets_met"], "results": cells}


def csv_text(record: dict) -> str:
    columns = ["threads", "context", "outcome"]
    stat_keys = ("median", "min", "max", "spread_percent", "samples", "noisy_over_5_percent")
    columns += [f"{arm}_{key}" for arm in ("native", "best_q8") for key in stat_keys]
    columns += ["native_over_best_q8", "winner_id", "winner_flags", "native_cpu_set",
                "winner_core_sets_matched", "winner_baseline_cpu_set", "read_ceiling_tps",
                "percent_of_ceiling", *TARGET_KEYS, "noisy_over_5_percent"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for cell in record["results"]:
        row = {key: cell[key] for key in ("threads", "context", "outcome", "native_over_best_q8",
                                        "winner_core_sets_matched", "read_ceiling_tps", "percent_of_ceiling",
                                        "noisy_over_5_percent")}
        for arm in ("native", "best_q8"):
            row.update({f"{arm}_{key}": cell[f"{arm}_tps"][key] for key in stat_keys})
        row.update(cell["targets"])
        row["winner_id"] = cell["winner"]["id"]
        row["winner_flags"] = json.dumps(cell["winner"], sort_keys=True, separators=(",", ":"))
        for key in ("native_cpu_set", "winner_baseline_cpu_set"):
            row[key] = json.dumps(cell[key], separators=(",", ":"))
        writer.writerow(row)
    return output.getvalue()


def link(path: Path, readme: Path) -> str:
    return Path(os.path.relpath(path, readme.parent)).as_posix()


def replace_body(text: str, name: str, body: str) -> str:
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    require(text.count(start) == 1 and text.count(end) == 1, f"requires exactly one {name} marker pair")
    left, right = text.index(start) + len(start), text.index(end)
    require(left <= right, f"reversed {name} markers")
    return text[:left] + "\n" + body.strip() + "\n" + text[right:]


def render_readme(text: str, summary: dict, record: dict, readme: Path,
                  input_path: Path, svg: Path, csv_path: Path, json_path: Path) -> str:
    # Validate both marker pairs together before replacement; nested pairs are invalid.
    positions = []
    for name in ("FINAL_RESULT", "FINAL_TABLE"):
        start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
        require(text.count(start) == 1 and text.count(end) == 1, f"requires exactly one {name} marker pair")
        positions += [text.index(start), text.index(end)]
    require(positions == sorted(set(positions)), "README marker pairs must be ordered and disjoint")
    counts = record["counts"]
    disposition = "met" if record["numeric_targets_met"] else "not met"
    result = (f"**Current result:** The [final matrix]({link(input_path, readme)}) has "
              f"{counts['native']} native wins, {counts['Q8']} Q8_0 wins and {counts['tie']} ties by median; "
              f"the requested numerical targets are **{disposition}**. "
              f"{record['noisy_cell_count']} cells have >5% spread; none are discarded.")
    proof = summary.get("quality_eligibility") or {}
    if proof.get("approved") is True or (proof.get("gate") or {}).get("approved") is True:
        quality_file = proof.get("file")
        require(isinstance(quality_file, str) and not Path(quality_file).is_absolute() and
                ".." not in Path(quality_file).parts, "quality link must be repository-relative")
        result += (f" The [quality comparison]({quality_file}) approves KL/agreement, "
                   "not better perplexity or downstream task accuracy.")
    table = ["| Threads | Context 128 | Context 1024 | Context 4096 |",
             "|---:|---:|---:|---:|"]
    cells = {(cell["threads"], cell["context"]): cell for cell in record["results"]}
    for thread in THREADS:
        values = []
        for context in CONTEXTS:
            cell = cells[thread, context]
            values.append(f"{cell['native_tps']['median']:.2f}/{cell['best_q8_tps']['median']:.2f} "
                          f"({cell['native_over_best_q8']:.2f}×)")
        table.append(f"|{thread}|" + "|".join(values) + "|")
    table += ["", "MEAS: native/best Q8_0 median tokens/s (ratio). 64 measured tokens, five repeats per "
              "invocation, ABAB: ten interleaved samples per winning arm. Best of nine configurations, "
              "selected on these same samples; no independent holdout.", "",
              f"Native loses {counts['Q8']} cells and ties {counts['tie']}; {record['noisy_cell_count']} "
              f"cells have >5% spread. [Every cell, including losses]({link(json_path, readme)}) / "
              f"[CSV]({link(csv_path, readme)}): ranges, samples, winner flags/core matches, "
              "requested targets and noise. Read-ceiling percentages are EXT estimates, "
              "not measured DRAM utilization.", "",
              f"![Final decode rates and measured ranges]({link(svg, readme)})"]
    text = replace_body(text, "FINAL_RESULT", result)
    text = replace_body(text, "FINAL_TABLE", "\n".join(table))
    require(len(text.split()) <= 900, "rendered README exceeds 900 words")
    require(BANNED.search(text) is None, "rendered README contains banned vocabulary")
    require(len(re.findall(r"\bpreregistered\b", text, re.IGNORECASE)) <= 1, "repeated preregistered wording")
    return text


def publish(contents: dict[Path, bytes]) -> None:
    """Stage every validated output first, then replace each file atomically; README last."""
    staged = []
    try:
        for destination, content in contents.items():
            if destination.exists() and destination.read_bytes() == content:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, filename = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            temporary = Path(filename)
            staged.append((temporary, destination))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(destination.stat().st_mode & 0o777 if destination.exists() else 0o644)
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def validate_protocol(summary: dict, input_path: Path) -> None:
    """Reload the retained approval, including every linked quality report."""
    protocol = json.loads(input_path.with_name("protocol.json").read_text())
    identity = digest({key: value for key, value in protocol.items() if key != "id"})
    require(protocol["id"] == identity == summary["protocol_id"], "retained protocol identity differs")
    require(protocol["development"] is False, "retained protocol is development-only")
    require(protocol.get("source_model") == summary.get("source_model"), "retained source model differs")
    require(protocol.get("quality_eligibility") == summary.get("quality_eligibility"),
            "summary quality approval differs from retained protocol")
    if protocol.get("quality_eligibility"):
        require(protocol["native"]["kernel"] in ("vnni", "vnni16"), "unexpected quality approval")
    check_quality_eligibility(protocol, input_path.parent)


def finalize(input_path: Path, readme: Path, svg: Path, csv_path: Path, json_path: Path) -> dict:
    paths = [path.resolve() for path in (input_path, readme, svg, csv_path, json_path)]
    require(len(set(paths)) == len(paths), "input and output paths must be distinct")
    source = input_path.read_bytes()
    try:
        summary = json.loads(source)
        rows = validate(summary)
        validate_protocol(summary, input_path)
        identity = {"file": link(input_path, readme), "sha256": hashlib.sha256(source).hexdigest(),
                    "protocol_id": summary["protocol_id"]}
        record = outcomes(summary, rows, identity)
        readme_text = render_readme(readme.read_bytes().decode("utf-8"), summary, record, readme,
                                   input_path, svg, csv_path, json_path)
        # Reuse the existing renderer, with deterministic cell order, not a second plotting convention.
        svg_text = figure({**summary, "results": rows})
        contents = {svg: svg_text.encode(), csv_path: csv_text(record).encode(),
                    json_path: (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
                    readme: readme_text.encode()}
    except (KeyError, TypeError, IndexError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid final summary: {exc}") from exc
    publish(contents)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/v2/final/summary.json")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--svg", type=Path, default=ROOT / "results/v2/final/decode.svg")
    parser.add_argument("--csv", type=Path, default=ROOT / "results/v2/final/cell-outcomes.csv")
    parser.add_argument("--json", type=Path, default=ROOT / "results/v2/final/cell-outcomes.json")
    args = parser.parse_args(argv)
    try:
        record = finalize(args.input, args.readme, args.svg, args.csv, args.json)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"schema": record["schema"], "counts": record["counts"],
                      "numeric_targets_met": record["numeric_targets_met"]}, sort_keys=True))


if __name__ == "__main__":
    main()
