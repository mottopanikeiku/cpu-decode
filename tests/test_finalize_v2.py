"""SIM fixtures only: publication tests do not load models or measure performance."""
from copy import deepcopy
import csv
import hashlib
import io
import json

import pytest

from tools import finalize_v2 as finalizer
from tools import measure_v2
from tools.portable import portable
from test_vnni16_quality_gate import fixture16
from tools.figure_v2 import figure
from tools.measure_v2 import CONTEXTS, THREADS, candidates


README = """# Synthetic publication fixture

Unchanged introduction and [quality scope](quality.json).

<!-- FINAL_RESULT_START -->
**Not yet run.**
<!-- FINAL_RESULT_END -->

## Final matrix

<!-- FINAL_TABLE_START -->
**Not yet run.**
<!-- FINAL_TABLE_END -->

Unchanged limitations: one synthetic fixture, not a model run.

Written with AI coding assistance.
"""


def synthetic_stats(median, spread=2.0, samples=10):
    lo, hi = median * (1 - spread / 200), median * (1 + spread / 200)
    actual_spread = 100 * (hi - lo) / median
    return {"median": median, "min": lo, "max": hi, "samples": samples,
            "spread_percent": actual_spread, "noisy_over_5_percent": actual_spread > 5}


def synthetic_summary():
    """Fifteen SIM cells: five native wins, five Q8 wins, five exact ties."""
    rows = []
    thresholds = dict(zip(finalizer.TARGET_KEYS, (1.05, 85.0, 75.0)))
    for thread in THREADS:
        for index, context in enumerate(CONTEXTS):
            native = (69.58, 60.0, 65.13)[index]
            matched_cpus = list(range(thread))
            measured = []
            for candidate_index, candidate in enumerate(candidates([50])):
                q8 = 65.13 if candidate_index == 8 else 50.0 + candidate_index
                baseline_cpus = list(range(12)) if candidate["affinity"] == "defaults" else matched_cpus
                measured.append({
                    "candidate": candidate,
                    "native_tps": synthetic_stats(native, 8.0 if context == 1024 else 2.0),
                    "baseline_tps": synthetic_stats(q8, 6.0 if context == 4096 else 2.0),
                    "invocations_per_engine": 2,
                    "baseline_process_cpu_set": baseline_cpus,
                    "comparison_core_sets_matched": matched_cpus == baseline_cpus,
                    "raw": [{"engine": engine, "round": round_id,
                             "command": ["native", "--steps", "64", "--repeats", "5"] if engine == "native"
                             else ["llama-bench", "-n", "64", "-r", "5"]}
                            for round_id in range(2) for engine in ("native", "llama")],
                })
            winner = measured[-1]
            ratio, percent = native / 65.13, native
            rows.append({
                "threads": thread, "context": context, "cpu_set": matched_cpus,
                "native_tps": winner["native_tps"], "best_baseline_tps": winner["baseline_tps"],
                "winner": winner, "candidates": measured, "missing_candidates": [],
                "native_samples_all_candidates": 90,
                "winner_core_sets_matched": winner["comparison_core_sets_matched"],
                "winner_baseline_cpu_set": winner["baseline_process_cpu_set"],
                "native_over_best_baseline": ratio, "read_ceiling_tps": 100.0,
                "percent_of_ceiling": percent, "read_GB_per_s": synthetic_stats(10.0, samples=5),
                "targets": {finalizer.TARGET_KEYS[0]: ratio >= thresholds[finalizer.TARGET_KEYS[0]],
                            finalizer.TARGET_KEYS[1]: percent >= 85 if context == 128 else None,
                            finalizer.TARGET_KEYS[2]: percent >= 75 if context == 4096 else None},
                "noisy_over_5_percent": context in (1024, 4096),
            })
    return {
        "schema": "cpu-decode-v2-summary", "fixture_kind": "SIM", "protocol_id": "a" * 64,
        "complete_final_matrix": True, "complete_requested_matrix": True,
        "full_fifteen_cell_target_matrix": True, "target_eligible": True, "development": False,
        "matrix_scope": "full-fifteen-cell", "rates_kind": "MEAS", "read_ceiling_kind": "EXT",
        "source_model": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
        "requested_threads": THREADS, "requested_contexts": CONTEXTS,
        "expected_cells": [[t, c] for t in THREADS for c in CONTEXTS],
        "expected_candidate_windows": 135, "expected_bandwidth_windows": 5,
        "missing_cells": [], "missing_bandwidth_threads": [], "failures": [],
        "thresholds": thresholds, "target_predicates": {key: False for key in finalizer.TARGET_KEYS},
        "numeric_targets_met": False, "results": rows,
        "quality_eligibility": None,
    }


@pytest.fixture
def publication(tmp_path):
    input_path = tmp_path / "results/v2/final/summary.json"
    input_path.parent.mkdir(parents=True)
    summary = synthetic_summary()
    protocol = {"development": False, "native": {"kernel": "simd512x4"},
                "source_model": summary["source_model"], "quality_eligibility": None}
    protocol["id"] = finalizer.digest(protocol)
    summary["protocol_id"] = protocol["id"]
    input_path.write_text(json.dumps(summary, indent=2) + "\n")
    input_path.with_name("protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    readme = tmp_path / "README.md"
    readme.write_text(README)
    return (input_path, readme, input_path.with_name("decode.svg"),
            input_path.with_name("cell-outcomes.csv"), input_path.with_name("cell-outcomes.json"))


@pytest.fixture
def approved_publication(publication, monkeypatch):
    root = publication[1].parent
    quality_dir = root / "results/v2"
    path, native, artifacts, _, _ = fixture16(quality_dir)
    quality_path = path.with_name("quality-vnni16-final.json")
    path.rename(quality_path)
    proof = portable(measure_v2.load_quality_eligibility(quality_path, native, artifacts), {root: "."})
    protocol = {"development": False, "native": native, "artifacts": artifacts,
                "source_model": proof["selection"]["oracle_identity"]["verified_source"],
                "quality_eligibility": proof}
    protocol["id"] = finalizer.digest(protocol)
    summary = json.loads(publication[0].read_text())
    summary.update(protocol_id=protocol["id"], source_model=protocol["source_model"], quality_eligibility=proof)
    publication[0].write_text(json.dumps(summary, indent=2) + "\n")
    publication[0].with_name("protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    monkeypatch.setattr(measure_v2, "ROOT", root)
    return publication


def run(publication):
    return finalizer.finalize(*publication)


def protected_bytes(publication):
    return {path: path.read_bytes() if path.exists() else None for path in publication[1:]}


def change_summary(publication, mutation):
    summary = json.loads(publication[0].read_text())
    mutation(summary)
    publication[0].write_text(json.dumps(summary))


def test_every_cell_stats_flags_targets_noise_and_input_identity(publication):
    record = run(publication)
    summary = json.loads(publication[0].read_text())
    assert record == json.loads(publication[4].read_text())
    assert record["schema"] == "cpu-decode-v2-cell-outcomes"
    assert record["input"] == {"file": "results/v2/final/summary.json",
                               "sha256": hashlib.sha256(publication[0].read_bytes()).hexdigest(),
                               "protocol_id": summary["protocol_id"]}
    assert record["counts"] == {"native": 5, "Q8": 5, "tie": 5}
    assert record["cell_count"] == 15 and record["noisy_cell_count"] == 10
    assert record["candidate_count_per_cell"] == 9 and record["samples_per_winning_arm"] == 10
    assert record["thresholds"] == summary["thresholds"]
    assert record["target_predicates"] == summary["target_predicates"]
    assert record["numeric_targets_met"] is False
    csv_rows = list(csv.DictReader(io.StringIO(publication[3].read_text())))
    assert len(csv_rows) == 15
    for original, cell, flat in zip(summary["results"], record["results"], csv_rows, strict=True):
        assert cell["native_tps"] == original["native_tps"]
        assert cell["best_q8_tps"] == original["best_baseline_tps"]
        assert cell["winner"] == original["winner"]["candidate"]
        assert cell["native_cpu_set"] == original["cpu_set"]
        assert cell["winner_core_sets_matched"] is original["winner_core_sets_matched"]
        assert cell["winner_baseline_cpu_set"] == original["winner_baseline_cpu_set"]
        assert cell["targets"] == original["targets"]
        assert cell["percent_of_ceiling"] == original["percent_of_ceiling"]
        assert cell["noisy_over_5_percent"] is original["noisy_over_5_percent"]
        expected_outcome = {128: "native", 1024: "Q8", 4096: "tie"}[cell["context"]]
        assert cell["outcome"] == flat["outcome"] == expected_outcome
        assert int(flat["threads"]) == cell["threads"] and int(flat["context"]) == cell["context"]
        assert json.loads(flat["winner_flags"]) == cell["winner"]
        assert flat["winner_id"] == cell["winner"]["id"]
        assert json.loads(flat["native_cpu_set"]) == cell["native_cpu_set"]
        assert json.loads(flat["winner_baseline_cpu_set"]) == cell["winner_baseline_cpu_set"]
        assert flat["winner_core_sets_matched"] == str(cell["winner_core_sets_matched"])
        for arm in ("native", "best_q8"):
            for key in ("median", "min", "max", "spread_percent", "samples"):
                assert float(flat[f"{arm}_{key}"]) == cell[f"{arm}_tps"][key]
            assert flat[f"{arm}_noisy_over_5_percent"] == str(cell[f"{arm}_tps"]["noisy_over_5_percent"])
        assert float(flat["native_over_best_q8"]) == cell["native_tps"]["median"] / cell["best_q8_tps"]["median"]
        assert float(flat["read_ceiling_tps"]) == cell["read_ceiling_tps"]
        assert float(flat["percent_of_ceiling"]) == cell["percent_of_ceiling"]
        for key in finalizer.TARGET_KEYS:
            assert flat[key] == ("" if cell["targets"][key] is None else str(cell["targets"][key]))
        assert flat["noisy_over_5_percent"] == str(cell["noisy_over_5_percent"])
    assert publication[2].read_text() == figure(summary)
    assert "Full final matrix" in publication[2].read_text()


def outside_markers(text):
    for name in ("FINAL_RESULT", "FINAL_TABLE"):
        start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
        left, right = text.index(start) + len(start), text.index(end)
        text = text[:left] + text[right:]
    return text


def test_readme_compact_table_captions_losses_quality_and_preservation(approved_publication):
    publication = approved_publication
    run(publication)
    text = publication[1].read_text()
    assert outside_markers(text) == outside_markers(README)
    assert len(text.split()) <= 900
    assert text.endswith("Written with AI coding assistance.\n")
    assert "5 native wins, 5 Q8_0 wins and 5 ties" in text
    assert "targets are **not met**" in text
    assert "10 cells have >5% spread" in text
    assert "not better perplexity or downstream task accuracy" in text
    assert "results/v2/quality-vnni16-final.json" in text
    assert "Native loses 5 cells and ties 5" in text
    assert "64 measured tokens, five repeats" in text and "ABAB: ten interleaved samples" in text
    assert "nine configurations" in text and "no independent holdout" in text
    assert "Read-ceiling percentages are EXT estimates" in text
    assert "results/v2/final/cell-outcomes.csv" in text
    assert "results/v2/final/cell-outcomes.json" in text
    assert "results/v2/final/decode.svg" in text
    for thread in THREADS:
        assert f"|{thread}|69.58/65.13 (1.07×)|60.00/65.13 (0.92×)|65.13/65.13 (1.00×)|" in text
    assert "**Not yet run.**" not in text


def test_repeat_command_identical_and_reordered_rows_deterministic(publication):
    run(publication)
    before = protected_bytes(publication)
    before_mtimes = {path: path.stat().st_mtime_ns for path in publication[1:]}
    run(publication)
    assert protected_bytes(publication) == before
    assert {path: path.stat().st_mtime_ns for path in publication[1:]} == before_mtimes
    change_summary(publication, lambda s: s["results"].reverse())
    run(publication)
    after = protected_bytes(publication)
    assert all(after[path] == before[path] for path in publication[1:4])
    previous, current = json.loads(before[publication[4]]), json.loads(after[publication[4]])
    assert current["input"]["sha256"] != previous["input"]["sha256"]
    del current["input"]["sha256"], previous["input"]["sha256"]
    assert current == previous


@pytest.mark.parametrize("fault", [
    "partial", "requested_partial", "development", "subset", "missing_cell", "duplicate_cell",
    "missing_candidate", "duplicate_candidate", "candidate_flags", "candidate_set_drift", "missing_records",
    "candidate_samples", "native_samples", "q8_samples", "invalid_range", "zero_rate", "negative_rate",
    "nan_rate", "infinite_rate", "spread", "noise", "cell_noise", "ratio", "ceiling_percent", "ceiling",
    "winner", "winning_stats", "target", "aggregate_target", "target_disposition", "core_match",
    "failures", "missing_bandwidth", "missing_candidates", "sampling_steps", "sampling_repeats", "abab",
    "source", "schema", "missing_field", "protocol_id",
])
@pytest.mark.parametrize("existing_outputs", [False, True])
def test_invalid_summary_never_changes_any_publication(publication, fault, existing_outputs):
    if existing_outputs:
        run(publication)
    before = protected_bytes(publication)
    def mutate(summary):
        row = summary["results"][0]
        entry = row["candidates"][0]
        winner = row["winner"]
        if fault == "partial": summary["complete_final_matrix"] = False
        elif fault == "requested_partial": summary["complete_requested_matrix"] = False
        elif fault == "development": summary["development"] = True
        elif fault == "subset": summary["matrix_scope"] = "explicit-subset"
        elif fault == "missing_cell": summary["results"].pop()
        elif fault == "duplicate_cell": summary["results"][-1] = deepcopy(row)
        elif fault == "missing_candidate": row["candidates"].pop(0)
        elif fault == "duplicate_candidate": row["candidates"][1] = deepcopy(entry)
        elif fault == "candidate_flags": entry["candidate"]["flash_attn"] = "invented"
        elif fault == "candidate_set_drift": entry["candidate"]["poll"] = 0
        elif fault == "missing_records": del entry["raw"]
        elif fault == "candidate_samples": entry["baseline_tps"]["samples"] = 9
        elif fault == "native_samples": winner["native_tps"]["samples"] = 9
        elif fault == "q8_samples": winner["baseline_tps"]["samples"] = 9
        elif fault == "invalid_range": entry["native_tps"]["min"] = 1000
        elif fault == "zero_rate": entry["native_tps"]["median"] = 0
        elif fault == "negative_rate": entry["native_tps"]["min"] = -1
        elif fault == "nan_rate": entry["baseline_tps"]["median"] = float("nan")
        elif fault == "infinite_rate": entry["baseline_tps"]["max"] = float("inf")
        elif fault == "spread": entry["native_tps"]["spread_percent"] = 1
        elif fault == "noise": entry["native_tps"]["noisy_over_5_percent"] = True
        elif fault == "cell_noise": row["noisy_over_5_percent"] = True
        elif fault == "ratio": row["native_over_best_baseline"] = 1.2
        elif fault == "ceiling_percent": row["percent_of_ceiling"] = 80
        elif fault == "ceiling": row["read_ceiling_tps"] = None
        elif fault == "winner": row["winner"] = deepcopy(entry)
        elif fault == "winning_stats": row["native_tps"]["median"] += 1
        elif fault == "target": row["targets"][finalizer.TARGET_KEYS[0]] = False
        elif fault == "aggregate_target": summary["target_predicates"][finalizer.TARGET_KEYS[0]] = True
        elif fault == "target_disposition": summary["numeric_targets_met"] = True
        elif fault == "core_match": entry["comparison_core_sets_matched"] = not entry["comparison_core_sets_matched"]
        elif fault == "failures": summary["failures"] = [{"error": "synthetic failure"}]
        elif fault == "missing_bandwidth": summary["missing_bandwidth_threads"] = [12]
        elif fault == "missing_candidates": row["missing_candidates"] = ["off-pinned-poll50"]
        elif fault == "sampling_steps": entry["raw"][0]["command"][2] = "63"
        elif fault == "sampling_repeats": entry["raw"][1]["command"][4] = "4"
        elif fault == "abab": entry["raw"].reverse()
        elif fault == "source": summary["source_model"]["model_id"] = "Qwen/Qwen2.5-1.5B-Instruct"
        elif fault == "schema": summary["schema"] = "other"
        elif fault == "missing_field": del row["best_baseline_tps"]
        elif fault == "protocol_id": summary["protocol_id"] = ""
    change_summary(publication, mutate)
    with pytest.raises(ValueError):
        run(publication)
    assert protected_bytes(publication) == before
    assert not list(publication[0].parent.glob(".*"))


@pytest.mark.parametrize("fault", ["missing_marker", "duplicate_marker", "reversed_marker", "nested_markers",
                                   "overflow", "banned"])
def test_invalid_readme_never_changes_any_publication(publication, fault):
    text = README
    if fault == "missing_marker": text = text.replace("<!-- FINAL_TABLE_END -->", "")
    elif fault == "duplicate_marker": text += "<!-- FINAL_RESULT_START -->"
    elif fault == "reversed_marker":
        text = text.replace("FINAL_RESULT_START", "TEMP").replace("FINAL_RESULT_END", "FINAL_RESULT_START").replace("TEMP", "FINAL_RESULT_END")
    elif fault == "nested_markers":
        text = text.replace("<!-- FINAL_RESULT_END -->", "").replace("<!-- FINAL_TABLE_START -->", "<!-- FINAL_TABLE_START --><!-- FINAL_RESULT_END -->")
    elif fault == "overflow": text += " word" * 901
    elif fault == "banned": text += " seamless"
    publication[1].write_text(text)
    before = protected_bytes(publication)
    with pytest.raises(ValueError):
        run(publication)
    assert protected_bytes(publication) == before
    assert not list(publication[0].parent.glob(".*"))


def test_figure_failure_is_before_any_writes(publication, monkeypatch):
    before = protected_bytes(publication)
    def broken_figure(summary):
        raise ValueError("synthetic rendering failure")
    monkeypatch.setattr(finalizer, "figure", broken_figure)
    with pytest.raises(ValueError, match="rendering failure"):
        run(publication)
    assert protected_bytes(publication) == before


def test_all_files_staged_before_replacement_and_readme_last(publication, monkeypatch):
    original = finalizer.os.replace
    replacements = []
    def replace(source, destination):
        if not replacements:
            assert len(list(publication[0].parent.glob(".*"))) == 3
            assert len(list(publication[1].parent.glob(".README.md.*"))) == 1
            assert protected_bytes(publication) == before
        replacements.append(destination)
        original(source, destination)
    before = protected_bytes(publication)
    monkeypatch.setattr(finalizer.os, "replace", replace)
    run(publication)
    assert replacements == [publication[2], publication[3], publication[4], publication[1]]


def test_path_collisions_and_bad_json_do_not_write(publication):
    before = protected_bytes(publication)
    with pytest.raises(ValueError, match="distinct"):
        finalizer.finalize(*publication[:3], publication[2], publication[4])
    assert protected_bytes(publication) == before
    publication[0].write_text("not JSON")
    with pytest.raises(ValueError):
        run(publication)
    assert protected_bytes(publication) == before


def test_cli_defaults_and_errors(publication, monkeypatch, capsys):
    monkeypatch.setattr(finalizer, "ROOT", publication[1].parent)
    finalizer.main([])
    output = json.loads(capsys.readouterr().out)
    assert output["counts"] == {"native": 5, "Q8": 5, "tie": 5}
    assert output["schema"] == "cpu-decode-v2-cell-outcomes"
    before = protected_bytes(publication)
    change_summary(publication, lambda s: s.update(complete_final_matrix=False))
    with pytest.raises(SystemExit) as error:
        finalizer.main([])
    assert error.value.code == 2
    assert "complete_final_matrix must be true" in capsys.readouterr().err
    assert protected_bytes(publication) == before


def test_outcome_is_strict_median_not_requested_margin(publication):
    def mutate(summary):
        row = summary["results"][2]
        for entry in [*row["candidates"], row["winner"]]:
            entry["native_tps"] = synthetic_stats(65.14)
        row["native_tps"] = synthetic_stats(65.14)
        row["native_over_best_baseline"] = 65.14 / 65.13
        row["percent_of_ceiling"] = 65.14
        # Q8 spread still makes this cell noisy. The 1.05 target remains false.
    change_summary(publication, mutate)
    record = run(publication)
    assert record["counts"] == {"native": 6, "Q8": 5, "tie": 4}
    assert record["results"][2]["outcome"] == "native"
    assert record["results"][2]["targets"]["native_over_best_baseline"] is False


def test_met_custom_targets_do_not_hide_q8_wins(publication):
    def mutate(summary):
        thresholds = dict(zip(finalizer.TARGET_KEYS, (0.90, 60.0, 60.0)))
        summary["thresholds"] = thresholds
        for row in summary["results"]:
            row["targets"] = {
                finalizer.TARGET_KEYS[0]: row["native_over_best_baseline"] >= 0.90,
                finalizer.TARGET_KEYS[1]: row["percent_of_ceiling"] >= 60 if row["context"] == 128 else None,
                finalizer.TARGET_KEYS[2]: row["percent_of_ceiling"] >= 60 if row["context"] == 4096 else None,
            }
        summary["target_predicates"] = {key: True for key in finalizer.TARGET_KEYS}
        summary["numeric_targets_met"] = True
    change_summary(publication, mutate)
    record = run(publication)
    assert record["counts"] == {"native": 5, "Q8": 5, "tie": 5}
    assert "targets are **met**" in publication[1].read_text()
    assert "Native loses 5 cells" in publication[1].read_text()
    assert record["numeric_targets_met"] is True


def test_no_quality_claim_from_speed_alone(publication):
    change_summary(publication, lambda s: s.update(quality_eligibility=None))
    run(publication)
    text = publication[1].read_text()
    assert "approves KL/agreement" not in text
    assert "Unchanged introduction and [quality scope](quality.json)." in text


@pytest.mark.parametrize("fault", ["protocol", "copied_approval", "quality_missing", "quality_changed",
                                   "calibration", "q8_calibration", "heldout_f16", "heldout_f32", "q8_heldout"])
def test_stale_quality_never_changes_publication(approved_publication, fault):
    publication = approved_publication
    protocol_path = publication[0].with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    if fault == "protocol":
        protocol["native"]["rope"] = "direct"
        protocol_path.write_text(json.dumps(protocol))
    elif fault == "copied_approval":
        change_summary(publication, lambda summary: summary["quality_eligibility"].update(quality_sha256="changed"))
    else:
        quality_dir = publication[1].parent / "results/v2"
        if fault.startswith("quality_"):
            path = quality_dir / "quality-vnni16-final.json"
        else:
            path = quality_dir / protocol["quality_eligibility"]["gate"]["reports"][fault]["report"]
        if fault == "quality_missing":
            path.unlink()
        else:
            path.write_text("{}")
    before = protected_bytes(publication)
    with pytest.raises((ValueError, OSError)):
        run(publication)
    assert protected_bytes(publication) == before


def test_repository_readme_has_room_for_complete_matrix():
    summary = synthetic_summary()
    summary["quality_eligibility"] = {
        "file": "results/v2/quality-vnni16-final.json", "gate": {"approved": True}}
    rows = finalizer.validate(summary)
    record = finalizer.outcomes(summary, rows, {"file": "synthetic", "sha256": "a" * 64,
                                               "protocol_id": summary["protocol_id"]})
    root = finalizer.ROOT
    rendered = finalizer.render_readme((root / "README.md").read_text(), summary, record,
        root / "README.md", root / "results/v2/final/summary.json",
        root / "results/v2/final/decode.svg", root / "results/v2/final/cell-outcomes.csv",
        root / "results/v2/final/cell-outcomes.json")
    assert len(rendered.split()) <= 900
    assert rendered.endswith("Written with AI coding assistance.\n")
