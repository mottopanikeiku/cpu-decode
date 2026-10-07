"""SIM fixtures only: these tests do not measure or publish model performance."""
from copy import deepcopy
import csv
import hashlib
import io
import json

import numpy as np
import pytest

from tools import cloud_summary as cloud
from tools.portable import ROOT


def synthetic_raw(native_seconds=None, llama_seconds=None):
    """Six complete SIM cells, with no model downloads or generated real claims."""
    native_seconds = [1.0] * 16 if native_seconds is None else native_seconds
    llama_seconds = [1.0] * 16 if llama_seconds is None else llama_seconds
    assert len(native_seconds) == len(llama_seconds)
    cells = []
    for threads in cloud.THREADS:
        for context in cloud.CONTEXTS:
            shared = {"threads": threads, "context": context, "steps": 128,
                      "cpu_set": list(range(threads))}
            cells.append({
                "threads": threads, "context": context, "steps": 128,
                "native_settings": {**shared, "backend": "native", "model": "$MODEL/native",
                                    "kernel": "vnni16", "requested_kernel": "auto", "poll": None},
                "llama_settings": {**shared, "backend": "llama", "model": "$MODEL/model.gguf",
                                   "flash": "off", "repack": True, "poll": 50},
                "pairs": [{"id": index, "block": index // 2, "order": "AB" if index % 2 == 0 else "BA",
                           "native": {"seconds": a, "tokens": list(range(128)), "next_token": 42},
                           "llama": {"seconds": b, "tokens": list(range(1, 129)), "next_token": 43}}
                          for index, (a, b) in enumerate(zip(native_seconds, llama_seconds))],
            })
    return {"design_sha256": "d" * 64, "fixture_kind": "SIM", "cells": cells,
            "environment": {"cpu": "synthetic", "container_cpu": 8, "memory_gib": 8,
                            "build_flags": {"GGML_OPENMP": False}},
            "artifacts": {"native": {"sha256": "a" * 64}, "llama": {"sha256": "b" * 64}},
            "cost": {"kind": "SIM", "usd": 0.0},
            "pilot": [{"kind": "SIM", "settings": {"flash": "off"}}]}


@pytest.mark.parametrize("ratio,decision", [(1.0, "inconclusive"), (2.0, "native-win"),
                                            (0.5, "llama-win")])
def test_identical_and_known_ratio(ratio, decision):
    raw = synthetic_raw(llama_seconds=[ratio] * 16)
    before = deepcopy(raw)
    summary = cloud.summarize(raw)
    assert raw == before
    assert summary["fixture_kind"] == "SIM"
    assert summary["design_sha256"] == raw["design_sha256"]
    for key in ("environment", "artifacts", "cost", "pilot"):
        assert summary[key] == raw[key]
        assert summary[key] is not raw[key]
    assert len(summary["cells"]) == 6
    for cell in summary["cells"]:
        assert cell["native_over_llama"] == {"median": ratio, "ci95": [ratio, ratio]}
        assert cell["decision"] == decision
        assert cell["pairs"] == 16 and cell["blocks"] == 8
        assert cell["native"]["tokens_per_second"]["median"] == 128.0
        assert cell["llama"]["tokens_per_second"]["median"] == 128.0 / ratio
        assert cell["native"]["samples"] == cell["llama"]["samples"] == 16
        assert [pair["native_over_llama"] for pair in cell["paired_ratios"]] == [ratio] * 16
        assert all(pair["native_next_token"] == 42 and pair["llama_next_token"] == 43
                   for pair in cell["paired_ratios"])
        assert all(pair["native_tokens"] == list(range(128)) and
                   pair["llama_tokens"] == list(range(1, 129)) for pair in cell["paired_ratios"])
        assert cell["paired_ratios"][0]["native_tokens"] is not raw["cells"][0]["pairs"][0]["native"]["tokens"]
    assert summary["statistics"]["bootstrap_seed"] == 20261007
    assert summary["statistics"]["bootstrap_draws"] == 20_000
    assert summary["statistics"]["confidence_level"] == 0.95
    assert summary["statistics"]["percentile_method"] == "linear"


def test_primary_estimand_is_paired_not_ratio_of_engine_medians():
    native = [1.0, 2.0, 3.0, 100.0] * 4
    llama = [2.0, 4.0, 9.0, 100.0] * 4
    cell = cloud.summarize(synthetic_raw(native, llama))["cells"][0]
    assert cell["native_over_llama"]["median"] == 2.0
    assert [pair["native_over_llama"] for pair in cell["paired_ratios"]] == [2.0, 2.0, 3.0, 1.0] * 4
    assert cell["llama"]["seconds"] == {"median": 6.5, "min": 2.0, "max": 100.0}
    assert cell["native"]["seconds"] == {"median": 2.5, "min": 1.0, "max": 100.0}
    assert cell["native"]["tokens_per_second"]["median"] == pytest.approx((128 / 2 + 128 / 3) / 2)
    assert cell["native"]["tokens_per_second"]["min"] == 128 / 100
    assert cell["native"]["tokens_per_second"]["max"] == 128
    unpaired = cell["native"]["tokens_per_second"]["median"] / cell["llama"]["tokens_per_second"]["median"]
    assert unpaired == pytest.approx(30 / 13)
    assert cell["native_over_llama"]["median"] != pytest.approx(unpaired)


def test_block_sampling_preserves_both_observations():
    # Every complete block contains one low and one high ratio. Whole-block
    # resampling must retain the balance; independent pair resampling does not.
    ratios = np.array([[0.5, 2.0]] * 8)
    assert cloud.block_bootstrap_ci(ratios) == [1.25, 1.25]
    rng = np.random.default_rng(20261007)
    flattened = ratios.reshape(-1)
    indices = rng.integers(0, len(flattened), size=(20_000, len(flattened)))
    independent_ci = np.percentile(np.median(flattened[indices], axis=1), [2.5, 97.5])
    assert independent_ci.tolist() == [0.5, 2.0]


def test_block_sampling_matches_independent_reference_and_is_reproducible():
    ratios = np.array([[0.4, 0.8], [0.6, 1.2], [0.9, 1.1], [1.0, 2.0],
                       [1.5, 3.0], [2.0, 4.0], [3.0, 0.5], [5.0, 0.7]])
    rng = np.random.default_rng(20261007)
    indices = rng.integers(0, 8, size=(20_000, 8))
    # Explicitly concatenate each selected block's two observations, rather
    # than using the implementation's vectorized indexing and chunking.
    medians = np.array([np.median(np.concatenate([ratios[index] for index in draw]))
                        for draw in indices])
    expected = np.percentile(medians, [2.5, 97.5], method="linear").tolist()
    actual = cloud.block_bootstrap_ci(ratios)
    assert actual == expected
    assert cloud.block_bootstrap_ci(ratios) == actual


def test_ci_straddling_one_is_inconclusive_even_with_median_above_one():
    ratios = [0.5] * 6 + [2.0] * 10
    cell = cloud.summarize(synthetic_raw(llama_seconds=ratios))["cells"][0]
    assert cell["native_over_llama"]["median"] == 2.0
    assert cell["native_over_llama"]["ci95"][0] < 1 < cell["native_over_llama"]["ci95"][1]
    assert cell["decision"] == "inconclusive"


@pytest.mark.parametrize("interval", [[1.0, 2.0], [0.5, 1.0], [1.0, 1.0]])
def test_ci_touching_one_is_inconclusive(interval, monkeypatch):
    # SIM boundary intervals isolate the decision rule, not the estimator.
    monkeypatch.setattr(cloud, "block_bootstrap_ci", lambda ratios: interval)
    cells = cloud.summarize(synthetic_raw())["cells"]
    assert all(cell["decision"] == "inconclusive" for cell in cells)


def test_more_than_minimum_complete_blocks_are_all_used():
    raw = synthetic_raw([1.0] * 20, [2.0] * 20)
    rows = cloud.validate(raw)
    assert len(rows[0]["pairs"]) == 20
    cell = cloud.summarize(raw)["cells"][0]
    assert cell["pairs"] == 20 and cell["blocks"] == 10
    assert cell["native"]["samples"] == 20
    assert len(cell["paired_ratios"]) == 20
    assert cell["native_over_llama"] == {"median": 2.0, "ci95": [2.0, 2.0]}


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "threads", "context", "steps", "bool_threads"])
def test_invalid_matrix(change):
    raw = synthetic_raw()
    if change == "missing":
        raw["cells"].pop()
    elif change == "duplicate":
        raw["cells"][-1] = deepcopy(raw["cells"][0])
    elif change == "extra":
        raw["cells"].append(deepcopy(raw["cells"][0]))
    elif change == "bool_threads":
        raw["cells"][0]["threads"] = True
    else:
        raw["cells"][0][change] = {"threads": 8, "context": 1024, "steps": 64}[change]
    with pytest.raises(ValueError):
        cloud.validate(raw)


def test_cell_order_does_not_change_summary():
    raw = synthetic_raw()
    expected = cloud.summarize(raw)
    raw["cells"].reverse()
    assert cloud.summarize(raw) == expected


@pytest.mark.parametrize("change", ["ABAB", "BAAB", "reversed", "block_split", "duplicate_id", "skip_id",
                                    "float_id", "bool_block", "too_few", "odd"])
def test_invalid_interleaving(change):
    raw = synthetic_raw()
    pairs = raw["cells"][0]["pairs"]
    if change == "ABAB":
        pairs[1]["order"] = "AB"
    elif change == "BAAB":
        pairs[0]["order"], pairs[1]["order"] = "BA", "AB"
    elif change == "reversed":
        pairs.reverse()
    elif change == "block_split":
        pairs[1]["block"] = 1
    elif change == "duplicate_id":
        pairs[1]["id"] = 0
    elif change == "skip_id":
        pairs[0]["id"] = 1
    elif change == "float_id":
        pairs[0]["id"] = 0.0
    elif change == "bool_block":
        pairs[0]["block"] = False
    else:
        del pairs[14 if change == "too_few" else 15:]
    with pytest.raises(ValueError):
        cloud.validate(raw)


@pytest.mark.parametrize("backend", ["native", "llama"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf"), True, "1", None, 10 ** 1000])
def test_invalid_seconds(backend, value):
    raw = synthetic_raw()
    raw["cells"][0]["pairs"][0][backend]["seconds"] = value
    with pytest.raises(ValueError, match="seconds"):
        cloud.validate(raw)


@pytest.mark.parametrize("backend", ["native", "llama"])
@pytest.mark.parametrize("value", [0, -128, 127, 128, 129, 128.0, float("nan"), float("inf"), True, "128", None,
                                 [], [0] * 127, [0] * 129, [True] * 128, [0.0] * 128,
                                 [-1] * 128, ["0"] * 128, [None] * 128,
                                 [float("nan")] * 128, [float("inf")] * 128])
def test_invalid_consumed_token_ids(backend, value):
    raw = synthetic_raw()
    raw["cells"][0]["pairs"][0][backend]["tokens"] = value
    with pytest.raises(ValueError, match="tokens"):
        cloud.validate(raw)


@pytest.mark.parametrize("value", [-1, 42.0, True, None])
def test_invalid_next_token(value):
    raw = synthetic_raw()
    raw["cells"][0]["pairs"][0]["native"]["next_token"] = value
    with pytest.raises(ValueError, match="next_token"):
        cloud.validate(raw)


@pytest.mark.parametrize("key", ["native", "llama"])
def test_missing_backend_sample(key):
    raw = synthetic_raw()
    del raw["cells"][0]["pairs"][0][key]
    with pytest.raises(ValueError, match="sample"):
        cloud.validate(raw)


@pytest.mark.parametrize("backend,key,value", [
    ("native", "kernel", "auto"), ("native", "kernel", "unknown"),
    ("llama", "flash", "unknown"), ("llama", "poll", 0), ("llama", "poll", 50.0),
    ("llama", "poll", None), ("native", "poll", 50), ("native", "poll", False),
    ("native", "threads", 2), ("llama", "context", 4096), ("native", "steps", 64),
    ("native", "backend", "llama"), ("llama", "model", ""),
    ("native", "cpu_set", [True]), ("native", "cpu_set", [-1]),
    ("llama", "cpu_set", [1]), ("native", "cpu_set", [0, 0]),
])
def test_mismatched_or_invalid_settings(backend, key, value):
    raw = synthetic_raw()
    raw["cells"][0][f"{backend}_settings"][key] = value
    with pytest.raises(ValueError):
        cloud.validate(raw)


@pytest.mark.parametrize("backend,key", [("native", "kernel"), ("native", "cpu_set"),
                                       ("llama", "flash"), ("llama", "poll"), ("llama", "cpu_set")])
def test_required_meaningful_setting_identity(backend, key):
    raw = synthetic_raw()
    del raw["cells"][0][f"{backend}_settings"][key]
    with pytest.raises(ValueError):
        cloud.validate(raw)


@pytest.mark.parametrize("kernel", cloud.NATIVE_KERNELS)
def test_actual_resolved_native_kernels(kernel):
    raw = synthetic_raw()
    raw["cells"][0]["native_settings"]["kernel"] = kernel
    assert cloud.validate(raw)[0]["native_settings"]["kernel"] == kernel


def test_settings_may_vary_by_cell_and_cpu_order_is_not_identity():
    raw = synthetic_raw()
    raw["cells"][1]["native_settings"]["kernel"] = "simd256"
    raw["cells"][1]["llama_settings"]["flash"] = "on"
    raw["cells"][-1]["llama_settings"]["cpu_set"].reverse()
    assert len(cloud.validate(raw)) == 6


@pytest.mark.parametrize("location", ["sample", "pair"])
@pytest.mark.parametrize("backend", ["native", "llama"])
def test_optional_per_sample_settings_must_match_cell(location, backend):
    raw = synthetic_raw()
    cell = raw["cells"][0]
    config = deepcopy(cell[f"{backend}_settings"])
    if location == "sample":
        cell["pairs"][0][backend]["settings"] = config
    else:
        cell["pairs"][0][f"{backend}_settings"] = config
    cloud.validate(raw)
    config["model"] = "$MODEL/different"
    with pytest.raises(ValueError, match="mismatched"):
        cloud.validate(raw)


@pytest.mark.parametrize("key,value", [("design_sha256", "d" * 63), ("design_sha256", None),
                                      ("environment", {}), ("artifacts", {}), ("cells", None)])
def test_missing_or_invalid_raw_identity(key, value):
    raw = synthetic_raw()
    raw[key] = value
    with pytest.raises(ValueError):
        cloud.validate(raw)


@pytest.mark.parametrize("ratios", [np.ones((7, 2)), np.ones((8, 3)), np.ones(16),
                                    np.full((8, 2), np.nan), np.full((8, 2), np.inf),
                                    np.zeros((8, 2)), -np.ones((8, 2))])
def test_invalid_bootstrap_input(ratios):
    with pytest.raises(ValueError):
        cloud.block_bootstrap_ci(ratios)


def test_invalid_derived_ratio_is_not_published():
    raw = synthetic_raw()
    raw["cells"][0]["pairs"][0]["native"]["seconds"] = 1e-300
    raw["cells"][0]["pairs"][0]["llama"]["seconds"] = 1e300
    with pytest.raises(ValueError, match="paired ratio"):
        cloud.summarize(raw)


def test_file_outputs_preserve_identity_and_are_byte_reproducible(tmp_path):
    source = tmp_path / "raw.json"
    output = tmp_path / "summary.json"
    table = tmp_path / "summary.csv"
    raw = synthetic_raw()
    raw["artifacts"]["native"]["path"] = str(ROOT / "build/cloud-bench")
    source.write_text(json.dumps(raw) + "\n")
    original_bytes = source.read_bytes()
    summary = cloud.summarize_file(source, output, table)
    assert summary["raw_result"]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert summary["design_sha256"] == raw["design_sha256"]
    assert summary["artifacts"]["native"]["path"] == "./build/cloud-bench"
    assert summary["artifacts"]["native"]["sha256"] == "a" * 64
    assert summary["cost"] == raw["cost"]
    assert json.loads(output.read_text()) == summary
    first_output, first_table = output.read_bytes(), table.read_bytes()
    assert source.read_bytes() == original_bytes
    assert cloud.summarize_file(source, output, table) == summary
    assert (output.read_bytes(), table.read_bytes()) == (first_output, first_table)
    rows = list(csv.DictReader(io.StringIO(table.read_text())))
    assert len(rows) == 6
    assert [(int(row["threads"]), int(row["context"])) for row in rows] == [
        (threads, context) for threads in (1, 2, 4) for context in (128, 4096)]
    assert all(row["decision"] == "inconclusive" and float(row["native_over_llama_median"]) == 1
               for row in rows)
    assert float(rows[0]["native_seconds_median"]) == 1.0
    assert json.loads(rows[0]["native_settings"])["kernel"] == "vnni16"


def test_cli_prints_optional_markdown_without_creating_documents(tmp_path, capsys):
    source, output, table = [tmp_path / name for name in ("raw.json", "summary.json", "summary.csv")]
    source.write_text(json.dumps(synthetic_raw()))
    argv = ["--input", str(source), "--output", str(output), "--csv", str(table)]
    cloud.main(argv)
    assert capsys.readouterr().out == ""
    cloud.main([*argv, "--markdown"])
    printed = capsys.readouterr().out
    assert printed.startswith("| Threads | Context |")
    assert printed.count("inconclusive") == 6
    assert set(path.name for path in tmp_path.iterdir()) == {"raw.json", "summary.json", "summary.csv"}


@pytest.mark.parametrize("collision", ["source_json", "source_csv", "json_csv"])
def test_output_path_collisions_do_not_modify_source(tmp_path, collision):
    source, output, table = [tmp_path / name for name in ("raw.json", "summary.json", "summary.csv")]
    source.write_text(json.dumps(synthetic_raw()))
    original = source.read_bytes()
    if collision == "source_json":
        output = source
    elif collision == "source_csv":
        table = source
    else:
        table = output
    with pytest.raises(ValueError, match="distinct"):
        cloud.summarize_file(source, output, table)
    assert source.read_bytes() == original


@pytest.mark.parametrize("bad_input", ["incomplete", "nonfinite_sample", "nonfinite_metadata"])
def test_invalid_input_does_not_replace_existing_outputs(tmp_path, bad_input):
    source, output, table = [tmp_path / name for name in ("raw.json", "summary.json", "summary.csv")]
    raw = synthetic_raw()
    if bad_input == "incomplete":
        raw["cells"].pop()
    elif bad_input == "nonfinite_sample":
        raw["cells"][0]["pairs"][0]["llama"]["seconds"] = float("nan")
    else:
        # JSON exponent overflow is not parse_constant; allow_nan=False on the
        # retained metadata must still reject it before either output is saved.
        raw["cost"]["usd"] = "EXPONENT_PLACEHOLDER"
    text = json.dumps(raw).replace('"EXPONENT_PLACEHOLDER"', '1e999')
    source.write_text(text)
    output.write_text("existing summary\n")
    table.write_text("existing CSV\n")
    with pytest.raises(ValueError):
        cloud.summarize_file(source, output, table)
    assert output.read_text() == "existing summary\n"
    assert table.read_text() == "existing CSV\n"
