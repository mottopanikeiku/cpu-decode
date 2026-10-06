"""Analytical quality statistics and real pinned corpus integrity (no model loads)."""
import copy
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from tools.corpus_v2 import (
    CORPUS_SHA256, FORMAT_CHOICES, LICENSE, LICENSE_URL, POLICY, SOURCES,
    digest_json, load_manifest, validate_manifest, window_alignment, write_json,
)
from tools.download_model import file_hash
from tools.quality_v2 import (
    checked_logits, choose_format, heldout_comparison, log_probabilities,
    position_metric, preflight_native_model, read_selection, summarize,
    validate_case, validate_heldout_settings, validate_reports,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results/v2/corpus.json"


def test_full_vocabulary_forward_kl_cross_entropy_and_perplexity():
    p = np.array([0.5, 0.25, 0.125, 0.125])
    q = np.array([0.25, 0.5, 0.125, 0.125])
    row = position_metric(np.log(p), np.log(q), 1)
    assert row["kl_reference_candidate_nats"] == pytest.approx(0.25 * math.log(2))
    assert row["reference_cross_entropy_nats"] == pytest.approx(math.log(4))
    assert row["candidate_cross_entropy_nats"] == pytest.approx(math.log(2))
    assert row["reference_top1"] == 0
    assert row["candidate_top1"] == 1
    assert not row["top1_match"]
    result = summarize([row])
    assert result["reference_perplexity"] == pytest.approx(4)
    assert result["perplexity"] == pytest.approx(2)
    # Confirm KL orientation with an asymmetric pair, not only swapped p/q.
    actual = position_metric(np.log([0.9, 0.1]), np.log([0.5, 0.5]), 0)
    assert actual["kl_reference_candidate_nats"] == pytest.approx(0.9 * math.log(1.8) + 0.1 * math.log(0.2))


def test_identical_logits_and_large_offsets_have_zero_kl():
    logits = np.array([10000.0, 9999.0, 9998.0])
    row = position_metric(logits, logits - 100000.0, 2)
    assert row["kl_reference_candidate_nats"] == pytest.approx(0.0, abs=1e-14)
    assert row["top1_match"]
    assert np.exp(log_probabilities(logits)).sum() == pytest.approx(1)


def test_global_position_weighting_p99_and_geometric_perplexity():
    # Unequal window lengths: one position with CE log(2), three with CE log(4).
    first = position_metric(np.log([0.5, 0.5]), np.log([0.5, 0.5]), 0)
    second = position_metric(np.log([0.5, 0.5]), np.log([0.25, 0.75]), 0)
    result = summarize([first] + [second] * 3)
    assert result["positions"] == 4
    assert result["mean_kl_reference_candidate_nats"] == pytest.approx(second["kl_reference_candidate_nats"] * 0.75)
    assert result["next_token_cross_entropy_nats"] == pytest.approx((math.log(2) + 3 * math.log(4)) / 4)
    assert result["perplexity"] == pytest.approx((2 * 4 ** 3) ** 0.25)
    assert result["perplexity"] != pytest.approx((2 + 4) / 2)
    # Linear p99 over 100 positions is 98.01, not a max or mean of window p99s.
    rows = [dict(first, kl_reference_candidate_nats=float(index)) for index in range(100)]
    assert summarize(rows)["p99_kl_reference_candidate_nats"] == pytest.approx(98.01)
    with pytest.raises(ValueError, match="No scored"):
        summarize([])


@pytest.mark.parametrize("reference,candidate,target", [
    ([0, np.nan], [0, 1], 0), ([0, 1], [0, np.inf], 0),
    ([0, 1], [0, 1, 2], 0), ([0, 1], [0, 1], 2),
    ([0, 1], [0, 1], -1), ([0, 1], [0, 1], True),
])
def test_invalid_logits_or_targets_rejected(reference, candidate, target):
    with pytest.raises(ValueError):
        position_metric(reference, candidate, target)


def test_next_token_target_alignment_includes_last_extra_target():
    tokens = list(range(513))
    inputs, positions, targets = window_alignment({"tokens": tokens})
    assert inputs == tokens[:512]
    assert positions == list(range(256, 512))
    assert targets == list(range(257, 513))
    assert len(targets) == 256
    for position, target in zip(positions, targets, strict=True):
        assert target == tokens[position + 1]
        assert inputs[:position + 1] == tokens[:position + 1]
        assert target not in inputs[:position + 1]
    for invalid in (tokens[:-1], tokens + [513], tokens[:-1] + [True]):
        with pytest.raises(ValueError, match="Invalid window"):
            window_alignment({"tokens": invalid})


def test_real_corpus_pins_licenses_partition_and_scored_count():
    corpus = load_manifest(CORPUS)
    assert file_hash(CORPUS) == CORPUS_SHA256
    assert corpus["policy"] == POLICY
    assert SOURCES["calibration"]["id"] != SOURCES["heldout"]["id"]
    counts = {split: 0 for split in SOURCES}
    covered = {split: set() for split in SOURCES}
    for window in corpus["windows"]:
        inputs, positions, targets = window_alignment(window)
        assert len(inputs) == 512
        assert targets == window["tokens"][257:]
        assert digest_json(window["tokens"]) == window["tokens_sha256"]
        indices = set(range(window["source_token_start"], window["source_token_start"] + 513))
        assert covered[window["split"]].isdisjoint(indices)
        covered[window["split"]].update(indices)
        counts[window["split"]] += len(positions)
    assert counts == {"calibration": 512, "heldout": 2048}
    for split, source in corpus["sources"].items():
        assert source["license"] == LICENSE
        assert source["license_url"] == LICENSE_URL
        assert source["sha256"] == SOURCES[split]["sha256"]
        assert file_hash(CORPUS.parent / source["excerpt"]) == source["excerpt_sha256"]


@pytest.mark.parametrize("changed", ["source", "license", "tokens", "partition", "offset", "count", "policy"])
def test_corpus_rejects_pin_license_hash_or_partition_changes(changed):
    data = copy.deepcopy(load_manifest(CORPUS))
    if changed == "source":
        data["sources"]["heldout"]["sha256"] = "0" * 64
    elif changed == "license":
        del data["sources"]["heldout"]["license"]
    elif changed == "tokens":
        data["windows"][0]["tokens"][0] += 1
    elif changed == "partition":
        data["windows"][0]["split"] = "heldout"
    elif changed == "offset":
        data["windows"][1]["source_token_start"] = 0
    elif changed == "count":
        data["windows"].pop()
    else:
        data["policy"] = dict(data["policy"], priming_tokens=0)
    with pytest.raises(ValueError):
        validate_manifest(data)


def test_manifest_pin_prevents_rehashing_altered_tokens(tmp_path):
    data = copy.deepcopy(load_manifest(CORPUS))
    data["windows"][0]["tokens"][0] += 1
    data["windows"][0]["tokens_sha256"] = digest_json(data["windows"][0]["tokens"])
    destination = tmp_path / "corpus.json"
    write_json(destination, data)
    with pytest.raises(ValueError, match="manifest hash changed"):
        load_manifest(destination)


def test_excerpt_text_corruption_rejected(tmp_path):
    corpus = load_manifest(CORPUS)
    for source in corpus["sources"].values():
        (tmp_path / source["excerpt"]).write_bytes((CORPUS.parent / source["excerpt"]).read_bytes())
    (tmp_path / corpus["sources"]["heldout"]["excerpt"]).write_text("changed")
    with pytest.raises(ValueError, match="text hash"):
        validate_manifest(corpus, tmp_path)


def test_raw_float32_size_and_hash_validation(tmp_path):
    path = tmp_path / "logits.bin"
    np.array([[0, 1], [2, 3]], dtype="<f4").tofile(path)
    record = {"shape": [2, 2], "sha256": file_hash(path)}
    array = checked_logits(path, record)
    assert array.tolist() == [[0, 1], [2, 3]]
    del array
    with pytest.raises(ValueError, match="byte count"):
        checked_logits(path, dict(record, shape=[1, 2]))
    path.write_bytes(b"\0" * 16)
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_logits(path, record)


def calibration_choices():
    """Small analytic comparison records, not engine/oracle measurement claims."""
    reports = []
    row = position_metric(np.log([0.5, 0.5]), np.log([0.4, 0.6]), 0)
    for index, (group, dtype) in enumerate(FORMAT_CHOICES):
        settings = {"group_size": group, "scale_dtype": dtype, "weight_dtype": "int8",
            "kv_dtype": "f16", "kernel": "scalar", "attention": "blocked",
            "scheduler": "pool", "threads": 1, "cpu_set": [0]}
        reports.append({"label": f"g{group}{dtype}", "split": "calibration", "backend": "native",
            "settings": settings, "aggregate": summarize([dict(row, kl_reference_candidate_nats=0.01 * (index + 1))]),
            "model_identity": {"sha256": f"weights-{group}-{dtype}"}})
    return reports


def test_calibration_selects_predetermined_budget_choices_not_heldout():
    reports = calibration_choices()
    assert choose_format(reports)["label"] == "g32f16"
    assert all(8 + (16 if dtype == "f16" else 32) / group <= 8.5 for group, dtype in FORMAT_CHOICES)
    reports[0]["split"] = "heldout"
    with pytest.raises(ValueError, match="never heldout"):
        choose_format(reports)
    reports = calibration_choices()
    reports[-1]["settings"]["threads"] = 2
    with pytest.raises(ValueError, match="identical execution"):
        choose_format(reports)
    with pytest.raises(ValueError, match="exactly the four"):
        choose_format(reports[:-1])


def test_calibration_tie_breaks_top1_then_cross_entropy_then_label():
    reports = calibration_choices()
    for report in reports:
        report["aggregate"]["mean_kl_reference_candidate_nats"] = 0.1
    reports[2]["aggregate"]["top1_agreement"] = 1
    assert choose_format(reports) is reports[2]
    for report in reports:
        report["aggregate"]["top1_agreement"] = 1
    reports[3]["aggregate"]["next_token_cross_entropy_nats"] = 0.1
    assert choose_format(reports) is reports[3]
    for report in reports:
        report["aggregate"]["next_token_cross_entropy_nats"] = 0.1
    assert choose_format(reports)["label"] == min(report["label"] for report in reports)


def selected_decision():
    chosen = calibration_choices()[0]
    return {"split": "calibration", "corpus_sha256": CORPUS_SHA256,
        "policy": POLICY["format_selection"], "chosen": dict(chosen["settings"],
        model_identity=chosen["model_identity"])}


def test_heldout_accepts_only_selected_weights_or_original_v1(tmp_path):
    decision = selected_decision()
    chosen = decision["chosen"]
    settings = {"group_size": 32, "scale_dtype": "f16", "weight_dtype": "int8"}
    validate_heldout_settings(settings, chosen["model_identity"], decision)
    validate_heldout_settings(dict(settings, group_size=0, scale_dtype="f32"), {"sha256": "v1"}, decision)
    with pytest.raises(ValueError, match="not selected"):
        validate_heldout_settings(settings, {"sha256": "different-weights"}, decision)
    with pytest.raises(ValueError, match="not selected"):
        validate_heldout_settings(dict(settings, group_size=64), chosen["model_identity"], decision)
    path = tmp_path / "selection.json"
    write_json(path, decision)
    assert read_selection(path, CORPUS_SHA256) == decision
    with pytest.raises(ValueError, match="decision"):
        read_selection(path, "different-corpus")
    with pytest.raises(FileExistsError):
        write_json(path, decision, exclusive=True)


def test_unselected_safetensors_format_rejected_before_engine_execution(tmp_path):
    # A real safetensors container with one int8 matrix exercises header preflight.
    header = {"__metadata__": {"group_size": "64", "scale_dtype": "f16"},
        "matrix.weight": {"dtype": "I8", "shape": [1, 2], "data_offsets": [0, 2]}}
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x01\x02")
    with pytest.raises(ValueError, match="not selected"):
        preflight_native_model(tmp_path, {"sha256": "unselected"}, selected_decision())


def test_oracle_record_rejects_shifted_targets_or_context():
    window = load_manifest(CORPUS)["windows"][0]
    inputs, positions, targets = window_alignment(window)
    record = {"split": window["split"], "tokens_sha256": window["tokens_sha256"],
        "input_tokens": inputs, "logit_positions": positions, "targets": targets, "shape": [256, 151936]}
    validate_case(record, window)
    shifted = dict(record, targets=window["tokens"][256:512])
    with pytest.raises(ValueError, match="alignment"):
        validate_case(shifted, window)
    with pytest.raises(ValueError, match="alignment"):
        validate_case(dict(record, input_tokens=inputs[1:]), window)


def report_alignment_rows(corpus, split):
    row = position_metric([0, 0], [0, 0], 0)
    return [dict(row, window_id=window["id"], input_position=position, target=target)
        for window in corpus["windows"] if window["split"] == split
        for position, target in zip(*window_alignment(window)[1:], strict=True)]


def test_comparison_rejects_incomplete_or_shifted_positions_and_wrong_aggregate():
    corpus = load_manifest(CORPUS)
    rows = report_alignment_rows(corpus, "heldout")
    report = {"split": "heldout", "corpus_sha256": CORPUS_SHA256, "label": "alignment-case",
        "oracle_identity": {"corpus_sha256": CORPUS_SHA256}, "positions": rows, "aggregate": summarize(rows)}
    validate_reports([report], corpus, CORPUS_SHA256, "heldout")
    incomplete = dict(report, positions=rows[:-1], aggregate=summarize(rows[:-1]))
    with pytest.raises(ValueError, match="actual position count"):
        validate_reports([incomplete], corpus, CORPUS_SHA256, "heldout")
    altered = copy.deepcopy(report)
    altered["positions"][0]["target"] += 1
    with pytest.raises(ValueError, match="alignment"):
        validate_reports([altered], corpus, CORPUS_SHA256, "heldout")
    with pytest.raises(ValueError, match="globally weighted"):
        validate_reports([dict(report, aggregate=dict(report["aggregate"], positions=2047))], corpus, CORPUS_SHA256, "heldout")


def test_kv_fixed_weight_pair_and_vnni_measured_quality_rule():
    decision = selected_decision()
    chosen = calibration_choices()[0]
    row = position_metric(np.log([0.6, 0.4]), np.log([0.55, 0.45]), 0)
    fp32 = dict(chosen, split="heldout", label="fp32", positions=[row], aggregate=summarize([row]),
        settings=dict(chosen["settings"], kv_dtype="f32"))
    f16 = dict(fp32, label="f16", settings=dict(fp32["settings"], kv_dtype="f16"))
    v1 = dict(fp32, label="v1", settings=dict(fp32["settings"], group_size=0, scale_dtype="f32"), model_identity={"sha256": "v1"})
    q8 = dict(fp32, backend="llama", label="q8_0", settings={"weight_dtype": "Q8_0"})
    vnni = dict(f16, label="vnni", settings=dict(f16["settings"], kernel="vnni"))
    reports = [fp32, f16, v1, q8, vnni]
    result = heldout_comparison(reports, decision)
    assert result["kv_comparisons"][0]["top1_agreement_between_kv_paths"] == 1
    assert result["vnni_decisions"][0]["retained"]
    vnni["aggregate"] = dict(vnni["aggregate"], mean_kl_reference_candidate_nats=q8["aggregate"]["mean_kl_reference_candidate_nats"] + 0.01)
    assert not heldout_comparison(reports, decision)["vnni_decisions"][0]["retained"]
    vnni["aggregate"] = dict(q8["aggregate"], top1_agreement=0)
    assert heldout_comparison(reports, decision)["vnni_decisions"][0]["decision"] == "rejected quality tradeoff"
    for key in ("p99_kl_reference_candidate_nats", "perplexity"):
        vnni["aggregate"] = dict(q8["aggregate"], **{key: q8["aggregate"][key] + 0.01})
        assert not heldout_comparison(reports, decision)["vnni_decisions"][0]["retained"]
    # The same quality rule applies to a v1-weight VNNI candidate too.
    v1_vnni = dict(v1, label="v1-vnni", settings=dict(v1["settings"], kernel="vnni"),
        aggregate=dict(q8["aggregate"], top1_agreement=0))
    v1_result = heldout_comparison([fp32, f16, v1, q8, v1_vnni], decision)
    assert v1_result["vnni_decisions"][0]["label"] == "v1-vnni"
    assert not v1_result["vnni_decisions"][0]["retained"]
    with pytest.raises(ValueError, match="fixed-weight"):
        heldout_comparison([f16, v1, q8], decision)
    with pytest.raises(ValueError, match="Q8_0"):
        heldout_comparison([fp32, f16, v1], decision)
    changed = dict(fp32, model_identity={"sha256": "changed"})
    with pytest.raises(ValueError, match="not selected"):
        heldout_comparison([changed, f16, v1, q8], decision)
