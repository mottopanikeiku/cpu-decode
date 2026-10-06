"""Analytical quality statistics and real pinned corpus integrity (no model loads)."""
import copy
import hashlib
import json
import math
import os
import shlex
import struct
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import tools.quality_v2 as quality_v2
from tools.corpus_v2 import (
    CORPUS_SHA256, FORMAT_CHOICES, LICENSE, LICENSE_URL, POLICY, SOURCES,
    digest_json, load_manifest, prepare, protect_destination, validate_manifest, window_alignment, write_json,
)
from tools.download_model import file_hash
from tools.quality_v2 import (
    archived_v1_identity, checked_logits, choose_format, compare, evaluate, heldout_comparison, log_probabilities,
    oracle, position_metric, preflight_native_model, read_selection, require_reader_identity,
    resolved_upstream_libraries, summarize, upstream_build_identity,
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
            "scheduler": "pool", "affinity": "strict", "threads": 1, "cpu_set": [0]}
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
    validate_heldout_settings(dict(settings, group_size=0, scale_dtype="f32"), archived_v1_identity(), decision)
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
    v1 = dict(fp32, label="v1", settings=dict(fp32["settings"], group_size=0, scale_dtype="f32"), model_identity=archived_v1_identity())
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


def test_result_destination_guard_handles_parent_traversal_and_symlinks(tmp_path):
    result_root = tmp_path / "results"
    legacy = result_root / "legacy"
    legacy.mkdir(parents=True)
    (result_root / "v2").mkdir()
    assert protect_destination(result_root / "v2/new.json", tmp_path) == (result_root / "v2/new.json").resolve()
    assert protect_destination(tmp_path / "external/raw", tmp_path) == (tmp_path / "external/raw").resolve()
    for path in (result_root / "old.json", result_root / "v2/../old.json", result_root):
        with pytest.raises(ValueError, match="overwrite v1"):
            protect_destination(path, tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError, match="overwrite v1"):
        protect_destination(alias / "old.json", tmp_path)
    (result_root / "v2/escape").symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError, match="overwrite v1"):
        protect_destination(result_root / "v2/escape/old.json", tmp_path)


def test_every_quality_stage_rejects_archived_destinations_before_inputs(tmp_path):
    archived = ROOT / "results/llama-quality.json"
    before = file_hash(archived)
    # Minimal arguments demonstrate guards execute before imports/model/input reads,
    # including an oracle invocation that could otherwise reuse complete raw data.
    for stage in (oracle, evaluate, compare):
        with pytest.raises(ValueError, match="overwrite v1"):
            stage(Namespace(output=archived))
    with pytest.raises(ValueError, match="overwrite v1"):
        oracle(Namespace(output=tmp_path / "new.json", raw_dir=ROOT / "results/raw"))
    with pytest.raises(ValueError, match="overwrite v1"):
        evaluate(Namespace(output=tmp_path / "new.json", raw_dir=ROOT / "results/raw"))
    with pytest.raises(ValueError, match="overwrite v1"):
        compare(Namespace(output=tmp_path / "new.json", split="calibration", selection=archived))
    with pytest.raises(ValueError, match="overwrite v1"):
        prepare(tmp_path / "absent-model", ROOT / "results", tmp_path / "raw")
    with pytest.raises(ValueError, match="overwrite v1"):
        prepare(tmp_path / "absent-model", tmp_path / "output", ROOT / "results/raw")
    with pytest.raises(ValueError, match="overwrite v1"):
        write_json(archived, {})
    assert file_hash(archived) == before


@pytest.mark.parametrize("backend", ["native", "llama"])
@pytest.mark.parametrize("window_index", [0, -1], ids=["first-window", "later-window"])
@pytest.mark.parametrize("suffix", [".bin", ".json"])
@pytest.mark.parametrize("kind", ["file", "symlink", "dangling-symlink"])
def test_evaluation_preflights_all_raw_outputs_before_subprocess(
        tmp_path, monkeypatch, backend, window_index, suffix, kind):
    corpus = load_manifest(CORPUS)
    windows = [window for window in corpus["windows"] if window["split"] == "calibration"]
    args = Namespace(output=tmp_path / "quality.json", raw_dir=tmp_path / "raw",
        corpus=CORPUS, split="calibration", selection=tmp_path / "selection.json",
        label="candidate", backend=backend, model=tmp_path / "candidate-model",
        engine=tmp_path / "engine", reader=tmp_path / "reader",
        artifact_manifest=tmp_path / "artifact-manifest.json", threads=2,
        kernel="scalar", kv="f16", attention="blocked", scheduler="pool",
        affinity="unpinned", cpu_set=None)
    args.raw_dir.mkdir()
    oracle_path = args.raw_dir / "reference.bin"
    oracle_path.write_bytes(b"recorded oracle bytes")
    args.engine.write_bytes(b"engine fixture")
    artifact = {"sha256": "candidate hash", "bytes": 1}
    source = {"fixture": "verified source"}
    args.artifact_manifest.write_text(json.dumps({
        "llama_commit": quality_v2.LLAMA_COMMIT, "llama_repository": quality_v2.LLAMA_URL,
        "source_model": source, "artifacts": {"Q8_0": artifact}}))
    metadata = {"identity": {"verified_source": source}, "windows": [
        {"id": window["id"], "logits": oracle_path.name, "shape": [256, 2],
            "input_tokens": window_alignment(window)[0]} for window in windows]}
    monkeypatch.setattr(quality_v2, "read_oracle", lambda *_: metadata)
    monkeypatch.setattr(quality_v2, "model_identity",
        lambda *_: {"files": {args.model.name: artifact}})
    monkeypatch.setattr(quality_v2, "preflight_native_model", lambda *_: {})
    reader_calls = []

    def reader_identity(*_):
        reader_calls.append(True)
        return {}

    monkeypatch.setattr(quality_v2, "reader_build_identity", reader_identity)
    monkeypatch.setattr(quality_v2, "checked_logits", lambda *_: np.zeros((256, 2)))
    subprocess_calls = []

    def unexpected_subprocess(*command, **kwargs):
        subprocess_calls.append(command)
        raise AssertionError("Raw collisions must be rejected before any subprocess")

    monkeypatch.setattr(quality_v2.subprocess, "run", unexpected_subprocess)
    collision = args.raw_dir / f"{windows[window_index]['id']}-{args.label}{suffix}"
    recorded = b"recorded candidate bytes must not change"
    target = tmp_path / "recorded-output"
    if kind == "file":
        collision.write_bytes(recorded)
    else:
        if kind == "symlink":
            target.write_bytes(recorded)
        collision.symlink_to(target)
    before = set(args.raw_dir.iterdir())

    with pytest.raises(ValueError, match="raw output already exists"):
        evaluate(args)

    assert subprocess_calls == []
    assert reader_calls == []
    assert set(args.raw_dir.iterdir()) == before
    assert oracle_path.read_bytes() == b"recorded oracle bytes"
    assert not args.output.exists()
    if kind == "dangling-symlink":
        assert collision.is_symlink()
        assert not target.exists()
    else:
        assert collision.read_bytes() == recorded
        if kind == "symlink":
            assert collision.is_symlink()
            assert target.read_bytes() == recorded


def test_make_raw_uses_resolved_results_path_and_preserves_override(tmp_path, monkeypatch):
    # Evaluate only the variable definitions; no project recipe or uv environment
    # setup runs. The shim executes the real inline Python with this interpreter.
    uv = tmp_path / "uv"
    uv.write_text(f'#!/bin/sh\nshift 2\nexec {shlex.quote(sys.executable)} "$@"\n')
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    definitions = "\n".join((ROOT / "Makefile").read_text().splitlines()[:5])
    makefile = definitions + "\n.PHONY: print-raw\nprint-raw:\n\t@printf '%s\\n' '$(RAW)'\n"

    def raw_for(results, override=None):
        command = ["make", "--no-print-directory", "-f", "-", "print-raw",
            "CACHE=external", f"RESULTS={results}"]
        if override is not None:
            command.append(f"RAW={override}")
        return subprocess.run(command, input=makefile, cwd=tmp_path, check=True,
            capture_output=True, text=True).stdout.strip()

    first = "results/v2/run-a/reproduction"
    second = "results/v2/run-b/reproduction"
    destination = (tmp_path / first).resolve()
    expected = "external/quality-v2-" + hashlib.sha256(str(destination).encode()).hexdigest()
    assert raw_for(first) == expected
    assert raw_for(second) != expected
    assert raw_for(destination) == expected
    assert raw_for("results/v2/run-a/../run-a/reproduction") == expected
    destination.parent.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(destination.parent, target_is_directory=True)
    assert raw_for(alias / "reproduction") == expected
    assert raw_for("results/v2/run with spaces/reproduction") != expected
    assert raw_for(first, "external/explicit-raw") == "external/explicit-raw"


@pytest.mark.parametrize("changed", ["weights", "config", "source"])
def test_v1_baseline_requires_archived_weights_config_and_oracle_source(changed):
    selection = selected_decision()
    settings = {"group_size": 0, "scale_dtype": "f32", "weight_dtype": "int8"}
    identity = archived_v1_identity()
    source = json.loads((ROOT / "results/quantized-manifest.json").read_text())["source"]
    selection["oracle_identity"] = {"verified_source": copy.deepcopy(source)}
    if changed == "weights":
        identity["files"]["model.safetensors"]["sha256"] = "0" * 64
    elif changed == "config":
        identity["files"]["config.json"]["sha256"] = "0" * 64
    else:
        selection["oracle_identity"]["verified_source"]["revision"] = "changed"
    with pytest.raises(ValueError, match="archived v1|Archived v1"):
        validate_heldout_settings(settings, identity, selection)


def test_changed_v1_container_rejected_at_model_preflight(tmp_path):
    header = {"__metadata__": {"quantization": "symmetric-per-row-int8"},
        "matrix.weight": {"dtype": "I8", "shape": [1, 2], "data_offsets": [0, 2]}}
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x01\x02")
    with pytest.raises(ValueError, match="archived v1"):
        preflight_native_model(tmp_path, {"sha256": "changed"}, selected_decision())


def upstream_fixture(tmp_path):
    build = tmp_path / "build"
    directory = build / "bin"
    directory.mkdir(parents=True)
    libraries = {}
    lines = []
    for name in ("libllama.so.0", "libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0"):
        path = directory / (name + ".1")
        path.write_bytes(name.encode())
        alias = directory / name
        alias.symlink_to(path.name)
        libraries[name] = path
        lines.append(f"\t{name} => {alias} (0x0123)")
    (build / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\nGGML_NATIVE:BOOL=ON\nGGML_BACKEND_DL:BOOL=OFF\n")
    (build / "flags.make").write_text("CXX_FLAGS = -O3 -march=native\n")
    return build, libraries, "\n".join(lines)


def test_ldd_resolves_actual_symlink_targets_and_requires_all_upstream_dependencies(tmp_path):
    _, libraries, text = upstream_fixture(tmp_path)
    assert resolved_upstream_libraries(text) == libraries
    with pytest.raises(ValueError, match="dynamically link"):
        resolved_upstream_libraries("\n".join(text.splitlines()[:-1]))
    with pytest.raises(ValueError, match="Unresolved"):
        resolved_upstream_libraries(text + "\nlibggml-missing.so => not found")


@pytest.mark.parametrize("changed", ["libllama.so.0", "libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0", "cache", "flags"])
def test_library_or_build_changes_detected_between_windows(tmp_path, changed):
    build, libraries, _ = upstream_fixture(tmp_path)
    before = upstream_build_identity(libraries)
    require_reader_identity(before, upstream_build_identity(libraries))
    if changed == "cache":
        with (build / "CMakeCache.txt").open("a") as stream:
            stream.write("GGML_OPENMP:BOOL=ON\n")
    elif changed == "flags":
        (build / "flags.make").write_text("CXX_FLAGS = -O2\n")
    else:
        libraries[changed].write_bytes(b"changed-library-content")
    with pytest.raises(ValueError, match="identity changed"):
        require_reader_identity(before, upstream_build_identity(libraries))


def test_unrecorded_dynamic_backend_build_rejected(tmp_path):
    build, libraries, _ = upstream_fixture(tmp_path)
    (build / "CMakeCache.txt").write_text("GGML_BACKEND_DL:BOOL=ON\n")
    with pytest.raises(ValueError, match="unrecorded dynamically"):
        upstream_build_identity(libraries)


@pytest.mark.parametrize("name", ["libggml-cuda.so", "libggml-cpu-extra.so", "libllama-extra.so"])
def test_unknown_resolved_upstream_dependency_rejected(tmp_path, name):
    _, libraries, text = upstream_fixture(tmp_path)
    path = next(iter(libraries.values()))
    with pytest.raises(ValueError, match="Unknown upstream"):
        resolved_upstream_libraries(text + f"\n{name} => {path} (0x1234)")


def test_identical_libraries_in_different_builds_keep_distinct_portable_identity(tmp_path):
    from tools.portable import portable
    left = tmp_path / "first"
    right = tmp_path / "second"
    _, left_libraries, _ = upstream_fixture(left)
    _, right_libraries, _ = upstream_fixture(right)
    first = upstream_build_identity(left_libraries)
    second = upstream_build_identity(right_libraries)
    assert {name: row["sha256"] for name, row in first["shared_libraries"].items()} == {
        name: row["sha256"] for name, row in second["shared_libraries"].items()}
    first = portable(first, quality_v2.reader_identity_locations(first))
    second = portable(second, quality_v2.reader_identity_locations(second))
    assert first["build"]["root"] == second["build"]["root"] == "$LLAMA_BUILD"
    assert all(row["resolved_path"].startswith("$LLAMA_BUILD/") for row in first["shared_libraries"].values())
    assert str(tmp_path) not in json.dumps(first)
    with pytest.raises(ValueError, match="identity changed"):
        require_reader_identity(first, second)


@pytest.mark.parametrize("variable", ["LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"])
def test_reader_identity_binds_loader_environment_without_publishing_paths(tmp_path, monkeypatch, variable):
    _, _, text = upstream_fixture(tmp_path)
    reader = tmp_path / "relocated-reader"
    reader.write_bytes(b"reader")
    monkeypatch.setattr(subprocess, "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 0, text, ""))
    monkeypatch.delenv(variable, raising=False)
    first = quality_v2.reader_build_identity(reader)
    loader_path = str(tmp_path / "private-loader-location")
    monkeypatch.setenv(variable, loader_path)
    second = quality_v2.reader_build_identity(reader)
    assert loader_path not in json.dumps(second["loader_environment_sha256"])
    assert first["loader_environment_sha256"][variable] != second["loader_environment_sha256"][variable]
    with pytest.raises(ValueError, match="identity changed"):
        require_reader_identity(first, second)


def test_reader_compiler_setting_is_bound_and_portable(tmp_path):
    from tools.portable import portable
    build, libraries, _ = upstream_fixture(tmp_path)
    compiler = tmp_path / "private-toolchain" / "c++"
    with (build / "CMakeCache.txt").open("a") as stream:
        stream.write(f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n")
    identity = upstream_build_identity(libraries)
    assert identity["build"]["settings"]["CMAKE_CXX_COMPILER"] == str(compiler)
    public = portable(identity, quality_v2.reader_identity_locations(identity))
    assert public["build"]["settings"]["CMAKE_CXX_COMPILER"] == "$LLAMA_CXX_COMPILER"
    assert str(tmp_path) not in json.dumps(public)
