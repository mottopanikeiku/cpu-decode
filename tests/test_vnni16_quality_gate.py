"""Synthetic VNNI16 decisions and mutations; no native/model processes or timings."""
from argparse import Namespace
import json

import pytest

from test_quality_v2 import CORPUS, calibration_sweep
from tools.corpus_v2 import CORPUS_SHA256, POLICY, digest_json, load_manifest, window_alignment, write_json
from tools.download_model import file_hash
from tools.measure_v2 import check_quality_eligibility, load_quality_eligibility, validate_quality_eligibility
from tools.quality_v2 import (FORMAT_SELECTION_POLICY, choose_format, compare, linked16_record,
    strict16_decision, summarize, vnni16_gate)
from tools.summarize_v2 import native_samples


def fixture16(tmp_path):
    corpus = load_manifest(CORPUS)
    sweep = calibration_sweep()
    for report in sweep:
        report["oracle_identity"]["verified_source"] = {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "fixture": "synthetic source only"}
        report["model_identity"]["sha256"] = digest_json(report["model_identity"]["files"])
    sweep[3]["aggregate"]["top1_agreement"] = 0.0  # Fixed g64f16 chosen before the new kernel.
    chosen = choose_format(sweep)
    assert chosen["settings"]["group_size"] == 64 and chosen["settings"]["scale_dtype"] == "f16"
    selection = {"split": "calibration", "corpus_sha256": CORPUS_SHA256,
        "policy": FORMAT_SELECTION_POLICY, "oracle_identity": chosen["oracle_identity"],
        "chosen": {"label": chosen["label"], "group_size": 64, "scale_dtype": "f16",
            "settings": chosen["settings"], "aggregate": chosen["aggregate"],
            "model_identity": chosen["model_identity"],
            "weights_artifact_bytes": chosen["model_identity"]["files"]["model.safetensors"]["bytes"]},
        "evidence": sweep}
    selection_path = tmp_path / "format-selection.json"
    write_json(selection_path, selection)
    selection_sha = file_hash(selection_path)
    records = {}
    for role in ("calibration", "q8_calibration", "heldout_f16", "heldout_f32", "q8_heldout"):
        q8 = role.startswith("q8_")
        split = "calibration" if "calibration" in role else "heldout"
        kv = "f32" if role == "heldout_f32" else "f16"
        settings = ({"weight_dtype": "Q8_0", "kv_dtype": "f16", "flash_attention": "auto", "threads": 1}
            if q8 else {**chosen["settings"], "kernel": "vnni16", "kv_dtype": kv,
                "activation_dtype": "int16", "activation_group_size": 64})
        rows, windows = [], []
        for window in corpus["windows"]:
            if window["split"] != split:
                continue
            inputs, positions, targets = window_alignment(window)
            command = (["$LLAMA_LOGITS", "--model", "$CANDIDATE_MODEL", "--threads", "1"] if q8 else
                ["$ENGINE", "logits", "--model", "$CANDIDATE_MODEL", "--kernel", "vnni16",
                    "--kv", kv, "--threads", "1", "--attention", "blocked", "--scheduler", "pool",
                    "--affinity", "strict", "--cpu-set", "0"])
            command += ["--tokens", ",".join(map(str, inputs)), "--logits-start", str(POLICY["priming_tokens"]),
                "--output", f"$RAW/{window['id']}-{role}"]
            windows.append({"id": window["id"], "tokens_sha256": window["tokens_sha256"],
                "command": command, "metadata_sha256": "synthetic metadata",
                "logits": {"sha256": "synthetic logits", "filename": f"{window['id']}-{role}.bin"}})
            for index, (position, target) in enumerate(zip(positions, targets, strict=True)):
                rows.append({"window_id": window["id"], "input_position": position, "target": target,
                    "source_target_token": window["source_token_start"] + position + 1,
                    "context_tokens": position + 1, "kl_reference_candidate_nats": .2 if q8 else .01,
                    "reference_cross_entropy_nats": 1.0, "candidate_cross_entropy_nats": 1.0 if q8 else 3.0,
                    "top1_match": index % 2 == 0 if q8 else True, "candidate_top1": target})
        report = {"label": role, "backend": "llama" if q8 else "native", "split": split,
            "corpus_sha256": CORPUS_SHA256, "oracle_identity": selection["oracle_identity"],
            "selection_sha256": None if role == "q8_calibration" else selection_sha,
            "settings": settings, "model_identity": sweep[-1]["model_identity"] if q8 else chosen["model_identity"],
            "binary_identity": ({**sweep[-1]["binary_identity"], "reader_binary_sha256": "reader",
                "shared_libraries": {"fixture": "synthetic"}, "build": {"fixture": "synthetic"},
                "artifact_manifest_sha256": "preparation"} if q8 else
                {"engine_binary_sha256": "engine", "engine_location_sha256": "engine-location",
                    "model_location_sha256": "model-location"}),
            "positions": rows, "windows": windows, "aggregate": summarize(rows)}
        path = tmp_path / (role + ".json")
        write_json(path, report)
        records[role] = linked16_record(path, tmp_path)
    gate = vnni16_gate(selection, selection_sha, records)
    evidence = [{"report": record["report"], "sha256": record["sha256"], **{key: record["data"][key]
        for key in ("label", "split", "backend", "settings", "aggregate", "model_identity", "binary_identity", "oracle_identity")}}
        for role, record in records.items() if role.startswith("heldout_") or role == "q8_heldout"]
    quality = {"split": "heldout", "corpus_sha256": CORPUS_SHA256, "selection": selection,
        "selection_sha256": selection_sha, "evidence": evidence,
        "comparison": {"vnni16_gate": gate}}
    path = tmp_path / "quality.json"
    write_json(path, quality)
    native = {key: value for key, value in records["heldout_f16"]["data"]["settings"].items()
        if key not in ("threads", "cpu_set", "weight_dtype", "activation_dtype", "activation_group_size")}
    native["rope"] = "cached"
    files = chosen["model_identity"]["files"]
    artifacts = {"weights": {"sha256": files["model.safetensors"]["sha256"], "model_location_sha256": "model-location"},
        "config": {"sha256": files["config.json"]["sha256"]},
        "engine": {"sha256": "engine", "location_sha256": "engine-location"},
        "gguf": {"sha256": sweep[-1]["model_identity"]["files"]["q8.gguf"]["sha256"]}}
    return path, native, artifacts, records, selection_path


@pytest.mark.parametrize("metric", ["mean_kl_reference_candidate_nats", "p99_kl_reference_candidate_nats", "top1_agreement"])
def test16_equality_is_rejected_on_each_metric(metric):
    candidate = {"label": "synthetic", "aggregate": {"mean_kl_reference_candidate_nats": .01,
        "p99_kl_reference_candidate_nats": .02, "top1_agreement": .9, "perplexity": 100}}
    q8 = {"aggregate": {"mean_kl_reference_candidate_nats": .1,
        "p99_kl_reference_candidate_nats": .2, "top1_agreement": .8, "perplexity": 2}}
    assert strict16_decision(candidate, q8)["retained"]
    candidate["aggregate"][metric] = q8["aggregate"][metric]
    assert not strict16_decision(candidate, q8)["retained"]


def test16_both_stages_recomputed_and_ppl_display_only(tmp_path):
    path, native, artifacts, records, selection_path = fixture16(tmp_path)
    proof = load_quality_eligibility(path, native, artifacts)
    assert proof["gate"]["approved"]
    assert proof["gate"]["calibration"]["candidate"]["perplexity"] > proof["gate"]["calibration"]["q8_0"]["perplexity"]
    protocol = {"development": False, "native": native, "artifacts": artifacts,
        "source_model": proof["selection"]["oracle_identity"]["verified_source"], "quality_eligibility": proof}
    check_quality_eligibility(protocol, tmp_path)
    args = Namespace(split="calibration", selection=selection_path, corpus=CORPUS,
        reports=[tmp_path / records[key]["report"] for key in ("calibration", "q8_calibration")],
        output=tmp_path / "cal-decision.json")
    compare(args)
    decision = json.loads(args.output.read_text())["vnni16_gate"]
    assert decision["calibration"]["retained"] and not decision["approved"]


def test16_cached_quality_cannot_approve_direct_rope(tmp_path):
    path, native, artifacts, _, _ = fixture16(tmp_path)
    proof = load_quality_eligibility(path, native, artifacts)
    native["rope"] = "direct"
    with pytest.raises(ValueError, match="cached RoPE"):
        validate_quality_eligibility(proof, native, artifacts)
    with pytest.raises(ValueError, match="cached RoPE"):
        load_quality_eligibility(path, native, artifacts)


@pytest.mark.parametrize("role", ["calibration", "heldout_f16"])
def test16_false_metric_cannot_be_overridden_by_approved_flag(tmp_path, role):
    path, native, artifacts, records, _ = fixture16(tmp_path)
    proof = load_quality_eligibility(path, native, artifacts)
    report = proof["gate"]["reports"][role]["data"]
    for row in report["positions"]:
        row["kl_reference_candidate_nats"] = .2
    report["aggregate"] = summarize(report["positions"])
    assert proof["gate"]["approved"]
    with pytest.raises(ValueError, match="rejected"):
        validate_quality_eligibility(proof, native, artifacts)

def test16_f32_cache_control_does_not_select_shipping_path(tmp_path):
    _, _, _, records, selection_path = fixture16(tmp_path)
    report = records["heldout_f32"]["data"]
    for row in report["positions"]:
        row["kl_reference_candidate_nats"] = .3
        row["top1_match"] = False
    report["aggregate"] = summarize(report["positions"])
    gate = vnni16_gate(json.loads(selection_path.read_text()), file_hash(selection_path), records)
    assert gate["approved"] and gate["heldout"][0]["retained"]
    assert not gate["heldout"][1]["retained"]



@pytest.mark.parametrize("role", ["calibration", "q8_calibration", "heldout_f16", "heldout_f32", "q8_heldout"])
def test16_every_linked_report_is_rehashed(tmp_path, role):
    path, native, artifacts, records, _ = fixture16(tmp_path)
    (tmp_path / records[role]["report"]).write_text("{}")
    with pytest.raises(ValueError, match="hash/content"):
        load_quality_eligibility(path, native, artifacts)


@pytest.mark.parametrize("mutation", ["binary", "source", "corpus", "settings", "selection", "activation", "positions", "aggregate", "command", "q8", "missing", "flag"])
def test16_mutated_frozen_proof_rejected(tmp_path, mutation):
    path, native, artifacts, _, _ = fixture16(tmp_path)
    proof = load_quality_eligibility(path, native, artifacts)
    report = proof["gate"]["reports"]["heldout_f32"]["data"]
    if mutation == "binary":
        report["binary_identity"]["engine_binary_sha256"] = "changed"
    elif mutation == "source":
        report["oracle_identity"] = {"changed": True}
    elif mutation == "corpus":
        report["corpus_sha256"] = "changed"
    elif mutation == "settings":
        report["settings"]["attention"] = "scalar"
    elif mutation == "selection":
        report["selection_sha256"] = "changed"
    elif mutation == "activation":
        report["settings"]["activation_dtype"] = "int8"
    elif mutation == "positions":
        report["positions"][0]["target"] += 1
    elif mutation == "aggregate":
        report["aggregate"]["top1_agreement"] = 0.0
    elif mutation == "command":
        report["windows"][0]["command"][0] = "$OTHER_ENGINE"
    elif mutation == "q8":
        proof["gate"]["reports"]["q8_heldout"]["data"]["binary_identity"] = {"changed": True}
    elif mutation == "missing":
        del proof["gate"]["reports"]["q8_calibration"]
    else:
        proof["gate"]["approved"] = False
    with pytest.raises((ValueError, KeyError)):
        validate_quality_eligibility(proof, native, artifacts)


@pytest.mark.parametrize("artifact,field", [("weights", "sha256"), ("config", "sha256"), ("engine", "sha256"),
    ("engine", "location_sha256"), ("weights", "model_location_sha256"), ("gguf", "sha256")])
def test16_timing_artifact_or_path_mismatch_rejected(tmp_path, artifact, field):
    path, native, artifacts, _, _ = fixture16(tmp_path)
    artifacts[artifact][field] = "changed"
    with pytest.raises(ValueError, match="differs"):
        load_quality_eligibility(path, native, artifacts)


def test16_no_final_window_or_summary_without_proof(tmp_path):
    native = {"kernel": "vnni16"}
    protocol = {"development": False, "native": native, "artifacts": {}}
    with pytest.raises(ValueError, match="no retained"):
        check_quality_eligibility(protocol, tmp_path)
    with pytest.raises(ValueError, match="no retained"):
        native_samples({"data": {"kernel": "vnni16"}}, {}, protocol, native)
    with pytest.raises(ValueError, match="no retained"):
        native_samples({"data": {"kernel": "vnni16"}}, {"schema": "cpu-decode-v2-ablation"}, protocol, native)
    protocol["native"] = {"kernel": "simd512x4"}
    check_quality_eligibility(protocol, tmp_path)  # FP32 remains independent of integer quality policy.


def test16_compare_requires_explicit_calibration_and_keeps_controls(tmp_path, monkeypatch):
    import tools.quality_v2 as quality
    path, native, artifacts, records, selection_path = fixture16(tmp_path)
    row = json.loads((tmp_path / records["heldout_f16"]["report"]).read_text())
    row["label"] = "historical-row"
    row["settings"].update(kernel="simd512x4", group_size=0, scale_dtype="f32")
    row["model_identity"] = {"files": {"synthetic": {"sha256": "row", "bytes": 1}}}
    monkeypatch.setattr(quality, "archived_v1_identity", lambda source: row["model_identity"])
    row_path = tmp_path / "historical-row.json"
    write_json(row_path, row)
    args = Namespace(split="heldout", selection=selection_path, corpus=CORPUS,
        reports=[row_path, *[tmp_path / records[key]["report"] for key in ("heldout_f16", "heldout_f32", "q8_heldout")]],
        output=tmp_path / "quality-compared.json",
        vnni16_calibration_report=None, q8_calibration_report=None)
    before = {report: report.read_bytes() for report in args.reports}
    with pytest.raises(ValueError, match="explicit"):
        compare(args)
    assert not args.output.exists()
    assert all(report.read_bytes() == content for report, content in before.items())
    args.vnni16_calibration_report = tmp_path / records["calibration"]["report"]
    args.q8_calibration_report = tmp_path / records["q8_calibration"]["report"]
    compare(args)
    proof = load_quality_eligibility(args.output, native, artifacts)
    summary = json.loads(args.output.read_text())
    assert proof["gate"]["approved"]
    assert any(entry["label"] == "historical-row" for entry in summary["evidence"])
    assert summary["comparison"]["vnni_decisions"] == []


def test16_rejected_calibration_summary_is_retained_not_format_reselection(tmp_path):
    _, _, _, records, selection_path = fixture16(tmp_path)
    cal_path = tmp_path / records["calibration"]["report"]
    report = json.loads(cal_path.read_text())
    for row in report["positions"]:
        row["kl_reference_candidate_nats"] = .2
    report["aggregate"] = summarize(report["positions"])
    write_json(cal_path, report)
    selection_before = selection_path.read_bytes()
    args = Namespace(split="calibration", selection=selection_path, corpus=CORPUS,
        reports=[cal_path, tmp_path / records["q8_calibration"]["report"]],
        output=tmp_path / "rejected-calibration.json")
    compare(args)
    gate = json.loads(args.output.read_text())["vnni16_gate"]
    assert not gate["calibration"]["retained"] and not gate["approved"]
    assert cal_path.exists() and selection_path.read_bytes() == selection_before


def test16_runner_refuses_missing_two_stage_gate(tmp_path, monkeypatch):
    from tools import run_final_v2 as runner
    path, native, artifacts, _, selection_path = fixture16(tmp_path)
    summary = json.loads(path.read_text())
    del summary["comparison"]["vnni16_gate"]
    write_json(path, summary)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"synthetic weights")
    (model / "config.json").write_bytes(b"synthetic config")
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"source": summary["selection"]["oracle_identity"]["verified_source"]})
    args = Namespace(quality=path, format_selection=selection_path, model_manifest=manifest,
        kernel="vnni16", affinity="strict", engine=tmp_path / "engine", gguf=tmp_path / "gguf", model=model)
    real_hash = runner.file_hash
    monkeypatch.setattr(runner, "file_hash", lambda item: "engine" if item == args.engine else
        "q8 artifact" if item == args.gguf else real_hash(item))
    with pytest.raises(ValueError, match="calibration AND heldout"):
        runner.check_quality_selection(args)
