"""Synthetic protocol tests; these are not performance measurements."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import os
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from tools.figure_v2 import figure
from tools.download_model import file_hash
from tools.measure_v2 import baseline_cpu_set, candidates, check_quality_eligibility, cpu_mask, cpu_order, digest, engine_command, execute, freeze, llama_command, load_quality_eligibility, native_cpu_set, validate_quality_eligibility
from tools.summarize_v2 import profile_bytes, select_best, spread, summarize, summarize_candidate


@pytest.fixture
def protocol():
    p = {"development": False, "threads": [1, 2, 4, 6, 12], "contexts": [128, 1024, 4096],
         "steps": 64, "repeats": 5, "rounds": 2, "warmup_steps": 1, "tokens": "1,2,3",
         "cpu_order": [0, 1, 4, 2, 3, 5, 6, 7, 10, 8, 9, 11], "candidates": candidates([50]),
         "cpu_metadata": {"allowed_cpu_ids": list(range(12))},
         "native": {"kernel": "simd512x4", "kv_dtype": "f16", "attention": "blocked", "scheduler": "pool",
                    "affinity": "strict", "rope": "cached", "group_size": 32, "scale_dtype": "f16"},
         "model_geometry": {"layers": 24, "kv_heads": 2, "head_dim": 64},
         "llama_commit": "6c73b3e12dc501de35fe5f6979960d06921a2f6c",
         "selection": "highest pooled median baseline; paired native only"}
    p["id"] = digest(p)
    return p


def byte_counts(p, context=128, scale_bytes=3200):
    kv = 24 * 2 * 64 * 2 * 2
    counts = {"matrix_weights": 100000, "scales": scale_bytes, "norm_bias": 100,
              "embedding": 900, "kv_write": kv, "kv_read_min": kv * (context + (p["steps"] + 1) / 2)}
    counts["total_min"] = sum(counts.values())
    # These are subsets, not additional traffic.
    counts["lm_head"] = 50000
    counts["lm_head_scales"] = scale_bytes / 2
    return counts


def window(p, thread=2, context=128):
    w = {"schema": "cpu-decode-v2-window", "protocol_id": p["id"], "threads": thread,
         "context": context, "cpu_set": native_cpu_set(p, thread), "candidates": p["candidates"],
         "native_settings": p["native"], "environment": {"nice": 19}, "invocations": []}
    for candidate_index, candidate in enumerate(p["candidates"]):
        for round_id in range(2):
            native_samples = [{"tokens_per_second": rate, "seconds": p["steps"] / rate,
                               "step_seconds": [1 / rate] * p["steps"], "generated_tokens": [1] * p["steps"],
                               "bytes_per_token": byte_counts(p, context)} for rate in [100, 101, 102, 103, 104]]
            native = {"threads": thread, "context": context, "steps": p["steps"], "repeats": 5,
                      "warmup_steps": 1, "prompt_tokens": [1, 2, 3], "cpu_set": w["cpu_set"],
                      **p["native"], "samples": native_samples}
            rate = 20 + candidate_index * 10
            llama = {"n_threads": thread, "n_depth": context, "n_gen": p["steps"], "n_prompt": 0,
                     "type_k": "f16", "type_v": "f16", "n_gpu_layers": 0,
                     "flash_attn": {"on": 1, "off": 0, "auto": -1}[candidate["flash_attn"]],
                     "cpu_strict": candidate["affinity"] == "pinned", "cpu_mask": cpu_mask(p["cpu_order"][:thread]) if candidate["affinity"] == "pinned" else "0x0",
                     "poll": candidate["poll"], "repack": True, "backends": "CPU", "model_type": "qwen2 Q8_0",
                     "build_commit": p["llama_commit"][:8], "cpu_info": "test CPU", "model_size": 531000000,
                     "samples_ts": [rate] * 5, "samples_ns": [p["steps"] * 1e9 / rate] * 5}
            for engine, raw, command in [("native", native, engine_command(Path("engine"), Path("model"), p, thread, context)),
                                         ("llama", [llama], llama_command(Path("llama"), Path("gguf"), p, thread, context, candidate))]:
                w["invocations"].append({"candidate_id": candidate["id"], "engine": engine, "round": round_id,
                                         "returncode": 0,
                                         "success": True, "error": None, "data": raw, "command": command,
                                         "process_cpu_set": native_cpu_set(p, thread) if engine == "native" else baseline_cpu_set(p, thread, candidate),
                                         "stdout_file": f"{candidate['id']}-{round_id}-{engine}.stdout.txt",
                                         "stderr_file": f"{candidate['id']}-{round_id}-{engine}.stderr.txt"})
    return w


def bandwidth(p, thread=2):
    cpus = native_cpu_set(p, thread)
    raw = {"schema": "cpu-decode-v2-bandwidth", "protocol_id": p["id"], "threads": thread,
           "cpu_set": cpus, "environment": {"nice": 19}, "invocations": []}
    for kernel, rate in [("simd256", 10), ("simd512", 12)]:
        data = {"kind": "read_bandwidth", "threads": thread, "kernel": kernel,
                "array_bytes": 256 * 1024 * 1024, "passes": 128,
                "samples": [{"GB_per_s": rate, "seconds": 256 * 1024 * 1024 * 128 / rate / 1e9} for _ in range(5)]}
        raw["invocations"].append({"data": data, "success": True, "returncode": 0,
                                   "command": ["taskset", "-c", ",".join(map(str, cpus)), "bandwidth", "--kernel", kernel],
                                   "environment_overrides": {"OMP_PLACES": ",".join(f"{{{x}}}" for x in cpus), "OMP_PROC_BIND": "true" if p["native"]["affinity"] == "strict" else "false", "OMP_DYNAMIC": "false"},
                                   "stdout_file": kernel + ".stdout.txt", "stderr_file": kernel + ".stderr.txt"})
    return raw


def save_inputs(directory, p, w):
    for name, raw in [("protocol", p), ("window-t2-c128-all", w), ("bandwidth-t2", bandwidth(p))]:
        (directory / f"{name}.json").write_text(json.dumps(raw))


def test_cpu_order_preserves_fast_physical_then_smt(protocol):
    metadata = {"allowed_cpu_ids": list(range(12)), "preferred_cpu_ids": protocol["cpu_order"]}
    assert cpu_order(metadata, None, 12) == protocol["cpu_order"]
    assert cpu_mask(cpu_order(metadata, None, 12)[:4]) == "0x17"
    for explicit in ["0,0", "0,12", "0"]:
        with pytest.raises(ValueError, match="CPU order"):
            cpu_order(metadata, explicit, 2)
    assert len(candidates([0, 50])) == 18
    with pytest.raises(ValueError):
        candidates([50, 50])


def test_final_settings_reject_short_sampling_and_f32(tmp_path):
    args = SimpleNamespace(model=tmp_path, gguf=tmp_path, llama=tmp_path, model_manifest=tmp_path,
                           kv="f16", steps=63, repeats=5, development=False)
    with pytest.raises(ValueError, match=">=64"):
        freeze(args, {})
    args.steps, args.repeats = 64, 4
    with pytest.raises(ValueError, match=">=5"):
        freeze(args, {})
    args.repeats, args.kv = 5, "f32"
    with pytest.raises(ValueError, match="F16 KV"):
        freeze(args, {})


def test_best_measured_baseline_not_default_or_weakest(tmp_path, protocol):
    save_inputs(tmp_path, protocol, window(protocol))
    result = summarize(tmp_path, allow_partial=True)
    cell = result["results"][0]
    assert cell["winner"]["candidate"]["id"] == "auto-defaults-poll50"
    assert cell["best_baseline_tps"]["median"] == 100
    assert not cell["winner_core_sets_matched"]
    assert cell["winner_baseline_cpu_set"] == list(range(12))
    assert cell["native_tps"]["median"] == 102
    assert cell["native_tps"]["samples"] == cell["best_baseline_tps"]["samples"] == 10
    assert cell["native_samples_all_candidates"] == 90
    assert cell["native_over_best_baseline"] == pytest.approx(102 / 100)
    assert cell["read_ceiling_tps"] == pytest.approx(12e9 / byte_counts(protocol)["total_min"])
    assert cell["percent_of_ceiling"] == pytest.approx(100 * 102 * byte_counts(protocol)["total_min"] / 12e9)
    assert not result["complete_final_matrix"]
    assert result["targets_all_cells"] == {"native_over_best_baseline": False, "percent_of_ceiling": False}
    assert not result["failures"]
    assert ET.fromstring(figure(result)).tag.endswith("svg")


def test_best_uses_median_not_mean_and_reports_noise():
    rows = [{"candidate": {"id": "outlier"}, "baseline_tps": spread([10, 10, 10, 10, 1000])},
            {"candidate": {"id": "steady"}, "baseline_tps": spread([20] * 5)}]
    assert select_best(rows)["candidate"]["id"] == "steady"
    assert rows[0]["baseline_tps"]["noisy_over_5_percent"]
    assert not spread([100, 101, 102, 103, 104])["noisy_over_5_percent"]
    assert spread([100, 100, 100, 100, 106])["spread_percent"] == 6
    for bad in [[], [0], [float("nan")], [float("inf")]]:
        with pytest.raises(ValueError):
            spread(bad)


@pytest.mark.parametrize("engine,key,value", [
    ("native", "warmup_steps", 2), ("native", "kv_dtype", "f32"),
    ("native", "group_size", 0), ("native", "scale_dtype", "f32"),
    ("native", "scheduler", "openmp"), ("native", "attention", "scalar"),
    ("native", "context", 1024), ("native", "cpu_set", [0, 4]),
    ("llama", "n_gen", 16), ("llama", "n_depth", 1024),
    ("llama", "n_threads", 6), ("llama", "type_k", "f32"),
    ("llama", "type_v", "f32"), ("llama", "flash_attn", -1),
    ("llama", "cpu_strict", False), ("llama", "cpu_mask", "0x17"),
    ("llama", "n_gpu_layers", 1), ("llama", "repack", False),
    ("llama", "build_commit", "deadbeef"), ("llama", "model_type", "Q4_K"),
])
def test_unfair_settings_rejected(protocol, engine, key, value):
    w = window(protocol)
    run = next(r for r in w["invocations"] if r["engine"] == engine)
    raw = run["data"] if engine == "native" else run["data"][0]
    raw[key] = value
    with pytest.raises(ValueError):
        summarize_candidate(w, protocol, protocol["candidates"][0])


@pytest.mark.parametrize("fault", ["order", "samplecount", "tokens", "command", "exit", "process", "warmupflag", "priority"])
def test_sampling_and_commands_rejected(protocol, fault):
    w = window(protocol)
    if fault == "order":
        w["invocations"][1], w["invocations"][2] = w["invocations"][2], w["invocations"][1]
    elif fault == "samplecount":
        w["invocations"][1]["data"][0]["samples_ts"].pop()
    elif fault == "tokens":
        w["invocations"][0]["data"]["samples"][0]["step_seconds"].pop()
    elif fault == "command":
        command = w["invocations"][1]["command"]
        command[command.index("-ctv") + 1] = "f32"
    elif fault == "exit":
        w["invocations"][1]["returncode"] = 1
    elif fault == "process":
        w["invocations"][0]["returncode"] = 1
    elif fault == "warmupflag":
        w["invocations"][1]["command"].append("--no-warmup")
    elif fault == "priority":
        w["environment"]["nice"] = 0
    with pytest.raises(ValueError):
        summarize_candidate(w, protocol, protocol["candidates"][0])


def test_profile_uses_selected_scale_bytes_and_f16_not_v1(protocol):
    geometry = protocol["model_geometry"]
    a = {"bytes_per_token": byte_counts(protocol, scale_bytes=3200)}
    b = {"bytes_per_token": byte_counts(protocol, scale_bytes=12345)}
    fa = profile_bytes([a], geometry, "f16", 128, 64)
    fb = profile_bytes([b], geometry, "f16", 128, 64)
    assert fb["total_min"] - fa["total_min"] == 12345 - 3200
    assert fa["kv_write"] == 12288
    assert fa["kv_read_min"] == 12288 * 160.5
    with pytest.raises(ValueError, match="KV bytes"):
        profile_bytes([a], geometry, "f32", 128, 64)
    duplicate = deepcopy(a)
    duplicate["bytes_per_token"]["total_min"] += duplicate["bytes_per_token"]["lm_head"]
    with pytest.raises(ValueError, match="double counts"):
        profile_bytes([duplicate], geometry, "f16", 128, 64)


def test_failed_winning_candidate_remains_visible_and_never_successful(tmp_path, protocol):
    w = window(protocol)
    w["invocations"][-1]["success"] = False
    w["invocations"][-1]["returncode"] = 137
    w["invocations"][-1]["error"] = "nonzero exit 137"
    save_inputs(tmp_path, protocol, w)
    summary = summarize(tmp_path, allow_partial=True)
    assert len(summary["failures"]) == 1
    assert summary["failures"][0]["candidate"] == "auto-defaults-poll50"
    assert summary["results"][0]["missing_candidates"] == ["auto-defaults-poll50"]
    assert not summary["complete_final_matrix"]
    assert all(not value for value in summary["targets_all_cells"].values())


def test_bandwidth_mismatched_core_set_invalidates_ceiling(tmp_path, protocol):
    save_inputs(tmp_path, protocol, window(protocol))
    raw = bandwidth(protocol)
    raw["invocations"][0]["command"][2] = "0,4"
    (tmp_path / "bandwidth-t2.json").write_text(json.dumps(raw))
    summary = summarize(tmp_path, allow_partial=True)
    assert summary["results"][0]["read_ceiling_tps"] is None
    assert "same selected core set" in summary["failures"][0]["error"]


def test_nonzero_output_retained_not_parsed_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 9, '{"samples":[1]}', 'error in /private/model'))
    result = execute(["engine"], tmp_path / "failure", {Path("/private/model"): "$MODEL"}, time.monotonic() + 10)
    assert result["returncode"] == 9 and not result["success"] and result["data"] is None
    assert (tmp_path / "failure.stdout.txt").read_text() == '{"samples":[1]}'
    assert (tmp_path / "failure.stderr.txt").read_text() == 'error in $MODEL'
    assert json.loads((tmp_path / "failure.json").read_text())["error"] == "nonzero exit 9"


def test_protocol_mutation_is_not_accepted(tmp_path, protocol):
    save_inputs(tmp_path, protocol, window(protocol))
    protocol["native"]["scheduler"] = "openmp"
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="digest"):
        summarize(tmp_path, allow_partial=True)


def test_all_cell_targets_need_every_cell_and_candidate(tmp_path, protocol):
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    for thread in protocol["threads"]:
        (tmp_path / f"bandwidth-t{thread}.json").write_text(json.dumps(bandwidth(protocol, thread)))
        for context in protocol["contexts"]:
            (tmp_path / f"window-t{thread}-c{context}-all.json").write_text(json.dumps(window(protocol, thread, context)))
    summary = summarize(tmp_path, target_ceiling_percent=1)
    assert summary["complete_final_matrix"]
    assert summary["targets_all_cells"] == {"native_over_best_baseline": True, "percent_of_ceiling": True}
    path = tmp_path / "window-t12-c4096-all.json"
    raw = json.loads(path.read_text())
    winner_id = protocol["candidates"][-1]["id"]
    for run in raw["invocations"]:
        if run["candidate_id"] == winner_id and run["engine"] == "native":
            for sample in run["data"]["samples"]:
                sample["tokens_per_second"] = 50
                sample["seconds"] = protocol["steps"] / 50
                sample["step_seconds"] = [1 / 50] * protocol["steps"]
    path.write_text(json.dumps(raw))
    summary = summarize(tmp_path, target_ceiling_percent=1)
    assert summary["complete_final_matrix"]
    assert not summary["targets_all_cells"]["native_over_best_baseline"]
    path.unlink()
    summary = summarize(tmp_path, target_ceiling_percent=1)
    assert not summary["complete_final_matrix"]
    assert summary["missing_cells"] == [(12, 4096)]
    assert all(not value for value in summary["targets_all_cells"].values())


def test_removing_stronger_baseline_from_protocol_is_rejected(tmp_path, protocol):
    protocol["candidates"].pop()
    protocol["id"] = digest({k: v for k, v in protocol.items() if k != "id"})
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="candidate set"):
        summarize(tmp_path)


def test_malformed_output_and_timeout_are_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, '{"rate":NaN}', ''))
    result = execute(["engine"], tmp_path / "malformed", {}, time.monotonic() + 10)
    assert not result["success"]
    assert "non-finite" in result["error"]
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=b"partial", stderr=b"timeout log")
    monkeypatch.setattr(subprocess, "run", timed_out)
    result = execute(["engine"], tmp_path / "timeout", {}, time.monotonic() + 10)
    assert not result["success"] and result["returncode"] is None
    assert (tmp_path / "timeout.stdout.txt").read_text() == "partial"
    assert (tmp_path / "timeout.stderr.txt").read_text() == "timeout log"


@pytest.mark.parametrize("affinity", ["pinned", "unpinned", "defaults"])
def test_process_affinity_is_real_not_just_echoed_flags(protocol, affinity):
    w = window(protocol, thread=1)
    candidate = next(c for c in protocol["candidates"] if c["affinity"] == affinity)
    runs = [r for r in w["invocations"] if r["candidate_id"] == candidate["id"] and r["engine"] == "llama"]
    expected = list(range(12)) if affinity == "defaults" else [0]
    assert all(r["command"][:3] == ["taskset", "-c", ",".join(map(str, expected))] for r in runs)
    assert summarize_candidate(w, protocol, candidate)["baseline_process_cpu_set"] == expected
    runs[0]["command"] = runs[0]["command"][3:]
    with pytest.raises(ValueError, match="process affinity"):
        summarize_candidate(w, protocol, candidate)


def quality_fixture(tmp_path, protocol):
    native = {**protocol["native"], "kernel": "vnni"}
    artifacts = {"weights": {"sha256": "weights"}, "config": {"sha256": "config"}, "engine": {"sha256": "engine"}}
    metrics = {"mean_kl_reference_candidate_nats": 0.1, "p99_kl_reference_candidate_nats": 0.2, "top1_agreement": 0.9, "perplexity": 4.0}
    identity = {"files": {"model.safetensors": {"sha256": "weights"}, "config.json": {"sha256": "config"}}}
    report = {"split": "heldout", "backend": "native", "selection_sha256": "selection",
              "settings": native, "aggregate": metrics, "model_identity": identity,
              "binary_identity": {"engine_binary_sha256": "engine"}}
    report_path = tmp_path / "heldout-vnni.json"
    report_path.write_text(json.dumps(report))
    quality = {"split": "heldout", "selection_sha256": "selection", "selection": {"chosen": {"model_identity": identity}},
               "evidence": [{"label": "vnni", "report": report_path.name, "sha256": file_hash(report_path), "settings": native, "aggregate": metrics},
                            {"label": "q8_0", "aggregate": metrics}],
               "comparison": {"q8_0_label": "q8_0", "vnni_decisions": [{"label": "vnni", "retained": True, "candidate": metrics, "q8_0": metrics}]}}
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(quality))
    return path, native, artifacts


@pytest.mark.parametrize("metric,worse", [("mean_kl_reference_candidate_nats", 0.11), ("p99_kl_reference_candidate_nats", 0.21),
                                       ("top1_agreement", 0.89), ("perplexity", 4.01)])
def test_vnni_retained_flag_cannot_override_worse_metric(tmp_path, protocol, metric, worse):
    path, native, artifacts = quality_fixture(tmp_path, protocol)
    proof = load_quality_eligibility(path, native, artifacts)
    proof["decision"]["candidate"][metric] = worse
    with pytest.raises(ValueError, match="all four"):
        validate_quality_eligibility(proof, native, artifacts)


@pytest.mark.parametrize("field", ["weights_sha256", "chosen_weights_sha256", "config_sha256", "engine_binary_sha256"])
def test_vnni_eligibility_binds_actual_artifact_and_binary(tmp_path, protocol, field):
    path, native, artifacts = quality_fixture(tmp_path, protocol)
    proof = load_quality_eligibility(path, native, artifacts)
    proof[field] = "different"
    with pytest.raises(ValueError, match="differs|weights"):
        validate_quality_eligibility(proof, native, artifacts)


def test_final_vnni_rejects_missing_rejected_or_mutated_decision(tmp_path, protocol):
    path, native, artifacts = quality_fixture(tmp_path, protocol)
    p = {**protocol, "native": native, "artifacts": artifacts}
    with pytest.raises(ValueError, match="no retained"):
        check_quality_eligibility(p, tmp_path)
    p["quality_eligibility"] = load_quality_eligibility(path, native, artifacts)
    check_quality_eligibility(p, tmp_path)
    quality = json.loads(path.read_text())
    quality["comparison"]["vnni_decisions"][0]["retained"] = False
    path.write_text(json.dumps(quality))
    with pytest.raises(ValueError, match="rejected"):
        check_quality_eligibility(p, tmp_path)
    p["development"] = True
    check_quality_eligibility(p, tmp_path)


def test_vnni_linked_report_and_execution_path_are_bound(tmp_path, protocol):
    path, native, artifacts = quality_fixture(tmp_path, protocol)
    changed = {**native, "attention": "scalar"}
    with pytest.raises(ValueError, match="exactly one"):
        load_quality_eligibility(path, changed, artifacts)
    (tmp_path / "heldout-vnni.json").write_text("{}")
    with pytest.raises(ValueError, match="report hash"):
        load_quality_eligibility(path, native, artifacts)


def test_freeze_keeps_nice19_requirement(tmp_path, monkeypatch):
    args = SimpleNamespace(model=tmp_path, gguf=tmp_path, llama=tmp_path, model_manifest=tmp_path,
                           kv="f16", steps=64, repeats=5, development=False)
    monkeypatch.setattr(os, "getpriority", lambda *args: 0)
    with pytest.raises(ValueError, match="nice 19"):
        freeze(args, {})


def test_summary_rejects_final_vnni_without_eligibility(tmp_path, protocol):
    protocol["native"]["kernel"] = "vnni"
    protocol["id"] = digest({k: v for k, v in protocol.items() if k != "id"})
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="no retained"):
        summarize(tmp_path)


def test_unpinned_native_inherits_full_mask_and_keeps_baseline_categories(tmp_path, protocol):
    protocol["native"]["affinity"] = "unpinned"
    protocol["id"] = digest({k: v for k, v in protocol.items() if k != "id"})
    w = window(protocol)
    native_runs = [r for r in w["invocations"] if r["engine"] == "native"]
    assert w["cpu_set"] == list(range(12))
    assert all("--cpu-set" not in r["command"] for r in native_runs)
    assert all(r["command"][r["command"].index("--affinity") + 1] == "unpinned" for r in native_runs)
    save_inputs(tmp_path, protocol, w)
    summary = summarize(tmp_path, allow_partial=True)
    cell = summary["results"][0]
    assert cell["native_affinity"] == "unpinned"
    assert cell["cpu_set"] == list(range(12)) and cell["selected_cpu_set"] == [0, 1]
    assert cell["winner_core_sets_matched"]
    assert not next(c for c in cell["candidates"] if c["candidate"]["affinity"] == "pinned")["comparison_core_sets_matched"]
    assert not summary["failures"]
    native_runs[0]["command"] += ["--cpu-set", "0,1"]
    with pytest.raises(ValueError, match="inherit"):
        summarize_candidate(w, protocol, protocol["candidates"][0])


def test_interactive_native_overlap_is_rejected(protocol):
    w = window(protocol)
    w["invocations"][0]["command"] += ["--interactive", "1"]
    w["invocations"][0]["request"] = "run"
    with pytest.raises(ValueError, match="exit before baseline"):
        summarize_candidate(w, protocol, protocol["candidates"][0])
