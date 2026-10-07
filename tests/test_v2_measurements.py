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
from tools.measure_v2 import baseline_cpu_set, candidates, check_baseline_identity, check_quality_eligibility, cpu_mask, cpu_order, digest, engine_command, execute, execute_baseline, freeze, llama_command, load_quality_eligibility, native_cpu_set, validate_quality_eligibility
from tools.summarize_v2 import profile_bytes, select_best, spread, summarize, summarize_candidate
from tools.portable import portable
from tools.quality_v2 import reader_build_identity, reader_identity_locations
from tools.measure_v2 import matrix_dimensions, matrix_list


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
         "baseline_build_identity": {"reader_binary_sha256": "synthetic-reader", "shared_libraries": {}},
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
                                         **({"baseline_identity_before": deepcopy(p["baseline_build_identity"]),
                                             "baseline_identity_after": deepcopy(p["baseline_build_identity"])} if engine == "llama" else {}),
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
    assert not any(result["target_predicates"].values())
    assert cell["targets"] is None
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
    assert not any(summary["target_predicates"].values())


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
    summary = summarize(tmp_path, target_short_best_ceiling_percent=1, target_long_all_ceiling_percent=1)
    assert summary["complete_final_matrix"]
    assert all(summary["target_predicates"].values())
    assert summary["numeric_targets_met"] and summary["threshold_basis"] == "custom"
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
    summary = summarize(tmp_path, target_short_best_ceiling_percent=1, target_long_all_ceiling_percent=1)
    assert summary["complete_final_matrix"]
    assert not summary["target_predicates"]["native_over_best_baseline"]
    scaling = next(r for r in summary["thread12_scaling"]["comparisons"] if r["context"] == 4096)
    assert scaling["12_over_6"] == pytest.approx(50 / 102)
    assert scaling["decrease_percent_vs_strongest_lower"] == pytest.approx(100 * (1 - 50 / 102))
    assert summary["thread12_scaling"]["acceptance"] is None
    path.unlink()
    summary = summarize(tmp_path, target_short_best_ceiling_percent=1, target_long_all_ceiling_percent=1)
    assert not summary["complete_final_matrix"]
    assert summary["missing_cells"] == [(12, 4096)]
    assert not any(summary["target_predicates"].values())
    assert all(row["targets"] is None for row in summary["results"])


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


@pytest.mark.parametrize("existing", ["protocol.json", "cpu-discovery.json", "cpu-discovery.stdout.txt", "cpu-discovery.stderr.txt"])
def test_freeze_replay_preserves_all_discovery_evidence(tmp_path, monkeypatch, existing):
    names = ["protocol.json", "cpu-discovery.json", "cpu-discovery.stdout.txt", "cpu-discovery.stderr.txt"]
    evidence = tmp_path / existing
    evidence.write_bytes(b"original evidence")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    args = SimpleNamespace(model=tmp_path, gguf=tmp_path, llama=tmp_path, model_manifest=tmp_path,
                           kv="f16", steps=64, repeats=5, development=False, output=tmp_path)
    monkeypatch.setattr(os, "getpriority", lambda *a: 19)
    def no_execution(*a, **kw):
        raise AssertionError("rejected freeze must not launch discovery or any subprocess")
    monkeypatch.setattr(subprocess, "run", no_execution)
    with pytest.raises(FileExistsError):
        freeze(args, {})
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()} == before
    assert all((tmp_path / name).exists() == (name == existing) for name in names)


def resolved_baseline_fixture(tmp_path, monkeypatch):
    # The executable is deliberately outside the actual dependencies' build tree.
    executable = tmp_path / "launch" / "renamed-benchmark"
    executable.parent.mkdir()
    executable.write_bytes(b"benchmark")
    build = tmp_path / "upstream" / "build"
    library_dir = build / "bin"
    library_dir.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\nGGML_NATIVE:BOOL=ON\nGGML_BACKEND_DL:BOOL=OFF\n")
    (build / "compile_commands.json").write_text("[]")
    libraries = {}
    for name in ["libllama.so", "libllama-bench-impl.so", "libllama-common.so.0", "libggml.so", "libggml-base.so", "libggml-cpu.so"]:
        libraries[name] = library_dir / name
        libraries[name].write_bytes(name.encode())
    def fake_ldd(command, **kwargs):
        assert command == ["ldd", str(executable.resolve())]
        text = "\n".join(f"{name} => {path} (0x1234)" for name, path in libraries.items())
        return subprocess.CompletedProcess(command, 0, text, "")
    monkeypatch.setattr(subprocess, "run", fake_ldd)
    return executable, build, libraries, fake_ldd


@pytest.mark.parametrize("library_name", ["libggml-cpu.so", "libllama-bench-impl.so", "libllama-common.so.0"])
def test_resolved_library_hash_not_adjacent_file_or_stamp(tmp_path, monkeypatch, library_name):
    executable, build, libraries, _ = resolved_baseline_fixture(tmp_path, monkeypatch)
    locations = {tmp_path: "$FIXTURE"}
    identity = reader_build_identity(executable)
    p = {"baseline_build_identity": portable(identity, {**locations, **reader_identity_locations(identity)})}
    assert check_baseline_identity(executable, p, locations) == p["baseline_build_identity"]
    (executable.parent / "libggml-cpu.so").write_bytes(b"unused neighbor")
    check_baseline_identity(executable, p, locations)
    library = libraries[library_name]
    old = library.stat()
    original = library.read_bytes()
    library.write_bytes(b"x" * len(original))
    os.utime(library, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError, match="identity changed"):
        check_baseline_identity(executable, p, locations)
    library.write_bytes(original)
    (build / "compile_commands.json").write_text('[{"command":"changed flags"}]')
    with pytest.raises(ValueError, match="identity changed"):
        check_baseline_identity(executable, p, locations)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_baseline_identity_checked_around_each_invocation(tmp_path, monkeypatch, phase):
    executable, _, libraries, fake_ldd = resolved_baseline_fixture(tmp_path, monkeypatch)
    locations = {tmp_path: "$FIXTURE"}
    identity = reader_build_identity(executable)
    p = {"baseline_build_identity": portable(identity, {**locations, **reader_identity_locations(identity)})}
    library = libraries["libggml-cpu.so"]
    launches = []
    def run(command, **kwargs):
        if command[0] == "ldd":
            return fake_ldd(command, **kwargs)
        launches.append(command)
        library.write_bytes(b"changed during invocation")
        return subprocess.CompletedProcess(command, 0, '[{"rate":100}]', "raw stderr")
    monkeypatch.setattr(subprocess, "run", run)
    stem = tmp_path / "baseline"
    if phase == "before":
        library.write_bytes(b"changed before invocation")
        with pytest.raises(ValueError, match="identity changed"):
            execute_baseline(["benchmark"], executable, p, stem, locations, time.monotonic() + 10)
        assert not launches
        assert not stem.with_suffix(".json").exists()
    else:
        record = execute_baseline(["benchmark"], executable, p, stem, locations, time.monotonic() + 10)
        assert launches == [["benchmark"]]
        assert not record["success"]
        assert "baseline identity changed" in record["error"]
        assert record["data"] == [{"rate": 100}]
        assert stem.with_suffix(".stdout.txt").read_text() == '[{"rate":100}]'
        assert stem.with_suffix(".stderr.txt").read_text() == "raw stderr"
        assert not json.loads(stem.with_suffix(".json").read_text())["success"]


@pytest.mark.parametrize("phase", ["before", "after"])
def test_summary_rejects_unbound_baseline_identity(protocol, phase):
    w = window(protocol)
    run = next(r for r in w["invocations"] if r["engine"] == "llama")
    run[f"baseline_identity_{phase}"]["reader_binary_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        summarize_candidate(w, protocol, protocol["candidates"][0])


def save_target_matrix(directory, p, short_percentages, long_percentages):
    (directory / "protocol.json").write_text(json.dumps(p))
    for index, thread in enumerate(p["threads"]):
        (directory / f"bandwidth-t{thread}.json").write_text(json.dumps(bandwidth(p, thread)))
        for context in p["contexts"]:
            w = window(p, thread, context)
            percentage = (short_percentages[index] if context == 128 else
                          long_percentages[index] if context == 4096 else 80)
            rate = percentage / 100 * 12e9 / byte_counts(p, context)["total_min"]
            for run in w["invocations"]:
                if run["engine"] == "native":
                    for sample in run["data"]["samples"]:
                        sample["tokens_per_second"] = rate
                        sample["seconds"] = p["steps"] / rate
                        sample["step_seconds"] = [1 / rate] * p["steps"]
            (directory / f"window-t{thread}-c{context}-all.json").write_text(json.dumps(w))


@pytest.mark.parametrize("short,long,short_met,long_met", [
    ([60, 60, 86, 60, 60], [76] * 5, True, True),
    ([84] * 5, [76] * 5, False, True),
    ([86] * 5, [76, 76, 76, 74, 76], True, False),
    ([60] * 5, [60] * 5, False, False),
])
def test_default_targets_are_short_best_and_long_all(tmp_path, protocol, short, long, short_met, long_met):
    save_target_matrix(tmp_path, protocol, short, long)
    summary = summarize(tmp_path)
    assert summary["complete_final_matrix"]
    assert summary["thresholds"] == {"native_over_best_baseline": 1,
                                     "short_best_percent_of_ceiling": 85,
                                     "long_all_percent_of_ceiling": 75}
    assert summary["threshold_basis"] == "user"
    assert summary["target_predicates"] == {"native_over_best_baseline": True,
                                            "short_best_percent_of_ceiling": short_met,
                                            "long_all_percent_of_ceiling": long_met}
    assert summary["numeric_targets_met"] == (short_met and long_met)
    assert all(r["targets"]["long_all_percent_of_ceiling"] is None for r in summary["results"] if r["context"] != 4096)
    assert all(r["targets"]["short_best_percent_of_ceiling"] is None for r in summary["results"] if r["context"] != 128)
    assert summary["thread12_scaling"]["acceptance"] is None


@pytest.mark.parametrize("fault", ["missing_cell", "missing_candidate", "missing_bandwidth", "development"])
def test_context_targets_never_claim_incomplete_or_development_matrix(tmp_path, protocol, fault):
    if fault == "development":
        protocol["development"] = True
        protocol["id"] = digest({k: v for k, v in protocol.items() if k != "id"})
    save_target_matrix(tmp_path, protocol, [86] * 5, [76] * 5)
    path = tmp_path / "window-t12-c4096-all.json"
    if fault == "missing_cell":
        path.unlink()
    elif fault == "missing_candidate":
        raw = json.loads(path.read_text())
        raw["candidates"].pop()
        path.write_text(json.dumps(raw))
    elif fault == "missing_bandwidth":
        (tmp_path / "bandwidth-t12.json").unlink()
    summary = summarize(tmp_path, allow_partial=True)
    assert not summary["complete_final_matrix"]
    assert not summary["numeric_targets_met"]
    assert not any(summary["target_predicates"].values())
    assert all(row["targets"] is None for row in summary["results"])


@pytest.mark.parametrize("kwargs", [
    {"target_ratio": float("nan")}, {"target_short_best_ceiling_percent": 0},
    {"target_long_all_ceiling_percent": float("inf")},
])
def test_custom_target_thresholds_reject_invalid_values(tmp_path, kwargs):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize(tmp_path, **kwargs)


@pytest.mark.parametrize("threads,contexts", [([1, 2, 4, 6, 12], [128, 1024, 4096]), ([2, 6], [128, 4096])])
def test_freeze_uses_resolved_build_for_relocated_benchmark(tmp_path, monkeypatch, protocol, threads, contexts):
    executable, build, libraries, _ = resolved_baseline_fixture(tmp_path, monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text(json.dumps(
        {"num_hidden_layers": 24, "num_key_value_heads": 2, "hidden_size": 128, "num_attention_heads": 2}))
    native = tmp_path / "native"
    native.mkdir()
    engine = native / "engine"
    engine.write_bytes(b"engine")
    (native / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nCPU_DECODE_NATIVE:BOOL=ON\nCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n")
    (native / "compile_commands.json").write_text("[]")
    gguf = tmp_path / "baseline.gguf"
    gguf.write_bytes(b"Q8")
    bw = native / "bandwidth"
    bw.write_bytes(b"bandwidth")
    manifest = tmp_path / "manifest.json"
    source = {"fixture": "same pinned model"}
    manifest.write_text(json.dumps({"source": source, "group_size": 32, "scale_dtype": "f16",
        "weights": {"sha256": file_hash(model / "model.safetensors")},
        "config_sha256": file_hash(model / "config.json")}))
    preparation = tmp_path / "preparation.json"
    preparation.write_text(json.dumps({"llama_commit": protocol["llama_commit"], "source_model": source,
        "build": {"type": "Release", "native_cpu": True, "gpu": False},
        "artifacts": {"Q8_0": {"sha256": file_hash(gguf)}}}))
    output = tmp_path / "evidence"
    output.mkdir()
    args = SimpleNamespace(model=model, engine=engine, llama=executable, gguf=gguf, bandwidth=bw,
        model_manifest=manifest, preparation=preparation, output=output, development=False,
        kv="f16", steps=64, repeats=5, cpu_order=None, polls="50", tokens="1,2,3",
        kernel="simd512x4", attention="blocked", scheduler="pool", affinity="strict", rope="cached",
        thread_counts=threads, context_lengths=contexts)
    monkeypatch.setattr(os, "getpriority", lambda *a: 19)
    monkeypatch.setattr("tools.measure_v2.observation", lambda command: {"command": command})
    monkeypatch.setattr("tools.measure_v2.environment", lambda locations: {"fixture": "isolated host observation"})
    cpus = {"allowed_cpu_ids": list(range(max(threads))), "preferred_cpu_ids": list(range(max(threads)))}
    monkeypatch.setattr("tools.measure_v2.execute", lambda *a, **kw: {"success": True, "data": cpus})
    locations = {tmp_path: "$FIXTURE"}
    freeze(args, locations)
    frozen = json.loads((output / "protocol.json").read_text())
    identity = frozen["baseline_build_identity"]
    assert identity["build"]["root"] == "$LLAMA_BUILD"
    assert identity["build"]["compile_flags_sha256"] == {"compile_commands.json": file_hash(build / "compile_commands.json")}
    assert identity["shared_libraries"]["libggml-cpu.so"]["resolved_path"] == "$LLAMA_BUILD/bin/libggml-cpu.so"
    assert identity["shared_libraries"]["libggml-cpu.so"]["sha256"] == file_hash(libraries["libggml-cpu.so"])
    assert identity["reader_binary_sha256"] == file_hash(executable)
    assert frozen["source_model"] == source
    assert frozen["threads"] == threads and frozen["contexts"] == contexts
    assert len(frozen["cpu_order"]) == max(threads)
    assert str(tmp_path) not in json.dumps(frozen)
    check_baseline_identity(executable, frozen, {tmp_path: "$FIXTURE"})


@pytest.mark.parametrize("value", ["", "2,2", "0,6", "-1,2", "2,", "two,6"])
def test_matrix_options_reject_invalid_lists(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match="matrix lists"):
        matrix_list(value)


def test_matrix_options_preserve_explicit_order():
    assert matrix_list("6,2") == [6, 2]
    assert matrix_dimensions([6, 2], [4096, 128]) == ([6, 2], [4096, 128])


@pytest.mark.parametrize("threads,contexts", [([], [128]), ([2, 2], [128]),
                                             ([True], [128]), ([2], [0]),
                                             ([2], [128, 128]), ("2,6", [128])])
def test_frozen_matrix_dimensions_reject_duplicates_and_invalid_types(threads, contexts):
    with pytest.raises(ValueError, match="matrix dimensions"):
        matrix_dimensions(threads, contexts)


def test_summary_asserts_requested_dimensions_without_overriding_protocol(tmp_path, protocol):
    protocol.update(threads=[2, 6], contexts=[128, 4096])
    protocol["id"] = digest({key: value for key, value in protocol.items() if key != "id"})
    save_target_matrix(tmp_path, protocol, [86, 86], [76, 76])
    summary = summarize(tmp_path, thread_counts=[2, 6], context_lengths=[128, 4096])
    assert summary["complete_requested_matrix"] and summary["matrix_scope"] == "explicit-subset"
    assert not summary["numeric_targets_met"] and not summary["target_eligible"]
    assert summary["expected_candidate_windows"] == 36 and summary["expected_bandwidth_windows"] == 2
    assert summary["thread12_scaling"]["comparisons"] == []
    with pytest.raises(ValueError, match="differs from frozen matrix"):
        summarize(tmp_path, thread_counts=[1, 2, 4, 6, 12])
    with pytest.raises(ValueError, match="differs from frozen matrix"):
        summarize(tmp_path, context_lengths=[128])


@pytest.mark.parametrize("outside", ["thread", "context", "bandwidth"])
def test_subset_summary_rejects_foreign_cells_and_bandwidth(tmp_path, protocol, outside):
    protocol.update(threads=[2, 6], contexts=[128, 4096])
    protocol["id"] = digest({key: value for key, value in protocol.items() if key != "id"})
    save_target_matrix(tmp_path, protocol, [86, 86], [76, 76])
    raw = (bandwidth(protocol, 1) if outside == "bandwidth" else
           window(protocol, 1 if outside == "thread" else 2, 1024 if outside == "context" else 128))
    (tmp_path / "foreign.json").write_text(json.dumps(raw))
    summary = summarize(tmp_path)
    assert not summary["complete_requested_matrix"]
    assert summary["failures"]
    assert all("outside frozen matrix" in failure["error"] for failure in summary["failures"])


def test_full_larger_model_matrix_is_not_original_model_user_target(tmp_path, protocol):
    protocol["source_model"] = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct"}
    protocol["id"] = digest({key: value for key, value in protocol.items() if key != "id"})
    save_target_matrix(tmp_path, protocol, [86] * 5, [76] * 5)
    summary = summarize(tmp_path)
    assert summary["complete_requested_matrix"]
    assert not summary["full_fifteen_cell_target_matrix"]
    assert not summary["numeric_targets_met"]
