"""Synthetic runner tests: no model loads, benchmark processes or performance claims."""
from copy import deepcopy
import json
import signal
from types import SimpleNamespace

import pytest

from test_v2_measurements import bandwidth, protocol, window
from tools import run_final_v2 as runner
from tools.measure_v2 import digest, execute
from tools.summarize_v2 import summarize


@pytest.fixture
def args(tmp_path):
    output = tmp_path / "final"
    output.mkdir()
    result = SimpleNamespace(output=output, kernel="simd512x4", affinity="strict", cpu_order=None,
                             tokens="1,2,3", wrapper="/private/scheduler bench", plan=False,
                             thread_counts=[1, 2, 4, 6, 12], context_lengths=[128, 1024, 4096])
    for key in ["model", "model_manifest", "format_selection", "llama", "gguf", "preparation",
                "quality", "engine", "bandwidth"]:
        setattr(result, key, tmp_path / key)
    manifest = {"group_size": 32, "scale_dtype": "f16", "weights": {"sha256": "weights"},
                "config_sha256": "config"}
    chosen = {"label": "g32f16", "group_size": 32, "scale_dtype": "f16", "model_identity": {"files": {
        "model.safetensors": {"sha256": "weights"}, "config.json": {"sha256": "config"}}}}
    for key, value in [("model_manifest", manifest), ("format_selection", {"split": "calibration", "chosen": chosen}),
                       ("preparation", {}), ("quality", {"split": "heldout"})]:
        runner.atomic_json(getattr(result, key), value)
    result.engine.write_bytes(b"synthetic engine identity")
    report = {"label": "chosen-f16", "split": "heldout", "backend": "native",
        "selection_sha256": runner.file_hash(result.format_selection), "model_identity": chosen["model_identity"],
        "binary_identity": {"engine_binary_sha256": runner.file_hash(result.engine)},
        "settings": {"kernel": result.kernel, "kv_dtype": "f16", "attention": "blocked", "scheduler": "pool",
                     "affinity": result.affinity, "group_size": 32, "scale_dtype": "f16"},
        "aggregate": {"top1_agreement": 1}, "oracle_identity": {"source": "synthetic"}}
    report_path = tmp_path / "heldout-native.json"
    runner.atomic_json(report_path, report)
    entry = {key: report[key] for key in ["label", "split", "backend", "model_identity", "binary_identity",
                                        "settings", "aggregate", "oracle_identity"]}
    runner.atomic_json(result.quality, {"split": "heldout",
        "selection_sha256": runner.file_hash(result.format_selection), "selection": runner.load(result.format_selection),
        "evidence": [{**entry, "report": report_path.name, "sha256": runner.file_hash(report_path)}]})
    return result


def unit_raw(unit, protocol):
    if unit["stage"] == "bandwidth":
        return bandwidth(protocol, unit["threads"])
    raw = window(protocol, unit["threads"], unit["context"])
    raw["candidates"] = [unit["candidate"]]
    raw["invocations"] = [run for run in raw["invocations"] if run["candidate_id"] == unit["candidate"]["id"]]
    return raw


def save_unit(attempt, unit, protocol):
    raw = unit_raw(unit, protocol)
    for index, invocation in enumerate(raw["invocations"]):
        for key in ["stdout_file", "stderr_file"]:
            path = attempt / f"{index}-{key}.txt"
            path.write_text("synthetic raw\n")
            invocation[key] = str(path)
    runner.atomic_json(attempt / (unit["name"] + ".json"), raw)
    return raw


def accepted_attempt(args, unit, protocol):
    attempt, _ = runner.new_attempt(args.output, unit["name"], lambda path: [])
    save_unit(attempt, unit, protocol)
    runner.disposition(attempt, "accepted", assets=runner.assets(attempt))
    return attempt


def test_plan_full_matrix_and_command_bounds(args):
    result = runner.plan(args)
    windows = [u for u in result["units"] if u["stage"] == "window"]
    assert len(windows) == 135
    assert {(u["threads"], u["context"]) for u in windows} == {
        (t, c) for t in [1, 2, 4, 6, 12] for c in [128, 1024, 4096]}
    assert len({u["name"] for u in result["units"]}) == 140
    for thread in [1, 2, 4, 6, 12]:
        for context in [128, 1024, 4096]:
            cell = [u for u in windows if (u["threads"], u["context"]) == (thread, context)]
            assert {u["candidate"]["flash_attn"] for u in cell} == {"on", "off", "auto"}
            assert {u["candidate"]["affinity"] for u in cell} == {"pinned", "unpinned", "defaults"}
            assert len(cell) == 9
    for unit in result["units"]:
        assert unit["command"][:10] == ["/private/scheduler", "bench", "timeout", "--foreground",
                                       "--signal=TERM", "--kill-after=5s", "1770s", "nice", "-n", "19"]
        if unit["stage"] == "window":
            assert unit["command"][unit["command"].index("--candidate") + 1] != "all"
    assert result["freeze"][:3] == ["nice", "-n", "19"]
    assert result["window_deadline_seconds"] < 1800
    assert result["hard_command_limit_seconds"] < 1800
    assert result["model_processes"] == 135 * 4
    assert result["measured_tokens_per_engine"] == 135 * 2 * 5 * 64
    assert result["wall_time_estimate"].startswith("Unknown")
    assert result["kind"] == "INFERENCE"


def test_public_default_and_quoted_wrapper(args):
    args.wrapper = ""
    assert runner.plan(args)["units"][0]["command"][:8] == [
        "timeout", "--foreground", "--signal=TERM", "--kill-after=5s", "1770s", "nice", "-n", "19"]
    args.wrapper = "'/private/path with spaces/scheduler' bench"
    assert runner.plan(args)["units"][0]["command"][0] == "/private/path with spaces/scheduler"


@pytest.mark.parametrize("field", ["format_selection", "model_manifest", "preparation", "quality"])
def test_resume_rejects_changed_input_hash(args, field):
    original = runner.specification(args)
    path = getattr(args, field)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="identity/settings changed"):
        runner.require_same(original, runner.specification(args))


@pytest.mark.parametrize("field,value", [("kernel", "simd256"), ("affinity", "unpinned"),
                                        ("wrapper", "/other/scheduler bench"), ("tokens", "4,5"),
                                        ("cpu_order", "1,0,2,3,4,5,6,7,8,9,10,11"),
                                        ("thread_counts", [2, 6]), ("context_lengths", [128, 4096])])
def test_resume_rejects_changed_settings(args, field, value):
    original = runner.specification(args)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="identity/settings changed"):
        runner.require_same(original, runner.specification(args))


def test_selection_rejects_format_guess_and_heldout(args):
    selection = runner.load(args.format_selection)
    manifest = runner.load(args.model_manifest)
    selection["chosen"]["group_size"] = 64
    with pytest.raises(ValueError, match="chosen format"):
        runner.validate_selection(selection, manifest)
    selection = runner.load(args.format_selection)
    selection["chosen"]["model_identity"]["files"]["model.safetensors"]["sha256"] = "other"
    with pytest.raises(ValueError, match="artifact identity"):
        runner.validate_selection(selection, manifest)
    selection["split"] = "heldout"
    with pytest.raises(ValueError, match="calibration"):
        runner.validate_selection(selection, manifest)


def test_resume_skips_valid_unit_and_recovers_atomic_publication(args, protocol):
    unit = runner.units()[0]
    attempt = accepted_attempt(args, unit, protocol)
    assert runner.recover_unit(args, unit, protocol)
    assert runner.load(args.output / (unit["name"] + ".json")) == runner.load(attempt / (unit["name"] + ".json"))
    assert runner.recover_unit(args, unit, protocol)
    assert len(list((args.output / "attempts" / unit["name"]).iterdir())) == 1


def test_partial_attempt_is_linked_and_never_combined(args, protocol):
    unit = runner.units()[0]
    first, _ = runner.new_attempt(args.output, unit["name"], lambda path: [])
    raw = save_unit(first, unit, protocol)
    raw["invocations"] = raw["invocations"][:2]
    runner.atomic_json(first / (unit["name"] + ".json"), raw)
    assert not runner.recover_unit(args, unit, protocol)
    assert runner.load(first / "attempt.json")["disposition"] == "interrupted"
    assert runner.load(first / (unit["name"] + ".json"))["invocations"] == raw["invocations"]
    second = accepted_attempt(args, unit, protocol)
    assert first != second and runner.recover_unit(args, unit, protocol)
    published = runner.load(args.output / (unit["name"] + ".json"))
    assert published == runner.load(second / (unit["name"] + ".json"))
    assert len(published["invocations"]) == 4
    statuses = runner.dispositions(args.output)
    assert {r["disposition"] for r in statuses} == {"interrupted", "accepted"}
    assert all(r["raw_directory"] for r in statuses)


def test_duplicate_successful_attempts_are_rejected(args, protocol):
    unit = runner.units()[0]
    accepted_attempt(args, unit, protocol)
    accepted_attempt(args, unit, protocol)
    with pytest.raises(ValueError, match="duplicate accepted"):
        runner.recover_unit(args, unit, protocol)


def test_changed_accepted_raw_and_unlinked_canonical_are_rejected(args, protocol):
    unit = runner.units()[0]
    attempt = accepted_attempt(args, unit, protocol)
    (attempt / "0-stdout_file.txt").write_text("changed")
    with pytest.raises(ValueError, match="identity/settings changed"):
        runner.recover_unit(args, unit, protocol)
    other = runner.units()[1]
    runner.atomic_json(args.output / (other["name"] + ".json"), unit_raw(other, protocol))
    with pytest.raises(ValueError, match="no accepted attempt"):
        runner.recover_unit(args, other, protocol)


def test_collection_failure_retains_attempt_and_resume_uses_new_directory(args, protocol, monkeypatch):
    unit = runner.units()[0]
    monkeypatch.setattr(runner, "check_current", lambda *a: None)
    def failed(command, attempt, aliases):
        save_unit(attempt, unit, protocol)
        (attempt / "failure.stdout.txt").write_text("failure raw")
        return 137
    monkeypatch.setattr(runner, "run_process", failed)
    assert not runner.collect(args, unit, protocol)
    assert not (args.output / (unit["name"] + ".json")).exists()
    first = args.output / "attempts" / unit["name"] / "000001"
    assert runner.load(first / "attempt.json")["disposition"] == "failed"
    assert runner.load(first / "attempt.json")["returncode"] == 137
    def good(command, attempt, aliases):
        save_unit(attempt, unit, protocol)
        return 0
    monkeypatch.setattr(runner, "run_process", good)
    assert runner.collect(args, unit, protocol)
    assert (first / "failure.stdout.txt").read_text() == "failure raw"
    assert runner.recover_unit(args, unit, protocol)


def test_interruption_disposition_and_durable_inflight_logs(args, protocol, monkeypatch):
    unit = runner.units()[0]
    monkeypatch.setattr(runner, "check_current", lambda *a: None)
    def interrupted(command, attempt, aliases):
        (attempt / "active.stdout.txt").write_text("partial raw")
        raise KeyboardInterrupt()
    monkeypatch.setattr(runner, "run_process", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.collect(args, unit, protocol)
    attempt = args.output / "attempts" / unit["name"] / "000001"
    assert runner.load(attempt / "attempt.json")["disposition"] == "interrupted"
    assert (attempt / "active.stdout.txt").read_text() == "partial raw"
    assert not runner.recover_unit(args, unit, protocol)
    def interrupted_model(command, **kwargs):
        kwargs["stdout"].write("model partial")
        kwargs["stdout"].flush()
        kwargs["stderr"].write("model stderr")
        kwargs["stderr"].flush()
        raise KeyboardInterrupt()
    import tools.measure_v2 as measure
    monkeypatch.setattr(measure.subprocess, "run", interrupted_model)
    import time
    with pytest.raises(KeyboardInterrupt):
        execute(["not-executed"], attempt / "inflight", {}, time.monotonic() + 10)
    assert (attempt / "inflight.stdout.txt").read_text() == "model partial"
    assert (attempt / "inflight.stderr.txt").read_text() == "model stderr"


def test_process_group_cleanup_and_memory_budget(args, monkeypatch):
    attempt = args.output / "fake-process"
    attempt.mkdir()
    observed = {}
    class FakeProcess:
        pid = 12345
        def wait(self, timeout=None):
            return 0
    def popen(command, **kwargs):
        observed.update(kwargs)
        return FakeProcess()
    signals = []
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert runner.run_process(["never-executed"], attempt, runner.locations(args)) == 0
    assert observed["env"]["PP_MEM"] == "2000M"
    assert observed["start_new_session"] is True
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


def test_freeze_is_once_and_changed_protocol_rejected(args, protocol, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "check_current", lambda *a: None)
    def fake_freeze(command, attempt, aliases):
        calls.append(command)
        runner.atomic_json(attempt / "protocol.json", protocol)
        return 0
    monkeypatch.setattr(runner, "run_process", fake_freeze)
    assert runner.freeze_once(args) == protocol
    assert runner.freeze_once(args) == protocol
    assert len(calls) == 1
    changed = deepcopy(protocol)
    changed["steps"] = 128
    changed["id"] = digest({k: v for k, v in changed.items() if k != "id"})
    runner.atomic_json(args.output / "protocol.json", changed)
    with pytest.raises(ValueError, match="identity/settings changed"):
        runner.freeze_once(args)


def test_changed_resolved_library_aborts_before_new_window(args, protocol, monkeypatch):
    from tools.quality_v2 import require_reader_identity
    original = {"reader_binary_sha256": "reader", "shared_libraries": {"libllama.so": {"sha256": "original"}}}
    changed = deepcopy(original)
    changed["shared_libraries"]["libllama.so"]["sha256"] = "changed"
    monkeypatch.setattr(runner, "verify_protocol", lambda *a: None)
    monkeypatch.setattr(runner, "check_artifacts", lambda *a: require_reader_identity(original, changed))
    with pytest.raises(ValueError):
        runner.collect(args, runner.units()[0], protocol)
    assert not (args.output / "attempts").exists()


def test_no_aggregate_acceptance_until_all_140_units_and_all_noise_kept(args, protocol):
    runner.atomic_json(args.output / "protocol.json", protocol)
    for unit in runner.units()[:-1]:
        raw = unit_raw(unit, protocol)
        if unit["stage"] == "window" and unit == runner.units()[0]:
            native = raw["invocations"][0]["data"]["samples"][0]
            native.update(tokens_per_second=150, seconds=64 / 150, step_seconds=[1 / 150] * 64)
        runner.atomic_json(args.output / (unit["name"] + ".json"), raw)
    partial = summarize(args.output)
    assert not partial["complete_final_matrix"]
    assert not partial["numeric_targets_met"]
    assert all(r["targets"] is None for r in partial["results"])
    assert len(partial["noisy_candidates"]) >= 1
    final_unit = runner.units()[-1]
    runner.atomic_json(args.output / (final_unit["name"] + ".json"), unit_raw(final_unit, protocol))
    complete = summarize(args.output)
    assert complete["complete_final_matrix"]
    assert len(complete["results"]) == 15
    assert all(len(row["candidates"]) == 9 for row in complete["results"])
    assert len(complete["bandwidth"]) == 5
    assert all(c["native_tps"]["samples"] == c["baseline_tps"]["samples"] == 10
               for row in complete["results"] for c in row["candidates"])
    assert len(complete["noisy_candidates"]) >= 1


def test_dry_run_does_not_read_models_or_launch_commands(args, monkeypatch, capsys):
    args.plan = True
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "specification", lambda *a: pytest.fail("dry run read identity inputs"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("dry run launched a process"))
    runner.main()
    assert json.loads(capsys.readouterr().out)["timing_commands"] == 140


def test_quality_selection_must_match_explicit_new_format(args):
    runner.check_quality_selection(args)
    quality = runner.load(args.quality)
    quality["selection"]["chosen"]["label"] = "other-format"
    runner.atomic_json(args.quality, quality)
    with pytest.raises(ValueError, match="explicitly supplied format"):
        runner.check_quality_selection(args)


@pytest.mark.parametrize("mutation", ["engine", "report", "settings", "model", "duplicate"])
def test_final_quality_binds_binary_report_and_execution_path(args, mutation):
    quality = runner.load(args.quality)
    if mutation == "engine":
        args.engine.write_bytes(b"different engine")
    elif mutation == "report":
        (args.quality.parent / quality["evidence"][0]["report"]).write_text("{}")
    elif mutation == "duplicate":
        quality["evidence"].append(deepcopy(quality["evidence"][0]))
        runner.atomic_json(args.quality, quality)
    else:
        entry = quality["evidence"][0]
        if mutation == "settings":
            entry["settings"]["kv_dtype"] = "f32"
        else:
            entry["model_identity"] = {"files": {}}
        runner.atomic_json(args.quality, quality)
    with pytest.raises(ValueError):
        runner.check_quality_selection(args)


@pytest.mark.parametrize("stage", ["window", "freeze"])
def test_resume_retains_attempt_missing_initial_checkpoint(args, protocol, monkeypatch, stage):
    unit = runner.units()[0]
    name = unit["name"] if stage == "window" else "freeze"
    abandoned = args.output / "attempts" / name / "000001"
    abandoned.mkdir(parents=True)
    (abandoned / "attempt.json.tmp").write_text("partial checkpoint")
    if stage == "window":
        assert not runner.recover_unit(args, unit, protocol)
        accepted = accepted_attempt(args, unit, protocol)
        assert runner.recover_unit(args, unit, protocol)
    else:
        accepted, _ = runner.new_attempt(args.output, "freeze", lambda path: [])
        runner.atomic_json(accepted / "protocol.json", protocol)
        runner.disposition(accepted, "accepted", assets=runner.assets(accepted))
        monkeypatch.setattr(runner, "check_current", lambda *a: None)
        assert runner.freeze_once(args) == protocol
    assert accepted.name == "000002"
    assert runner.load(abandoned / "attempt.json")["disposition"] == "interrupted"
    assert (abandoned / "attempt.json.tmp").read_text() == "partial checkpoint"


def test_orphan_cleanup_checks_pid_start_before_signalling(monkeypatch):
    signals = []
    monkeypatch.setattr(runner, "process_start", lambda pid: "new-start")
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    runner.stop_orphan({"process_pid": 12345, "process_start": "old-start"})
    assert not signals
    runner.stop_orphan({"process_pid": 12345, "process_start": "new-start"})
    assert signals == [(12345, signal.SIGKILL)]


@pytest.mark.parametrize("defect", ["partial", "duplicate", "short_samples", "foreign_candidate"])
def test_zero_exit_is_not_acceptance_of_invalid_samples(args, protocol, monkeypatch, defect):
    unit = runner.units()[0]
    monkeypatch.setattr(runner, "check_current", lambda *a: None)
    def invalid(command, attempt, aliases):
        raw = save_unit(attempt, unit, protocol)
        if defect == "partial":
            raw["invocations"] = raw["invocations"][:2]
        elif defect == "duplicate":
            raw["invocations"].append(deepcopy(raw["invocations"][0]))
        elif defect == "short_samples":
            raw["invocations"][0]["data"]["samples"].pop()
        else:
            raw["candidates"] = [runner.units()[1]["candidate"]]
        runner.atomic_json(attempt / (unit["name"] + ".json"), raw)
        return 0
    monkeypatch.setattr(runner, "run_process", invalid)
    assert not runner.collect(args, unit, protocol)
    attempt = args.output / "attempts" / unit["name"] / "000001"
    record = runner.load(attempt / "attempt.json")
    assert record["disposition"] == "failed" and record["returncode"] == 0
    assert record["raw_directory"] == str(attempt.relative_to(args.output))
    assert (attempt / (unit["name"] + ".json")).exists()
    assert not (args.output / (unit["name"] + ".json")).exists()


def test_subset_plan_keeps_all_candidates_and_bounded_chunks(args):
    args.thread_counts, args.context_lengths = [2, 6], [128, 4096]
    result = runner.plan(args)
    assert result["matrix_scope"] == "explicit-subset"
    assert result["timing_commands"] == 38
    assert result["candidate_windows"] == 36
    assert result["bandwidth_windows"] == 2
    assert result["model_processes"] == 144
    assert result["bandwidth_processes"] == 4
    assert result["measured_tokens_per_engine"] == 23040
    assert {(u["threads"], u["context"]) for u in result["units"] if u["stage"] == "window"} == {
        (2, 128), (2, 4096), (6, 128), (6, 4096)}
    freeze = result["freeze"]
    assert freeze[freeze.index("--thread-counts") + 1] == "2,6"
    assert freeze[freeze.index("--context-lengths") + 1] == "128,4096"
    assert len({u["name"] for u in result["units"]}) == 38
    assert result["hard_command_limit_seconds"] < 1800


def test_subset_protocol_validation_rejects_dimension_drift(args, protocol):
    args.thread_counts, args.context_lengths = [2, 6], [128, 4096]
    protocol.update(threads=args.thread_counts, contexts=args.context_lengths,
                    model_manifest=runner.load(args.model_manifest))
    protocol["id"] = runner.digest({k: v for k, v in protocol.items() if k != "id"})
    runner.verify_protocol(protocol, args)
    protocol["contexts"] = [128]
    protocol["id"] = runner.digest({k: v for k, v in protocol.items() if k != "id"})
    with pytest.raises(ValueError, match="contexts"):
        runner.verify_protocol(protocol, args)


@pytest.mark.parametrize("fault", ["missing_bandwidth", "missing_cell", "missing_candidate", "duplicate"])
def test_subset_complete_requires_all_38_units(args, protocol, fault):
    protocol.update(threads=[2, 6], contexts=[128, 4096],
                    source_model={"model_id": "Qwen/Qwen2.5-1.5B-Instruct"})
    protocol["id"] = runner.digest({k: v for k, v in protocol.items() if k != "id"})
    runner.atomic_json(args.output / "protocol.json", protocol)
    for unit in runner.units(protocol["threads"], protocol["contexts"]):
        runner.atomic_json(args.output / (unit["name"] + ".json"), unit_raw(unit, protocol))
    complete = summarize(args.output)
    assert not complete["complete_final_matrix"] and complete["complete_requested_matrix"]
    assert len(complete["results"]) == 4 and len(complete["bandwidth"]) == 2
    assert all(len(row["candidates"]) == 9 for row in complete["results"])
    assert complete["matrix_scope"] == "explicit-subset"
    assert not complete["full_fifteen_cell_target_matrix"] and not complete["target_eligible"]
    assert not complete["numeric_targets_met"] and not any(complete["target_predicates"].values())
    assert all(row["targets"] is None for row in complete["results"])
    if fault == "missing_bandwidth":
        (args.output / "bandwidth-t6.json").unlink()
    elif fault == "missing_cell":
        for path in args.output.glob("window-t6-c4096-*.json"):
            path.unlink()
    elif fault == "missing_candidate":
        unit = runner.units([6], [4096])[0]
        (args.output / (unit["name"] + ".json")).unlink()
    else:
        unit = runner.units([6], [4096])[0]
        runner.atomic_json(args.output / "duplicate.json", unit_raw(unit, protocol))
    incomplete = summarize(args.output)
    assert not incomplete["complete_requested_matrix"] and not incomplete["numeric_targets_met"]


def test_fixed_format_selection_binds_target_and_origin(args):
    manifest = runner.load(args.model_manifest)
    source = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "pinned"}
    manifest.update(group_size=64, source=source)
    original = runner.load(args.format_selection)
    selection = {"schema": "fixed-format-transfer-v1", "split": "fixed_format_transfer",
                 "chosen": {**original["chosen"], "label": "g64f16", "group_size": 64},
                 "origin_selection": original, "verified_source": source,
                 "oracle_identity": {"verified_source": source}}
    runner.validate_selection(selection, manifest)
    with pytest.raises(ValueError, match="target-model calibration"):
        runner.validate_selection(original, manifest)
    changed = deepcopy(selection)
    changed["verified_source"] = {"model_id": "wrong"}
    with pytest.raises(ValueError, match="target source"):
        runner.validate_selection(changed, manifest)
    changed = deepcopy(selection)
    changed["origin_selection"]["split"] = "heldout"
    with pytest.raises(ValueError, match="original calibration"):
        runner.validate_selection(changed, manifest)
    changed = deepcopy(selection)
    changed["chosen"]["model_identity"]["files"]["config.json"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="artifact identity"):
        runner.validate_selection(changed, manifest)


def config_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    models = []
    for name in ["0.5", "s1"]:
        directory = tmp_path / name
        directory.mkdir()
        model = directory / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"synthetic")
        (model / "config.json").write_text("{}")
        argv = ["--model", str(model), "--kernel", "simd512x4", "--affinity", "strict"]
        for key in ["model-manifest", "format-selection", "llama", "gguf", "preparation", "quality", "engine", "bandwidth"]:
            path = directory / key
            path.write_text("{}")
            argv += ["--" + key, str(path)]
        (directory / "model-manifest").write_text(json.dumps(
            {"source": {"model_id": f"Qwen/Qwen2.5-{'0.5' if name == '0.5' else '1.5'}B-Instruct"}}))
        output = tmp_path / ("results/v2/final" if name == "0.5" else "results/v2/s1/final")
        argv += ["--output", str(output)]
        if name == "s1":
            argv += ["--thread-counts", "2,6", "--context-lengths", "128,4096"]
        models.append({"name": name, "optional": name == "s1", "argv": argv})
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({"models": models}))
    return path


def test_combined_plan_never_reads_identity_or_starts_processes(tmp_path, monkeypatch, capsys):
    path = config_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "missing_inputs", lambda *a: pytest.fail("plan checked readiness"))
    monkeypatch.setattr(runner, "run_model", lambda *a: pytest.fail("plan ran model"))
    runner.run_config(path, dry_run=True)
    output = json.loads(capsys.readouterr().out)
    assert output["timing_commands_if_all_ready"] == 178
    assert [model["timing_commands"] for model in output["models"]] == [140, 38]


def test_combined_runs_prepared_models_sequentially_and_resumes(tmp_path, monkeypatch, capsys):
    path = config_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner.os, "setpriority", lambda *a: None)
    calls = []
    monkeypatch.setattr(runner, "run_model", lambda args: calls.append(args.output))
    runner.run_config(path)
    assert calls == [tmp_path / "results/v2/final", tmp_path / "results/v2/s1/final"]
    statuses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [status["model"] for status in statuses] == ["0.5", "s1"]
    runner.run_config(path)
    assert calls[2:] == calls[:2]  # Same single-model resume path, not a separate sampling implementation.


def test_combined_optional_not_ready_is_visible_but_existing_resume_is_not_skipped(tmp_path, monkeypatch, capsys):
    path = config_fixture(tmp_path, monkeypatch)
    (tmp_path / "s1/quality").unlink()
    monkeypatch.setattr(runner.os, "setpriority", lambda *a: None)
    calls = []
    monkeypatch.setattr(runner, "run_model", lambda args: calls.append(args.output))
    runner.run_config(path)
    assert len(calls) == 1
    statuses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert statuses[1] == {"model": "s1", "status": "skipped-not-ready", "missing_inputs": ["quality"]}
    output = tmp_path / "results/v2/s1/final"
    output.mkdir(parents=True)
    (output / "runner.json").write_text("{}")
    with pytest.raises(ValueError, match="missing prepared inputs"):
        runner.run_config(path)


def test_combined_does_not_skip_present_invalid_s1_or_continue_failed_primary(tmp_path, monkeypatch):
    path = config_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner.os, "setpriority", lambda *a: None)
    calls = []
    def fail_s1(args):
        calls.append(args.output)
        if len(calls) == 2:
            raise ValueError("invalid quality binding")
    monkeypatch.setattr(runner, "run_model", fail_s1)
    with pytest.raises(ValueError, match="invalid quality binding"):
        runner.run_config(path)
    assert len(calls) == 2
    calls.clear()
    def fail_primary(args):
        calls.append(args.output)
        raise SystemExit("incomplete primary")
    monkeypatch.setattr(runner, "run_model", fail_primary)
    with pytest.raises(SystemExit, match="incomplete primary"):
        runner.run_config(path)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["wrong_order", "duplicate_output", "narrow_primary", "broad_s1", "unknown_field"])
def test_combined_config_rejects_wrong_scope_or_overlapping_outputs(tmp_path, monkeypatch, fault):
    path = config_fixture(tmp_path, monkeypatch)
    config = runner.load(path)
    if fault == "wrong_order":
        config["models"].reverse()
    elif fault == "duplicate_output":
        argv = config["models"][1]["argv"]
        argv[argv.index("--output") + 1] = str(tmp_path / "results/v2/final")
    elif fault == "narrow_primary":
        config["models"][0]["argv"] += ["--thread-counts", "2,6"]
    elif fault == "broad_s1":
        argv = config["models"][1]["argv"]
        argv[argv.index("--context-lengths") + 1] = "128,1024,4096"
    else:
        config["models"][1]["prepare"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        runner.configured_models(path)


def test_fixed_transfer_uses_shared_validator_and_preserves_binary_report_binding(args, monkeypatch):
    original = runner.load(args.format_selection)
    source = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "pinned"}
    decision = {"schema": "fixed-format-transfer-v1", "split": "fixed_format_transfer",
                "chosen": {**original["chosen"], "label": "g64f16", "group_size": 64},
                "origin_selection": original, "origin_selection_sha256": "synthetic-origin-hash",
                "corpus_sha256": "synthetic-corpus", "verified_source": source,
                "oracle_identity": {"verified_source": source}}
    runner.atomic_json(args.format_selection, decision)
    quality = runner.load(args.quality)
    report_path = args.quality.parent / quality["evidence"][0]["report"]
    report = runner.load(report_path)
    report["settings"]["group_size"] = 64
    report["selection_sha256"] = runner.file_hash(args.format_selection)
    runner.atomic_json(report_path, report)
    entry = quality["evidence"][0]
    entry.update(settings=report["settings"], sha256=runner.file_hash(report_path))
    quality.update(selection=decision, selection_sha256=report["selection_sha256"])
    runner.atomic_json(args.quality, quality)
    validated = []
    monkeypatch.setattr("tools.quality_v2.read_selection",
                        lambda path, corpus_hash: validated.append((path, corpus_hash)) or decision)
    runner.check_quality_selection(args)
    assert validated == [(args.format_selection, "synthetic-corpus")]
    args.engine.write_bytes(b"changed final engine")
    with pytest.raises(ValueError, match="engine binary differs"):
        runner.check_quality_selection(args)
