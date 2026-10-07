"""Synthetic ELF/checkpoint/native fixtures; no model, performance or timing claims.

Native subprocess execution and allowed CPU topology are mocked. Corpus checks,
file hashing, output protection and report serialization use the real helpers.
Full-matrix fixtures use sparse zero-filled binaries, not model logits, and never execute the ELF.
"""
from argparse import ArgumentTypeError, Namespace
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest

from tools import check_threads as checker
from tools.corpus_v2 import digest_json, load_manifest, window_alignment
from tools.download_model import file_hash
from tools.portable import ROOT

CORPUS = ROOT / "results/v2/corpus.json"


def synthetic_metadata(requested, inputs, vocab, model):
    kernel = requested["kernel"]
    return {**requested, "fixture": "synthetic metadata, not a real checkpoint run",
        "activation_dtype": "int16" if kernel == "vnni16" else "float32",
        "activation_group_size": 64 if kernel == "vnni16" else 0,
        "model": str(model), "tokens": inputs, "prompt_tokens": inputs,
        "logit_positions": list(range(checker.POSITIONS)), "shape": [checker.POSITIONS, vocab],
        "vocab_size": vocab, "dtype": "float32", "layout": "row-major little-endian",
        "kv_capacity": checker.POSITIONS}


@pytest.fixture
def synthetic_args(tmp_path, monkeypatch):
    engine = tmp_path / "synthetic-engine"
    engine.write_bytes(b"\x7fELFsynthetic fixture; never executed")
    engine.chmod(0o700)
    model = tmp_path / "synthetic-model"
    model.mkdir()
    corpus = load_manifest(CORPUS)
    inputs = window_alignment(next(window for window in corpus["windows"] if window["split"] == "calibration"))[0][:129]
    vocab = max(inputs) + 1
    (model / "config.json").write_text(json.dumps({"vocab_size": vocab, "fixture": "synthetic config"}))
    header = {"__metadata__": {"group_size": "64", "scale_dtype": "f16", "quantization": "synthetic fixture"},
        "synthetic.weight": {"shape": [2, 64], "dtype": "I8", "data_offsets": [0, 128]}}
    encoded = json.dumps(header).encode()
    (model / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(128))
    allowed = set(range(12))
    monkeypatch.setattr(checker.os, "sched_getaffinity", lambda pid: allowed)
    # Non-sorted synthetic order detects accidental sorting or topology reselection.
    cpus = list(reversed(sorted(allowed)))
    return Namespace(engine=engine, model=model, corpus=CORPUS, raw_dir=tmp_path / "raw",
                     output=tmp_path / "report.json", cpu_order=cpus)


def install_synthetic_native(monkeypatch, args, mutate=None, failures=None):
    state = {"calls": [], "active": False, "closed": []}
    vocab = json.loads((args.model / "config.json").read_text())["vocab_size"]

    def run(command, *, cwd, stdout, stderr, check):
        assert not state["active"], "Only one native process may be active"
        assert cwd == ROOT and check is False
        # Every previous blocking run has returned and its output streams closed.
        assert all(out.closed and err.closed for out, err in state["closed"])
        state["active"] = True
        state["calls"].append(command)
        options = dict(zip(command[2::2], command[3::2], strict=True))
        assert command[0] == str(args.engine.resolve()) and command[1] == "logits"
        assert options["--logits-start"] == "0"
        prefix = Path(options["--output"])
        thread = int(options["--threads"])
        kernel, kv = options["--kernel"], options["--kv"]
        requested = {"group_size": 64, "scale_dtype": "f16", "weight_dtype": "int8",
            "kernel": kernel, "kv_dtype": kv, "threads": thread,
            "cpu_set": [int(item) for item in options["--cpu-set"].split(",")],
            "attention": options["--attention"], "scheduler": options["--scheduler"],
            "affinity": options["--affinity"], "rope": options["--rope"]}
        assert requested["cpu_set"] == args.cpu_order[:thread]
        stdout.write("Synthetic native stdout only\n")
        stderr.write("Synthetic native stderr only\n")
        exit_code = (failures or {}).get((kernel, kv, thread), 0)
        if exit_code == 0:
            inputs = [int(item) for item in options["--tokens"].split(",")]
            assert len(inputs) == checker.POSITIONS
            actual = synthetic_metadata(requested, inputs, vocab, args.model.resolve())
            binary = prefix.with_suffix(".bin")
            # Sparse fixture: exact byte count, streamed hashes, no large allocation.
            with binary.open("xb") as stream:
                stream.write(f"{kernel}:{kv}".encode())
                stream.truncate(checker.POSITIONS * vocab * 4)
            if mutate:
                mutate(actual, binary, kernel, kv, thread)
            with prefix.with_suffix(".json").open("x") as stream:
                json.dump(actual, stream)
        state["active"] = False
        state["closed"].append((stdout, stderr))
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr(checker.subprocess, "run", run)
    return state


def cli_args(args):
    return ["--engine", str(args.engine), "--model", str(args.model), "--raw-dir", str(args.raw_dir),
            "--output", str(args.output), "--cpu-order", ",".join(map(str, args.cpu_order))]


def test_synthetic_full_matrix_sequential_cpu_prefixes_and_portable_report(monkeypatch, synthetic_args, capsys):
    args = synthetic_args
    state = install_synthetic_native(monkeypatch, args)
    assert checker.main(cli_args(args)) == 0
    report = json.loads(args.output.read_text())
    assert report["schema"] == "checkpoint-thread-invariance-v1"
    assert report["status"] == "pass" and report["bitwise_invariant"] is True
    assert report["processes_completed"] == report["matrix"]["expected_processes"] == len(state["calls"]) == 20
    assert len(report["pairs"]) == 4 and all(pair["bitwise_invariant"] for pair in report["pairs"])
    assert len({pair["baseline_sha256"] for pair in report["pairs"]}) == 4  # Not a cross-kernel/KV comparison.
    window = next(window for window in load_manifest(CORPUS)["windows"] if window["split"] == "calibration")
    assert report["workload"]["input_tokens"] == window_alignment(window)[0][:129]
    assert report["workload"]["input_tokens_sha256"] == digest_json(report["workload"]["input_tokens"])
    assert report["workload"]["logit_positions"] == list(range(129))
    assert report["identity"]["engine"]["sha256"] == file_hash(args.engine)
    assert report["identity"]["corpus"]["sha256"] == file_hash(CORPUS)
    assert report["identity"]["model"]["files"]["config.json"]["sha256"] == file_hash(args.model / "config.json")
    assert report["tool_identity"]["files"]["tools/check_threads.py"] == file_hash(ROOT / "tools/check_threads.py")
    assert report["source_identity"]["sha256"] == digest_json(report["source_identity"]["files"])
    assert str(args.output.parent) not in args.output.read_text()
    assert "tokens_per_second" not in args.output.read_text() and "elapsed" not in args.output.read_text()
    expected_cells = [(kernel, kv, thread) for kernel in checker.KERNELS for kv in checker.KV_DTYPES for thread in checker.THREADS]
    assert [(run["actual_settings"]["kernel"], run["actual_settings"]["kv_dtype"], run["actual_settings"]["threads"])
            for run in report["runs"]] == expected_cells
    for run in report["runs"]:
        assert run["validated"] and run["bitwise_equal_t1"] is True and run["errors"] == []
        assert run["actual_settings"]["cpu_set"] == args.cpu_order[:run["requested"]["threads"]]
        assert run["number_logits"] == 129 and run["byte_count"] == 129 * run["vocab"] * 4
        assert run["command"][0] == "$ENGINE" and run["actual_metadata"]["model"] == "$MODEL"
        for suffix, record in run["artifacts"].items():
            path = args.raw_dir / (run["id"] + suffix)
            assert record == {"path": "$RAW/" + path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}
    assert all(out.closed and err.closed for out, err in state["closed"])
    assert json.loads(capsys.readouterr().out)["processes_completed"] == 20
    previous_report = args.output.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        checker.check_threads(args)
    assert args.output.read_bytes() == previous_report and len(state["calls"]) == 20


def test_synthetic_whole_bin_mismatch_is_explicit_failure_and_retains_all_raw(monkeypatch, synthetic_args):
    args = synthetic_args

    def mismatch(actual, binary, kernel, kv, thread):
        if (kernel, kv, thread) == ("vnni16", "f32", 6):
            with binary.open("r+b") as stream:
                stream.seek(binary.stat().st_size - 1)
                stream.write(b"\x01")  # Last byte detects a whole-bin, not sampled, comparison.

    state = install_synthetic_native(monkeypatch, args, mutate=mismatch)
    assert checker.main(cli_args(args)) == 1
    report = json.loads(args.output.read_text())
    assert report["status"] == "failure" and report["bitwise_invariant"] is False
    assert report["processes_completed"] == len(state["calls"]) == 20
    mismatches = [run for run in report["runs"] if run["bitwise_equal_t1"] is False]
    assert len(mismatches) == 1 and mismatches[0]["id"] == "vnni16-f32-t6"
    assert mismatches[0]["validated"] and "Whole-bin SHA256" in mismatches[0]["errors"][0]
    assert all(record is not None for run in report["runs"] for record in run["artifacts"].values())
    assert [pair["bitwise_invariant"] for pair in report["pairs"]] == [True, True, True, False]


@pytest.fixture
def small_case(tmp_path):
    requested = {"group_size": 64, "scale_dtype": "f16", "weight_dtype": "int8", "kernel": "vnni16",
        "kv_dtype": "f16", "threads": 2, "cpu_set": [4, 1], "affinity": "strict", "scheduler": "pool",
        "attention": "blocked", "rope": "cached"}
    inputs, vocab = [3] * 129, 16
    model = tmp_path / "synthetic-model"
    actual = synthetic_metadata(requested, inputs, vocab, model)
    binary = tmp_path / "synthetic.bin"
    binary.write_bytes(bytes(129 * vocab * 4))
    return actual, requested, inputs, vocab, model, binary


def test_synthetic_validation_accepts_matching_metadata(small_case):
    actual, requested, inputs, vocab, model, binary = small_case
    assert checker.validate_actual(actual, requested, inputs, vocab, model, binary) == {
        **requested, "activation_dtype": "int16", "activation_group_size": 64}


@pytest.mark.parametrize("key,value,match", [
    ("shape", [128, 16], "shape"), ("shape", [129, 17], "shape"),
    ("vocab_size", 17, "vocabulary"), ("dtype", "float16", "dtype"),
    ("layout", "row-major big-endian", "layout"), ("kernel", "simd512x4", "kernel"),
    ("kv_dtype", "f32", "kv_dtype"), ("threads", 1, "threads"),
    ("cpu_set", [1, 4], "cpu_set"), ("cpu_set", [4], "cpu_set"),
    ("affinity", "unpinned", "affinity"), ("scheduler", "openmp", "scheduler"),
    ("attention", "scalar", "attention"), ("rope", "direct", "rope"),
    ("weight_dtype", "bf16", "weight_dtype"), ("group_size", 32, "group_size"),
    ("scale_dtype", "f32", "scale_dtype"), ("activation_dtype", "int8", "activation"),
    ("activation_group_size", 32, "activation"), ("kv_capacity", 130, "capacity"),
    ("model", "different-model", "model path"), ("tokens", [3] * 128, "input tokens"),
    ("prompt_tokens", [2] * 129, "input tokens"), ("logit_positions", list(range(1, 130)), "positions"),
])
def test_synthetic_metadata_mutations_rejected(small_case, key, value, match):
    actual, requested, inputs, vocab, model, binary = small_case
    actual[key] = value
    with pytest.raises(ValueError, match=match):
        checker.validate_actual(actual, requested, inputs, vocab, model, binary)


@pytest.mark.parametrize("delta", [-4, 4])
def test_synthetic_binary_byte_count_rejected(small_case, delta):
    actual, requested, inputs, vocab, model, binary = small_case
    with binary.open("r+b") as stream:
        stream.truncate(129 * vocab * 4 + delta)
    with pytest.raises(ValueError, match="byte count"):
        checker.validate_actual(actual, requested, inputs, vocab, model, binary)


def test_synthetic_simd_activation_metadata_rejected(small_case):
    actual, requested, inputs, vocab, model, binary = small_case
    requested["kernel"] = actual["kernel"] = "simd512x4"
    with pytest.raises(ValueError, match="activation"):
        checker.validate_actual(actual, requested, inputs, vocab, model, binary)


def test_synthetic_nonzero_native_exits_retained_without_false_baselines(monkeypatch, synthetic_args):
    args = synthetic_args
    failures = {(kernel, kv, thread): 7 for kernel in checker.KERNELS for kv in checker.KV_DTYPES for thread in checker.THREADS}
    state = install_synthetic_native(monkeypatch, args, failures=failures)
    report = checker.check_threads(args)
    assert report["status"] == "failure" and not report["bitwise_invariant"]
    assert len(state["calls"]) == report["processes_completed"] == 20
    assert all(pair["baseline_sha256"] is None and not pair["bitwise_invariant"] for pair in report["pairs"])
    for run in report["runs"]:
        assert run["returncode"] == 7 and not run["validated"] and run["bitwise_equal_t1"] is None
        assert run["errors"] == ["Native subprocess exited 7"]
        assert run["artifacts"][".stdout.txt"] and run["artifacts"][".stderr.txt"]
        assert run["artifacts"][".bin"] is None and run["artifacts"][".json"] is None

def test_synthetic_valid_run_without_valid_t1_never_becomes_a_baseline(monkeypatch, synthetic_args):
    args = synthetic_args
    failures = {(kernel, kv, thread): 7 for kernel in checker.KERNELS for kv in checker.KV_DTYPES for thread in checker.THREADS}
    failures[("simd512x4", "f16", 2)] = 0
    install_synthetic_native(monkeypatch, args, failures=failures)
    report = checker.check_threads(args)
    candidate = next(run for run in report["runs"] if run["id"] == "simd512x4-f16-t2")
    assert candidate["validated"] and candidate["baseline_sha256"] is None
    assert candidate["bitwise_equal_t1"] is None and "baseline is unavailable" in candidate["errors"][0]
    assert report["status"] == "failure" and report["bitwise_invariant"] is False


@pytest.mark.parametrize("kind", ["malformed-json", "missing-json", "missing-bin", "metadata"])
def test_synthetic_native_output_errors_written_as_failure(monkeypatch, synthetic_args, kind):
    args = synthetic_args
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        options = dict(zip(command[2::2], command[3::2], strict=True))
        prefix = Path(options["--output"])
        if kind == "malformed-json":
            prefix.with_suffix(".json").write_text("not JSON")
        elif kind in ("missing-bin", "metadata"):
            requested = {"group_size": 64, "scale_dtype": "f16", "weight_dtype": "int8",
                "kernel": options["--kernel"], "kv_dtype": options["--kv"], "threads": int(options["--threads"]),
                "cpu_set": [int(item) for item in options["--cpu-set"].split(",")],
                "attention": "blocked", "scheduler": "pool", "affinity": "strict", "rope": "cached"}
            inputs = [int(item) for item in options["--tokens"].split(",")]
            vocab = json.loads((args.model / "config.json").read_text())["vocab_size"]
            actual = synthetic_metadata(requested, inputs, vocab, args.model.resolve())
            if kind == "metadata":
                actual["kernel"] = "synthetic-wrong-kernel"
            prefix.with_suffix(".json").write_text(json.dumps(actual))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(checker.subprocess, "run", run)
    report = checker.check_threads(args)
    assert report["status"] == "failure" and report["processes_completed"] == len(calls) == 20
    assert all(not run["validated"] and run["errors"] and run["bitwise_equal_t1"] is None for run in report["runs"])
    assert json.loads(args.output.read_text()) == report
    if kind == "missing-bin":
        assert all("No such file" in run["errors"][0] for run in report["runs"])
    if kind == "metadata":
        assert all("kernel differs" in run["errors"][0] for run in report["runs"])


@pytest.mark.parametrize("case", ["existing-report", "existing-raw", "raw-file", "report-symlink", "raw-symlink",
    "v1-report", "v1-raw", "artifact-conflict", "report-ancestor", "not-elf", "not-executable", "missing-engine",
    "cpu-unavailable", "cpu-limited", "cpu-short", "cpu-duplicate", "cpu-negative", "format", "vocabulary", "corpus"])
def test_synthetic_path_and_input_preflight_starts_no_native(monkeypatch, synthetic_args, tmp_path, case):
    args = synthetic_args
    original_report = None
    if case == "existing-report":
        args.output.write_text("preserved prior report")
        original_report = args.output.read_bytes()
    elif case == "existing-raw":
        args.raw_dir.mkdir()
        (args.raw_dir / "prior.bin").write_bytes(b"preserved raw")
    elif case == "raw-file":
        args.raw_dir.write_text("preserved raw path")
    elif case in ("report-symlink", "raw-symlink"):
        path = args.output if case == "report-symlink" else args.raw_dir
        path.symlink_to(tmp_path / "nonexistent-target")
    elif case == "v1-report":
        args.output = ROOT / "results/never-created-thread-check.json"
    elif case == "v1-raw":
        args.raw_dir = ROOT / "results/never-created-thread-raw"
    elif case == "artifact-conflict":
        args.output = args.raw_dir / "simd512x4-f16-t1.bin"
    elif case == "report-ancestor":
        args.raw_dir = args.output / "raw"
    elif case == "not-elf":
        args.engine.write_bytes(b"synthetic non-ELF")
    elif case == "not-executable":
        args.engine.chmod(0o600)
    elif case == "missing-engine":
        args.engine = tmp_path / "absent-engine"
    elif case == "cpu-unavailable":
        args.cpu_order[0] = max(os.sched_getaffinity(0)) + 1000
    elif case == "cpu-limited":
        monkeypatch.setattr(checker.os, "sched_getaffinity", lambda pid: set(range(4)))
    elif case == "cpu-short":
        args.cpu_order = args.cpu_order[:6]
    elif case == "cpu-duplicate":
        args.cpu_order[0] = args.cpu_order[1]
    elif case == "cpu-negative":
        args.cpu_order[0] = -1
    elif case == "format":
        weight = args.model / "model.safetensors"
        content = weight.read_bytes().replace(b'"64"', b'"32"')
        weight.write_bytes(content)
    elif case == "vocabulary":
        (args.model / "config.json").write_text(json.dumps({"vocab_size": 1}))
    elif case == "corpus":
        args.corpus = tmp_path / "modified-corpus.json"
        args.corpus.write_text("{}")

    def unexpected_run(*args, **kwargs):
        pytest.fail("Native subprocess started before preflight completed")

    monkeypatch.setattr(checker.subprocess, "run", unexpected_run)
    with pytest.raises((ValueError, OSError)):
        checker.check_threads(args)
    if original_report is not None:
        assert args.output.read_bytes() == original_report
    if case == "existing-raw":
        assert (args.raw_dir / "prior.bin").read_bytes() == b"preserved raw"
    if case == "raw-file":
        assert args.raw_dir.read_text() == "preserved raw path"
    if case not in ("existing-raw", "raw-file", "raw-symlink"):
        assert not args.raw_dir.exists()


@pytest.mark.parametrize("value", ["", "0,1", "0,1,2,3,4,5,6,7,8,9,10,10", "-1,1,2,3,4,5,6,7,8,9,10,11",
    "0,1,2,3,4,5,6,7,8,9,10, 11", "0,1,2,3,4,5,6,7,8,9,10,x"])
def test_cpu_order_parser_rejects_invalid_input(value):
    with pytest.raises(ArgumentTypeError):
        checker.parse_cpu_order(value)


def test_cpu_order_parser_preserves_explicit_order():
    assert checker.parse_cpu_order("0,1,4,2,3,5,6,7,10,8,9,11") == [0, 1, 4, 2, 3, 5, 6, 7, 10, 8, 9, 11]


def test_cli_requires_explicit_inputs():
    with pytest.raises(SystemExit) as exc:
        checker.main([])
    assert exc.value.code == 2
