"""Run the complete final matrix sequentially, retaining every interrupted attempt."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import uuid

from tools.download_model import file_hash
from tools.measure_v2 import (CONTEXTS, DEFAULT_TOKENS, ROOT, THREADS, candidates,
                              check_artifacts, check_quality_eligibility, digest, stamp)
from tools.portable import portable
from tools.summarize_v2 import summarize_bandwidth, summarize_candidate

MEMORY_BUDGET = "2000M"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def units() -> list[dict]:
    result = []
    for thread in THREADS:
        for context in CONTEXTS:
            for candidate in candidates([50]):
                name = f"window-t{thread}-c{context}-{candidate['id']}"
                result.append({"name": name, "stage": "window", "threads": thread,
                               "context": context, "candidate": candidate})
        result.append({"name": f"bandwidth-t{thread}", "stage": "bandwidth", "threads": thread})
    return result


def measure_command(args: argparse.Namespace, stage: str, output: Path, unit: dict | None = None) -> list[str]:
    command = [sys.executable, "-m", "tools.measure_v2", stage,
               "--model", str(args.model), "--model-manifest", str(args.model_manifest),
               "--engine", str(args.engine), "--llama", str(args.llama), "--gguf", str(args.gguf),
               "--bandwidth", str(args.bandwidth), "--preparation", str(args.preparation),
               "--quality", str(args.quality), "--output", str(output)]
    if stage == "freeze":
        command += ["--kernel", args.kernel, "--affinity", args.affinity, "--kv", "f16",
                    "--attention", "blocked", "--scheduler", "pool", "--rope", "cached",
                    "--polls", "50", "--steps", "64", "--repeats", "5", "--tokens", args.tokens]
        if args.cpu_order:
            command += ["--cpu-order", args.cpu_order]
    if unit:
        command += ["--threads", str(unit["threads"])]
        if unit["stage"] == "window":
            command += ["--contexts", str(unit["context"]), "--candidate", unit["candidate"]["id"]]
    # The wrapper is acquired for one candidate or bandwidth window, never the whole sweep.
    prefix = shlex.split(args.wrapper) if stage != "freeze" else []
    if stage != "freeze":
        # Start the hard limit after wrapper acquisition, so queue waits do not consume it.
        # Foreground mode keeps descendants in the driver's group for interrupt cleanup.
        prefix += ["timeout", "--foreground", "--signal=TERM", "--kill-after=5s", "1770s"]
    return prefix + ["nice", "-n", "19"] + command


def plan(args: argparse.Namespace) -> dict:
    commands = [{**unit, "command": measure_command(args, unit["stage"],
                 args.output / "attempts" / unit["name"] / "000001", unit)} for unit in units()]
    return {"kind": "INFERENCE", "freeze": measure_command(args, "freeze", args.output / "attempts/freeze/000001"),
            "units": commands, "timing_commands": len(commands), "candidate_windows": 135,
            "bandwidth_windows": 5, "model_processes": 540, "samples_per_engine_per_candidate": 10,
            "measured_tokens_per_engine": 86400, "bandwidth_processes": 10,
            "window_deadline_seconds": 1740, "hard_command_limit_seconds": 1775,
            "memory_budget": MEMORY_BUDGET,
            "sum_window_deadlines_hours": len(commands) * 1740 / 3600,
            "wall_time_estimate": "Unknown; deadlines are limits, not measured runtime; queue waits are unbounded",
            "summary": ["nice", "-n", "19", sys.executable, "-m", "tools.summarize_v2", "--input", str(args.output),
                        "--output", str(args.output / "summary.json")],
            "figure": ["nice", "-n", "19", sys.executable, "-m", "tools.figure_v2", "--input", str(args.output / "summary.json"),
                       "--output", str(args.output / "decode.svg")]}


def validate_selection(selection: dict, manifest: dict) -> None:
    if selection.get("split") != "calibration":
        raise ValueError("format selection must use calibration, not heldout")
    chosen = selection["chosen"]
    for key in ["group_size", "scale_dtype"]:
        if chosen[key] != manifest[key]:
            raise ValueError(f"chosen format differs from manifest: {key}")
    files = chosen["model_identity"]["files"]
    if (files["model.safetensors"]["sha256"] != manifest["weights"]["sha256"] or
            files["config.json"]["sha256"] != manifest["config_sha256"]):
        raise ValueError("chosen format artifact identity differs from manifest")


def specification(args: argparse.Namespace) -> dict:
    validate_selection(load(args.format_selection), load(args.model_manifest))
    files = {name: {"sha256": file_hash(getattr(args, name)),
                   "location_sha256": digest({"path": str(getattr(args, name).resolve())})}
             for name in ["format_selection", "model_manifest", "preparation", "quality"]}
    tools = {name: file_hash(ROOT / "tools" / name) for name in
             ["run_final_v2.py", "measure_v2.py", "summarize_v2.py", "figure_v2.py", "quality_v2.py", "portable.py"]}
    return {"schema": "cpu-decode-v2-runner", "files": files, "tools": tools,
            "locations_sha256": digest({name: str(getattr(args, name).resolve()) for name in
                                        ["model", "engine", "llama", "gguf", "bandwidth", "output"]}),
            "wrapper_sha256": digest({"argv": shlex.split(args.wrapper)}), "memory_budget": MEMORY_BUDGET,
            "kernel": args.kernel, "affinity": args.affinity, "cpu_order": args.cpu_order, "tokens": args.tokens,
            "units": units(), "steps": 64, "repeats": 5, "rounds": 2,
            "chosen_label": load(args.format_selection)["chosen"]["label"]}


def require_same(actual: dict, expected: dict) -> None:
    if actual != expected:
        raise ValueError("runner identity/settings changed; retain this run and use a new output directory")


def locations(args: argparse.Namespace) -> dict:
    result = {Path.home(): "$HOME", args.output: str(args.output.relative_to(ROOT))
              if args.output.is_relative_to(ROOT) else "$OUTPUT", sys.executable: "$PYTHON"}
    for name, alias in [("model", "$INT8"), ("gguf", "$GGUF"), ("llama", "$LLAMA_BENCH"),
                        ("engine", "$ENGINE"), ("bandwidth", "$BANDWIDTH"),
                        ("model_manifest", "$MODEL_MANIFEST"), ("format_selection", "$FORMAT_SELECTION"),
                        ("preparation", "$PREPARATION"), ("quality", "$QUALITY")]:
        result[getattr(args, name)] = alias
    for index, token in enumerate(shlex.split(args.wrapper)):
        if token.startswith("/"):
            result[Path(token)] = f"$WRAPPER_COMPONENT_{index}"
    return result


def process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_orphan(record: dict) -> None:
    pid, started = record.get("process_pid"), record.get("process_start")
    if pid and started and process_start(pid) == started:
        try:
            if os.getpgid(pid) != pid:
                raise ValueError("retained driver is no longer in its own process group")
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def portable_logs(attempt: Path, aliases: dict) -> None:
    for path in attempt.rglob("*.txt"):
        path.write_text(portable(path.read_text(errors="replace"), aliases))


def run_process(command: list[str], attempt: Path, aliases: dict) -> int:
    """Retain driver logs; an interrupt stops the wrapper and every model descendant."""
    process = None
    try:
        with (attempt / "driver.stdout.txt").open("x") as out, (attempt / "driver.stderr.txt").open("x") as err:
            process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                       env={**os.environ, "PP_MEM": MEMORY_BUDGET}, start_new_session=True)
            if (attempt / "attempt.json").exists():
                atomic_json(attempt / "attempt.json", {**load(attempt / "attempt.json"),
                            "process_pid": process.pid, "process_start": process_start(process.pid)})
            return process.wait()
    finally:
        if process is not None:
            # Also remove a descendant left behind by a timed-out driver.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        portable_logs(attempt, aliases)


@contextmanager
def interrupted_as_exception():
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    old = {sig: signal.signal(sig, interrupt) for sig in [signal.SIGTERM, signal.SIGHUP]}
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def new_attempt(output: Path, name: str, command_factory) -> tuple[Path, list[str]]:
    base = output / "attempts" / name
    base.mkdir(parents=True, exist_ok=True)
    number = max([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()] or [0]) + 1
    attempt = base / f"{number:06d}"
    attempt.mkdir()
    command = command_factory(attempt)
    atomic_json(attempt / "attempt.json", {"unit": name, "disposition": "running",
                                         "raw_directory": str(attempt.relative_to(output))})
    return attempt, command


def attempt_record(attempt: Path, output: Path) -> dict:
    checkpoint = attempt / "attempt.json"
    if not checkpoint.exists():
        # No process can start until new_attempt has written this checkpoint.
        atomic_json(checkpoint, {"unit": attempt.parent.name, "disposition": "interrupted",
            "raw_directory": str(attempt.relative_to(output)),
            "error": "interrupted before initial checkpoint; retained directory is excluded from sampling"})
    return load(checkpoint)


def disposition(attempt: Path, status: str, error: str | None = None, **details) -> None:
    atomic_json(attempt / "attempt.json", {**load(attempt / "attempt.json"), "disposition": status,
                                           "error": error, **details})


def verify_protocol(protocol: dict, args: argparse.Namespace) -> None:
    if digest({k: v for k, v in protocol.items() if k != "id"}) != protocol["id"]:
        raise ValueError("protocol changed after freeze")
    expected = {"threads": THREADS, "contexts": CONTEXTS, "candidates": candidates([50]),
                "steps": 64, "repeats": 5, "rounds": 2, "warmup_steps": 1,
                "tokens": args.tokens, "development": False}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"final protocol mismatch: {key}")
    native = protocol["native"]
    for key, value in {"kernel": args.kernel, "affinity": args.affinity, "kv_dtype": "f16",
                       "attention": "blocked", "scheduler": "pool", "rope": "cached"}.items():
        if native.get(key) != value:
            raise ValueError(f"final native setting mismatch: {key}")
    validate_selection(load(args.format_selection), protocol["model_manifest"])


def check_quality_selection(args: argparse.Namespace) -> None:
    quality = load(args.quality)
    if (quality.get("split") != "heldout" or
            quality.get("selection_sha256") != file_hash(args.format_selection) or
            quality.get("selection") != load(args.format_selection)):
        raise ValueError("heldout quality must bind the explicitly supplied format selection")
    chosen = quality["selection"]["chosen"]
    settings = {"kernel": args.kernel, "kv_dtype": "f16", "attention": "blocked",
                "scheduler": "pool", "affinity": args.affinity,
                "group_size": chosen["group_size"], "scale_dtype": chosen["scale_dtype"]}
    matching = [entry for entry in quality.get("evidence", [])
                if entry.get("backend") == "native" and entry.get("model_identity") == chosen["model_identity"]
                and all(entry.get("settings", {}).get(key) == value for key, value in settings.items())]
    if len(matching) != 1:
        raise ValueError("final quality needs exactly one matching chosen-format F16 report")
    entry = matching[0]
    if Path(entry["report"]).name != entry["report"]:
        raise ValueError("quality report must be a sibling filename")
    path = args.quality.parent / entry["report"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError("final heldout report hash changed")
    report = load(path)
    for key in ["label", "split", "backend", "settings", "aggregate", "model_identity", "binary_identity", "oracle_identity"]:
        if report.get(key) != entry.get(key):
            raise ValueError(f"final heldout evidence differs from linked report: {key}")
    if report.get("split") != "heldout" or report.get("selection_sha256") != quality["selection_sha256"]:
        raise ValueError("final heldout report split/selection differs")
    if report["binary_identity"].get("engine_binary_sha256") != file_hash(args.engine):
        raise ValueError("final heldout engine binary differs from timing engine")


def check_current(args: argparse.Namespace, protocol: dict) -> None:
    verify_protocol(protocol, args)
    check_quality_selection(args)
    check_artifacts(args, protocol, locations(args))
    for name, path in [("weights", args.model / "model.safetensors"), ("config", args.model / "config.json"),
                       ("gguf", args.gguf), ("bandwidth", args.bandwidth)]:
        identity = protocol["artifacts"][name]
        if file_hash(path) != identity["sha256"] or stamp(path) != {k: identity[k] for k in ["bytes", "mtime_ns"]}:
            raise ValueError(f"frozen artifact changed: {name}")
    if not set(protocol["cpu_metadata"]["allowed_cpu_ids"]).issubset(os.sched_getaffinity(0)):
        raise ValueError("full frozen CPU set is unavailable")
    check_quality_eligibility(protocol, args.output)


def raw_path(value: str, args: argparse.Namespace) -> Path:
    path = Path(value.replace("$OUTPUT", str(args.output)).replace("$HOME", str(Path.home())))
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_unit(raw: dict, unit: dict, protocol: dict, attempt: Path, args: argparse.Namespace) -> None:
    if raw.get("threads") != unit["threads"]:
        raise ValueError("unit thread mismatch")
    if unit["stage"] == "window":
        if (raw.get("schema") != "cpu-decode-v2-window" or raw.get("context") != unit["context"] or
                raw.get("candidates") != [unit["candidate"]]):
            raise ValueError("unit must contain exactly its one candidate and context")
        summarize_candidate(raw, protocol, unit["candidate"])
    else:
        if raw.get("schema") != "cpu-decode-v2-bandwidth":
            raise ValueError("unit bandwidth schema mismatch")
        summarize_bandwidth(raw, protocol)
    for invocation in raw["invocations"]:
        for key in ["stdout_file", "stderr_file"]:
            path = raw_path(invocation[key], args)
            if not path.is_relative_to(attempt.resolve()) or not path.is_file():
                raise ValueError("unit raw output is missing or outside its attempt")


def assets(attempt: Path) -> dict:
    return {str(path.relative_to(attempt)): file_hash(path) for path in sorted(attempt.rglob("*"))
            if path.is_file() and path.name != "attempt.json"}


def publish(output: Path, attempt: Path, filename: str) -> None:
    source = load(attempt / filename)
    destination = output / filename
    if destination.exists():
        require_same(load(destination), source)
    else:
        atomic_json(destination, source)


def recover_unit(args: argparse.Namespace, unit: dict, protocol: dict) -> bool:
    base = args.output / "attempts" / unit["name"]
    accepted = []
    for attempt in sorted(base.iterdir()) if base.exists() else []:
        record = attempt_record(attempt, args.output)
        if record["disposition"] == "accepted":
            require_same(assets(attempt), record["assets"])
            validate_unit(load(attempt / (unit["name"] + ".json")), unit, protocol, attempt, args)
            accepted.append(attempt)
        elif record["disposition"] in ["running", "completed"]:
            stop_orphan(record)
            portable_logs(attempt, locations(args))
            # No durable successful-validation checkpoint: never combine partial ABAB samples.
            disposition(attempt, "interrupted", "no accepted checkpoint; retained raw is excluded from sampling")
    if len(accepted) > 1:
        raise ValueError("duplicate accepted attempts; do not choose a favorable rerun")
    canonical = args.output / (unit["name"] + ".json")
    if accepted:
        publish(args.output, accepted[0], canonical.name)
        return True
    if canonical.exists():
        raise ValueError("published unit has no accepted attempt; refusing an unlinked sample set")
    return False


def collect(args: argparse.Namespace, unit: dict, protocol: dict) -> bool:
    check_current(args, protocol)
    attempt, command = new_attempt(args.output, unit["name"],
                                   lambda path: measure_command(args, unit["stage"], path, unit))
    atomic_json(attempt / "protocol.json", protocol)
    try:
        returncode = run_process(command, attempt, locations(args))
        disposition(attempt, "completed", returncode=returncode)
        if returncode != 0:
            raise ValueError(f"window driver exit {returncode}; see retained driver and invocation logs")
        raw = load(attempt / (unit["name"] + ".json"))
        validate_unit(raw, unit, protocol, attempt, args)
        check_current(args, protocol)
    except KeyboardInterrupt:
        disposition(attempt, "interrupted", "launch interrupted; partial samples excluded")
        raise
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        disposition(attempt, "failed", portable(str(exc), locations(args)))
        return False
    disposition(attempt, "accepted", assets=assets(attempt))
    publish(args.output, attempt, unit["name"] + ".json")
    return True


def freeze_once(args: argparse.Namespace) -> dict:
    path = args.output / "protocol.json"
    base = args.output / "attempts/freeze"
    accepted = []
    for attempt in sorted(base.iterdir()) if base.exists() else []:
        record = attempt_record(attempt, args.output)
        if record["disposition"] == "accepted":
            require_same(assets(attempt), record["assets"])
            accepted.append(attempt)
        if record["disposition"] in ["running", "completed"]:
            stop_orphan(record)
            portable_logs(attempt, locations(args))
            disposition(attempt, "interrupted", "no accepted freeze checkpoint; retained raw")
    if len(accepted) > 1:
        raise ValueError("duplicate accepted freezes")
    if accepted:
        publish(args.output, accepted[0], "protocol.json")
        protocol = load(path)
        check_current(args, protocol)
        return protocol
    if path.exists():
        raise ValueError("published protocol has no accepted freeze attempt")
    attempt, command = new_attempt(args.output, "freeze", lambda path: measure_command(args, "freeze", path))
    try:
        code = run_process(command, attempt, locations(args))
        disposition(attempt, "completed", returncode=code)
        if code != 0:
            raise ValueError(f"freeze driver exit {code}; see retained driver logs")
        protocol = load(attempt / "protocol.json")
        check_current(args, protocol)
    except BaseException as exc:
        disposition(attempt, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    portable(str(exc), locations(args)))
        raise
    disposition(attempt, "accepted", assets=assets(attempt))
    publish(args.output, attempt, "protocol.json")
    return protocol


def dispositions(output: Path) -> list[dict]:
    return [load(path) for path in sorted((output / "attempts").glob("*/*/attempt.json"))
            if path.parent.parent.name != "summary"]


def finish(args: argparse.Namespace, complete: bool) -> None:
    # Derived outputs are versioned too; a failed launch cannot overwrite a previous summary.
    attempt, _ = new_attempt(args.output, "summary", lambda path: [])
    commands = [["nice", "-n", "19", sys.executable, "-m", "tools.summarize_v2", "--input", str(args.output),
                 "--output", str(attempt / "summary.json")]]
    if not complete:
        commands[0].append("--allow-partial")
    try:
        code = run_process(commands[0], attempt, locations(args))
        disposition(attempt, "completed", returncode=code)
        if code != 0:
            raise ValueError(f"summary exit {code}; retained logs")
        summary = load(attempt / "summary.json")
        if bool(summary["complete_final_matrix"]) != complete:
            raise ValueError("summary completeness disagrees with accepted units")
        summary["runner_attempts"] = dispositions(args.output)
        atomic_json(attempt / "summary.json", summary)
        if complete:
            # Figure generation is light and has no reported timing subprocess.
            from tools.figure_v2 import figure
            (attempt / "decode.svg").write_text(figure(summary))
        disposition(attempt, "accepted", assets=assets(attempt))
        atomic_json(args.output / "summary.json", summary)
        if complete:
            temporary = args.output / ("decode." + uuid.uuid4().hex + ".tmp")
            temporary.write_text((attempt / "decode.svg").read_text())
            os.replace(temporary, args.output / "decode.svg")
    except BaseException as exc:
        disposition(attempt, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    portable(str(exc), locations(args)))
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "model-manifest", "format-selection", "llama", "gguf", "preparation", "quality"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--bandwidth", type=Path, default=Path("build/read-bandwidth"))
    parser.add_argument("--kernel", choices=["scalar", "simd256", "simd512", "simd512x4", "vnni"], required=True)
    parser.add_argument("--affinity", choices=["strict", "unpinned"], required=True)
    parser.add_argument("--cpu-order")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--wrapper", default="", help="Runtime argv prefix (shell quoting accepted; no shell execution); nice 19 is always appended")
    parser.add_argument("--output", type=Path, default=Path("results/v2/final"))
    parser.add_argument("--plan", "--dry-run", action="store_true", dest="plan")
    args = parser.parse_args(argv)
    for key in ["model", "model_manifest", "format_selection", "llama", "gguf", "preparation", "quality", "engine", "bandwidth", "output"]:
        setattr(args, key, getattr(args, key).resolve())
    if not args.output.is_relative_to(ROOT / "results/v2") or args.output == ROOT / "results/v2":
        parser.error("use a dedicated --output directory beneath results/v2")
    return args


def main() -> None:
    args = parse_args()
    if args.plan:
        print(json.dumps(plan(args), indent=2))
        return
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runner.lock").open("a") as lock, interrupted_as_exception():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = specification(args)
        state_path = args.output / "runner.json"
        if state_path.exists():
            require_same(load(state_path), spec)
        else:
            if list(args.output.glob("*.json")) or (args.output / "attempts").exists():
                raise ValueError("output has unowned prior records; choose a new directory")
            atomic_json(state_path, spec)
        protocol = freeze_once(args)
        successful = True
        for unit in units():
            check_current(args, protocol)
            if not recover_unit(args, unit, protocol) and not collect(args, unit, protocol):
                successful = False
        finish(args, successful)
        if not successful:
            raise SystemExit("incomplete matrix; retained attempts and partial summary; rerun the identical command to resume")


if __name__ == "__main__":
    main()
