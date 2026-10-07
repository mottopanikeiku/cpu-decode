"""Run the unchanged paired design only on an actual vnni16-capable host.

Prepare the named model Volume with cloud_assets_modal.py first. A separate
runtime-pilot invocation measures the first complete cell; size the full booking
as ceil(pilot function minutes * 6 * 1.3). GPU containers are optional host-CPU
fallbacks: both builds and all model computation remain CPU-only. Missing CPU
flags return immediately, before source checkout, model access or compilation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import hashlib
import uuid
import zipfile

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU_HOURLY = {"none": 0.0, "T4": 0.5904, "L4": 0.7992, "A100-40GB": 2.0988, "H100": 3.9492}
app = modal.App("cpu-decode-vnni-comparison")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential", "cmake", "git", "util-linux", "ca-certificates")
         .pip_install("uv==0.12.5", "numpy==2.3.3"))
UPLOADS = ("cloud_run.py", "cloud_summary.py", "cloud_bench.cpp", "cloud_host.py",
           "cloud_vnni_design.json", "cloud_targets.cmake", "portable.py")
if modal.is_local():
    for filename in UPLOADS:
        image = image.add_local_file(ROOT / "tools" / filename, "/assets/tools/" + filename, copy=True)


def parse_deadline(value: str) -> float | None:
    if not value:
        return None
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("deadline must include its UTC offset")
    return deadline.timestamp()


def remaining(started: float, minutes: int, deadline: float | None) -> float:
    seconds = minutes * 60 - (time.monotonic() - started)
    if deadline is not None:
        seconds = min(seconds, deadline - time.time())
    if seconds <= 40:
        raise TimeoutError("Not enough booked time remains for work and process cleanup")
    return seconds - 40


def pack_outputs(output: Path, raw: dict) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and not path.name.endswith((".stderr.txt", ".tmp")):
                if path.name == "raw.json":
                    archive.writestr(path.name, json.dumps(raw, indent=2, allow_nan=False) + "\n")
                else:
                    archive.write(path, path.name)
    return payload.getvalue()


def annotate(raw: dict, started: float, gpu: str, startup: dict) -> dict:
    count = len(raw["cells"])
    raw["matrix_status"] = {"complete": count == 6, "completed_cells": count, "planned_cells": 6}
    raw["environment"]["timing_container_startup"] = startup
    hourly = 8 * 0.047160 + 8 * 0.007992 + GPU_HOURLY[gpu]
    cost = raw.setdefault("cost", {})
    cost.update(requested_resources_hourly_usd=hourly,
                function_wall_minutes_before_export=(time.monotonic() - started) / 60)
    cost["function_resource_estimate_usd"] = cost["function_wall_minutes_before_export"] / 60 * hourly
    cost["note"] = "Requested-resource estimate, not an invoice; includes preparation, pilots and warmups. Separate total cost accounting includes probes, CPU staging, runtime pilot, rejected starts and client/image time."
    return raw


def stop_runner(runner):
    if runner.poll() is None:
        try:
            os.killpg(runner.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def enforce_deadline(runner, grace_seconds=35):
    """Escalate independently of a potentially blocked stdout reader."""
    stop_runner(runner)
    try:
        runner.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # A leader can exit while a descendant still holds the group's stdout.
    try:
        os.killpg(runner.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def persist_outputs(output: Path, checkpoint: Path, raw: dict, volume_name: str):
    checkpoint.mkdir(parents=True, exist_ok=True)
    for path in sorted(output.iterdir()):
        if path.is_file() and not path.name.endswith((".stderr.txt", ".tmp")):
            content = (json.dumps(raw, indent=2, allow_nan=False) + "\n").encode() if path.name == "raw.json" else path.read_bytes()
            temporary = checkpoint / (path.name + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(checkpoint / path.name)
    modal.Volume.from_name(volume_name).commit()


def unpack_outputs(destination: Path, payload: bytes):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            if Path(member.filename).name != member.filename:
                raise ValueError("Unexpected nested result path")
            (destination / member.filename).write_bytes(archive.read(member))


def read_checkpoint(checkpoint: Path, runtime_pilot: bool):
    import sys
    sys.path.insert(0, "/assets")
    from tools.cloud_run import read_completed
    design_path = Path("/assets/tools/cloud_vnni_design.json")
    return read_completed(checkpoint / "raw.json", hashlib.sha256(design_path.read_bytes()).hexdigest(),
                          "runtime-pilot" if runtime_pilot else "full-matrix", True)


def finish_checkpoint(checkpoint: Path, volume_name: str, raw: dict):
    from tools.cloud_summary import summarize_file
    summarize_file(checkpoint / "raw.json", checkpoint / "summary.json", checkpoint / "table.csv",
                   allow_partial=len(raw["cells"]) != 6)
    modal.Volume.from_name(volume_name).commit()


@app.function(image=image, cpu=1, memory=512, timeout=180,
              max_containers=1, single_use_containers=True)
def recover(run_name: str, volume_name: str, runtime_pilot: bool):
    checkpoint = Path("/cache/results") / run_name
    raw = read_checkpoint(checkpoint, runtime_pilot)
    if raw is None:
        return None
    finish_checkpoint(checkpoint, volume_name, raw)
    return pack_outputs(checkpoint, raw)


@app.function(image=image, cpu=8, memory=8192, timeout=1800,
              max_containers=1, single_use_containers=True)
def execute(gpu: str, minutes: int, runtime_pilot: bool, deadline: float | None,
            volume_name: str, run_name: str, container_run_id: str):
    import sys

    started = time.monotonic()
    sys.path.insert(0, "/assets/tools")
    from cloud_host import snapshot

    startup = snapshot()
    startup["requested_gpu"] = gpu
    if not startup["supports_vnni16"]:
        yield {"event": "rejected", "host": startup}
        return
    remaining(started, minutes, deadline)
    yield {"event": "accepted", "host": startup}
    design = json.loads(Path("/assets/tools/cloud_vnni_design.json").read_text())
    checkpoint = Path("/cache/results") / run_name
    stored = checkpoint / "raw.json"
    if stored.exists():
        previous = read_checkpoint(checkpoint, runtime_pilot)
        if len(previous["cells"]) == (1 if runtime_pilot else 6):
            finish_checkpoint(checkpoint, volume_name, previous)
            yield {"event": "complete", "payload": pack_outputs(checkpoint, previous)}
            return
    work = Path("/work/cloud")
    source = work / "native"
    output = work / "output"
    source.mkdir(parents=True)
    output.mkdir()
    if stored.exists():
        shutil.copyfile(stored, output / "raw.json")
    for command, cwd in (
        (["git", "init", str(source)], None),
        (["git", "remote", "add", "origin", design["native_repository"]], source),
        (["git", "fetch", "--depth", "1", "origin", design["native_commit"]], source),
        (["git", "checkout", "--detach", design["native_commit"]], source),
    ):
        subprocess.run(command, cwd=cwd, check=True, timeout=remaining(started, minutes, deadline))
    for filename in UPLOADS:
        shutil.copyfile(Path("/assets/tools") / filename, source / "tools" / filename)
    cmake = source / "CMakeLists.txt"
    cmake.write_text(cmake.read_text() + "\nif(CPU_DECODE_LLAMA_ROOT)\n  include(tools/cloud_targets.cmake)\nendif()\n")
    subprocess.run(["uv", "sync", "--locked"], cwd=source, check=True,
                   timeout=remaining(started, minutes, deadline))
    command = [str(source / ".venv/bin/python"), "-m", "tools.cloud_run", "--source", str(source),
               "--work", str(work), "--output", str(output), "--assets", "/cache", "--require-vnni",
               "--design", str(source / "tools/cloud_vnni_design.json"),
               "--budget-minutes", str(minutes), "--requested-gpu", gpu,
               "--resume", "--container-run-id", container_run_id]
    if runtime_pilot:
        command.append("--runtime-pilot")
    alarm_delay = remaining(started, minutes, deadline)
    runner = subprocess.Popen(command, cwd=source, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    alarm = None
    try:
        alarm = threading.Timer(alarm_delay, enforce_deadline, args=(runner,))
        alarm.daemon = True
        alarm.start()
        for line in runner.stdout:
            if line.strip() == "CLOUD_CELL_COMPLETE":
                raw = annotate(json.loads((output / "raw.json").read_text()), started, gpu, startup)
                persist_outputs(output, checkpoint, raw, volume_name)
                yield {"event": "cell", "payload": pack_outputs(output, raw)}
            else:
                print(line, end="", flush=True)
        if runner.wait():
            raise RuntimeError("Comparison stopped; any completed cells have already been returned")
    finally:
        if alarm is not None:
            alarm.cancel()
        stop_runner(runner)
        try:
            runner.wait(timeout=35)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(runner.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            runner.wait()
        runner.stdout.close()
    raw = annotate(json.loads((output / "raw.json").read_text()), started, gpu, startup)
    (output / "raw.json").write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    summary_command = [str(source / ".venv/bin/python"), "-m", "tools.cloud_summary",
                       "--input", str(output / "raw.json"), "--output", str(output / "summary.json"),
                       "--csv", str(output / "table.csv")]
    if runtime_pilot:
        summary_command.append("--allow-partial")
    subprocess.run(summary_command, cwd=source, check=True,
                   timeout=remaining(started, minutes, deadline))
    persist_outputs(output, checkpoint, raw, volume_name)
    yield {"event": "complete", "payload": pack_outputs(output, raw)}


@app.local_entrypoint()
def main(output: str = "results/v2/cloud-vnni/final", gpu: str = "none", minutes: int = 30,
         volume_name: str = "cpu-decode-day-vnni-assets", retries: int = 5,
         runtime_pilot: bool = False, deadline_utc: str = "",
         run_name: str = "", recover_only: bool = False, resume: bool = False):
    if gpu not in GPU_HOURLY or not 1 <= retries <= 5 or not 1 <= minutes <= 240:
        raise ValueError("Choose a supported GPU, 1..5 startup attempts and 1..240 booked minutes")
    if not volume_name.startswith("cpu-decode-day-"):
        raise ValueError("Use a cpu-decode-day-* model Volume")
    deadline = parse_deadline(deadline_utc)
    if deadline is not None and deadline - time.time() <= 40:
        raise TimeoutError("The experiment deadline has passed")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        if not resume:
            raise ValueError("Existing comparison needs an explicit --resume")
        local_raw = destination / "raw.json"
        if local_raw.exists():
            previous = json.loads(local_raw.read_text())
            design_sha = hashlib.sha256((ROOT / "tools/cloud_vnni_design.json").read_bytes()).hexdigest()
            if previous["design_sha256"] != design_sha or previous.get("run_mode") != ("runtime-pilot" if runtime_pilot else "full-matrix"):
                raise ValueError("Refusing to overwrite another comparison")
        elif any(path.name != "startup.json" for path in destination.iterdir()):
            raise ValueError("Refusing to overwrite unrelated files")
    destination.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(volume_name)
    run_name = run_name or ("runtime-pilot" if runtime_pilot else "vnni-final")
    if not run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Use an alphanumeric checkpoint name with optional hyphens or underscores")
    if recover_only:
        payload = recover.with_options(timeout=minutes * 60, volumes={"/cache": volume}).remote(run_name, volume_name, runtime_pilot)
        if payload is None:
            raise RuntimeError("No completed cells have been committed for this run")
        unpack_outputs(destination, payload)
        return
    selected = execute.with_options(gpu=None if gpu == "none" else gpu,
                                    timeout=minutes * 60, volumes={"/cache": volume})
    startup_path = destination / "startup.json"
    startup_attempts = json.loads(startup_path.read_text()) if startup_path.exists() else []
    prior_attempts = len(startup_attempts)
    retries = min(retries, 5 - len(startup_attempts))
    if retries <= 0:
        raise RuntimeError("The five timing startup attempts for this dataset are exhausted")
    for attempt in range(1, retries + 1):
        accepted = False
        complete = False
        for message in selected.remote_gen(gpu, minutes, runtime_pilot, deadline,
                                            volume_name, run_name, uuid.uuid4().hex):
            if message["event"] in ("rejected", "accepted"):
                startup_attempts.append({"attempt": prior_attempts + attempt, "event": message["event"], "host": message["host"]})
                temporary = destination / "startup.json.tmp"
                temporary.write_text(json.dumps(startup_attempts, indent=2, allow_nan=False) + "\n")
                temporary.replace(destination / "startup.json")
                accepted = message["event"] == "accepted"
                print(f"Timing start {prior_attempts + attempt}/5: {message['event']}; missing={message['host']['missing_flags']}", flush=True)
            else:
                unpack_outputs(destination, message["payload"])
                raw = json.loads((destination / "raw.json").read_text())
                print(f"Saved {len(raw['cells'])}/6 completed cells to {destination}", flush=True)
                complete = message["event"] == "complete"
        if accepted:
            if not complete:
                raise RuntimeError("Accepted timing stopped; retained cells are explicitly partial")
            expected = 1 if runtime_pilot else 6
            if len(raw["cells"]) != expected:
                raise RuntimeError("The accepted run returned the wrong completed-cell count")
            return
    raise RuntimeError("No timing container exposed the required ISA within the bounded startup attempts")
