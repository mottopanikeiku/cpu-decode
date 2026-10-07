"""Run and stream a pinned comparison from one ephemeral Modal CPU container.

Usage: modal run tools/cloud_modal.py --output results/v2/cloud
The design is committed before measurement. No GPU, volume, deployment, detached
job, or chosen region is used. Native builds happen inside execute(), not Image.
Completed cells are returned immediately; a timeout never discards them.
"""
from __future__ import annotations

from datetime import datetime
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from zoneinfo import ZoneInfo
import zipfile

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("cpu-decode-cloud-comparison")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential", "cmake", "git", "util-linux", "ca-certificates")
         .pip_install("uv==0.12.5"))
if modal.is_local():
    for filename in ("cloud_run.py", "cloud_summary.py", "cloud_bench.cpp", "cloud_design.json", "cloud_targets.cmake"):
        image = image.add_local_file(ROOT / "tools" / filename, "/assets/tools/" + filename, copy=True)


def cutoff_seconds():
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    return (now.replace(hour=7, minute=50, second=0, microsecond=0) - now).total_seconds()


def pack_outputs(output: Path, raw: dict):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and not path.name.endswith(".stderr.txt"):
                if path.name == "raw.json":
                    archive.writestr(path.name, json.dumps(raw, indent=2, allow_nan=False) + "\n")
                else:
                    archive.write(path, path.name)
    return payload.getvalue()


def annotate(raw: dict, started: float):
    count = len(raw["cells"])
    raw["matrix_status"] = {"complete": count == 6, "completed_cells": count, "planned_cells": 6}
    cost = raw.setdefault("cost", {"requested_resources_hourly_usd": 8 * 0.047160 + 8 * 0.007992})
    cost["function_wall_minutes_before_export"] = (time.time() - started) / 60
    cost["function_resource_estimate_usd"] = cost["function_wall_minutes_before_export"] / 60 * cost["requested_resources_hourly_usd"]
    cost["note"] = "Requested-resource estimate, not an invoice. Preparation, pilot and warmup time are included. The outer cost record also includes failed attempts, image/client time and a 10% margin."
    return raw


@app.function(image=image, cpu=8, memory=8192, timeout=2400, max_containers=1)
def execute():
    started = time.time()
    design = json.loads(Path("/assets/tools/cloud_design.json").read_text())
    if cutoff_seconds() <= 0:
        raise RuntimeError("The cloud comparison cutoff has passed")
    work = Path("/work/cloud")
    source = work / "native"
    output = work / "output"
    source.mkdir(parents=True)
    output.mkdir()
    commands = [(["git", "init", str(source)], None),
                (["git", "remote", "add", "origin", design["native_repository"]], source),
                (["git", "fetch", "--depth", "1", "origin", design["native_commit"]], source),
                (["git", "checkout", "--detach", design["native_commit"]], source)]
    for command, cwd in commands:
        subprocess.run(command, cwd=cwd, check=True, timeout=min(2400, cutoff_seconds()))
    for file in Path("/assets/tools").iterdir():
        shutil.copyfile(file, source / "tools" / file.name)
    # Keep the pinned engine's source list even after later repo changes. Only
    # append the shared optional benchmark target, rather than replacing CMake.
    cmake = source / "CMakeLists.txt"
    cmake.write_text(cmake.read_text() + "\nif(CPU_DECODE_LLAMA_ROOT)\n  include(tools/cloud_targets.cmake)\nendif()\n")
    subprocess.run(["uv", "sync", "--locked"], cwd=source, check=True,
                   timeout=min(2400 - (time.time() - started), cutoff_seconds()))
    runner = subprocess.Popen(
        [str(source / ".venv/bin/python"), "-m", "tools.cloud_run", "--source", str(source),
         "--work", str(work), "--output", str(output), "--design", str(source / "tools/cloud_design.json")],
        cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True)
    try:
        for line in runner.stdout:
            if line.strip() == "CLOUD_CELL_COMPLETE":
                raw = annotate(json.loads((output / "raw.json").read_text()), started)
                yield {"event": "cell", "payload": pack_outputs(output, raw)}
            else:
                print(line, end="", flush=True)
        if runner.wait():
            raise RuntimeError("Cloud runner failed; completed cells have already been returned")
    finally:
        if runner.poll() is None:
            try:
                os.killpg(runner.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                runner.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(runner.pid, signal.SIGKILL)
                runner.wait()
        runner.stdout.close()
    raw_path = output / "raw.json"
    raw = annotate(json.loads(raw_path.read_text()), started)
    raw_path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    subprocess.run([str(source / ".venv/bin/python"), "-m", "tools.cloud_summary", "--input", str(raw_path),
                    "--output", str(output / "summary.json"), "--csv", str(output / "table.csv")],
                   cwd=source, check=True, timeout=min(2400 - (time.time() - started), cutoff_seconds()))
    yield {"event": "complete", "payload": pack_outputs(output, raw)}


@app.local_entrypoint()
def main(output: str = "results/v2/cloud"):
    if cutoff_seconds() <= 0:
        raise RuntimeError("The cloud comparison cutoff has passed")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Refusing to overwrite an existing cloud comparison")
    completed = False
    for message in execute.remote_gen():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(message["payload"])) as archive:
            for member in archive.infolist():
                if Path(member.filename).name != member.filename:
                    raise ValueError("Unexpected nested result path")
                (destination / member.filename).write_bytes(archive.read(member))
        raw = json.loads((destination / "raw.json").read_text())
        print(f"Saved {len(raw['cells'])}/6 completed cloud cells to {destination}")
        completed = message["event"] == "complete"
    if not completed:
        raise RuntimeError("The complete matrix did not return; saved cells remain explicitly partial")
