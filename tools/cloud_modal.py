"""Run the pinned cloud CPU comparison in one ephemeral Modal container.

Usage: modal run tools/cloud_modal.py --output results/v2/cloud
The design is committed before measurement. No GPU, volume, deployment, detached
job, or chosen region is used. Native builds happen inside execute(), not Image.
"""
from __future__ import annotations

from datetime import datetime
import io
import json
from pathlib import Path
import shutil
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
    image = image.add_local_file(ROOT / "CMakeLists.txt", "/assets/CMakeLists.txt", copy=True)
    for filename in ("cloud_run.py", "cloud_summary.py", "cloud_bench.cpp", "cloud_design.json"):
        image = image.add_local_file(ROOT / "tools" / filename, "/assets/tools/" + filename, copy=True)

def cutoff_seconds():
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    return (now.replace(hour=7, minute=45, second=0, microsecond=0) - now).total_seconds()


@app.function(image=image, cpu=8, memory=8192, timeout=4200, max_containers=1)
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
        subprocess.run(command, cwd=cwd, check=True, timeout=min(4200, cutoff_seconds()))
    shutil.copyfile("/assets/CMakeLists.txt", source / "CMakeLists.txt")
    for file in Path("/assets/tools").iterdir():
        shutil.copyfile(file, source / "tools" / file.name)
    subprocess.run(["uv", "sync", "--locked"], cwd=source, check=True,
                   timeout=min(4200 - (time.time() - started), cutoff_seconds()))
    subprocess.run([str(source / ".venv/bin/python"), "-m", "tools.cloud_run", "--source", str(source),
                    "--work", str(work), "--output", str(output), "--design", str(source / "tools/cloud_design.json")],
                   cwd=source, check=True, timeout=min(4200 - (time.time() - started), cutoff_seconds()))
    raw_path = output / "raw.json"
    raw = json.loads(raw_path.read_text())
    cost = raw["cost"]
    cost["function_wall_minutes_before_summary"] = (time.time() - started) / 60
    cost["function_resource_estimate_usd"] = cost["function_wall_minutes_before_summary"] / 60 * cost["requested_resources_hourly_usd"]
    raw_path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    subprocess.run([str(source / ".venv/bin/python"), "-m", "tools.cloud_summary", "--input", str(raw_path),
                    "--output", str(output / "summary.json"), "--csv", str(output / "table.csv")],
                   cwd=source, check=True, timeout=min(4200 - (time.time() - started), cutoff_seconds()))
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and not path.name.endswith(".stderr.txt"):
                archive.write(path, path.name)
    return payload.getvalue()


@app.local_entrypoint()
def main(output: str = "results/v2/cloud"):
    if cutoff_seconds() <= 0:
        raise RuntimeError("The cloud comparison cutoff has passed")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Refusing to overwrite an existing cloud comparison")
    payload = execute.remote()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            if Path(member.filename).name != member.filename:
                raise ValueError("Unexpected nested result path")
            (destination / member.filename).write_bytes(archive.read(member))
    print(f"Saved cloud comparison to {destination}")
