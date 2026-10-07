"""Stage the pinned comparison's model artifacts in one ephemeral CPU container.

Usage (book 2 CPUs, 8 GiB and the same --minutes before running):
    modal run tools/cloud_assets_modal.py --minutes 10 \
        --volume-name cpu-decode-day-vnni-assets --output results/v2/day-assets.json

The named volume must be empty. It is mounted at /cache; the original
cloud_run.prepare(source=/work/native, work=/cache, output=/cache/manifests)
creates hf/ and artifacts/ and preserves all existing checksum checks. This is
preparation only: no comparison, pilot, sample, or generation is run.

Timing consumers mount the returned volume and read manifests/ offline. They
must compile both engines and the driver fresh on their actual timing host,
copying only upstream source (not artifacts/llama.cpp/<commit>/build). Staged
binaries are CPU preparation tools, never timing binaries. Returned paths are
relative to the volume root; manifest paths retain their existing conventions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ("cloud_run.py", "cloud_bench.cpp", "cloud_targets.cmake", "cloud_design.json")
MANIFESTS = ("source-model.json", "native-preparation.json", "llama-preparation.json")
app = modal.App("cpu-decode-day-assets")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential", "cmake", "git", "util-linux", "ca-certificates")
         .pip_install("uv==0.12.5"))
if modal.is_local():
    for filename in UPLOADS:
        image = image.add_local_file(ROOT / "tools" / filename, "/assets/tools/" + filename, copy=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(command, deadline, cwd=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Asset staging exhausted its booked time, including setup")
    print("+", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        result = process.wait(timeout=remaining)
        if result:
            raise subprocess.CalledProcessError(result, command)
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


@app.function(image=image, cpu=2, memory=8192, timeout=600,
              max_containers=1, single_use_containers=True)
def stage(volume_name: str, minutes: int):
    started = time.monotonic()
    started_utc = utc_now()
    deadline = started + minutes * 60
    cache = Path("/cache")
    if any(cache.iterdir()):
        raise ValueError("Asset staging requires a new or empty volume; existing contents are not overwritten")
    source = Path("/work/native")
    source.mkdir(parents=True)
    output = cache / "manifests"
    output.mkdir()
    design_path = Path("/assets/tools/cloud_design.json")
    design = json.loads(design_path.read_text())
    for command, cwd in (
        (["git", "init", str(source)], None),
        (["git", "remote", "add", "origin", design["native_repository"]], source),
        (["git", "fetch", "--depth", "1", "origin", design["native_commit"]], source),
        (["git", "checkout", "--detach", design["native_commit"]], source),
    ):
        run(command, deadline, cwd)
    # Preserve the pinned root source list and add only the optional driver.
    for filename in UPLOADS:
        shutil.copyfile(Path("/assets/tools") / filename, source / "tools" / filename)
    cmake = source / "CMakeLists.txt"
    cmake.write_text(cmake.read_text() + "\nif(CPU_DECODE_LLAMA_ROOT)\n  include(tools/cloud_targets.cmake)\nendif()\n")
    os.environ.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2"})
    run(["uv", "sync", "--locked"], deadline, source)
    # Use the pinned environment, rather than installing model dependencies in
    # the image. The default prepare path retains fresh-source/cache semantics.
    preparation = (
        "import json; from pathlib import Path; from tools.cloud_run import prepare; "
        "source = Path('/work/native'); "
        "design = json.loads((source / 'tools/cloud_design.json').read_text()); "
        "prepare(source, Path('/cache'), Path('/cache/manifests'), design)"
    )
    run([str(source / ".venv/bin/python"), "-c", preparation], deadline, source)
    manifests = {name: json.loads((output / name).read_text()) for name in MANIFESTS}
    native = manifests["native-preparation.json"]
    llama = manifests["llama-preparation.json"]
    expected_llama = json.loads((source / "results/llama-preparation.json").read_text())
    llama_prefix = f"artifacts/llama.cpp/{design['llama_commit']}/qwen-{design['model_revision']}"
    artifacts = {
        "source_weights": {
            "path": f"hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{design['model_revision']}/model.safetensors",
            "sha256": manifests["source-model.json"]["files"]["model.safetensors"]["sha256"],
            "expected_sha256": expected_llama["source_model"]["files"]["model.safetensors"]["sha256"],
        },
        "native_weights": {
            "path": "artifacts/g64f16/model.safetensors",
            "sha256": native["weights"]["sha256"],
            "expected_sha256": design["native_weights_sha256"],
        },
        "native_config": {"path": "artifacts/g64f16/config.json", "sha256": native["config_sha256"]},
        **{
            name: {"path": f"{llama_prefix}-{suffix}.gguf",
                   "sha256": llama["artifacts"][name]["sha256"],
                   "expected_sha256": expected_llama["artifacts"][name]["sha256"]}
            for name, suffix in (("BF16", "bf16"), ("Q8_0", "q8_0"))
        },
    }
    volume = modal.Volume.from_name(volume_name)
    volume.commit()
    return {
        "stage": "artifact_preparation_only",
        "volume": volume_name,
        "started_utc": started_utc,
        "committed_utc": utc_now(),
        "function_wall_seconds_including_setup_and_commit": time.monotonic() - started,
        "requested_resources": {"gpu": "none", "cpu_cores": 2, "memory_gib": 8,
                                "max_containers": 1, "single_use_containers": True,
                                "timeout_seconds": minutes * 60},
        "pins": {key: design[key] for key in ("native_commit", "llama_commit", "model_id", "model_revision")},
        "design_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
        "manifests": {name: "manifests/" + name for name in MANIFESTS},
        "artifacts": artifacts,
        "timing_binary_policy": "Compile both engines and the driver fresh on the actual timing host; do not copy or run staged builds.",
    }


@app.local_entrypoint()
def main(volume_name: str = "cpu-decode-day-vnni-assets",
         output: str = "results/v2/day-assets.json", minutes: int = 10):
    if not volume_name.startswith("cpu-decode-day-"):
        raise ValueError("Volume names must start with cpu-decode-day-")
    if minutes < 1:
        raise ValueError("Book a positive integer number of minutes")
    destination = Path(output)
    if destination.exists():
        raise ValueError("Refusing to overwrite an existing staging output")
    # local_entrypoint runs inside the ephemeral `modal run` app. Do not create
    # or resolve a named volume at import time or outside this invocation.
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    result = stage.with_options(timeout=minutes * 60, max_containers=1,
                                volumes={"/cache": volume}).remote(volume_name, minutes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved CPU artifact staging manifest to {destination}")
