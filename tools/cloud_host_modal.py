"""Find CPU ISA support with cheap, fresh, serial Modal containers.

Usage: modal run tools/cloud_host_modal.py --gpu none --count 12 \
    --minutes 1 --output results/v2/host-probes.json

Book the same GPU type, one CPU, 512 MiB, one concurrent container and timeout
in the caller's budget wrapper. Each returned snapshot is atomically saved in
an ordinary JSON array before the next probe. Fresh containers need not land
on distinct physical hosts. requested_gpu describes the request, not a measured
GPU model; no GPU computation, model download, package install or build occurs.

Resource and lifecycle APIs:
https://modal.com/docs/reference/modal.App#function
https://modal.com/docs/guide/dynamic-function-config
https://modal.com/docs/guide/resources
https://modal.com/docs/guide/gpu
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import modal


GPU_TYPES = ("none", "T4", "L4", "A100-40GB", "H100")
app = modal.App("cpu-decode-day-host-probes")
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    # Upload the stdlib helper explicitly: a remote import must not depend on
    # the client's tools package or on a separately cloned repository.
    image = image.add_local_file(Path(__file__).with_name("cloud_host.py"),
                                 "/assets/cloud_host.py", copy=False)


@app.function(image=image, cpu=1, memory=512, timeout=60,
              single_use_containers=True, max_containers=1)
def probe(requested_gpu: str, probe_index: int, probe_count: int) -> dict:
    import sys

    sys.path.insert(0, "/assets")
    from cloud_host import snapshot

    record = snapshot()
    record.update(requested_gpu=requested_gpu, probe_index=probe_index,
                  probe_count=probe_count)
    return record


def save_records(destination: Path, records: list[dict]) -> None:
    """Replace the JSON array without risking an earlier completed snapshot."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=destination.parent,
                                         prefix=destination.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(records, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@app.local_entrypoint()
def main(gpu: str = "none", count: int = 1,
         output: str = "results/v2/host-probes.json", minutes: int = 1):
    if gpu not in GPU_TYPES:
        raise ValueError("gpu must be one of: " + ", ".join(GPU_TYPES))
    if not 1 <= count <= 20:
        raise ValueError("count must be between 1 and 20")
    if minutes < 1:
        raise ValueError("minutes must be a positive integer")
    destination = Path(output)
    if destination.exists() and destination.stat().st_size:
        raise ValueError("Refusing to overwrite nonempty probe output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected_probe = probe.with_options(gpu=None if gpu == "none" else gpu,
                                        timeout=minutes * 60)
    records = []
    for index in range(1, count + 1):
        record = selected_probe.remote(gpu, index, count)
        records.append(record)
        save_records(destination, records)
        print(f"Saved probe {index}/{count}: model={record['model_name']!r}, "
              f"missing_flags={record['missing_flags']} to {destination}", flush=True)
