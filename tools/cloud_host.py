"""Inspect the exposed Linux CPU without importing a model or native library.

The flags are the intersection across /proc/cpuinfo processor records, not a
union. Missing CPU information never counts as support. The exposed logical
CPU count and allowed affinity are observations, not a container CPU quota.
Timing startup can reuse snapshot() before attempting the vnni16 kernel.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path


# Linux spells GCC's "avx512vnni" CPU feature as "avx512_vnni".
# Keep this in sync with parse_kernel("vnni16") in src/kernels.cpp.
REQUIRED_FLAGS = ("avx512f", "avx512_vnni", "avx512bw", "avx2", "f16c")


def parse_cpuinfo(text: str) -> dict:
    """Return JSON-compatible CPU fields from Linux /proc/cpuinfo text.

    Unknown or mixed model names are represented by model_name=None. Observed
    names remain in model_names. A processor record without flags contributes
    an empty set, so another processor cannot mask its missing ISA information.
    """
    processors = []
    fields = {}
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            if any(key in fields for key in ("processor", "flags", "model name")):
                processors.append(fields)
            fields = {}
        elif ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()

    flags = set(processors[0].get("flags", "").split()) if processors else set()
    for processor in processors[1:]:
        flags.intersection_update(processor.get("flags", "").split())
    model_names = sorted({processor["model name"] for processor in processors
                          if processor.get("model name")})
    model_name = (model_names[0] if len(model_names) == 1
                  and all(processor.get("model name") for processor in processors)
                  else None)
    missing_flags = [flag for flag in REQUIRED_FLAGS if flag not in flags]
    return {
        "model_name": model_name,
        "model_names": model_names,
        "flags": sorted(flags),
        "missing_flags": missing_flags,
        "supports_vnni16": not missing_flags,
        "parsed_cpu_count": len(processors),
    }


def snapshot() -> dict:
    """Observe this container's CPU information using only the standard library."""
    cpuinfo_error = None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError as error:
        cpuinfo = ""
        cpuinfo_error = str(error)
    record = parse_cpuinfo(cpuinfo)
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    record["exposed_cpu_count"] = os.cpu_count()
    try:
        record["allowed_affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        record["allowed_affinity"] = None
        record["affinity_error"] = str(error)
    if cpuinfo_error is not None:
        record["cpuinfo_error"] = cpuinfo_error
    return record
