"""Produce and identify offline per-row or group-scale int8 safetensors.

Existing output is verified against its recorded identity; it is never
silently overwritten.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tools.download_model import file_hash, verify_snapshot
from tools.portable import portable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--manifest", type=Path, default=Path("results/v2/quantized-manifest.json"))
    parser.add_argument("--group-size", type=int, choices=(0, 32, 64, 128), default=0)
    parser.add_argument("--scale-dtype", choices=("f16", "f32"), default="f32")
    args = parser.parse_args()
    source = verify_snapshot(args.source)
    manifest_path = args.output / "quantization.json"
    weights = args.output / "model.safetensors"
    if weights.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("group_size", 0) != args.group_size or manifest.get("scale_dtype", "f32") != args.scale_dtype:
            raise ValueError("Existing artifact uses a different quantization format")
        if manifest["source"]["revision"] != source["revision"] or manifest["weights"]["sha256"] != file_hash(weights):
            raise ValueError("Existing int8 artifact does not match its manifest")
        if manifest["config_sha256"] != file_hash(args.output / "config.json"):
            raise ValueError("Existing int8 configuration differs from its manifest")
    else:
        command = [str(args.engine), "quantize", "--model", str(args.source), "--output", str(args.output), "--group-size", str(args.group_size), "--scale-dtype", args.scale_dtype]
        subprocess.run(command, check=True)
        manifest = {"source": source, "format": "safetensors I8 matrices, group scales, FP32 norms/QKV biases",
                    "quantization": "symmetric round-away-from-zero; scale=max(abs(group))/127, rounded to stored scale dtype before weight rounding; zeros use scale1",
                    "group_size": args.group_size, "scale_dtype": args.scale_dtype,
                    "matrix_bits_per_weight": 8 + (16 if args.scale_dtype == "f16" else 32) / args.group_size if args.group_size else None,
                    "tied_head": True, "activation_dtype": "float32", "command": command,
                    "weights": {"path": str(weights.resolve()), "bytes": weights.stat().st_size, "sha256": file_hash(weights)},
                    "config_sha256": file_hash(args.output / "config.json")}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    locations = {args.source.resolve(): "$MODEL", args.output.resolve(): "$INT8", args.engine.resolve(): "build/cpu-decode"}
    for flag, name in [("--model", "$MODEL"), ("--output", "$INT8")]:
        manifest["command"][manifest["command"].index(flag) + 1] = name
    manifest["command"][0] = "build/cpu-decode"
    manifest["source"] = source
    manifest["group_size"] = args.group_size
    manifest["scale_dtype"] = args.scale_dtype
    manifest["weights"]["path"] = "model.safetensors"
    manifest = portable(manifest, locations)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
