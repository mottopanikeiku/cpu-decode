"""Produce and identify the engine's offline per-row int8 weight file.

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
    parser.add_argument("--manifest", type=Path, default=Path("results/quantized-manifest.json"))
    args = parser.parse_args()
    source = verify_snapshot(args.source)
    manifest_path = args.output / "quantization.json"
    weights = args.output / "model.safetensors"
    if weights.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source"]["revision"] != source["revision"] or manifest["weights"]["sha256"] != file_hash(weights):
            raise ValueError("Existing int8 artifact does not match its manifest")
        if manifest["config_sha256"] != file_hash(args.output / "config.json"):
            raise ValueError("Existing int8 configuration differs from its manifest")
    else:
        command = [str(args.engine), "quantize", "--model", str(args.source), "--output", str(args.output)]
        subprocess.run(command, check=True)
        manifest = {"source": source, "format": "safetensors I8 matrices, FP32 output-row scales/norms/QKV biases",
                    "quantization": "symmetric round-away-from-zero; scale=max(abs(row))/127, zeros use scale1",
                    "tied_head": True, "activation_dtype": "float32", "command": command,
                    "weights": {"path": str(weights.resolve()), "bytes": weights.stat().st_size, "sha256": file_hash(weights)},
                    "config_sha256": file_hash(args.output / "config.json")}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    locations = {args.source.resolve(): "$MODEL", args.output.resolve(): "$INT8", args.engine.resolve(): "build/cpu-decode"}
    for flag, name in [("--model", "$MODEL"), ("--output", "$INT8")]:
        manifest["command"][manifest["command"].index(flag) + 1] = name
    manifest["command"][0] = "build/cpu-decode"
    manifest["source"] = source
    manifest["weights"]["path"] = "model.safetensors"
    manifest = portable(manifest, locations)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
