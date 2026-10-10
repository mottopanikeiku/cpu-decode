"""Produce and identify one of the engine's offline block-quantized weight files.

Existing output is verified against its recorded identity and formats; it is
never silently overwritten.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

from tools.download_model import file_hash, verify_snapshot
from tools.portable import portable

FORMATS = ("q8", "q4")
DESCRIPTION = ("32-weight blocks with F16 scales (q8: int8; q4: 4-bit offset-8 levels, scale = signed extreme / -8); "
               "FP32 norms/biases; activations quantized per 32-element block at run time")


def stored_metadata(weights: Path) -> dict:
    """The safetensors header's __metadata__, read without loading any tensor."""
    with weights.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size)).get("__metadata__", {})


def check_formats(metadata: dict, weight_format: str, head_format: str, where: str) -> None:
    expected = {"quantization": "block32", "weight_format": weight_format, "head_format": head_format}
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"{where} formats {actual} differ from the requested {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=FORMATS, default="q8", help="Projection/embedding weight format")
    parser.add_argument("--head-format", choices=FORMATS, help="Tied LM-head (embedding) format; defaults to --format")
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    head_format = args.head_format or args.format
    source = verify_snapshot(args.source)
    manifest_path = args.output / "quantization.json"
    weights = args.output / "model.safetensors"
    if weights.exists():
        if not manifest_path.exists():
            raise ValueError(f"Existing artifact {weights} has no quantization.json; move it aside before quantizing")
        manifest = json.loads(manifest_path.read_text())
        if manifest["source"]["revision"] != source["revision"] or manifest["weights"]["sha256"] != file_hash(weights):
            raise ValueError("Existing quantized artifact does not match its manifest")
        if manifest["config_sha256"] != file_hash(args.output / "config.json"):
            raise ValueError("Existing quantized configuration differs from its manifest")
        check_formats({"quantization": "block32", **{key: manifest.get(key) for key in ("weight_format", "head_format")}}, args.format, head_format, "Existing manifest")
    else:
        command = [str(args.engine), "quantize", "--model", str(args.source), "--output", str(args.output), "--format", args.format, "--head-format", head_format]
        subprocess.run(command, check=True)
        manifest = {"source": source, "weight_format": args.format, "head_format": head_format, "format": DESCRIPTION,
                    "storage": "safetensors: I8 [rows, cols] (q8) or U8 [rows, cols/2] (q4) matrices with NAME.scales F16 [rows, cols/32]; F32 vectors",
                    "tied_head": True, "command": command,
                    "weights": {"path": str(weights.resolve()), "bytes": weights.stat().st_size, "sha256": file_hash(weights)},
                    "config_sha256": file_hash(args.output / "config.json")}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # The stored header is the engine's own record; it must agree with the manifest.
    check_formats(stored_metadata(weights), args.format, head_format, f"Stored {weights}")
    locations = {args.source.resolve(): "$MODEL", args.output.resolve(): "$QUANT", args.engine.resolve(): "build/cpu-decode"}
    for flag, name in [("--model", "$MODEL"), ("--output", "$QUANT")]:
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
