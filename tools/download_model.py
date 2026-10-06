#!/usr/bin/env python3
"""Download the pinned public Qwen snapshot and verify upstream blob identities."""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
# Upstream identities: https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct/revision/7ae557604adf67be50417f59c2c2f167def9a775?blobs=true
FILES = {
    "LICENSE": (11343, "git-sha1", "6634c8cc3133b3848ec74b9f275acaaa1ea618ab"),
    "README.md": (4917, "git-sha1", "4b8373851d093eb9f3017443f27781c6971eff24"),
    "config.json": (659, "git-sha1", "0dbb161213629a23f0fc00ef286e6b1e366d180f"),
    "generation_config.json": (242, "git-sha1", "dfc11073787daf1b0f9c0f1499487ab5f4c93738"),
    "merges.txt": (1671839, "git-sha1", "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0"),
    "model.safetensors": (988097824, "sha256", "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"),
    "tokenizer.json": (7031645, "git-sha1", "443909a61d429dff23010e5bddd28ff530edda00"),
    "tokenizer_config.json": (7305, "git-sha1", "07bfe0640cb5a0037f9322287fbfc682806cf672"),
    "vocab.json": (2776833, "git-sha1", "4783fe10ac3adce15ac8f358ef5462739852c569"),
}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new("sha1" if algorithm == "git-sha1" else algorithm)
    if algorithm == "git-sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path) -> dict:
    records = {}
    for name, (size, algorithm, expected) in FILES.items():
        path = snapshot / name
        actual = file_hash(path, algorithm)
        if path.stat().st_size != size or actual != expected:
            raise ValueError(f"Pinned model checksum mismatch: {path}")
        records[name] = {
            "bytes": size,
            "sha256": actual if algorithm == "sha256" else file_hash(path),
            "upstream_algorithm": algorithm,
            "upstream_digest": expected,
        }
    return {"model_id": MODEL_ID, "revision": REVISION, "stored_dtype": "bfloat16", "files": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, help="Optional cache override; otherwise use Hugging Face's standard environment settings")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/model-manifest.json"))
    args = parser.parse_args()
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, cache_dir=str(args.hf_home / "hub") if args.hf_home is not None else None, allow_patterns=list(FILES), local_files_only=args.offline, max_workers=2))
    manifest = verify_snapshot(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
