#!/usr/bin/env python3
"""Compare real Q8_0 llama.cpp logits with the four stored FP32 oracle cases.

Both paths consume the oracle's exact teacher-forced IDs, including its reference
continuation. This does not check independent llama.cpp greedy generation.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import subprocess
from pathlib import Path

from tools.download_model import MODEL_ID, REVISION, file_hash
from tools.portable import ROOT, portable
from tools.prepare_llama import LLAMA_COMMIT, LLAMA_URL
from tools.reference import position_metrics


def verify_inputs(args) -> tuple[dict, dict, dict]:
    """Check stored identities before starting any reader/model process."""
    manifest = json.loads(args.artifact_manifest.read_text())
    if manifest["llama_commit"] != LLAMA_COMMIT or manifest["llama_repository"] != LLAMA_URL:
        raise ValueError("Artifact manifest does not name the pinned llama.cpp revision")
    source = manifest["source_model"]
    if (source["model_id"], source["revision"], source["stored_dtype"]) != (MODEL_ID, REVISION, "bfloat16"):
        raise ValueError("Artifact manifest does not name the pinned BF16 source model")
    artifact = manifest["artifacts"]["Q8_0"]
    actual_hash = file_hash(args.model)
    if actual_hash != artifact["sha256"] or args.model.stat().st_size != artifact["bytes"]:
        raise ValueError("Supplied GGUF does not match the manifest Q8_0 artifact")

    oracle_path = args.reference_dir / "reference.json"
    oracle = json.loads(oracle_path.read_text())
    if (oracle["model_id"], oracle["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("Stored oracle does not name the pinned model revision")
    if (oracle["arithmetic"], oracle["weight_storage"], oracle["kv_dtype"]) != ("fp32", "bfloat16", "float32"):
        raise ValueError("Stored oracle must use FP32 arithmetic/KV on BF16 weights")
    if oracle["verified_source"] != source:
        raise ValueError("Stored oracle and GGUF manifest have different source identities")

    fixed = json.loads((ROOT / "configs/prompts.json").read_text())
    expected_ids = [case["id"] for case in fixed["prompts"]]
    if len(expected_ids) != 4 or [case["id"] for case in oracle["prompts"]] != expected_ids:
        raise ValueError("Stored oracle must contain exactly the four fixed prompt cases")
    if oracle["steps"] != fixed["greedy_steps"]:
        raise ValueError("Stored oracle has a different reference continuation length")
    for case in oracle["prompts"]:
        shape = case["shape"]
        tokens = case["tokens"]
        if (len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)
                or shape[0] != len(tokens)):
            raise ValueError(f"Invalid oracle logit shape for {case['id']}")
        if not tokens or any(type(token) is not int or not 0 <= token < shape[1] for token in tokens):
            raise ValueError(f"Invalid oracle token IDs for {case['id']}")
        if (not case["prompt_tokens"] or len(case["generated_tokens"]) != oracle["steps"]
                or tokens != case["prompt_tokens"] + case["generated_tokens"][:-1]):
            raise ValueError(f"Invalid oracle teacher-forced continuation for {case['id']}")
        name = case["logits"]
        if Path(name).name != name:
            raise ValueError("Oracle logits must name a file inside the reference directory")
        path = args.reference_dir / name
        if path.stat().st_size != shape[0] * shape[1] * 4 or file_hash(path) != case["sha256"]:
            raise ValueError(f"Stored oracle logit checksum/size mismatch for {case['id']}")

    identity = {"path": str(args.model.resolve()), "sha256": actual_hash, "bytes": args.model.stat().st_size}
    return manifest, oracle, identity


def summarize_positions(rows: list[dict]) -> dict:
    matches = sum(row["top1_match"] for row in rows)
    return {
        "positions": len(rows),
        "top1_matches": matches,
        "top1_agreement": matches / len(rows),
        "mean_kl_reference_candidate_nats": sum(row["kl_reference_candidate_nats"] for row in rows) / len(rows),
        "max_kl_reference_candidate_nats": max(row["kl_reference_candidate_nats"] for row in rows),
    }


def aggregate(cases: list[dict]) -> dict:
    return {
        "prompt_only": summarize_positions([
            row for case in cases for row in case["positions"][:len(case["prompt_tokens"])]
        ]),
        "prompt_and_reference_continuation": summarize_positions([
            row for case in cases for row in case["positions"]
        ]),
    }


def run_comparison(args) -> dict:
    import numpy as np

    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    manifest, oracle, identity = verify_inputs(args)
    cases = []
    for case in oracle["prompts"]:
        prefix = args.reference_dir.resolve() / f"{case['id']}-llama-q8_0"
        command = [
            str(args.reader.resolve()), "--model", str(args.model.resolve()),
            "--tokens", ",".join(map(str, case["tokens"])),
            "--output", str(prefix), "--threads", str(args.threads),
        ]
        subprocess.run(command, check=True)
        metadata_path = prefix.with_suffix(".json")
        candidate_path = prefix.with_suffix(".bin")
        metadata = json.loads(metadata_path.read_text())
        if metadata["shape"] != case["shape"] or metadata["tokens"] != case["tokens"]:
            raise ValueError(f"llama.cpp logits metadata does not match oracle inputs for {case['id']}")
        if (metadata["threads"] != args.threads or metadata["kv_dtype"] != "float16"
                or metadata["flash_attention"] != "auto"
                or metadata["model"] != args.model.name):
            raise ValueError("llama.cpp reader settings do not match the requested compared path")
        shape = tuple(case["shape"])
        if candidate_path.stat().st_size != shape[0] * shape[1] * 4:
            raise ValueError(f"llama.cpp logit file size does not match shape for {case['id']}")
        reference = np.memmap(args.reference_dir / case["logits"], dtype="<f4", mode="r", shape=shape)
        candidate = np.memmap(candidate_path, dtype="<f4", mode="r", shape=shape)
        metrics = position_metrics(reference, candidate, atol=0.0, rtol=0.0)
        del reference, candidate
        if metadata["argmax"] != [row["candidate_top1"] for row in metrics]:
            raise ValueError(f"llama.cpp argmax metadata disagrees with actual logits for {case['id']}")
        # The common function also calculates allclose; no tolerance/pass gate is
        # meaningful for this diagnostic comparison of different quantized paths.
        for row in metrics:
            del row["allclose"]
        record = {
            "id": case["id"],
            "prompt_tokens": case["prompt_tokens"],
            "teacher_forced_tokens": case["tokens"],
            "reference_generated_tokens": case["generated_tokens"],
            "positions": metrics,
            "reader_settings": metadata,
            "logits": {"path": str(candidate_path), "sha256": file_hash(candidate_path), "shape": case["shape"], "dtype": "little-endian float32"},
            "reader_metadata": {"path": str(metadata_path), "sha256": file_hash(metadata_path)},
            "command": command,
        }
        record["aggregate"] = aggregate([record])
        cases.append(record)

    result = {
        "label": "llama.cpp Q8_0",
        "quality_reporting_only": True,
        "quality_thresholds": None,
        "independent_greedy_generation_checked": False,
        "comparison": "Full-vocabulary logits at every position on the exact oracle teacher-forced tokens; KL(reference || candidate) in nats",
        "scope": "Four fixed prompts only; prompt-only includes every prompt position, and the larger scope also includes the oracle's reference continuation inputs",
        "compared_path": {"weight_quantization": "Q8_0", "kv_dtype": "float16", "flash_attention": "auto", "threads": args.threads, "batch_size": 1, "fresh_context_per_prompt": True},
        "identity": {
            "llama_commit": manifest["llama_commit"],
            "model_id": MODEL_ID,
            "model_revision": REVISION,
            "pin_provenance": "Preparation manifest; reader binary revision is not independently reported by the reader interface",
            "artifact": identity,
            "artifact_manifest": {"path": str(args.artifact_manifest.resolve()), "sha256": file_hash(args.artifact_manifest)},
            "oracle_metadata": {"path": str((args.reference_dir / "reference.json").resolve()), "sha256": file_hash(args.reference_dir / "reference.json")},
        },
        "oracle": oracle,
        "fixed_prompts": len(cases),
        "aggregate": aggregate(cases),
        "prompts": cases,
    }
    return portable(result, {
        args.model: "$GGUF", args.reader: "$LLAMA_LOGITS",
        args.reference_dir: "$REFERENCE_DIR", args.artifact_manifest: "$ARTIFACT_MANIFEST",
        args.output: "$OUTPUT", sys.executable: "$PYTHON",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reader", type=Path, default=Path("build/llama-logits"))
    parser.add_argument("--reference-dir", type=Path, default=Path("external/reference"))
    parser.add_argument("--artifact-manifest", type=Path, default=Path("results/llama-preparation.json"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/llama-quality.json"))
    args = parser.parse_args()
    result = run_comparison(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
