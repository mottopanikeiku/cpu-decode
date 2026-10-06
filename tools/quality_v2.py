#!/usr/bin/env python3
"""Streaming corpus quality: prepare, oracle, evaluate, and compare.

Public commands (set MODEL, GGUF, and format directories to local artifacts):
  nice -n 19 uv run python -m tools.quality_v2 prepare --model "$MODEL"
  nice -n 19 uv run python -m tools.quality_v2 oracle --model "$MODEL"
  nice -n 19 uv run python -m tools.quality_v2 evaluate --split calibration --model "$GROUP32F16" --label g32f16 --selection results/v2/format-selection.json --output results/v2/cal-g32f16.json
Repeat calibration evaluate for 64F32,64F16,128F16 using identical settings, then:
  nice -n 19 uv run python -m tools.quality_v2 evaluate --split calibration --backend llama --model "$GGUF" --threads 2 --label q8_0-calibration --selection results/v2/format-selection.json --output results/v2/cal-q8_0.json
  nice -n 19 uv run python -m tools.quality_v2 compare --split calibration --reports results/v2/cal-*.json --selection results/v2/format-selection.json --output results/v2/format-calibration.json
This exclusively creates the supplied selection BEFORE any held-out candidate.
Evaluate heldout with that same --selection, separately for v1, the chosen
format (F16 and FP32 KV, and desired kernels), and actual pinned Q8_0.
Use new output paths; historical selection, calibration and heldout files remain.
Raw whole-vocabulary logits live in ignored external/quality-v2, never results/.
All destinations under results/ must resolve inside results/v2/. The v1 baseline
must match archived weight/configuration hashes and the oracle's pinned source.
Q8_0 evaluation records resolved llama/ggml shared-library and upstream build
hashes, rechecking the complete reader identity before and after every window.
Format selection requires strictly lower calibration mean/p99 KL and strictly
higher top1 than Q8_0, then minimizes weights artifact bytes. PPL is reported,
not used to select a format. No passing format is an error, not a fallback.
VNNI is retained only when its
held-out mean KL, p99 KL, top1 agreement and perplexity are all at least as good
as actual Q8_0.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import gc
import json
import math
import re
import subprocess
from pathlib import Path
import struct

from tools.corpus_v2 import FORMAT_CHOICES, POLICY, SOURCES, digest_json, load_manifest, prepare, protect_destination, window_alignment, write_json
from tools.download_model import MODEL_ID, REVISION, file_hash, verify_snapshot
from tools.portable import ROOT, portable
from tools.prepare_llama import LLAMA_COMMIT, LLAMA_URL
from tools.reference import load_oracle, project_last, versions

FORMAT_SELECTION_POLICY = (
    "Among 32F16,64F32,64F16,128F16, require calibration mean KL and p99 KL "
    "strictly below actual Q8_0 and top1 strictly above; minimize weights artifact "
    "bytes, then mean KL, p99 KL, negative top1, label; perplexity is report-only; "
    "no fallback or heldout reselection"
)


def normalize_dtype(value: str) -> str:
    return {"float16": "f16", "float32": "f32", "F16": "f16", "F32": "f32"}.get(value, value)


def log_probabilities(row):
    import numpy as np

    row = np.asarray(row, dtype=np.float64)
    if row.ndim != 1 or not len(row) or not np.isfinite(row).all():
        raise ValueError("Expected finite whole-vocabulary logits")
    result = row - row.max()
    return result - np.log(np.exp(result).sum())


def position_metric(reference, candidate, target: int) -> dict:
    import numpy as np

    logp, logq = log_probabilities(reference), log_probabilities(candidate)
    if logp.shape != logq.shape or type(target) is not int or not 0 <= target < len(logp):
        raise ValueError("Vocabulary/next-token target mismatch")
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    ref_top, top = int(logp.argmax()), int(logq.argmax())
    return {"target": target, "reference_top1": ref_top, "candidate_top1": top,
        "top1_match": ref_top == top,
        "kl_reference_candidate_nats": max(0.0, float(np.dot(np.exp(logp), logp - logq))),
        "reference_cross_entropy_nats": float(-logp[target]),
        "candidate_cross_entropy_nats": float(-logq[target]),
        "max_abs_error": float(np.abs(difference).max()),
        "rmse": float(np.sqrt(np.mean(difference * difference)))}


def summarize(rows: list[dict]) -> dict:
    import numpy as np

    if not rows:
        raise ValueError("No scored positions")
    kl = [row["kl_reference_candidate_nats"] for row in rows]
    ref_ce = math.fsum(row["reference_cross_entropy_nats"] for row in rows) / len(rows)
    ce = math.fsum(row["candidate_cross_entropy_nats"] for row in rows) / len(rows)
    return {"positions": len(rows), "mean_kl_reference_candidate_nats": math.fsum(kl) / len(rows),
        "p99_kl_reference_candidate_nats": float(np.quantile(kl, 0.99, method="linear")),
        "p99_method": "numpy.quantile linear over all scored positions",
        "max_kl_reference_candidate_nats": max(kl),
        "top1_agreement": sum(row["top1_match"] for row in rows) / len(rows),
        "reference_next_token_cross_entropy_nats": ref_ce, "reference_perplexity": math.exp(ref_ce),
        "next_token_cross_entropy_nats": ce, "perplexity": math.exp(ce)}


def checked_logits(path: Path, record: dict, verify_hash=True):
    import numpy as np

    shape = record["shape"]
    if (len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape)
            or path.stat().st_size != shape[0] * shape[1] * 4):
        raise ValueError("Logit shape/byte count mismatch")
    if verify_hash and file_hash(path) != record["sha256"]:
        raise ValueError("Raw logits hash mismatch")
    return np.memmap(path, dtype="<f4", mode="r", shape=tuple(shape))


def oracle_identity(args, corpus: dict, source: dict) -> dict:
    return {"model_id": MODEL_ID, "revision": REVISION, "verified_source": source,
        "corpus_sha256": file_hash(args.corpus), "corpus_policy": corpus["policy"],
        "weight_storage": "bfloat16", "arithmetic": "fp32", "kv_dtype": "float32",
        "attention": "Transformers eager", "head_chunk_rows": args.head_chunk,
        "threads": args.threads, "versions": versions()}


def oracle(args) -> None:
    protect_destination(args.output)
    protect_destination(args.raw_dir)
    corpus = load_manifest(args.corpus)
    source = verify_snapshot(args.model)
    import numpy as np
    import torch

    identity = oracle_identity(args, corpus, source)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.raw_dir / "oracle.json"
    protect_destination(manifest_path)
    metadata = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"identity": identity, "windows": []}
    if metadata["identity"] != identity:
        raise ValueError("Existing oracle identity differs; use a new raw directory")
    by_id = {case["id"]: case for case in metadata["windows"]}
    wanted = [window for window in corpus["windows"] if args.split == "all" or window["split"] == args.split]
    pending = []
    for window in wanted:
        if window["id"] in by_id:
            record = by_id[window["id"]]
            validate_case(record, window)
            rows = checked_logits(raw_file(args.raw_dir, record["logits"]), record)
            del rows
        else:
            pending.append(window)
    if pending:
        model = load_oracle(args.model, args.threads, "fp32")
        with torch.inference_mode():
            for window in pending:
                inputs, positions, targets = window_alignment(window)
                name = f"{window['id']}-reference.bin"
                path = args.raw_dir / name
                protect_destination(path)
                shape = [len(positions), model.config.vocab_size]
                logits = np.memmap(path, dtype="<f4", mode="w+", shape=tuple(shape))
                priming = model.model(input_ids=torch.tensor([inputs[:POLICY['priming_tokens']]], dtype=torch.long), use_cache=True, return_dict=True)
                cache = priming.past_key_values
                del priming
                diagnostics = []
                for index, (position, target) in enumerate(zip(positions, targets, strict=True)):
                    result = model.model(input_ids=torch.tensor([[inputs[position]]], dtype=torch.long), past_key_values=cache, use_cache=True, return_dict=True)
                    cache = result.past_key_values
                    row = project_last(model, result.last_hidden_state, "fp32", args.head_chunk)
                    logits[index] = row.numpy()
                    logp = log_probabilities(row.numpy())
                    diagnostics.append({"window_id": window["id"], "input_position": position,
                        "target": target, "reference_top1": int(logp.argmax()),
                        "reference_cross_entropy_nats": float(-logp[target])})
                    del result, row, logp
                logits.flush()
                del logits, cache
                record = {"id": window["id"], "split": window["split"], "tokens_sha256": window["tokens_sha256"],
                    "input_tokens": inputs, "logit_positions": positions, "targets": targets,
                    "logits": name, "shape": shape, "sha256": file_hash(path), "positions": diagnostics}
                metadata["windows"].append(record)
                write_json(manifest_path, metadata)
                print(json.dumps({"oracle_window": window["id"], "scored_positions": len(positions)}, sort_keys=True), flush=True)
        del model
        gc.collect()
    splits = {}
    for split in SOURCES:
        rows = [row for case in metadata["windows"] if case["split"] == split for row in case["positions"]]
        if rows:
            ce = math.fsum(row["reference_cross_entropy_nats"] for row in rows) / len(rows)
            splits[split] = {"positions": len(rows), "next_token_cross_entropy_nats": ce, "perplexity": math.exp(ce)}
    summary = {"identity": identity, "oracle_metadata_sha256": file_hash(manifest_path),
        "splits": splits, "windows": [{key: value for key, value in case.items() if key != "input_tokens"} for case in metadata["windows"]]}
    write_json(args.output, summary)
    print(json.dumps({"reference": splits, "output": str(args.output)}, sort_keys=True), flush=True)


def raw_file(directory: Path, name: str) -> Path:
    if Path(name).name != name:
        raise ValueError("Raw artifact must name a file, not a path")
    return directory / name


def validate_case(record: dict, window: dict) -> None:
    inputs, positions, targets = window_alignment(window)
    if (record["tokens_sha256"] != window["tokens_sha256"] or record["input_tokens"] != inputs
            or record["logit_positions"] != positions or record["targets"] != targets
            or record["shape"][0] != len(targets) or record["split"] != window["split"]):
        raise ValueError("Oracle target/context alignment differs from protected corpus")


def read_oracle(args, corpus: dict) -> dict:
    metadata = json.loads((args.raw_dir / "oracle.json").read_text())
    identity = metadata["identity"]
    if (identity["corpus_sha256"] != file_hash(args.corpus)
            or (identity["model_id"], identity["revision"], identity["weight_storage"], identity["arithmetic"], identity["kv_dtype"])
            != (MODEL_ID, REVISION, "bfloat16", "fp32", "float32")):
        raise ValueError("Oracle identity does not match pinned FP32 reference/corpus")
    windows = {window["id"]: window for window in corpus["windows"]}
    if len({case["id"] for case in metadata["windows"]}) != len(metadata["windows"]):
        raise ValueError("Duplicate oracle windows")
    for case in metadata["windows"]:
        validate_case(case, windows[case["id"]])
    return metadata


def model_identity(model: Path, backend: str) -> dict:
    paths = [model] if backend == "llama" else [model / "config.json", model / "model.safetensors"]
    files = {path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size} for path in paths}
    return {"files": files, "sha256": digest_json(files)}


def archived_v1_identity(source: dict | None = None) -> dict:
    manifest = json.loads((ROOT / "results/quantized-manifest.json").read_text())
    archived_source = manifest["source"]
    if ((archived_source["model_id"], archived_source["revision"], archived_source["stored_dtype"])
            != (MODEL_ID, REVISION, "bfloat16") or source is not None and source != archived_source):
        raise ValueError("Archived v1 source differs from the pinned oracle")
    files = {"model.safetensors": {"sha256": manifest["weights"]["sha256"], "bytes": manifest["weights"]["bytes"]},
        "config.json": {"sha256": manifest["config_sha256"], "bytes": archived_source["files"]["config.json"]["bytes"]}}
    if files["config.json"]["sha256"] != archived_source["files"]["config.json"]["sha256"]:
        raise ValueError("Archived v1 configuration differs from its pinned source")
    return {"files": files, "sha256": digest_json(files)}


def resolved_upstream_libraries(ldd_output: str) -> dict[str, Path]:
    libraries = {}
    for line in ldd_output.splitlines():
        match = re.match(r"\s*(lib(?:llama|ggml)[^ ]*)\s+=>\s+(/.*?)\s+\(0x[0-9a-fA-F]+\)", line)
        if match:
            if not re.fullmatch(r"lib(?:llama|llama-bench-impl|llama-common|ggml|ggml-base|ggml-cpu)\.so(?:\.\d+)*", match.group(1)):
                raise ValueError(f"Unknown upstream CPU reader dependency: {match.group(1)}")
            if match.group(1) in libraries:
                raise ValueError("Duplicate upstream reader dependency")
            libraries[match.group(1)] = Path(match.group(2)).resolve(strict=True)
        elif re.match(r"\s*lib(?:llama|ggml)", line):
            raise ValueError("Unresolved upstream reader dependency")
    for prefix in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so"):
        if not any(name.startswith(prefix) for name in libraries):
            raise ValueError(f"Reader must dynamically link the pinned {prefix} dependency")
    return libraries


def upstream_build_identity(libraries: dict[str, Path]) -> dict:
    directories = {path.parent for path in libraries.values()}
    if len(directories) != 1:
        raise ValueError("Reader upstream dependencies come from different build directories")
    build_root = directories.pop().parent
    cache = build_root / "CMakeCache.txt"
    settings = {}
    for line in cache.read_text().splitlines():
        match = re.match(r"(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER):[^=]+=(.*)|(GGML_[^:]+):BOOL=(.*)", line)
        if match:
            settings[match.group(1) or match.group(3)] = match.group(2) if match.group(1) else match.group(4)
    if settings.get("GGML_BACKEND_DL") == "ON":
        raise ValueError("Reader identity requires directly linked CPU libraries, not unrecorded dynamically discovered backends")
    flags = {str(path.relative_to(build_root)): file_hash(path) for path in sorted(build_root.rglob("flags.make"))}
    commands = build_root / "compile_commands.json"
    if commands.exists():
        flags["compile_commands.json"] = file_hash(commands)
    if not flags:
        raise ValueError("Upstream build identity needs generated flags.make or compile_commands.json")
    return {"shared_libraries": {
        name: {"resolved_path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}
        for name, path in sorted(libraries.items())},
        "build": {"root": str(build_root), "root_sha256": digest_json(str(build_root)),
                  "cache_sha256": file_hash(cache), "settings": settings, "compile_flags_sha256": flags},
        "resolution": "ldd under the same inherited loader environment; resolved upstream files rehashed before and after every window"}


def reader_build_identity(reader: Path) -> dict:
    run = subprocess.run(["ldd", str(reader.resolve())], check=True, capture_output=True, text=True)
    libraries = resolved_upstream_libraries(run.stdout)
    return {"reader_binary_sha256": file_hash(reader), **upstream_build_identity(libraries),
            "loader_environment_sha256": {
                name: digest_json(os.environ.get(name))
                for name in ["LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"]}}


def reader_identity_locations(identity: dict) -> dict:
    """Alias the resolved dependency build, independent of reader placement."""
    root = Path(identity["build"]["root"])
    locations = {root.parent: "$LLAMA_ROOT", root: "$LLAMA_BUILD"}
    compiler = identity["build"]["settings"].get("CMAKE_CXX_COMPILER")
    if compiler and Path(compiler).is_absolute():
        locations[Path(compiler)] = "$LLAMA_CXX_COMPILER"
    return locations


def require_reader_identity(expected: dict, actual: dict) -> None:
    if expected != actual:
        raise ValueError("Reader or upstream shared-library/build identity changed between windows")


def read_selection(path: Path, corpus_hash: str) -> dict:
    selection = json.loads(path.read_text())
    if (selection["corpus_sha256"] != corpus_hash or selection["split"] != "calibration"
            or selection["policy"] not in (POLICY["format_selection"], FORMAT_SELECTION_POLICY)
            or (selection["chosen"]["group_size"], selection["chosen"]["scale_dtype"]) not in FORMAT_CHOICES):
        raise ValueError("Calibration decision does not match corpus/policy")
    if selection["policy"] == FORMAT_SELECTION_POLICY:
        selected = choose_format(selection["evidence"])
        chosen = selection["chosen"]
        if (chosen["label"] != selected["label"] or chosen["model_identity"] != selected["model_identity"]
                or chosen["settings"] != selected["settings"] or chosen["aggregate"] != selected["aggregate"]
                or chosen["weights_artifact_bytes"] != weights_artifact_bytes(selected)):
            raise ValueError("Calibration decision differs from its Q8_0 comparison evidence")
    return selection


def native_settings(metadata: dict) -> dict:
    return {"group_size": metadata["group_size"], "scale_dtype": normalize_dtype(metadata["scale_dtype"]),
        "kv_dtype": normalize_dtype(metadata["kv_dtype"]), "kernel": metadata["kernel"],
        "attention": metadata["attention"], "scheduler": metadata["scheduler"], "affinity": metadata["affinity"],
        "cpu_set": metadata["cpu_set"], "threads": metadata["threads"], "weight_dtype": metadata["weight_dtype"]}


def validate_heldout_settings(settings: dict, identity: dict, selection: dict) -> None:
    if settings["weight_dtype"] != "int8":
        raise ValueError("Heldout native candidates must be v1 int8 or the calibration-selected int8 format")
    if (settings["group_size"], settings["scale_dtype"]) == (0, "f32"):
        source = selection.get("oracle_identity", {}).get("verified_source")
        if identity != archived_v1_identity(source):
            raise ValueError("Per-row candidate does not match the archived v1 weight/configuration hashes")
        return
    chosen = selection["chosen"]
    if (settings["group_size"], settings["scale_dtype"], identity["sha256"]) != (chosen["group_size"], chosen["scale_dtype"], chosen["model_identity"]["sha256"]):
        raise ValueError("Heldout format/weights were not selected on calibration")


def preflight_native_model(model: Path, identity: dict, selection: dict | None, oracle_source: dict | None = None) -> dict:
    # Reject an unselected format before producing/inspecting held-out logits.
    with (model / "model.safetensors").open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 64 * 1024 * 1024:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(length))
    metadata = header.get("__metadata__", {})
    if metadata.get("quantization") == "symmetric-per-row-int8":
        settings = {"group_size": 0, "scale_dtype": "f32", "weight_dtype": "int8"}
    else:
        settings = {"group_size": int(metadata["group_size"]),
            "scale_dtype": normalize_dtype(metadata["scale_dtype"]), "weight_dtype": "int8"}
    matrices = [tensor for name, tensor in header.items() if name != "__metadata__" and len(tensor["shape"]) == 2 and not name.endswith(".scales")]
    if not matrices or any(tensor["dtype"] != "I8" for tensor in matrices):
        raise ValueError("Quality candidates must contain real int8 matrix weights")
    if (settings["group_size"], settings["scale_dtype"]) == (0, "f32") and identity != archived_v1_identity(oracle_source):
        raise ValueError("Per-row candidate does not match the archived v1 weight/configuration hashes")
    if selection is not None:
        validate_heldout_settings(settings, identity, selection)
    return settings


def evaluate(args) -> None:
    protect_destination(args.output)
    protect_destination(args.raw_dir)
    corpus = load_manifest(args.corpus)
    corpus_hash = file_hash(args.corpus)
    selection = read_selection(args.selection, corpus_hash) if args.split == "heldout" else None
    if args.split == "calibration" and args.selection.exists():
        raise ValueError("Calibration is closed after format selection")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise ValueError("Label must be a simple filename identifier")
    if args.output.exists():
        raise ValueError("Quality output already exists; do not overwrite recorded measurements")
    metadata = read_oracle(args, corpus)
    if selection is not None and selection["oracle_identity"] != metadata["identity"]:
        raise ValueError("Heldout oracle differs from the calibration oracle")
    cases = {case["id"]: case for case in metadata["windows"]}
    windows = [window for window in corpus["windows"] if window["split"] == args.split]
    if any(window["id"] not in cases for window in windows):
        raise ValueError("Oracle generation is incomplete for the requested split")
    # Both readers truncate PREFIX.bin/PREFIX.json: reject every collision before
    # any subprocess, so a later window cannot leave a partially written run.
    prefixes = [args.raw_dir.resolve() / f"{window['id']}-{args.label}" for window in windows]
    for prefix in prefixes:
        for suffix in (".bin", ".json"):
            path = prefix.with_suffix(suffix)
            destination = protect_destination(path)
            if destination.exists() or path.is_symlink():
                raise ValueError("Quality raw output already exists; do not overwrite recorded measurements")
    identity = model_identity(args.model, args.backend)
    pinned = None
    if args.backend == "llama":
        manifest = json.loads(args.artifact_manifest.read_text())
        if manifest["llama_commit"] != LLAMA_COMMIT or manifest["llama_repository"] != LLAMA_URL or manifest["source_model"] != metadata["identity"]["verified_source"]:
            raise ValueError("Baseline preparation/source pin differs from oracle")
        artifact = manifest["artifacts"]["Q8_0"]
        actual = identity["files"][args.model.name]
        if actual != {"sha256": artifact["sha256"], "bytes": artifact["bytes"]}:
            raise ValueError("GGUF is not the actual pinned Q8_0 artifact")
        reader_identity = reader_build_identity(args.reader)
        pinned = {"llama_commit": LLAMA_COMMIT, "artifact_manifest_sha256": file_hash(args.artifact_manifest),
            **reader_identity, "pin_provenance": "Preparation manifest; executable, resolved upstream libraries and build records independently hashed"}
    else:
        preflight = preflight_native_model(args.model, identity, selection, metadata["identity"]["verified_source"])
        pinned = {"engine_binary_sha256": file_hash(args.engine)}
    records, all_rows = [], []
    settings = None
    for window, prefix in zip(windows, prefixes, strict=True):
        if args.backend == "llama":
            require_reader_identity(reader_identity, reader_build_identity(args.reader))
        case = cases[window["id"]]
        reference = checked_logits(raw_file(args.raw_dir, case["logits"]), case)
        common = ["--model", str(args.model.resolve()), "--tokens", ",".join(map(str, case["input_tokens"])),
            "--output", str(prefix), "--threads", str(args.threads), "--logits-start", str(POLICY["priming_tokens"])]
        if args.backend == "native":
            command = [str(args.engine.resolve()), "logits", *common, "--kernel", args.kernel,
                "--kv", args.kv, "--attention", args.attention, "--scheduler", args.scheduler, "--affinity", args.affinity]
            if args.cpu_set:
                command += ["--cpu-set", args.cpu_set]
        else:
            command = [str(args.reader.resolve()), *common]
        subprocess.run(command, check=True)
        if args.backend == "llama":
            require_reader_identity(reader_identity, reader_build_identity(args.reader))
        actual = json.loads(prefix.with_suffix(".json").read_text())
        if (actual["shape"] != case["shape"] or actual["tokens"] != case["input_tokens"]
                or actual["logit_positions"] != case["logit_positions"]):
            raise ValueError("Candidate did not consume/score exactly the oracle context positions")
        current = native_settings(actual) if args.backend == "native" else {
            "weight_dtype": "Q8_0", "kv_dtype": normalize_dtype(actual["kv_dtype"]),
            "flash_attention": actual["flash_attention"], "threads": actual["threads"]}
        if current["threads"] != args.threads:
            raise ValueError("Candidate thread setting differs from request")
        if args.backend == "native":
            if any(current[key] != preflight[key] for key in preflight):
                raise ValueError("Native settings differ from safetensors format metadata")
            if (current["kv_dtype"], current["attention"], current["scheduler"], current["affinity"]) != (args.kv, args.attention, args.scheduler, args.affinity):
                raise ValueError("Native engine settings differ from request")
            if args.kernel != "auto" and current["kernel"] != args.kernel:
                raise ValueError("Native kernel differs from request")
            if args.cpu_set and sorted(current["cpu_set"]) != sorted(int(cpu) for cpu in args.cpu_set.split(",")):
                raise ValueError("Native CPU set differs from request")
            if selection is not None:
                validate_heldout_settings(current, identity, selection)
        elif current["kv_dtype"] != "f16" or current["flash_attention"] != "auto" or actual["model"] != args.model.name:
            raise ValueError("llama reader settings differ from pinned Q8_0 path")
        if settings is not None and current != settings:
            raise ValueError("Candidate settings changed between windows")
        settings = current
        path = prefix.with_suffix(".bin")
        raw_record = {"shape": case["shape"], "sha256": file_hash(path)}
        candidate = checked_logits(path, raw_record, verify_hash=False)
        rows = []
        for index, (position, target) in enumerate(zip(case["logit_positions"], case["targets"], strict=True)):
            row = position_metric(reference[index], candidate[index], target)
            row.update(window_id=window["id"], input_position=position,
                source_target_token=window["source_token_start"] + position + 1,
                context_tokens=position + 1)
            rows.append(row)
        del reference, candidate
        if actual["argmax"] != [row["candidate_top1"] for row in rows]:
            raise ValueError("Candidate argmax metadata differs from raw logits")
        records.append({"id": window["id"], "tokens_sha256": window["tokens_sha256"],
            "logits": dict(raw_record, filename=path.name), "metadata_sha256": file_hash(prefix.with_suffix(".json")),
            "command": portable(command, {args.model: "$CANDIDATE_MODEL", args.engine: "$ENGINE", args.reader: "$LLAMA_LOGITS", args.raw_dir: "$RAW"})})
        all_rows.extend(rows)
    result = {"label": args.label, "split": args.split, "backend": args.backend, "corpus_sha256": corpus_hash,
        "oracle_identity": metadata["identity"], "settings": settings, "model_identity": identity, "binary_identity": pinned,
        "selection_sha256": file_hash(args.selection) if selection is not None else None,
        "aggregate": summarize(all_rows), "windows": records, "positions": all_rows,
        "scope": f"Teacher-forced {args.split} next-token likelihood, full-vocabulary KL(reference||candidate), fresh cache per window; no independent generation or tuning on heldout"}
    if result["aggregate"]["positions"] != POLICY[f"{args.split}_windows"] * 256:
        raise ValueError("Incomplete scored corpus")
    if args.backend == "llama":
        result["binary_identity"] = portable(pinned, reader_identity_locations(reader_identity))
    write_json(args.output, result, exclusive=True)
    print(json.dumps({"label": args.label, "split": args.split, "aggregate": result["aggregate"]}, sort_keys=True))


def validate_reports(reports: list[dict], corpus: dict, corpus_hash: str, split: str) -> None:
    expected = [(window["id"], position, target) for window in corpus["windows"] if window["split"] == split
        for position, target in zip(*window_alignment(window)[1:], strict=True)]
    labels = set()
    for report in reports:
        if report["split"] != split or report["corpus_sha256"] != corpus_hash or report["label"] in labels:
            raise ValueError("Report split/corpus/label mismatch")
        labels.add(report["label"])
        positions = report["positions"]
        if [(row["window_id"], row["input_position"], row["target"]) for row in positions] != expected:
            raise ValueError("Report target/context alignment or actual position count changed")
        if report["aggregate"] != summarize(positions):
            raise ValueError("Report aggregate differs from globally weighted positions")
    if not reports or any(report["oracle_identity"] != reports[0]["oracle_identity"] for report in reports):
        raise ValueError("Reports do not use exactly the same oracle")


def weights_artifact_bytes(report: dict) -> int:
    files = report["model_identity"]["files"]
    if report["backend"] == "native":
        artifact = files["model.safetensors"]
    elif report["backend"] == "llama" and len(files) == 1:
        artifact = next(iter(files.values()))
    else:
        raise ValueError("Expected one Q8_0 weights artifact")
    if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
        raise ValueError("Weights artifact bytes must be a positive integer")
    return artifact["bytes"]


def beats_q8_calibration(report: dict, baseline: dict) -> bool:
    candidate, q8 = report["aggregate"], baseline["aggregate"]
    return (candidate["mean_kl_reference_candidate_nats"] < q8["mean_kl_reference_candidate_nats"]
        and candidate["p99_kl_reference_candidate_nats"] < q8["p99_kl_reference_candidate_nats"]
        and candidate["top1_agreement"] > q8["top1_agreement"])


def choose_format(reports: list[dict]) -> dict:
    native = [report for report in reports if report["backend"] == "native"]
    baselines = [report for report in reports if report["backend"] == "llama"]
    if (len(reports) != len(FORMAT_CHOICES) + 1 or len(native) != len(FORMAT_CHOICES)
            or len(baselines) != 1 or any(report["split"] != "calibration" for report in reports)):
        raise ValueError("Selection requires exactly the four native calibration choices and one Q8_0, never heldout")
    choices = {(report["settings"]["group_size"], report["settings"]["scale_dtype"]) for report in native}
    if choices != set(FORMAT_CHOICES):
        raise ValueError("Calibration must cover 32F16,64F32,64F16,128F16")
    fixed = ("kernel", "kv_dtype", "attention", "scheduler", "affinity", "threads", "cpu_set", "weight_dtype")
    if any(any(report["settings"][key] != native[0]["settings"][key] for key in fixed) for report in native):
        raise ValueError("Calibration choices must use identical execution settings")
    if any(report["binary_identity"] != native[0]["binary_identity"] for report in native):
        raise ValueError("Calibration choices must use the same native binary")
    if any(report["oracle_identity"] != native[0]["oracle_identity"] for report in reports):
        raise ValueError("Calibration choices and Q8_0 must use the same oracle/source")
    if native[0]["settings"]["weight_dtype"] != "int8":
        raise ValueError("Calibration must measure int8 format choices")
    baseline = baselines[0]
    if (baseline["settings"]["weight_dtype"] != "Q8_0"
            or baseline["settings"]["flash_attention"] != "auto"
            or baseline["settings"]["kv_dtype"] != "f16"
            or baseline["binary_identity"]["llama_commit"] != LLAMA_COMMIT):
        raise ValueError("Calibration requires the actual pinned Q8_0 reader path")
    if any(baseline["settings"][key] != native[0]["settings"][key] for key in ("threads", "kv_dtype")):
        raise ValueError("Q8_0 calibration must match native threads and KV dtype")
    for report in reports:
        weights_artifact_bytes(report)
    eligible = [report for report in native if beats_q8_calibration(report, baseline)]
    if not eligible:
        raise ValueError("No native format beats calibration Q8_0 on mean KL, p99 KL and top1; no fallback")
    return min(eligible, key=lambda report: (weights_artifact_bytes(report),
        report["aggregate"]["mean_kl_reference_candidate_nats"],
        report["aggregate"]["p99_kl_reference_candidate_nats"],
        -report["aggregate"]["top1_agreement"], report["label"]))


def calibration_table(reports: list[dict]) -> list[dict]:
    baseline = next(report for report in reports if report["backend"] == "llama")
    table = []
    for report in sorted(reports, key=lambda report: report["label"]):
        settings, aggregate = report["settings"], report["aggregate"]
        bits = (8.5 if report["backend"] == "llama" else
            8 + (16 if settings["scale_dtype"] == "f16" else 32) / settings["group_size"])
        table.append({"label": report["label"], "split": "calibration",
            "matrix_bits_per_weight": bits, "weights_artifact_bytes": weights_artifact_bytes(report),
            "mean_kl_reference_candidate_nats": aggregate["mean_kl_reference_candidate_nats"],
            "p99_kl_reference_candidate_nats": aggregate["p99_kl_reference_candidate_nats"],
            "top1_agreement": aggregate["top1_agreement"], "perplexity": aggregate["perplexity"],
            "beats_q8_0": report["backend"] == "native" and beats_q8_calibration(report, baseline)})
    return table


def heldout_comparison(reports: list[dict], selection: dict) -> dict:
    baselines = [report for report in reports if report["backend"] == "llama"]
    if len(baselines) != 1:
        raise ValueError("Heldout comparison requires exactly one actual pinned Q8_0 baseline")
    baseline = baselines[0]
    native = [report for report in reports if report["backend"] == "native"]
    for report in native:
        validate_heldout_settings(report["settings"], report["model_identity"], selection)
    if not any((report["settings"]["group_size"], report["settings"]["scale_dtype"]) == (0, "f32") for report in native):
        raise ValueError("Heldout comparison requires the unchanged v1 per-row baseline")
    chosen = [report for report in native if report["settings"]["group_size"] != 0]
    pairs = []
    for fp32 in chosen:
        if fp32["settings"]["kv_dtype"] != "f32":
            continue
        for f16 in chosen:
            if f16["settings"]["kv_dtype"] != "f16" or f16["model_identity"] != fp32["model_identity"]:
                continue
            keys = ("kernel", "attention", "scheduler", "affinity", "threads", "cpu_set")
            if any(f16["settings"][key] != fp32["settings"][key] for key in keys):
                continue
            pairs.append({"f16_label": f16["label"], "f32_label": fp32["label"], "fixed_weights_sha256": f16["model_identity"]["sha256"],
                "f16": f16["aggregate"], "f32": fp32["aggregate"],
                "mean_kl_delta_f16_minus_f32": f16["aggregate"]["mean_kl_reference_candidate_nats"] - fp32["aggregate"]["mean_kl_reference_candidate_nats"],
                "cross_entropy_delta_f16_minus_f32_nats": f16["aggregate"]["next_token_cross_entropy_nats"] - fp32["aggregate"]["next_token_cross_entropy_nats"],
                "top1_agreement_between_kv_paths": sum(a["candidate_top1"] == b["candidate_top1"] for a, b in zip(f16["positions"], fp32["positions"], strict=True)) / len(f16["positions"])})
    if not pairs:
        raise ValueError("Heldout comparison requires a fixed-weight F16 versus FP32 KV pair with identical execution settings")
    vnni = []
    for report in native:
        if report["settings"]["kernel"] != "vnni":
            continue
        measured = report["aggregate"]
        q8 = baseline["aggregate"]
        retained = (measured["mean_kl_reference_candidate_nats"] <= q8["mean_kl_reference_candidate_nats"]
            and measured["p99_kl_reference_candidate_nats"] <= q8["p99_kl_reference_candidate_nats"]
            and measured["top1_agreement"] >= q8["top1_agreement"]
            and measured["perplexity"] <= q8["perplexity"])
        vnni.append({"label": report["label"], "retained": retained,
            "decision": "retained" if retained else "rejected quality tradeoff",
            "rule": "Heldout mean KL <= Q8_0 mean KL AND p99 KL <= Q8_0 p99 KL AND top1 agreement >= Q8_0 top1 agreement AND perplexity <= Q8_0 perplexity",
            "candidate": measured, "q8_0": q8})
    return {"positions": baseline["aggregate"]["positions"], "q8_0_label": baseline["label"],
        "kv_comparisons": pairs, "vnni_decisions": vnni,
        "vnni_measured": bool(vnni), "thresholds": None,
        "tuning": "Format chosen on calibration before heldout; no heldout format reselection"}


def compare(args) -> None:
    protect_destination(args.output)
    if args.split == "calibration":
        protect_destination(args.selection)
    corpus = load_manifest(args.corpus)
    corpus_hash = file_hash(args.corpus)
    reports = [json.loads(path.read_text()) for path in args.reports]
    validate_reports(reports, corpus, corpus_hash, args.split)
    if args.output.exists():
        raise ValueError("Comparison output already exists")
    evidence = [{"label": report["label"], "report": path.name, "sha256": file_hash(path),
        "split": report["split"], "backend": report["backend"],
        "settings": report["settings"], "aggregate": report["aggregate"],
        "model_identity": report["model_identity"], "binary_identity": report["binary_identity"],
        "oracle_identity": report["oracle_identity"]} for path, report in zip(args.reports, reports, strict=True)]
    evidence.sort(key=lambda entry: entry["label"])
    if args.split == "calibration":
        if args.selection.exists():
            raise ValueError("Calibration decision already exists; never reselect after heldout")
        chosen = choose_format(reports)
        settings = chosen["settings"]
        bits = 8 + (16 if settings["scale_dtype"] == "f16" else 32) / settings["group_size"]
        decision = {"split": "calibration", "corpus_sha256": corpus_hash,
            "oracle_identity": chosen["oracle_identity"],
            "policy": FORMAT_SELECTION_POLICY, "chosen": {"label": chosen["label"],
                "group_size": settings["group_size"], "scale_dtype": settings["scale_dtype"],
                "matrix_bits_per_weight": bits, "weights_artifact_bytes": weights_artifact_bytes(chosen),
                "model_identity": chosen["model_identity"],
                "settings": settings, "aggregate": chosen["aggregate"]},
            "scope": "512 calibration positions only; PPL reported, not a selection criterion; no heldout reselection",
            "format_table": calibration_table(reports),
            "baseline": next(entry for entry in evidence if entry["backend"] == "llama"), "evidence": evidence}
        write_json(args.selection, decision, exclusive=True)
        summary = decision
    else:
        selection = read_selection(args.selection, corpus_hash)
        selection_hash = file_hash(args.selection)
        if any(report["selection_sha256"] != selection_hash for report in reports):
            raise ValueError("Heldout report was not evaluated after this exact calibration decision")
        if any(report["oracle_identity"] != selection["oracle_identity"] for report in reports):
            raise ValueError("Heldout oracle differs from the calibration oracle")
        summary = {"split": "heldout", "corpus_sha256": corpus_hash, "selection_sha256": selection_hash,
            "selection": selection, "evidence": evidence, "comparison": heldout_comparison(reports, selection)}
    write_json(args.output, summary, exclusive=True)
    print(json.dumps(summary, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="stage", required=True)
    preparation = commands.add_parser("prepare", help="Fix attributed source/text/token selection before measurements")
    preparation.add_argument("--model", type=Path, required=True)
    preparation.add_argument("--output-dir", type=Path, default=Path("results/v2"))
    preparation.add_argument("--raw-dir", type=Path, default=Path("external/quality-v2/sources"))
    reference = commands.add_parser("oracle", help="BF16 storage/FP32 arithmetic Transformers, streaming whole-vocabulary projection")
    reference.add_argument("--model", type=Path, required=True)
    reference.add_argument("--split", choices=("all", "calibration", "heldout"), default="all")
    reference.add_argument("--head-chunk", type=int, default=1024)
    reference.add_argument("--threads", type=int, default=1)
    reference.add_argument("--output", type=Path, default=Path("results/v2/oracle-summary.json"))
    evaluation = commands.add_parser("evaluate", help="Run one real native or pinned Q8_0 candidate on the exact oracle positions")
    evaluation.add_argument("--backend", choices=("native", "llama"), default="native")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    evaluation.add_argument("--reader", type=Path, default=Path("build/llama-logits"))
    evaluation.add_argument("--artifact-manifest", type=Path, default=Path("results/llama-preparation.json"))
    evaluation.add_argument("--label", required=True)
    evaluation.add_argument("--threads", type=int, default=1)
    evaluation.add_argument("--kernel", choices=("auto", "scalar", "simd256", "simd512", "simd512x4", "vnni"), default="scalar")
    evaluation.add_argument("--kv", choices=("f16", "f32"), default="f16")
    evaluation.add_argument("--attention", choices=("blocked", "scalar"), default="blocked")
    evaluation.add_argument("--scheduler", choices=("pool", "openmp"), default="pool")
    evaluation.add_argument("--affinity", choices=("strict", "unpinned"), default="strict")
    evaluation.add_argument("--cpu-set")
    comparison = commands.add_parser("compare", help="Select on calibration, or report heldout KV/kernel/Q8_0 comparisons")
    comparison.add_argument("--reports", type=Path, nargs="+", required=True)
    for command in (evaluation, comparison):
        command.add_argument("--split", choices=("calibration", "heldout"), required=True)
        command.add_argument("--selection", type=Path, default=Path("results/v2/format-selection.json"),
            help="Exact calibration decision path; use selection.json only to read historical decisions")
        command.add_argument("--output", type=Path, required=True)
    for command in (reference, evaluation, comparison):
        command.add_argument("--corpus", type=Path, default=Path("results/v2/corpus.json"))
    for command in (reference, evaluation):
        command.add_argument("--raw-dir", type=Path, default=Path("external/quality-v2"))
    args = parser.parse_args()
    if hasattr(args, "threads") and args.threads < 1 or hasattr(args, "head_chunk") and args.head_chunk < 1:
        parser.error("threads and head-chunk must be positive")
    if args.stage == "prepare":
        prepare(args.model, args.output_dir, args.raw_dir)
        print(json.dumps({"corpus_sha256": file_hash(args.output_dir / "corpus.json")}))
    elif args.stage == "oracle":
        oracle(args)
    elif args.stage == "evaluate":
        evaluate(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
