"""Long-context extension of the existing heldout KL metric, with sparse heads.

Run prepare first, then oracle-layer 0..23 (one model-layer job at a time),
oracle-head, native once per cache, and compare. No performance times are saved.
"""
import argparse
import json
import subprocess
from pathlib import Path

from tools.corpus_v2 import load_manifest
from tools.download_model import MODEL_ID, REVISION, file_hash, verify_snapshot
from tools.quality_v2 import position_metric, summarize
from tools.streamed_oracle import gather_embeddings, load_layer, write_logits


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(corpus):
    manifest = load_manifest(corpus)
    heldout = [window for window in manifest["windows"] if window["split"] == "heldout"]
    tokens = [token for window in heldout for token in window["tokens"]]
    positions = list(range(2048, 2080)) + list(range(4064, 4096))
    return {"model_id": MODEL_ID, "revision": REVISION,
            "corpus_sha256": file_hash(corpus), "source": manifest["sources"]["heldout"],
            "selection": "Concatenate all eight existing heldout windows in source order, with no resets. Score 2048..2079 and 4064..4095 (zero-based input positions). No prompt or cache tuning.",
            "tokens": tokens[:4097], "positions": positions,
            "reference": "Original BF16 snapshot widened to FP32; unmodified Transformers eager Qwen2 decoder layers; causal 128-token chunks; F32 KV.",
            "metric": "tools.quality_v2.position_metric and summarize; row at p predicts token p+1"}


def configured(model, threads):
    import torch
    from transformers import Qwen2Config

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = Qwen2Config.from_pretrained(str(model), local_files_only=True)
    config._attn_implementation = "eager"
    if config.model_type != "qwen2" or not config.tie_word_embeddings or config.use_sliding_window or config.rope_scaling:
        raise ValueError("Expected tied full-attention Qwen2 with standard RoPE")
    return config


def oracle_layer(args, plan):
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    config = configured(args.model, args.threads)
    if not 0 <= args.layer < config.num_hidden_layers:
        raise ValueError("Layer outside model")
    args.raw.mkdir(parents=True, exist_ok=True)
    source = args.raw / f"hidden-{args.layer:02d}.pt"
    target = args.raw / f"hidden-{args.layer + 1:02d}.pt"
    with torch.inference_mode():
        if args.layer == 0:
            identity = verify_snapshot(args.model)
            save(args.raw / "oracle-source.json", identity)
            hidden = gather_embeddings(args.model / "model.safetensors", config, plan["tokens"][:-1], 1024)
        else:
            hidden = torch.load(source, weights_only=True)
        if hidden.shape != (1, len(plan["tokens"]) - 1, config.hidden_size) or hidden.dtype != torch.float32:
            raise ValueError("Hidden state shape/dtype differs from plan")
        positions = torch.arange(hidden.shape[1])
        embeddings = Qwen2RotaryEmbedding(config).eval()(hidden, positions[None])
        layer = load_layer(args.model / "model.safetensors", config, args.layer)
        cache = DynamicCache()
        output = torch.empty_like(hidden)
        for start in range(0, hidden.shape[1], args.chunk):
            end = min(hidden.shape[1], start + args.chunk)
            # Each query sees its own position and all earlier keys, never future tokens.
            allowed = positions[:end][None, :] <= positions[start:end][:, None]
            mask = torch.zeros((end - start, end), dtype=torch.float32)
            mask.masked_fill_(~allowed, torch.finfo(torch.float32).min)
            output[:, start:end] = layer(
                hidden[:, start:end], attention_mask=mask[None, None],
                position_ids=positions[start:end][None], past_key_values=cache,
                use_cache=True, cache_position=positions[start:end],
                position_embeddings=tuple(value[:, start:end] for value in embeddings))
        torch.save(output, target)
    if source.exists():
        source.unlink()


def oracle_head(args, plan):
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from tools.reference import versions
    from tools.streamed_oracle import widened_tensor

    config = configured(args.model, args.threads)
    with torch.inference_mode():
        hidden = torch.load(args.raw / f"hidden-{config.num_hidden_layers:02d}.pt", weights_only=True)
        norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
        norm.load_state_dict({"weight": widened_tensor(args.model / "model.safetensors", "model.norm.weight")}, assign=True)
        logits = args.raw / "oracle.bin"
        write_logits(args.model, config, norm(hidden), plan["positions"], logits, 1024)
    save(args.raw / "oracle.json", {"model_id": MODEL_ID, "revision": REVISION,
         "plan_sha256": file_hash(args.plan), "logits_sha256": file_hash(logits),
         "shape": [len(plan["positions"]), config.vocab_size], "chunk": args.chunk,
         "threads": args.threads, "versions": versions(),
         "source": json.loads((args.raw / "oracle-source.json").read_text())})


def native(args, plan):
    prefix = args.raw / args.kv
    args.raw.mkdir(parents=True, exist_ok=True)
    command = [str(args.engine.resolve()), "logits", "--model", str(args.model.resolve()),
               "--tokens", ",".join(map(str, plan["tokens"][:-1])),
               "--logits-positions", ",".join(map(str, plan["positions"])),
               "--batch", "8", "--threads", str(args.threads), "--kernel", args.kernel,
               "--kv", args.kv, "--output", str(prefix.resolve())]
    subprocess.run(command, check=True)
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    metadata["model"] = "$INT8_MODEL"
    metadata["plan_sha256"] = file_hash(args.plan)
    metadata["binary_sha256"] = file_hash(args.engine)
    metadata["weights_sha256"] = file_hash(args.model / "model.safetensors")
    metadata["config_sha256"] = file_hash(args.model / "config.json")
    metadata["logits_sha256"] = file_hash(prefix.with_suffix(".bin"))
    save(prefix.with_suffix(".json"), metadata)


def compare(args, plan):
    import numpy as np

    oracle = json.loads((args.raw / "oracle.json").read_text())
    if oracle["plan_sha256"] != file_hash(args.plan) or oracle["logits_sha256"] != file_hash(args.raw / "oracle.bin"):
        raise ValueError("Oracle inputs/logits changed")
    shape = tuple(oracle["shape"])
    reference = np.memmap(args.raw / "oracle.bin", mode="r", dtype="<f4", shape=shape)
    result = {"plan_sha256": file_hash(args.plan), "model_id": MODEL_ID, "revision": REVISION,
              "positions": plan["positions"], "oracle": oracle, "timing": "not measured", "caches": {}}
    native_reference = np.memmap(args.raw / "f32.bin", mode="r", dtype="<f4", shape=shape)
    identities = []
    for cache in ("f32", "f16", "i8", "i8-centered"):
        metadata = json.loads((args.raw / f"{cache}.json").read_text())
        if (metadata["shape"] != list(shape) or metadata["logit_positions"] != plan["positions"] or
                metadata["plan_sha256"] != file_hash(args.plan) or metadata["logits_sha256"] != file_hash(args.raw / f"{cache}.bin") or
                metadata["kv_dtype"] != ("i8_centered" if cache == "i8-centered" else cache)):
            raise ValueError("Native output differs from plan")
        identities.append(tuple(metadata[key] for key in ("binary_sha256", "weights_sha256", "config_sha256", "kernel", "threads", "cpu_set")))
        candidate = np.memmap(args.raw / f"{cache}.bin", mode="r", dtype="<f4", shape=shape)
        rows, isolated = [], []
        for row, position in enumerate(plan["positions"]):
            target = plan["tokens"][position + 1]
            rows.append({"position": position, **position_metric(reference[row], candidate[row], target)})
            isolated.append({"position": position, **position_metric(native_reference[row], candidate[row], target)})
        keep = ("binary_sha256", "weights_sha256", "config_sha256", "kernel", "threads", "cpu_set", "kv_dtype", "kv_cache_bytes", "kv_capacity")
        item = {"settings": {key: metadata[key] for key in keep}, "aggregate": summarize(rows),
                "versus_native_f32": summarize(isolated), "rows": rows,
                "context_bands": {"2k": summarize(rows[:32]), "4k": summarize(rows[32:])},
                "logits_sha256": metadata["logits_sha256"]}
        for key in ("kv_key_mean_prefix", "kv_centered_warmup_keys"):
            if key in metadata:
                item[key] = metadata[key]
        result["caches"][cache] = item
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Native candidates must use identical weights, binary and execution settings")
    baseline = result["caches"]["f16"]["settings"]["kv_cache_bytes"]
    for item in result["caches"].values():
        item["bytes_saved_vs_f16"] = baseline - item["settings"]["kv_cache_bytes"]
        item["fraction_saved_vs_f16"] = item["bytes_saved_vs_f16"] / baseline
    save(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "oracle-layer", "oracle-head", "native", "compare"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--corpus", type=Path, default=Path("results/v2/corpus.json"))
    parser.add_argument("--plan", type=Path, default=Path("results/v3/long-context-inputs.json"))
    parser.add_argument("--raw", type=Path, default=Path("external/quality-v3"))
    parser.add_argument("--output", type=Path, default=Path("results/v3/kv-quality.json"))
    parser.add_argument("--layer", type=int)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--threads", type=int, choices=range(1, 3), default=2)
    parser.add_argument("--kernel", default="vnni16")
    parser.add_argument("--kv", choices=("f32", "f16", "i8", "i8-centered"), default="f16")
    args = parser.parse_args()
    if not 1 <= args.chunk <= 128:
        parser.error("chunk must be 1..128")
    if args.command == "prepare":
        save(args.plan, prepare(args.corpus))
        return
    plan = json.loads(args.plan.read_text())
    if plan != prepare(args.corpus):
        raise ValueError("Long-context inputs differ from the fixed heldout extension")
    {"oracle-layer": oracle_layer, "oracle-head": oracle_head, "native": native, "compare": compare}[args.command](args, plan)


if __name__ == "__main__":
    main()
