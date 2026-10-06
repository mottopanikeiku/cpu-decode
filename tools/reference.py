#!/usr/bin/env python3
"""Transformers FP32 arithmetic on BF16 weights, then native-engine comparison.

The oracle worker exits before any engine is loaded. No full FP32 weight copy
is retained. Raw logits stay outside git; small per-position metrics and token
IDs are written to JSON with portable artifact locations.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import gc
import importlib.metadata
import json
import subprocess
import types
from pathlib import Path

from tools.download_model import MODEL_ID, REVISION, file_hash, verify_snapshot
from tools.portable import portable


def versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "tokenizers", "huggingface-hub", "numpy")}


def load_oracle(model_path: Path, threads: int, arithmetic: str = "fp32"):
    """Use unmodified Qwen2 blocks, changing only weight storage conversion."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="eager", low_cpu_mem_usage=True).eval()
    if model.config.model_type != "qwen2" or not model.config.tie_word_embeddings:
        raise ValueError("Expected the pinned tied-head Qwen2 model")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("Embedding/head weights must be tied")
    if arithmetic == "fp32":
        def linear(module, inputs):
            # Only this projection is widened, and released immediately after GEMM.
            bias = module.bias.float() if module.bias is not None else None
            return functional.linear(inputs.float(), module.weight.float(), bias)

        def embedding(module, ids):
            # Gather BF16 rows first; never cast the entire vocabulary table.
            return functional.embedding(ids, module.weight, module.padding_idx).float()

        for module in model.model.modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = types.MethodType(linear, module)
        model.model.embed_tokens.forward = types.MethodType(embedding, model.model.embed_tokens)
        # RMSNorm weight promotion is naturally FP32 with FP32 hidden states.
        # RotaryEmbedding computes frequencies in FP32 in Transformers itself.
    return model


def project_last(model, hidden, arithmetic: str, head_chunk: int):
    import torch
    import torch.nn.functional as functional

    hidden = hidden[:, -1:, :]
    if arithmetic == "bf16":
        return model.lm_head(hidden).float()[0, 0]
    if hidden.dtype != torch.float32:
        raise ValueError("FP32 oracle unexpectedly produced non-FP32 activations")
    output = torch.empty(model.config.vocab_size, dtype=torch.float32)
    for start in range(0, model.config.vocab_size, head_chunk):
        end = min(start + head_chunk, model.config.vocab_size)
        output[start:end] = functional.linear(hidden, model.lm_head.weight[start:end].float())[0, 0]
    return output


def oracle_worker(args) -> None:
    source_manifest = verify_snapshot(args.model)
    import numpy as np
    import torch

    token_data = json.loads(args.tokens_file.read_text())
    model = load_oracle(args.model, args.threads, args.arithmetic)
    records = []
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for case in token_data["prompts"]:
            tokens = list(case["tokens"])
            count = len(tokens) + args.steps - 1
            path = args.raw_dir / f"{case['id']}-reference.bin"
            logits = np.memmap(path, dtype="<f4", mode="w+", shape=(count, model.config.vocab_size))
            cache = None
            generated = []
            for position in range(count):
                token = tokens[position] if position < len(tokens) else generated[position - len(tokens)]
                result = model.model(input_ids=torch.tensor([[token]], dtype=torch.long), past_key_values=cache, use_cache=True, return_dict=True)
                cache = result.past_key_values
                row = project_last(model, result.last_hidden_state, args.arithmetic, args.head_chunk)
                logits[position] = row.numpy()
                if position >= len(tokens) - 1:
                    generated.append(int(row.argmax()))
                del result, row
            logits.flush()
            del logits, cache
            records.append({"id": case["id"], "prompt_tokens": tokens, "generated_tokens": generated, "tokens": tokens + generated[:-1], "shape": [count, model.config.vocab_size], "logits": path.name, "sha256": file_hash(path)})
    del model
    gc.collect()
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "verified_source": source_manifest, "weight_storage": "bfloat16", "arithmetic": args.arithmetic, "attention": "Transformers eager", "kv_dtype": "float32" if args.arithmetic == "fp32" else "bfloat16", "head_chunk_rows": args.head_chunk if args.arithmetic == "fp32" else None, "threads": args.threads, "steps": args.steps, "stop_on_eos": False, "versions": versions(), "prompts": records}
    (args.raw_dir / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")


def position_metrics(reference, candidate, atol: float, rtol: float) -> list[dict]:
    """KL is KL(reference || candidate) over the complete vocabulary, in nats."""
    import numpy as np

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Logit arrays must have the same [positions, vocabulary] shape")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Non-finite logits")
    metrics = []
    for position, (ref, actual) in enumerate(zip(reference, candidate, strict=True)):
        ref = ref.astype(np.float64)
        actual = actual.astype(np.float64)
        logp = ref - ref.max()
        logq = actual - actual.max()
        logp -= np.log(np.exp(logp).sum())
        logq -= np.log(np.exp(logq).sum())
        difference = actual - ref
        metrics.append({"position": position, "reference_top1": int(ref.argmax()), "candidate_top1": int(actual.argmax()), "top1_match": bool(ref.argmax() == actual.argmax()), "kl_reference_candidate_nats": max(0.0, float(np.sum(np.exp(logp) * (logp - logq)))), "max_abs_error": float(np.abs(difference).max()), "rmse": float(np.sqrt(np.mean(difference * difference))), "allclose": bool(np.all(np.abs(difference) <= atol + rtol * np.abs(ref)))})
    return metrics


def compare_engine(args, oracle: dict, model: Path, label: str) -> dict:
    import numpy as np

    cases = []
    for case in oracle["prompts"]:
        prefix = args.raw_dir / f"{case['id']}-{label}"
        common = [str(args.engine.resolve()), "--model", str(model.resolve()), "--threads", str(args.threads), "--kernel", args.kernel, "--kv", "f32"]
        command = [common[0], "logits", *common[1:], "--tokens", ",".join(map(str, case["tokens"])), "--output", str(prefix)]
        subprocess.run(command, check=True)
        metadata = json.loads(prefix.with_suffix(".json").read_text())
        if metadata["shape"] != case["shape"] or metadata["tokens"] != case["tokens"]:
            raise ValueError("Engine logits metadata does not match reference inputs")
        reference = np.memmap(args.raw_dir / case["logits"], dtype="<f4", mode="r", shape=tuple(case["shape"]))
        candidate = np.memmap(prefix.with_suffix(".bin"), dtype="<f4", mode="r", shape=tuple(case["shape"]))
        metrics = position_metrics(reference, candidate, args.atol, args.rtol)
        del reference, candidate
        generated_path = args.raw_dir / f"{case['id']}-{label}-generated.json"
        generate_command = [common[0], "generate", *common[1:], "--tokens", ",".join(map(str, case["prompt_tokens"])), "--steps", str(args.steps), "--output", str(generated_path)]
        subprocess.run(generate_command, check=True)
        generation = json.loads(generated_path.read_text())
        if generation["prompt_tokens"] != case["prompt_tokens"] or len(generation["generated_tokens"]) != args.steps:
            raise ValueError("Engine generation did not execute the exact fixed token count")
        top1_fraction = sum(row["top1_match"] for row in metrics) / len(metrics)
        maximum_kl = max(row["kl_reference_candidate_nats"] for row in metrics)
        exact_greedy = generation["generated_tokens"] == case["generated_tokens"]
        if label == "bf16-fp32":
            passed = all(row["allclose"] for row in metrics) and exact_greedy
        else:
            # Quantization quality is a measurement, not an assumed guarantee.
            passed = (args.quant_max_kl is None or maximum_kl <= args.quant_max_kl) and (args.quant_min_top1 is None or top1_fraction >= args.quant_min_top1)
        cases.append({"id": case["id"], "prompt_tokens": case["prompt_tokens"], "teacher_forced_tokens": case["tokens"], "reference_generated_tokens": case["generated_tokens"], "candidate_generated_tokens": generation["generated_tokens"], "exact_greedy_match": exact_greedy, "top1_agreement": top1_fraction, "max_kl_reference_candidate_nats": maximum_kl, "mean_kl_reference_candidate_nats": sum(row["kl_reference_candidate_nats"] for row in metrics) / len(metrics), "passed": passed, "positions": metrics, "engine_settings": metadata, "commands": [command, generate_command]})
    return {"label": label, "model_directory": str(model.resolve()), "quality_reporting_only": label == "int8" and args.quant_max_kl is None and args.quant_min_top1 is None, "passed": all(case["passed"] for case in cases), "prompts": cases}


def run_comparison(args) -> dict:
    if args.tokens_file is None:
        args.tokens_file = args.raw_dir / "tokens.json"
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        # Isolate Transformers tokenization so the driver stays small.
        subprocess.run([sys.executable, "-m", "tools.tokenize", "--model", str(args.model.resolve()), "--prompts", str(args.prompts.resolve()), "--output", str(args.tokens_file.resolve())], check=True, cwd=Path(__file__).resolve().parents[1], stdout=subprocess.DEVNULL)
    worker_command = [sys.executable, "-m", "tools.reference", "--oracle-only", "--model", str(args.model.resolve()), "--tokens-file", str(args.tokens_file.resolve()), "--raw-dir", str(args.raw_dir.resolve()), "--arithmetic", args.arithmetic, "--threads", str(args.threads), "--steps", str(args.steps), "--head-chunk", str(args.head_chunk)]
    subprocess.run(worker_command, check=True, cwd=Path(__file__).resolve().parents[1])
    oracle = json.loads((args.raw_dir / "reference.json").read_text())
    comparisons = [compare_engine(args, oracle, args.model, "bf16-fp32")]
    if args.quant_model is not None:
        comparisons.append(compare_engine(args, oracle, args.quant_model, "int8"))
    return {"oracle": oracle, "tolerance": {"atol": args.atol, "rtol": args.rtol, "unquantized_requires_exact_greedy": True, "quant_max_position_kl_nats": args.quant_max_kl, "quant_min_top1_agreement": args.quant_min_top1, "quant_greedy_match_is_diagnostic": True}, "passed": all(item["passed"] for item in comparisons), "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--quant-model", type=Path)
    parser.add_argument("--prompts", type=Path, default=Path("configs/prompts.json"))
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/correctness.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("external/reference"))
    parser.add_argument("--arithmetic", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--head-chunk", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--kernel", choices=("scalar", "simd256", "simd512", "simd512x4"), default="scalar")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--quant-max-kl", type=float, help="Optional caller-chosen maximum per-position KL; otherwise reporting only")
    parser.add_argument("--quant-min-top1", type=float, help="Optional caller-chosen minimum top1 agreement; otherwise reporting only")
    parser.add_argument("--oracle-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1 or args.head_chunk < 1:
        parser.error("steps, threads and head-chunk must be positive")
    args.atol = args.atol if args.atol is not None else (0.003 if args.arithmetic == "fp32" else 0.5)
    args.rtol = args.rtol if args.rtol is not None else (0.0003 if args.arithmetic == "fp32" else 0.03)
    if args.oracle_only:
        if args.tokens_file is None:
            parser.error("oracle worker requires --tokens-file")
        oracle_worker(args)
        return
    result = run_comparison(args)
    locations = {args.model.resolve(): "$MODEL", args.engine.resolve(): "build/cpu-decode", args.raw_dir.resolve(): "$RAW"}
    if args.quant_model is not None:
        locations[args.quant_model.resolve()] = "$INT8"
    result = portable(result, locations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
