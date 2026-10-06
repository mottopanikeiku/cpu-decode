#!/usr/bin/env python3
"""Single-stream naive Transformers BF16 eager decode, excluding prefill.

Only pp-run bench may execute this timing script. Chunked untimed prefill
bounds attention memory at 4096 context. Each timed step includes a full
backbone forward, tied vocabulary head and greedy argmax, with a KV cache.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import statistics
import time
from pathlib import Path

from tools.download_model import MODEL_ID, REVISION
from tools.reference import load_oracle, project_last, versions
from tools.tokenize import repeat_tokens


def require_bench() -> None:
    pid = os.getpid()
    for _ in range(64):
        proc = Path("/proc") / str(pid)
        try:
            command = (proc / "cmdline").read_bytes().split(b"\0")
            if any(Path(arg.decode(errors="replace")).name == "pp-run" for arg in command if arg) and b"bench" in command:
                return
            status = (proc / "status").read_text().splitlines()
            pid = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
        except (OSError, StopIteration, ValueError):
            break
        if pid <= 1:
            break
    raise RuntimeError("Timings must run under /home/alp/Projects/profile-program/bin/pp-run bench")


def benchmark(args) -> dict:
    require_bench()
    os.nice(19)
    import torch

    seed = [int(token) for token in args.tokens.split(",")]
    context_tokens = repeat_tokens(seed, args.context)
    model = load_oracle(args.model, args.threads, "bf16")
    samples = []
    with torch.inference_mode():
        # One warm-up decode uses its own KV cache; never reuse it in samples.
        for repeat in range(-1, args.repeats):
            cache = None
            for start in range(0, len(context_tokens), args.prefill_chunk):
                ids = torch.tensor([context_tokens[start:start + args.prefill_chunk]], dtype=torch.long)
                result = model.model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
                cache = result.past_key_values
                hidden = result.last_hidden_state[:, -1:, :]
                del result
            next_token = int(project_last(model, hidden, "bf16", 1024).argmax())
            del hidden
            decoded = []
            begin = time.perf_counter_ns()
            for _ in range(args.steps):
                decoded.append(next_token)
                result = model.model(input_ids=torch.tensor([[next_token]], dtype=torch.long), past_key_values=cache, use_cache=True, return_dict=True)
                cache = result.past_key_values
                next_token = int(project_last(model, result.last_hidden_state, "bf16", 1024).argmax())
                del result
            elapsed = (time.perf_counter_ns() - begin) / 1e9
            del cache
            if repeat >= 0:
                samples.append({"repeat": repeat, "seconds": elapsed, "tokens_per_second": args.steps / elapsed, "decoded_token_ids": decoded, "next_token_id": next_token})
    rates = [sample["tokens_per_second"] for sample in samples]
    return {"engine": "transformers-eager-bf16", "model_id": MODEL_ID, "revision": REVISION, "model_directory": str(args.model.resolve()), "weight_storage": "bfloat16", "linear_and_kv_dtype": "bfloat16", "normalization_and_softmax_accumulation": "float32 (Transformers Qwen2)", "quantization": "none", "attention": "eager", "batch_size": 1, "threads": args.threads, "interop_threads": 1, "context": args.context, "steps": args.steps, "repeats": args.repeats, "warmup_repeats": 1, "prefill_chunk": args.prefill_chunk, "seed_token_ids": seed, "prompt_token_ids": context_tokens, "decode_policy": "greedy argmax, fixed step count, no EOS stop", "timed_operations": "Each step: forward one previously selected token, KV update, full tied head, argmax. Prefill and first token selection excluded.", "versions": versions(), "command": [sys.executable, *sys.argv], "samples": samples, "median_tokens_per_second": statistics.median(rates), "min_tokens_per_second": min(rates), "max_tokens_per_second": max(rates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", required=True, help="Same comma-separated seed IDs supplied to native bench")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--output", type=Path, help="Optional file; JSON is always printed to stdout")
    args = parser.parse_args()
    if min(args.context, args.threads, args.steps, args.repeats, args.prefill_chunk) < 1:
        parser.error("context, threads, steps, repeats and prefill-chunk must be positive")
    result = benchmark(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
