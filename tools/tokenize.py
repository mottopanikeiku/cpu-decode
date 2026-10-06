#!/usr/bin/env python3
"""Materialize pinned prompt token IDs once for every engine."""
import os
import sys

# Do not shadow Python's stdlib tokenize when this file is run directly.
if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from pathlib import Path

from tools.download_model import MODEL_ID, REVISION


def prompt_tokens(model: Path, prompts: Path) -> dict:
    from transformers import AutoTokenizer

    config = json.loads(prompts.read_text())
    if config["model_id"] != MODEL_ID or config["revision"] != REVISION:
        raise ValueError("Prompt configuration does not match the pinned model")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    cases = []
    for prompt in config["prompts"]:
        if prompt["chat"]:
            ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True, add_generation_prompt=True)
        else:
            ids = tokenizer.encode(prompt["text"], add_special_tokens=False)
        if not ids:
            raise ValueError(f"Empty prompt: {prompt['id']}")
        cases.append({**prompt, "tokens": ids})
    return {"model_id": MODEL_ID, "revision": REVISION, "greedy_steps": config["greedy_steps"], "prompts": cases, "benchmark": config["benchmark"], "tokenizer": "Transformers AutoTokenizer; no implicit BOS; explicit chat template only for chat case"}


def repeat_tokens(tokens: list[int], length: int) -> list[int]:
    if length < 1 or not tokens:
        raise ValueError("A positive context length and nonempty tokens are required")
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("configs/prompts.json"))
    parser.add_argument("--output", type=Path, default=Path("results/tokens.json"))
    parser.add_argument("--prompt-id", help="Also print this prompt's comma-separated IDs to stderr")
    args = parser.parse_args()
    result = prompt_tokens(args.model, args.prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.prompt_id:
        case = next(case for case in result["prompts"] if case["id"] == args.prompt_id)
        print(",".join(map(str, case["tokens"])), file=sys.stderr)


if __name__ == "__main__":
    main()
