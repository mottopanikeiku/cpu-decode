"""Tokenize text at the edge; all model arithmetic runs in the C++ engine.

Run under pp-run heavy. Use --chat for the pinned Qwen chat-template tokens.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="BF16 snapshot or quantized engine directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Original pinned Hugging Face snapshot")
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--kernel", choices=["scalar", "simd256", "simd512", "simd512x4"], default="simd512x4")
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    prompt = args.prompt
    if args.chat:
        prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        parser.error("prompt must produce at least one token")
    completed = subprocess.run([str(args.engine), "generate", "--model", str(args.model),
                                "--tokens", ",".join(map(str, ids)), "--steps", str(args.steps),
                                "--threads", str(args.threads), "--kernel", args.kernel],
                               check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)
    print(tokenizer.decode(result["generated_tokens"], skip_special_tokens=False))
    print(json.dumps({"prompt_tokens": ids, "generated_tokens": result["generated_tokens"]}))


if __name__ == "__main__":
    main()
