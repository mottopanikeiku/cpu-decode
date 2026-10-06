"""Aggregate, without discarding, the committed per-position correctness data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(raw: dict) -> dict:
    unquantized, quantized = raw["comparisons"]
    greedy_pairs = [(p["reference_generated_tokens"], p["candidate_generated_tokens"]) for p in unquantized["prompts"]]
    all_fp32 = [r for p in unquantized["prompts"] for r in p["positions"]]
    result = {"source_passed": raw["passed"], "oracle": raw["oracle"]["arithmetic"], "tolerance": raw["tolerance"],
              "fixed_prompts": len(unquantized["prompts"]),
              "unquantized": {"max_abs_logit_error": max(r["max_abs_error"] for r in all_fp32),
                              "positions": len(all_fp32), "allclose": all(r["allclose"] for r in all_fp32),
                              "greedy_tokens": sum(len(a) for a, _ in greedy_pairs),
                              "greedy_matching_tokens": sum(sum(x == y for x, y in zip(a, b, strict=True)) for a, b in greedy_pairs),
                              "all_greedy_sequences_identical": all(a == b for a, b in greedy_pairs)}, "quantized": {}}
    for scope in ["prompt_only", "prompt_and_reference_continuation"]:
        rows = [r for p in quantized["prompts"] for r in
                (p["positions"][:len(p["prompt_tokens"])] if scope == "prompt_only" else p["positions"])]
        result["quantized"][scope] = {"positions": len(rows), "top1_matches": sum(r["top1_match"] for r in rows),
                                     "top1_agreement": sum(r["top1_match"] for r in rows) / len(rows),
                                     "mean_kl_reference_candidate_nats": sum(r["kl_reference_candidate_nats"] for r in rows) / len(rows),
                                     "max_kl_reference_candidate_nats": max(r["kl_reference_candidate_nats"] for r in rows)}
    result["quantized"]["greedy_sequences_matching"] = sum(p["exact_greedy_match"] for p in quantized["prompts"])
    result["quantized"]["quality_is_reporting_only"] = quantized["quality_reporting_only"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/correctness.json"))
    parser.add_argument("--output", type=Path, default=Path("results/quality-summary.json"))
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    result["source"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
