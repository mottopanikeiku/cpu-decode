"""Aggregate, without discarding, the committed per-position correctness data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.reference import UNQUANTIZED

SCOPES = ("prompt_only", "prompt_and_reference_continuation")


def summarize_quantized(comparison: dict) -> dict:
    result = {"kv_dtype": comparison["kv_dtype"]}
    for scope in SCOPES:
        rows = [r for p in comparison["prompts"] for r in
                (p["positions"][:len(p["prompt_tokens"])] if scope == "prompt_only" else p["positions"])]
        result[scope] = {"positions": len(rows), "top1_matches": sum(r["top1_match"] for r in rows),
                         "top1_agreement": sum(r["top1_match"] for r in rows) / len(rows),
                         "mean_kl_reference_candidate_nats": sum(r["kl_reference_candidate_nats"] for r in rows) / len(rows),
                         "max_kl_reference_candidate_nats": max(r["kl_reference_candidate_nats"] for r in rows)}
    result["greedy_sequences_matching"] = sum(p["exact_greedy_match"] for p in comparison["prompts"])
    result["quality_is_reporting_only"] = comparison["quality_reporting_only"]
    return result


def summarize(raw: dict) -> dict:
    by_label = {comparison["label"]: comparison for comparison in raw["comparisons"]}
    if len(by_label) != len(raw["comparisons"]) or UNQUANTIZED not in by_label:
        raise ValueError(f"Correctness data needs unique labels including {UNQUANTIZED!r}")
    unquantized = by_label.pop(UNQUANTIZED)
    greedy_pairs = [(p["reference_generated_tokens"], p["candidate_generated_tokens"]) for p in unquantized["prompts"]]
    all_fp32 = [r for p in unquantized["prompts"] for r in p["positions"]]
    return {"source_passed": raw["passed"], "oracle": raw["oracle"]["arithmetic"], "tolerance": raw["tolerance"],
            "fixed_prompts": len(unquantized["prompts"]),
            "unquantized": {"kv_dtype": unquantized["kv_dtype"], "max_abs_logit_error": max(r["max_abs_error"] for r in all_fp32),
                            "positions": len(all_fp32), "allclose": all(r["allclose"] for r in all_fp32),
                            "greedy_tokens": sum(len(a) for a, _ in greedy_pairs),
                            "greedy_matching_tokens": sum(sum(x == y for x, y in zip(a, b, strict=True)) for a, b in greedy_pairs),
                            "all_greedy_sequences_identical": all(a == b for a, b in greedy_pairs)},
            "quantized": {label: summarize_quantized(comparison) for label, comparison in by_label.items()}}


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
