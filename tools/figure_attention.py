"""Plot retained development attention stages, without treating them as final timings."""
import argparse
import html
import json
from pathlib import Path
import statistics

STAGES = [("before", "Original K layout"), ("transposed", "Transposed K"),
          ("masked-exp", "Masked SIMD tails"), ("parallel-merge", "Parallel head merge"),
          ("claim1", "Singleton claims (rejected)")]


def summarize(directory: Path) -> list[dict]:
    rows = []
    for stage, label in STAGES:
        path = next((directory / stage).glob("ablation-*.json"))
        raw = json.loads(path.read_text())
        samples = [sample for invocation in raw["invocations"] for sample in invocation["data"]["samples"]]
        rates = [sample["tokens_per_second"] for sample in samples]
        median = statistics.median(rates)
        rows.append({"stage": stage, "label": label, "file": str(path), "samples": len(samples),
                     "median_tokens_per_second": median, "min_tokens_per_second": min(rates),
                     "max_tokens_per_second": max(rates), "spread_percent": 100 * (max(rates) - min(rates)) / median,
                     "median_attention_ms_per_token": statistics.median(
                         sample["operation_seconds"]["attention"] / len(sample["step_seconds"]) * 1000 for sample in samples)})
    return rows


def figure(rows: list[dict]) -> str:
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="375" viewBox="0 0 1020 375">',
                '<rect width="1020" height="375" fill="#ffffff"/>',
                '<g font-family="sans-serif" fill="#17212b">',
                '<text x="24" y="30" font-size="19">Attention development checks: one Qwen 0.5B, g64f16</text>',
                '<text x="24" y="54" font-size="13">6 threads · initial context 4096 · F16 KV · 16 decode steps × 3 repeats × 2 processes</text>',
                '<text x="330" y="85" font-size="14">Decode tokens/s (min–max)</text>',
                '<text x="690" y="85" font-size="14">Attention ms/token</text>']
    for index, row in enumerate(rows):
        y = 111 + 43 * index
        color = "#a43d35" if row["stage"] == "claim1" else "#29719c"
        label = html.escape(row["label"])
        rate, low, high = (row[key] for key in ["median_tokens_per_second", "min_tokens_per_second", "max_tokens_per_second"])
        attention = row["median_attention_ms_per_token"]
        noise = "*" if row["spread_percent"] > 5 else ""
        elements += [f'<text x="24" y="{y + 17}" font-size="13">{label}</text>',
                     f'<rect x="330" y="{y}" width="{rate * 4:.3f}" height="24" fill="{color}"/>',
                     f'<path d="M {330 + low * 4:.3f} {y + 12} H {330 + high * 4:.3f}" stroke="#17212b" stroke-width="2"/>',
                     f'<text x="592" y="{y + 17}" font-size="13">{rate:.2f}{noise}</text>',
                     f'<rect x="690" y="{y}" width="{attention * 40:.3f}" height="24" fill="{color}"/>',
                     f'<text x="920" y="{y + 17}" font-size="13">{attention:.3f}</text>']
    elements += ['<text x="24" y="344" font-size="12">* rate spread &gt;5%. All samples retained. Stages were not interleaved; this is not a causal decomposition.</text>',
                 '<text x="24" y="364" font-size="12">No strongest-baseline or read-ceiling claim is established by these development checks.</text>', '</g></svg>']
    return "\n".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/v2/attention-stages"))
    parser.add_argument("--output", type=Path, default=Path("results/v2/attention-stages.svg"))
    args = parser.parse_args()
    rows = summarize(args.input)
    args.output.write_text(figure(rows))
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
