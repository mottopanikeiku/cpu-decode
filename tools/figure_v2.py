"""Draw small SVG decode/context curves using only measured v2 summary values."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from tools.measure_v2 import ROOT


def figure(summary: dict) -> str:
    rows = summary["results"]
    if not rows:
        raise ValueError("no measured cells to plot")
    threads = sorted({r["threads"] for r in rows})
    width, height = max(600, 210 * len(threads) + 40), 360
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
             '<title id="title">CPU decode speed versus initial context</title>',
             '<desc id="description">Median native and best measured llama.cpp rates, with a format-specific read-bandwidth ceiling. Error bars show measured minimum and maximum. Asterisks mark spreads above five percent.</desc>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font:12px sans-serif;fill:#222}.heading{font-size:16px;font-weight:600}.axis{stroke:#999;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>']
    def text(x, y, value, extra=""):
        lines.append(f'<text x="{x:.1f}" y="{y:.1f}" {extra}>{html.escape(str(value))}</text>')
    text(20, 24, "CPU decode: median tokens/s by initial context", 'class="heading"')
    caption = ("Complete requested subset — not fifteen-cell target acceptance"
               if summary.get("matrix_scope") == "explicit-subset" and summary["complete_requested_matrix"]
               else "Full final matrix" if summary["complete_final_matrix"]
               else "Partial/development data — not a full-matrix result")
    text(20, 45, caption)
    series = [("native_tps", "#1766ac", "native", False),
              ("best_baseline_tps", "#c54e00", "best llama.cpp", False),
              ("read_ceiling_tps", "#555", "read ceiling (EXT)", True)]
    for index, (_, color, label, dashed) in enumerate(series):
        x = 20 + index * 210
        lines.append(f'<path d="M{x} 64h24" stroke="{color}" stroke-width="2" fill="none"' + (' stroke-dasharray="5 3"' if dashed else '') + '/>')
        text(x + 30, 68, label)
    panel_width = (width - 40) / len(threads)
    for index, thread in enumerate(threads):
        cell_rows = sorted([r for r in rows if r["threads"] == thread], key=lambda r: r["context"])
        left, top, bottom = 42 + index * panel_width, 102, 265
        right = left + panel_width - 52
        values = [r[key]["max"] for r in cell_rows for key in ["native_tps", "best_baseline_tps"]]
        values += [r["read_ceiling_tps"] for r in cell_rows if r["read_ceiling_tps"] is not None]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid plotted rate")
        maximum = max(values) * 1.12
        def x(context):
            return left + (math.log2(context) - 7) / 5 * (right - left)
        def y(rate):
            return bottom - rate / maximum * (bottom - top)
        text((left + right) / 2, 93, f"{thread} thread{'s' if thread != 1 else ''}", 'text-anchor="middle"')
        for tick in range(4):
            rate = maximum * tick / 3
            yy = y(rate)
            lines.append(f'<path d="M{left:.1f} {yy:.1f}H{right:.1f}" class="grid"/>')
            text(left - 5, yy + 4, f"{rate:.0f}", 'text-anchor="end"')
        lines.append(f'<path d="M{left:.1f} {top}V{bottom}H{right:.1f}" class="axis" fill="none"/>')
        for context in [128, 1024, 4096]:
            text(x(context), bottom + 18, str(context), 'text-anchor="middle"')
        for key, color, _, dashed in series:
            points = []
            for row in cell_rows:
                rate = row[key] if dashed else row[key]["median"]
                if rate is None:
                    continue
                xx, yy = x(row["context"]), y(rate)
                points.append(f"{xx:.2f},{yy:.2f}")
                if not dashed:
                    stats = row[key]
                    lines.append(f'<path d="M{xx:.2f} {y(stats["min"]):.2f}V{y(stats["max"]):.2f}" stroke="{color}"/>')
                    lines.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="3" fill="{color}"/>')
                    if stats["noisy_over_5_percent"]:
                        text(xx + 4, yy - 6, "*")
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"' + (' stroke-dasharray="5 3"' if dashed else '') + '/>')
    text(width / 2, 306, "Initial context (tokens, log₂ spacing); rates exclude loading and prefill", 'text-anchor="middle"')
    text(20, 327, "Whiskers: min–max. * spread >5%. Winner flags, sample counts, byte bound and raw logs: summary.json.")
    text(20, 347, "Best llama.cpp includes full-allowed defaults; differing winner CPU sets are disclosed in summary.json.")
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/v2/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("results/v2/decode.svg"))
    args = parser.parse_args()
    result_root = ROOT / "results"
    if args.output.resolve().is_relative_to(result_root) and not args.output.resolve().is_relative_to(result_root / "v2"):
        parser.error("v2 figure must not overwrite v1 results")
    output = figure(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)


if __name__ == "__main__":
    main()
