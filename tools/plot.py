"""Draw decode tokens/s against context as a standalone SVG (stdlib only)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from tools.download_model import MODEL_ID

WIDTH, HEIGHT = 720, 400
LEFT, RIGHT, TOP, BOTTOM = 64, 470, 48, 344
SECONDARY = ("q4h8-f16", "q4_0-f16")


def nice_step(maximum: float) -> float:
    raw = maximum / 5
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)


def describe(label: str, engine: bool) -> str:
    model, _, kv = label.partition("-")
    return f"engine {model}, {kv} KV" if engine else f"llama.cpp {model.upper()}, {kv} KV"


def render(summary: dict, threads: int | None) -> str:
    primary = summary["primary_pair"]
    results = summary["results"]
    if not results:
        raise ValueError("summary has no primary results")
    if threads is None:
        largest = max(r["context"] for r in results)
        threads = max((r for r in results if r["context"] == largest), key=lambda r: r["engine_tps"]["median"])["threads"]
    rows = sorted((r for r in results if r["threads"] == threads), key=lambda r: r["context"])
    if not rows:
        raise ValueError(f"summary has no results at threads={threads}")
    series = [
        {"id": "engine", "label": describe(primary["engine"], True), "color": "#1f6feb", "width": 2.5, "dash": None,
         "points": [(r["context"], r["engine_tps"]["median"]) for r in rows]},
        {"id": "llama", "label": describe(primary["llama"], False), "color": "#d1495b", "width": 2.5, "dash": None,
         "points": [(r["context"], r["llama_tps"]["median"]) for r in rows]},
        {"id": "ceiling", "label": f"read-bandwidth ceiling ({primary['engine']})", "color": "#555555", "width": 1.5, "dash": "6 4",
         "points": [(r["context"], r["bandwidth_ceiling_tps"]) for r in rows]},
    ]
    pairs = sorted((p for p in summary.get("pairs", []) if (p["engine_label"], p["llama_label"]) == SECONDARY and p["threads"] == threads),
                   key=lambda p: p["context"])
    if pairs:
        series += [
            {"id": "engine-secondary", "label": describe(SECONDARY[0], True), "color": "#1f6feb", "width": 1.2, "dash": None,
             "points": [(p["context"], p["engine_tps"]["median"]) for p in pairs]},
            {"id": "llama-secondary", "label": describe(SECONDARY[1], False), "color": "#d1495b", "width": 1.2, "dash": None,
             "points": [(p["context"], p["llama_tps"]["median"]) for p in pairs]},
        ]
    contexts = sorted({c for s in series for c, _ in s["points"]})
    low, high = math.log2(contexts[0]), math.log2(contexts[-1])

    def x(context: float) -> float:
        if high == low:
            return (LEFT + RIGHT) / 2
        return LEFT + (math.log2(context) - low) / (high - low) * (RIGHT - LEFT)

    step = nice_step(max(v for s in series for _, v in s["points"]))
    top = step * math.ceil(max(v for s in series for _, v in s["points"]) / step)

    def y(value: float) -> float:
        return BOTTOM - value / top * (BOTTOM - TOP)

    def fmt(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" font-family="sans-serif" font-size="12">',
           f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
           f'<text x="{(LEFT + RIGHT) / 2}" y="24" text-anchor="middle" font-size="15" fill="#111111">'
           f'{escape(MODEL_ID.split("/")[1])} greedy decode, {threads} thread{"s" if threads != 1 else ""}</text>']
    ticks = round(top / step)
    for i in range(ticks + 1):
        value = i * step
        out.append(f'<line x1="{LEFT}" y1="{fmt(y(value))}" x2="{RIGHT}" y2="{fmt(y(value))}" stroke="#e5e5e5" stroke-width="1"/>')
        out.append(f'<text x="{LEFT - 6}" y="{fmt(y(value) + 4)}" text-anchor="end" fill="#333333">{fmt(value)}</text>')
    for context in contexts:
        out.append(f'<line x1="{fmt(x(context))}" y1="{BOTTOM}" x2="{fmt(x(context))}" y2="{BOTTOM + 5}" stroke="#333333" stroke-width="1"/>')
        out.append(f'<text x="{fmt(x(context))}" y="{BOTTOM + 19}" text-anchor="middle" fill="#333333">{context}</text>')
    out.append(f'<line id="x-axis" x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#333333" stroke-width="1"/>')
    out.append(f'<line id="y-axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="#333333" stroke-width="1"/>')
    out.append(f'<text x="{(LEFT + RIGHT) / 2}" y="{BOTTOM + 40}" text-anchor="middle" fill="#111111">context (tokens already in the KV cache, log scale)</text>')
    out.append(f'<text x="18" y="{(TOP + BOTTOM) / 2}" text-anchor="middle" fill="#111111" transform="rotate(-90 18 {(TOP + BOTTOM) / 2})">decode tokens/s</text>')
    for s in series:
        points = " ".join(f"{fmt(x(c))},{fmt(y(v))}" for c, v in s["points"])
        dash = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
        out.append(f'<polyline id="{s["id"]}" points="{points}" fill="none" stroke="{s["color"]}" stroke-width="{s["width"]}"{dash}/>')
        if s["dash"] is None:
            radius = 4 if s["width"] > 2 else 2.5
            for c, v in s["points"]:
                out.append(f'<circle cx="{fmt(x(c))}" cy="{fmt(y(v))}" r="{radius}" fill="{s["color"]}"/>')
    for i, s in enumerate(series):
        row = TOP + 8 + 22 * i
        dash = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
        out.append(f'<line x1="{RIGHT + 16}" y1="{row}" x2="{RIGHT + 44}" y2="{row}" stroke="{s["color"]}" stroke-width="{s["width"]}"{dash}/>')
        out.append(f'<text x="{RIGHT + 50}" y="{row + 4}" font-size="11" fill="#111111">{escape(s["label"])}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--threads", type=int, help="default: the thread count with the highest engine median at the largest context")
    parser.add_argument("--output", type=Path, default=Path("results/tokens-vs-context.svg"))
    args = parser.parse_args()
    svg = render(json.loads(args.summary.read_text()), args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(args.output)


if __name__ == "__main__":
    main()
