"""Synthetic tests for aggregation and plotting, never a substitute for measured inputs."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.measure import fast_places
from tools.summarize import spread

ROOT = Path(__file__).resolve().parents[1]


def test_median_and_range() -> None:
    assert spread([2, 9, 4]) == {"median": 4, "min": 2, "max": 9}
    with pytest.raises(ValueError, match="empty"):
        spread([])


def test_fast_places_keeps_highest_clock_cores_with_smt_siblings() -> None:
    listing = "CPU CORE MAXMHZ\n0 0 5100.0\n1 1 5100.0\n2 2 3300.0\n3 0 5100.0\n4 1 5100.0\n5 2 3300.0\n"
    places, details = fast_places(listing)
    assert places == "{0,3},{1,4}"
    assert details["selected_cores"] == [0, 1]
    with pytest.raises(ValueError, match="only some CPUs"):
        fast_places("CPU CORE MAXMHZ\n0 0 5100.0\n1 1 -\n")
    with pytest.raises(ValueError, match="header"):
        fast_places("CPU CORE\n0 0\n")


def engine(threads: int, context: int, kv: str, rates: list[float], **fields) -> dict:
    return {"threads": threads, "context": context, "steps": 16, "repeats": 3, "warmup_steps": 1, "kernel": "neon", "kv_dtype": kv,
            "weight_format": "q8", "head_format": "q8", "weight_memory": "hugepage", "fused_projections": True, "prompt_tokens": [1, 2, 3],
            **fields,
            "samples": [{"tokens_per_second": r, "bytes_per_token": {"total_min": 1e9}, "operation_seconds": {"lm_head": 0.16}} for r in rates]}


def llama(threads: int, context: int, kv: str, rates: list[float], model_type: str = "qwen2 1B Q8_0") -> list[dict]:
    return [{"n_threads": threads, "n_depth": context, "n_gen": 16, "samples_ts": rates, "type_k": kv, "type_v": kv,
             "flash_attn": -1 if kv == "f16" else 0, "n_gpu_layers": 0, "model_type": model_type}]


def write(directory: Path, files: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (directory / f"{name}.json").write_text(json.dumps(data))


def summarize(tmp_path: Path) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "tools.summarize", "--input", str(tmp_path / "inputs"), "--ablations", str(tmp_path / "ablations"),
               "--output", str(tmp_path / "output")]
    return subprocess.run(command, capture_output=True, text=True, cwd=ROOT)


def test_summary_pairs_baselines_ablations_and_plot(tmp_path: Path) -> None:
    write(tmp_path / "inputs", {
        "bandwidth-t6-c0-neon": {"threads": 6, "kernel": "neon", "samples": [{"GB_per_s": r} for r in [10, 20, 15]]},
        "engine-q8-f16-t6-c128": engine(6, 128, "f16", [10, 12, 11]),
        "llama-q8_0-f16-t6-c128": llama(6, 128, "f16", [20, 22, 21]),
        "engine-q4h8-f16-t6-c128": engine(6, 128, "f16", [14, 16, 15], weight_format="q4"),
        "llama-q4_0-f16-t6-c128": llama(6, 128, "f16", [25, 30, 24], "qwen2 1B Q4_0"),
        "engine-q8-f32-t6-c128": engine(6, 128, "f32", [8, 9, 10]),
        "llama-q8_0-f32-t6-c128": llama(6, 128, "f32", [18, 18, 18]),
        "engine-q4-f16-t6-c128": engine(6, 128, "f16", [16, 16, 16], weight_format="q4", head_format="q4"),
    })
    write(tmp_path / "ablations", {
        "engine-q8-f16-t6-c128": engine(6, 128, "f16", [10, 10, 10]),
        "engine-q8-f16-unfused-t6-c128": engine(6, 128, "f16", [8, 8, 8], fused_projections=False),
        "engine-q8-f16-scalar-t6-c128": engine(6, 128, "f16", [2, 2, 2], kernel="scalar"),
        "engine-q4-f16-t6-c128": engine(6, 128, "f16", [16, 16, 16], weight_format="q4", head_format="q4"),
    })
    completed = summarize(tmp_path)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "output" / "summary.json").read_text())
    result = summary["results"][0]
    assert result["engine_tps"] == {"median": 11, "min": 10, "max": 12}
    assert result["llama_tps"]["median"] == 21
    assert result["bandwidth_ceiling_tps"] == 15
    assert result["percent_of_ceiling"] == pytest.approx(100 * 11 / 15)
    assert result["engine_over_llama"] == pytest.approx(11 / 21)
    assert result["operation_seconds_per_token"]["lm_head"] == pytest.approx(0.01)
    assert [(p["engine_label"], p["llama_label"], p["engine_over_llama"]) for p in summary["pairs"]] == [
        ("q8-f16", "q8_0-f16", pytest.approx(11 / 21)), ("q4h8-f16", "q4_0-f16", pytest.approx(15 / 25)),
        ("q8-f32", "q8_0-f32", pytest.approx(9 / 18)),
        ("q4-f16", "q4_0-f16", pytest.approx(16 / 25))]
    # q4/q4 differs from q8/q8 in two fields, so it pairs with nothing here.
    assert [(e["change"], e["ratio"]) for e in summary["ablation_effects"]] == [
        ("fused_projections: False → True", pytest.approx(10 / 8)), ("kernel: scalar → neon", pytest.approx(5))]

    svg = tmp_path / "plot.svg"
    subprocess.run([sys.executable, "-m", "tools.plot", "--summary", str(tmp_path / "output" / "summary.json"), "--output", str(svg)],
                   check=True, capture_output=True, cwd=ROOT)
    root = ET.parse(svg).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    lines = {e.get("id"): [tuple(map(float, p.split(","))) for p in e.get("points").split()] for e in root.iterfind("svg:polyline", namespace)}
    assert set(lines) == {"engine", "llama", "ceiling", "engine-secondary", "llama-secondary"}
    baseline = float(root.find("svg:line[@id='x-axis']", namespace).get("y1"))
    height = {name: baseline - points[0][1] for name, points in lines.items()}
    assert height["engine"] / height["ceiling"] == pytest.approx(11 / 15, rel=1e-3)
    assert height["llama"] / height["engine"] == pytest.approx(21 / 11, rel=1e-3)


def test_summary_rejects_llama_kv_mismatch(tmp_path: Path) -> None:
    write(tmp_path / "inputs", {
        "bandwidth-t6-c0-neon": {"threads": 6, "kernel": "neon", "samples": [{"GB_per_s": 1}]},
        "engine-q8-f16-t6-c128": engine(6, 128, "f16", [10, 12, 11]),
        "llama-q8_0-f16-t6-c128": llama(6, 128, "f32", [20, 22, 21]),
    })
    completed = summarize(tmp_path)
    assert completed.returncode != 0
    assert "KV types f32/f32 do not match engine kv_dtype f16" in completed.stderr
