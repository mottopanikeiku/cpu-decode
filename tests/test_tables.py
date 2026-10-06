"""Synthetic tests for aggregation, never a substitute for measured inputs."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.summarize import spread


def test_median_and_range() -> None:
    assert spread([2, 9, 4]) == {"median": 4, "min": 2, "max": 9}
    with pytest.raises(ValueError, match="empty"):
        spread([])


def test_summary_uses_matched_baselines_and_byte_bound(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    ablations = tmp_path / "ablations"
    output = tmp_path / "output"
    inputs.mkdir()
    ablations.mkdir()
    bandwidth = {"threads": 6, "kernel": "simd512", "samples": [{"GB_per_s": r} for r in [1, 2, 1.5]]}
    engine = {"threads": 6, "context": 128, "steps": 16, "repeats": 3, "warmup_steps": 1, "kernel": "simd512", "weight_dtype": "int8", "rope": "cached", "prompt_tokens": [1, 2, 3],
              "samples": [{"tokens_per_second": r, "bytes_per_token": {"total_min": 1000}, "operation_seconds": {"lm_head": 0.16}} for r in [10, 12, 11]]}
    llama = [{"n_threads": 6, "n_depth": 128, "n_gen": 16, "samples_ts": [20, 22, 21], "type_k": "f16", "type_v": "f16", "flash_attn": -1, "n_gpu_layers": 0, "model_type": "Q8_0"}]
    eager = {"threads": 6, "context": 128, "steps": 16, "warmup_steps": 1, "seed_token_ids": [1, 2, 3],
             "samples": [{"tokens_per_second": r} for r in [5, 7, 6]]}
    for name, data in [("bandwidth-t6-c0-simd512", bandwidth), ("engine-t6-c128-simd512", engine),
                       ("llama-t6-c128-baseline", llama), ("eager-t6-c128-baseline", eager)]:
        (inputs / f"{name}.json").write_text(json.dumps(data))
    command = [sys.executable, "-m", "tools.summarize", "--input", str(inputs), "--ablations", str(ablations), "--output", str(output), "--kernel", "simd512"]
    subprocess.run(command, check=True, capture_output=True, cwd=Path(__file__).resolve().parents[1])
    result = json.loads((output / "summary.json").read_text())["results"][0]
    assert result["engine_tps"] == {"median": 11, "min": 10, "max": 12}
    assert result["llama_tps"]["median"] == 21
    assert result["eager_tps"]["median"] == 6
    assert result["bandwidth_ceiling_tps"] == 1.5e6
    assert result["percent_of_ceiling"] == pytest.approx(100 * 11 / 1.5e6)
    assert result["engine_over_llama"] == pytest.approx(11 / 21)
    assert result["operation_seconds_per_token"]["lm_head"] == pytest.approx(0.01)
    for key, value in [("threads", 4), ("context", 1024), ("steps", 8), ("seed_token_ids", [3, 2, 1]), ("warmup_steps", 16)]:
        (inputs / "eager-t6-c128-baseline.json").write_text(json.dumps({**eager, key: value}))
        failed = subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
        assert failed.returncode != 0
        assert "Eager settings do not match" in failed.stderr
