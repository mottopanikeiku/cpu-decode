"""Small math tests plus opt-in full pinned-model integration.

Full integration after download/build:
  CPU_DECODE_MODEL=/snapshot CPU_DECODE_QUANT_MODEL=/int8-dir \
    nice -n 19 uv run pytest -q tests/test_reference.py
Full-model integration is strictly opt-in: CPU_DECODE_MODEL must be set.
The quantized integration requires CPU_DECODE_QUANT_MODEL explicitly.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.reference import position_metrics
from tools.tokenize import repeat_tokens


def test_position_metrics_exact() -> None:
    logits = np.array([[1.0, 0.0, -2.0], [-1.0, 3.0, 2.0]], dtype=np.float32)
    rows = position_metrics(logits, logits.copy(), 0.0, 0.0)
    assert [row["reference_top1"] for row in rows] == [0, 1]
    assert all(row["top1_match"] and row["allclose"] for row in rows)
    assert all(row["kl_reference_candidate_nats"] == 0.0 for row in rows)


def test_position_metrics_kl_direction_and_shift() -> None:
    reference = np.array([[np.log(0.8), np.log(0.2)]], dtype=np.float64)
    candidate = np.array([[np.log(0.5), np.log(0.5)]], dtype=np.float64)
    rows = position_metrics(reference, candidate, 0.0, 0.0)
    expected = 0.8 * np.log(0.8 / 0.5) + 0.2 * np.log(0.2 / 0.5)
    assert rows[0]["kl_reference_candidate_nats"] == pytest.approx(expected)
    shifted = position_metrics(reference, reference + 100.0, 0.0, 0.0)[0]
    assert shifted["kl_reference_candidate_nats"] == pytest.approx(0.0, abs=1e-14)
    assert not shifted["allclose"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_position_metrics_rejects_nonfinite(value) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        position_metrics(np.array([[0.0, value]]), np.array([[1.0, 0.0]]), 0.1, 0.1)


def test_repeat_tokens_matches_native_policy() -> None:
    assert repeat_tokens([7, 8, 9], 8) == [7, 8, 9, 7, 8, 9, 7, 8]
    with pytest.raises(ValueError):
        repeat_tokens([], 8)


def _check_chunked_fp32(tmp_path: Path) -> None:
    """Tiny synthetic weights test conversion mechanics, not Qwen correctness."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from tools.reference import load_oracle, project_last

    torch.manual_seed(123)
    config = Qwen2Config(vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    stored = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
    stored.save_pretrained(tmp_path)
    dense = Qwen2ForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True).float().eval()
    oracle = load_oracle(tmp_path, 1, "fp32")
    assert all(parameter.dtype == torch.bfloat16 for parameter in oracle.parameters())
    dense_cache = None
    oracle_cache = None
    with torch.inference_mode():
        for token in [1, 5, 3, 21, 7]:
            ids = torch.tensor([[token]])
            expected = dense(input_ids=ids, past_key_values=dense_cache, use_cache=True)
            actual = oracle.model(input_ids=ids, past_key_values=oracle_cache, use_cache=True)
            dense_cache = expected.past_key_values
            oracle_cache = actual.past_key_values
            logits = project_last(oracle, actual.last_hidden_state, "fp32", 7)
            torch.testing.assert_close(logits, expected.logits[0, -1], atol=1e-6, rtol=1e-5)
            assert int(logits.argmax()) == int(expected.logits[0, -1].argmax())


def test_chunked_fp32_oracle_equals_dense_fp32(tmp_path) -> None:
    # The pytest driver must not retain torch while full-model workers run.
    code = "import runpy, sys; from pathlib import Path; runpy.run_path(sys.argv[1])['_check_chunked_fp32'](Path(sys.argv[2]))"
    subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)], check=True, cwd=Path(__file__).resolve().parents[1])


def model_path() -> Path:
    explicit = os.environ.get("CPU_DECODE_MODEL")
    if not explicit:
        pytest.skip("Full-model integration requires explicit CPU_DECODE_MODEL")
    model = Path(explicit)
    if not (model / "model.safetensors").exists():
        pytest.fail(f"Requested model does not contain weights: {model}")
    return model


def integration(tmp_path: Path, quantized: bool) -> dict:
    root = Path(__file__).resolve().parents[1]
    model = model_path()
    engine = Path(os.environ.get("CPU_DECODE_ENGINE", str(root / "build" / "cpu-decode")))
    if not engine.exists():
        pytest.fail(f"Explicitly requested native engine does not exist: {engine}")
    output = tmp_path / "metrics.json"
    command = [sys.executable, "-m", "tools.reference", "--model", str(model), "--engine", str(engine), "--steps", "8", "--threads", "1", "--kernel", os.environ.get("CPU_DECODE_KERNEL", "scalar"), "--output", str(output), "--raw-dir", str(tmp_path / "raw")]
    if quantized:
        directory = os.environ.get("CPU_DECODE_QUANT_MODEL")
        if not directory:
            pytest.skip("Set CPU_DECODE_QUANT_MODEL to test the offline int8 model")
        command += ["--quant-model", directory]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr + completed.stdout[-4000:]
    metrics = json.loads(output.read_text())
    assert metrics["oracle"]["arithmetic"] == "fp32"
    assert metrics["oracle"]["weight_storage"] == "bfloat16"
    assert len(metrics["oracle"]["prompts"]) == 4
    assert metrics["passed"]
    return metrics


def test_pinned_bf16_weights_fp32_logits_and_exact_greedy(tmp_path) -> None:
    metrics = integration(tmp_path, False)
    for case in metrics["comparisons"][0]["prompts"]:
        assert len(case["reference_generated_tokens"]) == 8
        assert case["exact_greedy_match"]
        assert all(row["allclose"] for row in case["positions"])
        assert len(case["positions"]) == len(case["prompt_tokens"]) + 7


def test_pinned_int8_per_position_top1_and_kl(tmp_path) -> None:
    metrics = integration(tmp_path, True)
    quantized = metrics["comparisons"][1]
    assert quantized["label"] == "int8"
    assert quantized["quality_reporting_only"]
    for case in quantized["prompts"]:
        assert 0.0 <= case["top1_agreement"] <= 1.0
        for row in case["positions"]:
            assert np.isfinite(row["kl_reference_candidate_nats"])
            assert row["kl_reference_candidate_nats"] >= 0.0
            assert isinstance(row["top1_match"], bool)
    # Cross-check the produced file with the official parser, not only our loader.
    from safetensors import safe_open

    path = Path(os.environ["CPU_DECODE_QUANT_MODEL"]) / "model.safetensors"
    with safe_open(path, framework="numpy") as stored:
        assert "lm_head.weight" not in stored.keys()
        assert stored.metadata()["quantization"] == "symmetric-per-row-int8"
        assert stored.get_slice("model.embed_tokens.weight").get_shape() == [151936, 896]
        assert stored.get_tensor("model.layers.0.self_attn.k_proj.weight").dtype == np.int8
        assert stored.get_tensor("model.embed_tokens.weight.scales").dtype == np.float32


def test_explicit_integration_rejects_missing_engine(tmp_path, monkeypatch) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"")
    monkeypatch.setenv("CPU_DECODE_MODEL", str(tmp_path))
    monkeypatch.setenv("CPU_DECODE_ENGINE", str(tmp_path / "missing-engine"))
    with pytest.raises(pytest.fail.Exception, match="Explicitly requested native engine"):
        integration(tmp_path, False)


def test_oracle_rejects_edited_snapshot_before_loading(tmp_path) -> None:
    from types import SimpleNamespace
    from tools.reference import oracle_worker

    (tmp_path / "LICENSE").write_text("not the pinned license")
    with pytest.raises(ValueError, match="Pinned model checksum mismatch"):
        oracle_worker(SimpleNamespace(model=tmp_path))
