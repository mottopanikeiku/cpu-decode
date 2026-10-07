"""Offline statistics/oracle regressions and opt-in real greedy exactness."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.kv_quality_v3 import prepare
from tools.lookup_v3 import aggregate
from tools.time_v3 import spread

ROOT = Path(__file__).resolve().parents[1]


def test_long_context_plan_uses_only_existing_heldout():
    corpus = json.loads((ROOT / "results/v2/corpus.json").read_text())
    expected = [token for window in corpus["windows"] if window["split"] == "heldout" for token in window["tokens"]]
    plan = prepare(ROOT / "results/v2/corpus.json")
    assert plan["tokens"] == expected[:4097]
    assert plan["positions"] == list(range(2048, 2080)) + list(range(4064, 4096))
    assert all(position + 1 < len(plan["tokens"]) for position in plan["positions"])
    assert plan == json.loads((ROOT / "results/v3/long-context-inputs.json").read_text())


def test_acceptance_uses_total_proposed_tokens_not_mean_prompt_rates():
    cases = []
    for name, category, drafted, accepted in (("a", "copy-heavy", 4, 4), ("b", "copy-heavy", 12, 0), ("c", "open-ended", 0, 0)):
        cases.append({"id": name, "category": category, "exact_match": True,
                      "lookup": {"generated_tokens": [1] * 8,
                                 "lookup": {"drafted_tokens": drafted, "accepted_tokens": accepted,
                                            "verification_batches": int(drafted > 0), "forward_tokens": 7, "ordinary_steps": 2}}})
    result = aggregate(cases)
    assert result["all"]["acceptance_rate"] == 0.25
    assert result["copy-heavy"]["acceptance_rate"] == 0.25
    assert result["open-ended"]["acceptance_rate"] is None
    assert result["all"]["generated_tokens"] == 24
    with pytest.raises(ValueError):
        aggregate(cases + cases)


def test_timing_summary_reports_spread():
    assert spread([1, 3, 2])["median_seconds"] == 2
    assert spread([1, 3, 2])["max_seconds"] == 3
    with pytest.raises(ValueError):
        spread([0, 1])


def _oracle_chunks_equal_dense(tmp):
    from argparse import Namespace
    from unittest.mock import patch
    import numpy as np
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from tools.kv_quality_v3 import oracle_layer, oracle_head

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=37, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        tie_word_embeddings=True)
    stored = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
    model = tmp / "model"
    stored.save_pretrained(model)
    dense = Qwen2ForCausalLM.from_pretrained(model, dtype=torch.float32, attn_implementation="eager").eval()
    plan = {"tokens": [1, 5, 3, 21, 7, 3, 5, 1, 4], "positions": [0, 1, 6, 7]}
    plan_path = tmp / "plan.json"
    plan_path.write_text(json.dumps(plan))
    args = Namespace(model=model, raw=tmp / "raw", plan=plan_path, threads=1, layer=0, chunk=3)
    with patch("tools.kv_quality_v3.verify_snapshot", return_value={"synthetic": True}), patch("torch.set_num_interop_threads"):
        oracle_layer(args, plan)
        args.layer = 1
        oracle_layer(args, plan)
        oracle_head(args, plan)
    logits = np.memmap(args.raw / "oracle.bin", dtype="<f4", mode="r", shape=(4, config.vocab_size))
    with torch.inference_mode():
        expected = dense(torch.tensor([plan["tokens"][:-1]])).logits[0, plan["positions"]].numpy()
    np.testing.assert_allclose(logits, expected, atol=2e-6, rtol=2e-5)
    assert (args.raw / "hidden-02.pt").exists()
    assert not (args.raw / "hidden-01.pt").exists()


def test_long_context_chunk_mask_matches_unmodified_dense_oracle(tmp_path):
    code = "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['_oracle_chunks_equal_dense'](Path(sys.argv[2]))"
    subprocess.run([sys.executable, "-c", code, str(Path(__file__)), str(tmp_path)], check=True, cwd=ROOT)


@pytest.mark.parametrize("kv", ("f16", "i8", "i8-centered"))
def test_fixed_prompts_exact_greedy(tmp_path, kv):
    model = os.environ.get("CPU_DECODE_QUANT_MODEL")
    if not model:
        pytest.skip("Requires explicit CPU_DECODE_QUANT_MODEL")
    from tools.lookup_v3 import measure

    config = json.loads((ROOT / "configs/lookup-prompts.json").read_text())
    inputs = json.loads((ROOT / "results/v3/lookup-inputs.json").read_text())
    engine = Path(os.environ.get("CPU_DECODE_ENGINE", str(ROOT / "build/cpu-decode")))
    # Fixed copy and open-ended prompts; full public suite is in the result files.
    for case in (inputs["prompts"][0], inputs["prompts"][6]):
        result = measure(engine, Path(model), case, {**config, "steps": 32}, 1,
                         os.environ.get("CPU_DECODE_KERNEL", "auto"), kv)
        assert result["exact_match"]
        assert len(result["lookup"]["generated_tokens"]) == 32


@pytest.mark.parametrize("kv", ("f32", "f16", "i8", "i8-centered"))
def test_real_sparse_logits_batch_matches_single(tmp_path, kv):
    model = os.environ.get("CPU_DECODE_QUANT_MODEL")
    if not model:
        pytest.skip("Requires explicit CPU_DECODE_QUANT_MODEL")
    from tools.download_model import file_hash

    inputs = json.loads((ROOT / "results/v3/lookup-inputs.json").read_text())
    tokens = inputs["prompts"][0]["tokens"][:128]
    assert len(tokens) == 128
    positions = [0, 3, 63, 64, 65, 127]
    engine = Path(os.environ.get("CPU_DECODE_ENGINE", str(ROOT / "build/cpu-decode")))
    for batch in (1, 8):
        subprocess.run([str(engine), "logits", "--model", model, "--tokens", ",".join(map(str, tokens)),
                        "--logits-positions", ",".join(map(str, positions)), "--batch", str(batch),
                        "--threads", "1", "--kernel", os.environ.get("CPU_DECODE_KERNEL", "auto"),
                        "--kv", kv, "--output", str(tmp_path / f"logits-{batch}")], check=True)
        metadata = json.loads((tmp_path / f"logits-{batch}.json").read_text())
        assert metadata["logit_positions"] == positions
        assert metadata["shape"][0] == len(positions)
    assert file_hash(tmp_path / "logits-1.bin") == file_hash(tmp_path / "logits-8.bin")
