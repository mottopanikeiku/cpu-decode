import json
import os
import subprocess
import sys
from argparse import Namespace

import pytest

from tools.download_model import MODEL_ID, REVISION, file_hash
from tools.llama_quality import aggregate, verify_inputs
from tools.measure import execute
from tools.prepare_llama import LLAMA_COMMIT, LLAMA_URL


def test_quality_aggregates_separate_prompt_and_continuation_rows():
    cases = [
        {"prompt_tokens": [1, 2], "positions": [
            {"top1_match": True, "kl_reference_candidate_nats": 0.1},
            {"top1_match": False, "kl_reference_candidate_nats": 0.3},
            {"top1_match": True, "kl_reference_candidate_nats": 0.2},
        ]},
        {"prompt_tokens": [3], "positions": [
            {"top1_match": True, "kl_reference_candidate_nats": 0.5},
            {"top1_match": False, "kl_reference_candidate_nats": 0.4},
        ]},
    ]
    result = aggregate(cases)
    assert result["prompt_only"]["positions"] == 3
    assert result["prompt_only"]["top1_agreement"] == pytest.approx(2 / 3)
    assert result["prompt_only"]["mean_kl_reference_candidate_nats"] == pytest.approx(0.3)
    assert result["prompt_and_reference_continuation"]["positions"] == 5
    assert result["prompt_and_reference_continuation"]["top1_agreement"] == 0.6
    assert result["prompt_and_reference_continuation"]["mean_kl_reference_candidate_nats"] == pytest.approx(0.3)
    assert result["prompt_and_reference_continuation"]["max_kl_reference_candidate_nats"] == 0.5


@pytest.mark.parametrize("changed, expected", [
    ("llama_commit", "pinned llama.cpp revision"),
    ("model_revision", "pinned BF16 source model"),
])
def test_changed_preparation_pin_rejects_before_any_artifact_or_model_read(tmp_path, changed, expected):
    manifest = {
        "llama_commit": LLAMA_COMMIT,
        "llama_repository": LLAMA_URL,
        "source_model": {"model_id": MODEL_ID, "revision": REVISION, "stored_dtype": "bfloat16"},
    }
    if changed == "llama_commit":
        manifest["llama_commit"] = "edited"
    else:
        manifest["source_model"]["revision"] = "edited"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = Namespace(artifact_manifest=path, artifact="Q8_0", model=tmp_path / "absent.gguf", reference_dir=tmp_path / "absent-reference")
    with pytest.raises(ValueError, match=expected):
        verify_inputs(args)


def test_gguf_must_match_the_selected_artifact_not_another(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"q8 bytes")
    q8 = {"sha256": file_hash(model), "bytes": model.stat().st_size, "tensor_types": {}}
    manifest = {
        "llama_commit": LLAMA_COMMIT,
        "llama_repository": LLAMA_URL,
        "source_model": {"model_id": MODEL_ID, "revision": REVISION, "stored_dtype": "bfloat16"},
        "artifacts": {"Q8_0": q8, "Q4_0": {**q8, "sha256": "0" * 64}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = Namespace(artifact_manifest=path, artifact="Q4_0", model=model, reference_dir=tmp_path / "absent-reference")
    with pytest.raises(ValueError, match="manifest Q4_0 artifact"):
        verify_inputs(args)


def test_failed_measurement_retains_portable_command_stderr_and_returncode(tmp_path):
    model = tmp_path / "model"
    destination = tmp_path / "attempt.json"
    command = [sys.executable, "-c", "import sys; print('diagnostic'); print(sys.argv[1], file=sys.stderr); sys.exit(3)", str(model)]
    with pytest.raises(subprocess.CalledProcessError) as failure:
        execute(command, destination, {model: "$MODEL"}, dict(os.environ))
    assert failure.value.returncode == 3
    assert not destination.exists()
    record = json.loads(destination.with_suffix(".failure.json").read_text())
    assert record["returncode"] == 3
    assert record["command"][-1] == "$MODEL"
    assert record["stderr"] == "$MODEL\n"
    assert record["stdout"] == "diagnostic\n"
    assert destination.with_suffix(".stderr.txt").read_text() == "$MODEL\n"
