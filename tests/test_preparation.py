import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import download_model, prepare_llama, quantize


def test_registry_retains_original_pin_and_identical_tokenizer_blobs():
    assert download_model.model_pin() == {
        "revision": download_model.REVISION, "files": download_model.FILES,
    }
    assert download_model.model_pin(download_model.S1_MODEL_ID)["revision"] == download_model.S1_REVISION
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        assert download_model.S1_FILES[name] == download_model.FILES[name]
    assert download_model.S1_FILES["config.json"] != download_model.FILES["config.json"]
    assert download_model.S1_FILES["model.safetensors"] == (
        3087467144, "sha256", "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    )


@pytest.mark.parametrize("model_id", [download_model.MODEL_ID, download_model.S1_MODEL_ID])
def test_verify_snapshot_selects_pin_and_checks_bytes(tmp_path, monkeypatch, model_id):
    path = tmp_path / "config.json"
    path.write_bytes(b"configuration")
    files = {"config.json": (path.stat().st_size, "git-sha1", download_model.file_hash(path, "git-sha1"))}
    revision = download_model.model_pin(model_id)["revision"]
    monkeypatch.setitem(download_model.PINNED_MODELS, model_id, {"revision": revision, "files": files})
    manifest = download_model.verify_snapshot(tmp_path, model_id=model_id)
    assert manifest["model_id"] == model_id
    assert manifest["revision"] == revision
    assert manifest["files"]["config.json"]["sha256"] == download_model.file_hash(path)
    if model_id == download_model.MODEL_ID:
        assert download_model.verify_snapshot(tmp_path) == manifest
    path.write_bytes(b"edited config")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_model.verify_snapshot(tmp_path, model_id=model_id)


def test_unknown_model_rejected_before_snapshot_read(tmp_path):
    with pytest.raises(ValueError, match="Unsupported pinned model"):
        download_model.verify_snapshot(tmp_path / "absent", model_id="Qwen/moving")



@pytest.mark.parametrize("model_id, destination", [
    (download_model.MODEL_ID, "results/model-manifest.json"),
    (download_model.S1_MODEL_ID, "results/v2/s1/model-manifest.json"),
])
def test_download_cli_uses_only_registry_revision_and_separate_manifest(tmp_path, monkeypatch, model_id, destination):
    snapshot = tmp_path / "snapshot"
    calls = []

    def fetch(requested_id, **kwargs):
        calls.append((requested_id, kwargs))
        return str(snapshot)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fetch))
    monkeypatch.setattr(download_model, "verify_snapshot", lambda path, model_id=download_model.MODEL_ID: pinned_source(model_id))
    monkeypatch.setattr(sys, "argv", ["download_model", "--model-id", model_id, "--offline"])
    download_model.main()
    assert calls == [(model_id, {
        "revision": download_model.model_pin(model_id)["revision"],
        "cache_dir": None,
        "allow_patterns": list(download_model.model_pin(model_id)["files"]),
        "local_files_only": True,
        "max_workers": 2,
    })]
    assert json.loads((tmp_path / destination).read_text()) == pinned_source(model_id)

def pinned_source(model_id):
    return {"model_id": model_id, "revision": download_model.model_pin(model_id)["revision"],
            "stored_dtype": "bfloat16", "files": {}}


@pytest.fixture
def upstream_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    root = cache / "llama.cpp" / prepare_llama.LLAMA_COMMIT
    (root / "source").mkdir(parents=True)
    (root / "build" / "bin").mkdir(parents=True)
    for name in ("llama-bench", "llama-quantize"):
        binary = root / "build" / "bin" / name
        binary.write_bytes(name.encode())
        binary.chmod(0o755)
    monkeypatch.setattr(prepare_llama, "verify_snapshot", lambda model, model_id=download_model.MODEL_ID: pinned_source(model_id))
    monkeypatch.setattr(prepare_llama.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, stdout=prepare_llama.LLAMA_COMMIT + "\n"))
    return cache, root


def test_reused_preparation_separates_revision_artifacts_without_building(tmp_path, monkeypatch, upstream_cache):
    cache, root = upstream_cache
    legacy = root / "preparation.json"
    legacy.write_text("untouched original cache manifest")
    commands = []

    def produce(command, cwd=None):
        commands.append(command)
        if "--outfile" in command:
            Path(command[command.index("--outfile") + 1]).write_bytes(b"bf16 gguf")
        elif Path(command[0]).name == "llama-quantize":
            Path(command[-3]).write_bytes(b"q8 gguf")
        else:
            raise AssertionError("Reuse must not fetch, configure or build")

    monkeypatch.setattr(prepare_llama, "run", produce)
    for model_id in (download_model.MODEL_ID, download_model.S1_MODEL_ID):
        output = tmp_path / ("small.json" if model_id == download_model.MODEL_ID else "large.json")
        manifest = prepare_llama.prepare(tmp_path / "model", cache, 2, output, model_id=model_id, reuse_build=True)
        revision = download_model.model_pin(model_id)["revision"]
        assert manifest["source_model"] == pinned_source(model_id)
        assert manifest["build"]["reused"] is True
        assert (root / f"preparation-{revision}.json").is_file()
        assert (root / f"qwen-{revision}-bf16.gguf").is_file()
        assert (root / f"qwen-{revision}-q8_0.gguf").is_file()
        assert str(tmp_path) not in output.read_text()
        before = len(commands)
        assert prepare_llama.prepare(tmp_path / "model", cache, 2, output, model_id=model_id, reuse_build=True) == manifest
        assert len(commands) == before
    assert legacy.read_text() == "untouched original cache manifest"
    quantizers = [command for command in commands if Path(command[0]).name == "llama-quantize"]
    assert "--max-buffer-size" not in quantizers[0]
    assert quantizers[1][1:3] == ["--max-buffer-size", "128"]
    converters = [command for command in commands if "--outfile" in command]
    assert Path(converters[0][1]).name == "convert_hf_to_gguf.py"
    assert "--use-temp-file" in converters[0]
    assert converters[0][converters[0].index("--outtype") + 1] == "bf16"
    assert "--model-id" not in converters[0]
    assert Path(converters[1][1]).name == "streamed_gguf.py"
    assert converters[1][converters[1].index("--model-id") + 1] == download_model.S1_MODEL_ID
    assert "--use-temp-file" not in converters[1]
    assert converters[1][converters[1].index("--chunk-mib") + 1] == "4"
    assert converters[1][converters[1].index("--source") + 1] == str(root / "source")
    assert converters[1][converters[1].index("--outfile") + 1].endswith(".streamed.partial.gguf")


def test_reuse_rejects_wrong_upstream_commit_without_fetch(tmp_path, monkeypatch, upstream_cache):
    cache, _ = upstream_cache
    monkeypatch.setattr(prepare_llama.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="wrong\n"))
    monkeypatch.setattr(prepare_llama, "run", lambda *args: pytest.fail("Must not fetch/build"))
    with pytest.raises(ValueError, match="pinned llama.cpp commit"):
        prepare_llama.prepare(tmp_path / "model", cache, 2, tmp_path / "output.json", reuse_build=True)


def test_reuse_rejects_missing_binary_without_build(tmp_path, monkeypatch, upstream_cache):
    cache, root = upstream_cache
    (root / "build" / "bin" / "llama-quantize").unlink()
    monkeypatch.setattr(prepare_llama, "run", lambda *args: pytest.fail("Must not build"))
    with pytest.raises(ValueError, match="Missing existing upstream executable"):
        prepare_llama.prepare(tmp_path / "model", cache, 2, tmp_path / "output.json", reuse_build=True)


def test_preparation_rejects_unrecorded_existing_artifact(tmp_path, upstream_cache):
    cache, root = upstream_cache
    (root / f"qwen-{download_model.S1_REVISION}-bf16.gguf").write_bytes(b"unrecorded")
    with pytest.raises(ValueError, match="Unverified existing artifact"):
        prepare_llama.prepare(tmp_path / "model", cache, 2, tmp_path / "output.json", model_id=download_model.S1_MODEL_ID, reuse_build=True)


@pytest.mark.parametrize("fault", [None, "source", "commit", "BF16", "Q8_0"])
def test_verified_legacy_cache_migrates_only_after_all_checks(tmp_path, monkeypatch, upstream_cache, fault):
    cache, root = upstream_cache
    source = pinned_source(download_model.MODEL_ID)
    paths = {dtype: root / f"qwen-{source['revision']}-{suffix}.gguf"
             for dtype, suffix in (("BF16", "bf16"), ("Q8_0", "q8_0"))}
    for dtype, path in paths.items():
        path.write_bytes(f"synthetic {dtype} artifact".encode())
    record = {"source_model": source, "llama_commit": prepare_llama.LLAMA_COMMIT,
              "artifacts": {dtype: {"sha256": download_model.file_hash(path)}
                            for dtype, path in paths.items()}}
    if fault == "source":
        record["source_model"] = pinned_source(download_model.S1_MODEL_ID)
    elif fault == "commit":
        record["llama_commit"] = "wrong"
    elif fault in paths:
        record["artifacts"][fault]["sha256"] = "wrong"
    legacy = root / "preparation.json"
    original = json.dumps(record, indent=2).encode()
    legacy.write_bytes(original)
    revised = root / f"preparation-{source['revision']}.json"
    monkeypatch.setattr(prepare_llama, "run", lambda *args: pytest.fail("Verified reuse must not build or convert"))
    if fault is not None:
        with pytest.raises(ValueError, match="different source|Unverified existing artifact"):
            prepare_llama.prepare(tmp_path / "model", cache, 1, tmp_path / "output.json", reuse_build=True)
        assert legacy.read_bytes() == original
        assert not revised.exists()
    else:
        result = prepare_llama.prepare(tmp_path / "model", cache, 1, tmp_path / "output.json", reuse_build=True)
        assert not legacy.exists()
        assert json.loads(revised.read_text()) == result
        assert all(result["artifacts"][dtype]["sha256"] == record["artifacts"][dtype]["sha256"]
                   for dtype in paths)


@pytest.mark.parametrize("model_id", [download_model.MODEL_ID, download_model.S1_MODEL_ID])
def test_native_quantization_threads_model_id_and_preserves_private_manifest(tmp_path, monkeypatch, model_id):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "int8"
    public_manifest = tmp_path / "public.json"
    calls = []

    def verify(model, model_id=download_model.MODEL_ID):
        assert model == source
        calls.append(model_id)
        return pinned_source(model_id)

    def produce(command, check):
        assert check
        output.mkdir()
        (output / "model.safetensors").write_bytes(b"quantized")
        (output / "config.json").write_text("{}")

    monkeypatch.setattr(quantize, "verify_snapshot", verify)
    monkeypatch.setattr(quantize.subprocess, "run", produce)
    monkeypatch.setattr(sys, "argv", ["quantize", "--model-id", model_id, "--source", str(source),
        "--output", str(output), "--group-size", "64", "--scale-dtype", "f16", "--manifest", str(public_manifest)])
    quantize.main()
    assert (output / "quantization.json").is_file()
    private_before = (output / "quantization.json").read_bytes()
    quantize.main()
    assert (output / "quantization.json").read_bytes() == private_before
    assert calls == [model_id, model_id]
    manifest = json.loads(public_manifest.read_text())
    assert manifest["source"] == pinned_source(model_id)
    assert manifest["matrix_bits_per_weight"] == 8.25
    assert str(tmp_path) not in public_manifest.read_text()
    other = download_model.S1_MODEL_ID if model_id == download_model.MODEL_ID else download_model.MODEL_ID
    monkeypatch.setattr(quantize, "verify_snapshot", lambda *args, **kwargs: pinned_source(other))
    with pytest.raises(ValueError, match="pinned source/manifest"):
        quantize.main()


def test_failed_streamed_attempt_retries_only_its_own_partial(tmp_path, monkeypatch, upstream_cache):
    cache, root = upstream_cache
    stem = root / f"qwen-{download_model.S1_REVISION}-bf16"
    owned = stem.with_suffix(".streamed.partial.gguf")
    original_partial = stem.with_suffix(".partial.gguf")
    original_partial.write_bytes(b"preserve old converter partial")
    unrelated = root / f"qwen-{download_model.REVISION}-bf16.gguf"
    unrelated.write_bytes(b"preserve completed original")
    calls = []

    def fail(command, cwd=None):
        calls.append(command)
        assert Path(command[1]).name == "streamed_gguf.py"
        destination = Path(command[command.index("--outfile") + 1])
        assert destination == owned and not destination.exists()
        destination.write_bytes(b"incomplete streamed payload")
        raise subprocess.CalledProcessError(143, command)

    monkeypatch.setattr(prepare_llama, "run", fail)
    output = tmp_path / "preparation.json"
    with pytest.raises(subprocess.CalledProcessError):
        prepare_llama.prepare(tmp_path / "model", cache, 2, output,
                              model_id=download_model.S1_MODEL_ID, reuse_build=True)
    assert len(calls) == 1
    assert not output.exists()
    assert not (root / f"qwen-{download_model.S1_REVISION}-bf16.gguf").exists()
    assert not (root / f"preparation-{download_model.S1_REVISION}.json").exists()

    def finish(command, cwd=None):
        if "--outfile" in command:
            destination = Path(command[command.index("--outfile") + 1])
            assert destination == owned and not destination.exists()
            destination.write_bytes(b"complete bf16 gguf")
        else:
            assert Path(command[0]).name == "llama-quantize"
            assert command[1:3] == ["--max-buffer-size", "128"]
            Path(command[-3]).write_bytes(b"complete actual upstream q8")

    monkeypatch.setattr(prepare_llama, "run", finish)
    prepare_llama.prepare(tmp_path / "model", cache, 2, output,
                          model_id=download_model.S1_MODEL_ID, reuse_build=True)
    assert not owned.exists()
    assert original_partial.read_bytes() == b"preserve old converter partial"
    assert unrelated.read_bytes() == b"preserve completed original"


def test_q8_failure_keeps_completed_streamed_bf16_reusable(tmp_path, monkeypatch, upstream_cache):
    cache, root = upstream_cache
    output = tmp_path / "preparation.json"
    revision = download_model.S1_REVISION
    bf16 = root / f"qwen-{revision}-bf16.gguf"
    q8 = root / f"qwen-{revision}-q8_0.gguf"
    commands = []

    def fail_quantization(command, cwd=None):
        commands.append(command)
        if "--outfile" in command:
            Path(command[command.index("--outfile") + 1]).write_bytes(b"complete bf16 gguf")
        else:
            Path(command[-3]).write_bytes(b"failed q8 partial")
            raise subprocess.CalledProcessError(143, command)

    monkeypatch.setattr(prepare_llama, "run", fail_quantization)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_llama.prepare(tmp_path / "model", cache, 2, output,
                              model_id=download_model.S1_MODEL_ID, reuse_build=True)
    assert bf16.read_bytes() == b"complete bf16 gguf"
    assert not q8.exists() and not output.exists()
    private = json.loads((root / f"preparation-{revision}.json").read_text())
    assert set(private["artifacts"]) == {"BF16"}
    assert private["artifacts"]["BF16"]["sha256"] == download_model.file_hash(bf16)

    def finish(command, cwd=None):
        commands.append(command)
        assert Path(command[0]).name == "llama-quantize", "BF16 must not be converted again"
        Path(command[-3]).write_bytes(b"complete q8 gguf")

    monkeypatch.setattr(prepare_llama, "run", finish)
    manifest = prepare_llama.prepare(tmp_path / "model", cache, 2, output,
                                     model_id=download_model.S1_MODEL_ID, reuse_build=True)
    assert len(commands) == 3
    assert set(manifest["artifacts"]) == {"BF16", "Q8_0"}
    assert bf16.read_bytes() == b"complete bf16 gguf"
