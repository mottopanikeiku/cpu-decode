"""Synthetic tensors only; actual pinned gguf-py integration is opt-in by path.

Set CPU_DECODE_LLAMA_SOURCE to the pinned llama.cpp source checkout to exercise
its real writer/codec and compare against its unmodified Qwen2 converter.
"""
import json
import hashlib
import io
import os
import subprocess
import struct
import sys
import weakref
from pathlib import Path

import pytest

from tools import streamed_gguf as streamed


@pytest.mark.parametrize("shape,budget,expected", [
    ((5, 3), 24, [(0, 2), (2, 4), (4, 5)]),
    ((5,), 8, [(0, 2), (2, 4), (4, 5)]),
    ((3, 8960), 8960 * 4, [(0, 1), (1, 2), (2, 3)]),
    ((151936, 1536), 4 * 1024 * 1024, None),
])
def test_row_chunks_cover_in_order_with_bounded_widened_width(shape, budget, expected):
    ranges = streamed.row_ranges(shape, budget)
    if expected is not None:
        assert ranges == expected
    assert ranges[0][0] == 0 and ranges[-1][1] == shape[0]
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    width = shape[1] if len(shape) == 2 else 1
    assert all((end - start) * width * 4 <= budget for start, end in ranges)


@pytest.mark.parametrize("shape,budget", [
    ((), 16), ((0, 8), 32), ((2, -1), 32), ((2, 2, 2), 32),
    ((2, 8), 31), ((5,), 0),
])
def test_invalid_shapes_and_subrow_budgets_rejected(shape, budget):
    with pytest.raises(ValueError):
        streamed.row_ranges(shape, budget)


def pinned_config(model_id=streamed.S1_MODEL_ID):
    config = {
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "torch_dtype": "bfloat16", "hidden_size": 1536,
        "intermediate_size": 8960, "num_hidden_layers": 28,
        "num_attention_heads": 12, "num_key_value_heads": 2,
        "vocab_size": 151936, "tie_word_embeddings": True,
        "hidden_act": "silu", "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 32768,
        "use_sliding_window": False, "attention_dropout": 0.0,
        "bos_token_id": 151643, "eos_token_id": 151645,
        "initializer_range": 0.02, "max_window_layers": 21,
        "sliding_window": 32768, "transformers_version": "4.43.1",
        "use_cache": True,
    }
    if model_id == streamed.MODEL_ID:
        config.update(hidden_size=896, intermediate_size=4864,
                      num_hidden_layers=24, num_attention_heads=14)
    return config


def s1_config():
    return pinned_config()


@pytest.mark.parametrize("key,value", [
    ("architectures", ["Qwen2MoeForCausalLM"]), ("hidden_size", 896),
    ("torch_dtype", "float32"), ("tie_word_embeddings", False),
    ("quantization_config", {"quant_method": "gptq"}),
    ("rope_scaling", {"type": "linear", "factor": 2}),
    ("text_config", {"hidden_size": 1536}),
    ("rope_parameters", {"rope_theta": 1000000.0}),
])
@pytest.mark.parametrize("model_id", [streamed.MODEL_ID, streamed.S1_MODEL_ID])
def test_only_selected_verified_configuration_is_supported(key, value, model_id):
    config = pinned_config(model_id)
    streamed.validate_config(config, model_id)
    config[key] = 1024 if key == "hidden_size" else value
    with pytest.raises(ValueError, match=key):
        streamed.validate_config(config, model_id)


@pytest.mark.parametrize("model_id", [streamed.MODEL_ID, streamed.S1_MODEL_ID])
def test_config_rejects_missing_extra_and_other_pin(model_id):
    config = pinned_config(model_id)
    del config["bos_token_id"]
    with pytest.raises(ValueError, match="bos_token_id"):
        streamed.validate_config(config, model_id)
    config = pinned_config(model_id)
    config["arbitrary_setting"] = False
    with pytest.raises(ValueError, match="arbitrary_setting"):
        streamed.validate_config(config, model_id)
    other = streamed.S1_MODEL_ID if model_id == streamed.MODEL_ID else streamed.MODEL_ID
    with pytest.raises(ValueError, match="hidden_size"):
        streamed.validate_config(pinned_config(other), model_id)
    with pytest.raises(ValueError, match="Unsupported pinned model"):
        streamed.validate_config(config, "Qwen/unpinned")


def test_source_commit_rejected_before_import(tmp_path, monkeypatch):
    monkeypatch.setattr(streamed.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, stdout="wrong\n"))
    monkeypatch.setattr(streamed.importlib, "import_module", lambda name:
                        pytest.fail("Wrong source must not be imported"))
    with pytest.raises(ValueError, match="pinned llama.cpp commit"):
        streamed.load_upstream(tmp_path)


def test_convert_rejects_config_or_existing_output_before_loading(tmp_path, monkeypatch):
    output = tmp_path / "existing.gguf"
    output.write_bytes(b"preserve complete artifact")
    monkeypatch.setattr(streamed, "load_upstream", lambda *args: pytest.fail("Must not import"))
    with pytest.raises(FileExistsError):
        streamed.convert(tmp_path, tmp_path, output)
    assert output.read_bytes() == b"preserve complete artifact"
    config = s1_config()
    config["hidden_size"] = 896
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hidden_size"):
        streamed.convert(tmp_path, tmp_path, tmp_path / "new.gguf")
    assert not (tmp_path / "new.gguf").exists()


def test_index_and_chunks_release_safetensors_contexts(tmp_path, monkeypatch):
    import numpy as np
    import torch
    import safetensors
    from safetensors.torch import save_file

    matrix = torch.arange(35, dtype=torch.float32).reshape(7, 5).to(torch.bfloat16)
    vector = torch.tensor([-0.0, 1.0, -2.5, 0.125], dtype=torch.bfloat16)
    path = tmp_path / "model.safetensors"
    save_file({"matrix": matrix, "vector": vector}, path)
    original_open = safetensors.safe_open
    active = []
    contexts = []
    mapped_tensors = []
    requests = []

    class Slice:
        def __init__(self, source_slice, name):
            self.source_slice = source_slice
            self.name = name

        def get_shape(self):
            return self.source_slice.get_shape()

        def get_dtype(self):
            return self.source_slice.get_dtype()

        def __getitem__(self, rows):
            requests.append((self.name, rows.start, rows.stop))
            tensor = self.source_slice[rows]
            mapped_tensors.append(weakref.ref(tensor))
            return tensor

    class Context:
        def __init__(self, *args, **kwargs):
            assert not active, "Source mappings overlap"
            self.context = original_open(*args, **kwargs)
            contexts.append(weakref.ref(self))

        def __enter__(self):
            self.source = self.context.__enter__()
            active.append(self)
            return self

        def __exit__(self, *args):
            assert all(ref() is None for ref in mapped_tensors), "Mapped tensor escaped widening"
            active.pop()
            result = self.context.__exit__(*args)
            del self.source, self.context
            return result

        def keys(self):
            return self.source.keys()

        def get_slice(self, name):
            return Slice(self.source.get_slice(name), name)

        def get_tensor(self, name):
            pytest.fail("Never load a whole tensor")

    monkeypatch.setattr(safetensors, "safe_open", Context)
    specs = streamed.index_safetensors(path)
    assert list(specs) == ["matrix", "vector"]
    assert all(ref() is None for ref in contexts)
    chunks = []
    for start, end in streamed.row_ranges(specs["matrix"].shape, 40):
        value = streamed.load_chunk(specs["matrix"], start, end)
        assert not active and all(ref() is None for ref in contexts)
        assert value.dtype == np.float32 and value.flags.c_contiguous
        np.testing.assert_array_equal(value, matrix[start:end].float().numpy())
        chunks.append(value)
    np.testing.assert_array_equal(np.concatenate(chunks), matrix.float().numpy())
    assert requests == [("matrix", 0, 2), ("matrix", 2, 4), ("matrix", 4, 6), ("matrix", 6, 7)]
    actual_vector = streamed.load_chunk(specs["vector"], 0, 4)
    assert actual_vector.tobytes() == vector.float().numpy().tobytes()
    assert not active and all(ref() is None for ref in contexts)
    with pytest.raises(ValueError, match="row range"):
        streamed.load_chunk(specs["matrix"], 0, 8)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int32"])
def test_source_non_bf16_rejected(tmp_path, dtype):
    import torch
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    save_file({"matrix": torch.ones((3, 4), dtype=getattr(torch, dtype))}, path)
    with pytest.raises(ValueError, match="original BF16"):
        streamed.index_safetensors(path)


def _check_actual_writer(tmp_path, source, model_id=streamed.S1_MODEL_ID):
    import numpy as np
    import torch
    from safetensors.torch import save_file

    qwen2, gguf = streamed.load_upstream(source)
    torch.set_num_threads(1)
    config = pinned_config(model_id)
    config.update(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, vocab_size=7)
    (tmp_path / "config.json").write_text(json.dumps(config))
    # Deliberately cross row boundaries and include values whose bytes matter.
    matrix = (torch.arange(56, dtype=torch.float32).reshape(7, 8) / 7).to(torch.bfloat16)
    matrix[0, :4] = torch.tensor([-0.0, -2.5, 0.125, 3.0], dtype=torch.bfloat16)
    tensors = {
        "model.embed_tokens.weight": matrix,
        "model.norm.weight": torch.tensor([-0.0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": matrix[:4].clone(),
        "model.layers.0.self_attn.q_proj.bias": matrix[0].clone(),
        "model.layers.0.input_layernorm.weight": matrix[1].clone(),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    # Only the tiny synthetic tokenizer is supplied here. All production
    # metadata/vocabulary methods remain inherited, unchanged, in the adapter.
    def synthetic_vocab(self):
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_token_list([str(i) for i in range(7)])

    qwen2.set_vocab = synthetic_vocab
    adapted = streamed.make_streamed_model_class(qwen2, gguf)
    assert adapted.model_arch is qwen2.model_arch
    assert adapted.write is qwen2.write
    assert adapted.prepare_metadata is qwen2.prepare_metadata
    assert adapted.set_vocab is qwen2.set_vocab
    assert adapted.set_gguf_parameters is qwen2.set_gguf_parameters
    assert adapted.modify_tensors is qwen2.modify_tensors
    reference = tmp_path / "reference.gguf"
    qwen2(tmp_path, gguf.LlamaFileType.MOSTLY_BF16, reference,
          hparams=config, use_temp_file=False).write()
    expected = reference.read_bytes()
    for row_count in (1, 2, 3, 8):
        output = tmp_path / f"streamed-{row_count}.gguf"
        model = adapted(tmp_path, gguf.LlamaFileType.MOSTLY_BF16, output,
                        hparams=config, use_temp_file=False)
        model.chunk_bytes = row_count * 8 * 4
        assert all(isinstance(spec, streamed.TensorSpec) for spec in model.model_tensors.values())
        model.write()
        assert model.gguf_writer.temp_file is None
        assert output.read_bytes() == expected
        record = streamed.compare_gguf(source, reference, output)
        assert record["equivalent"] is True
        assert record["whole_file_equal"] is True
        assert record["tensor_metadata_equal"] is True
        assert record["mismatches"] == []
        assert record["artifacts"]["$REFERENCE"]["sha256"] == hashlib.sha256(expected).hexdigest()
        assert str(tmp_path) not in json.dumps(record)
        reader = gguf.GGUFReader(output, mode="r")
        assert len(reader.tensors) == len(tensors)
        for tensor in reader.tensors:
            # Delegate the actual HF->GGUF name map, not a fabricated one.
            name = next(name for name in tensors if model.map_tensor_name(name) == tensor.name)
            original = tensors[name]
            if original.ndim == 1:
                assert tensor.tensor_type == gguf.GGMLQuantizationType.F32
                assert tensor.data.tobytes() == original.float().numpy().tobytes()
            else:
                assert tensor.tensor_type == gguf.GGMLQuantizationType.BF16
                assert tensor.data.tobytes() == original.view(torch.int16).numpy().astype("<i2").tobytes()
        del reader
    # Synthetic valid GGUFs with changed bytes: never evidence about real models.
    reference_reader = gguf.GGUFReader(reference, mode="r")
    tensor = reference_reader.tensors[0]
    info = tensor.field
    mutations = {
        "payload": (tensor.data_offset, None),
        "tensor-name": (info.offset + info.parts[0].nbytes, b"x"),
        "tensor-shape": (info.offset + sum(part.nbytes for part in info.parts[:3]), struct.pack("<Q", 4)),
        "tensor-type": (info.offset + sum(part.nbytes for part in info.parts[:4]),
                        struct.pack("<I", int(gguf.GGMLQuantizationType.I16))),
    }
    vocab = reference_reader.fields["tokenizer.ggml.tokens"]
    mutations["vocabulary"] = (vocab.offset + sum(part.nbytes for part in vocab.parts) - 1, b"x")
    architecture = reference_reader.fields["general.architecture"]
    mutations["metadata"] = (architecture.offset + sum(part.nbytes for part in architecture.parts) - 1, b"x")
    for label, (offset, replacement) in mutations.items():
        changed = bytearray(expected)
        if replacement is None:
            changed[offset] ^= 1
        else:
            changed[offset:offset + len(replacement)] = replacement
        candidate = tmp_path / f"synthetic-mismatch-{label}.gguf"
        candidate.write_bytes(changed)
        record = streamed.compare_gguf(source, reference, candidate)
        assert record["equivalent"] is False, label
        assert record["whole_file_equal"] is False
        assert record["mismatches"], label
        section = "metadata" if label in ("metadata", "vocabulary") else "tensors"
        assert any(item["section"] == section for item in record["mismatches"])
        assert candidate.read_bytes() == changed and reference.read_bytes() == expected
    cli_record = tmp_path / "synthetic-cli-mismatch.json"
    result = subprocess.run(
        [sys.executable, str(Path(streamed.__file__)), "--source", str(source),
         "--compare", str(reference), str(candidate), "--record", str(cli_record)],
        capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(cli_record.read_text()) == record
    assert json.loads(result.stdout) == record
    reference_reader.data._mmap.close()
    del reference_reader, tensor, info, vocab, architecture
    # Extra trailing/padding bytes do not change any metadata or tensor payload.
    reordered_layout = tmp_path / "synthetic-layout-difference.gguf"
    reordered_layout.write_bytes(expected + b"\0" * 32)
    record = streamed.compare_gguf(source, reference, reordered_layout)
    assert record["equivalent"] is True and record["whole_file_equal"] is False
    assert record["tensor_metadata_equal"] is True
    assert record["basis"] == "tensor-and-all-metadata-sha256"
    # The upstream BF16 codec also defines special-value behavior; compare it
    # directly so chunking cannot silently change NaN/Inf encodings.
    spec = streamed.TensorSpec(tmp_path / "special.safetensors", "special", (3, 4))
    special = torch.tensor([[float("inf"), -float("inf"), float("nan"), -0.0],
                            [1, -1, 0.125, -2.5], [3, 4, 5, 6]], dtype=torch.bfloat16)
    save_file({"special": special}, spec.path)
    callbacks = [lambda start=start, end=end: streamed.load_chunk(spec, start, end)
                 for start, end in streamed.row_ranges(spec.shape, 16)]
    chunked = gguf.LazyChunkedTensor(callbacks, spec.shape, np.float32).quantize(gguf.GGMLQuantizationType.BF16)
    actual = tmp_path / "special.bin"
    with actual.open("wb") as stream:
        chunked.tofile(stream)
    assert actual.read_bytes() == gguf.quants.quantize(special.float().numpy(), gguf.GGMLQuantizationType.BF16).tobytes()


@pytest.mark.parametrize("model_id", [streamed.MODEL_ID, streamed.S1_MODEL_ID])
def test_real_pinned_writer_matches_upstream_bytes_on_synthetic_source(tmp_path, model_id):
    source = os.environ.get("CPU_DECODE_LLAMA_SOURCE")
    if not source:
        pytest.skip("Set CPU_DECODE_LLAMA_SOURCE for the actual pinned upstream API integration")
    code = "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['_check_actual_writer'](Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])"
    subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path), source, model_id],
                   check=True, cwd=Path(__file__).resolve().parents[1])


@pytest.mark.parametrize("model_id", [streamed.MODEL_ID, streamed.S1_MODEL_ID])
def test_convert_requires_selected_verified_snapshot_before_upstream_import(tmp_path, monkeypatch, model_id):
    (tmp_path / "config.json").write_text(json.dumps(pinned_config(model_id)))
    calls = []

    def reject(model, model_id):
        calls.append((model, model_id))
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(streamed, "verify_snapshot", reject)
    monkeypatch.setattr(streamed, "load_upstream", lambda *args: pytest.fail("Must not import"))
    output = tmp_path / "new.gguf"
    with pytest.raises(ValueError, match="checksum mismatch"):
        streamed.convert(tmp_path, tmp_path, output, model_id=model_id)
    assert calls == [(tmp_path, model_id)]
    assert not output.exists()


@pytest.mark.parametrize("model_id", [None, streamed.MODEL_ID, streamed.S1_MODEL_ID])
def test_conversion_cli_preserves_default_and_explicit_model_id(tmp_path, monkeypatch, model_id):
    calls = []
    monkeypatch.setattr(streamed, "convert", lambda *args, **kwargs: calls.append((args, kwargs)))
    command = ["streamed_gguf", "--source", str(tmp_path / "source"),
               "--model", str(tmp_path / "model"), "--outfile", str(tmp_path / "fresh.gguf")]
    if model_id is not None:
        command.extend(["--model-id", model_id])
    monkeypatch.setattr(sys, "argv", command)
    streamed.main()
    assert calls == [((tmp_path / "source", tmp_path / "model", tmp_path / "fresh.gguf", 4),
                      {"model_id": model_id or streamed.S1_MODEL_ID})]


def test_hash_region_reads_bounded_payload_and_detects_truncation(monkeypatch):
    monkeypatch.setattr(streamed, "HASH_CHUNK_BYTES", 7)
    payload = b"synthetic raw tensor payload" * 13

    class BoundedStream(io.BytesIO):
        def read(self, count=-1):
            assert 0 < count <= 7
            return super().read(count)

    assert streamed.hash_region(BoundedStream(payload), 3, 91) == hashlib.sha256(payload[3:94]).hexdigest()
    with pytest.raises(ValueError, match="Truncated"):
        streamed.hash_region(BoundedStream(payload), 0, len(payload) + 1)
    with pytest.raises(ValueError, match="Invalid"):
        streamed.hash_region(BoundedStream(payload), -1, 1)


def test_comparison_cli_records_false_and_preserves_inputs(tmp_path, monkeypatch, capsys):
    reference = tmp_path / "synthetic-reference.gguf"
    candidate = tmp_path / "synthetic-candidate.gguf"
    reference.write_bytes(b"synthetic reference")
    candidate.write_bytes(b"synthetic changed")
    output = tmp_path / "comparison.json"
    record = {"equivalent": False, "mismatches": [{"section": "tensors", "name": "synthetic"}]}
    calls = []

    def comparison(source, left, right):
        calls.append((source, left, right))
        return record

    monkeypatch.setattr(streamed, "compare_gguf", comparison)
    monkeypatch.setattr(sys, "argv", ["streamed_gguf", "--source", str(tmp_path),
                        "--compare", str(reference), str(candidate), "--record", str(output)])
    with pytest.raises(SystemExit) as status:
        streamed.main()
    assert status.value.code == 1
    assert calls == [(tmp_path, reference, candidate)]
    assert json.loads(output.read_text()) == record
    assert json.loads(capsys.readouterr().out) == record
    assert reference.read_bytes() == b"synthetic reference"
    assert candidate.read_bytes() == b"synthetic changed"
    with pytest.raises(FileExistsError):
        streamed.main()
    assert len(calls) == 1 and json.loads(output.read_text()) == record


def test_comparison_cli_error_record_is_portable_and_not_equivalent(tmp_path, monkeypatch):
    output = tmp_path / "error.json"

    def reject(*args):
        raise ValueError(f"invalid synthetic artifact at {tmp_path}")

    monkeypatch.setattr(streamed, "compare_gguf", reject)
    monkeypatch.setattr(sys, "argv", ["streamed_gguf", "--source", str(tmp_path),
                        "--compare", "reference.gguf", "candidate.gguf", "--record", str(output)])
    with pytest.raises(SystemExit) as status:
        streamed.main()
    assert status.value.code == 1
    record = json.loads(output.read_text())
    assert record["equivalent"] is False and record["error_type"] == "ValueError"
    assert "artifacts" not in record and str(tmp_path) not in output.read_text()
