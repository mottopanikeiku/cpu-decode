"""Compare actual Q8_0 tensor storage with native per-step traffic estimates."""
import argparse
import json
import statistics
import sys
from pathlib import Path

from tools.download_model import REVISION, file_hash
from tools.portable import portable
from tools.prepare_llama import LLAMA_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", type=Path, default=Path("external/llama.cpp") / LLAMA_COMMIT)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("results/llama-preparation.json"))
    parser.add_argument("--native-results", type=Path, default=Path("results/measurements"))
    parser.add_argument("--output", type=Path, default=Path("results/traffic.json"))
    args = parser.parse_args()
    gguf_path = args.gguf or args.llama_root / f"qwen-{REVISION}-q8_0.gguf"
    manifest = json.loads(args.manifest.read_text())
    digest = file_hash(gguf_path)
    if manifest["llama_commit"] != LLAMA_COMMIT or digest != manifest["artifacts"]["Q8_0"]["sha256"]:
        raise ValueError("Traffic input does not match the pinned Q8_0 preparation")
    sys.path.insert(0, str(args.llama_root / "source/gguf-py"))
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(gguf_path)
    matrices = [tensor for tensor in reader.tensors if len(tensor.shape) == 2]
    vectors = [tensor for tensor in reader.tensors if len(tensor.shape) == 1]
    if any(t.tensor_type != GGMLQuantizationType.Q8_0 for t in matrices) or any(t.tensor_type != GGMLQuantizationType.F32 for t in vectors):
        raise ValueError("Expected all Q8_0 matrices and FP32 auxiliary vectors")
    matrix_elements = sum(t.n_elements for t in matrices)
    matrix_bytes = sum(t.n_bytes for t in matrices)
    auxiliary_bytes = sum(t.n_bytes for t in vectors)
    embedding = next(t for t in matrices if t.name == "token_embd.weight")
    embedding_row_bytes = embedding.n_bytes // int(embedding.shape[1])
    q8_weight_bytes = matrix_bytes + auxiliary_bytes + embedding_row_bytes
    layers = int(reader.fields["qwen2.block_count"].contents())
    heads = int(reader.fields["qwen2.attention.head_count"].contents())
    kv_heads = int(reader.fields["qwen2.attention.head_count_kv"].contents())
    head_dim = int(reader.fields["qwen2.embedding_length"].contents()) // heads
    f32_kv_bytes = layers * kv_heads * head_dim * 2 * 4
    rows = []
    for context in [128, 1024, 4096]:
        path = args.native_results / f"engine-t6-c{context}-simd512x4.json"
        raw = json.loads(path.read_text())
        traffic = {key: statistics.mean(s["bytes_per_token"][key] for s in raw["samples"]) for key in raw["samples"][0]["bytes_per_token"]}
        if traffic["matrix_weights"] != matrix_elements or traffic["kv_write"] != f32_kv_bytes:
            raise ValueError("Native and GGUF model geometry differ")
        native_weights = traffic["matrix_weights"] + traffic["scales"] + traffic["norm_bias"] + traffic["embedding"]
        rows.append({"context": context, "steps": raw["steps"], "native_weight_bytes": native_weights, "q8_weight_bytes": q8_weight_bytes,
                     "native_kv_read_min_bytes": traffic["kv_read_min"], "q8_f16_kv_read_min_bytes": traffic["kv_read_min"] / 2,
                     "native_total_min_bytes": traffic["total_min"], "q8_total_min_bytes": q8_weight_bytes + traffic["kv_read_min"] / 2 + f32_kv_bytes / 2,
                     "native_matrix_bits_per_weight_including_scales": 8 * (traffic["matrix_weights"] + traffic["scales"]) / matrix_elements,
                     "q8_matrix_bits_per_weight_including_scales": 8 * matrix_bytes / matrix_elements, "native_source": str(path)})
    output = {"definition": "Storage-level lower bounds, not DRAM counters. KV reads are unique-head estimates averaged over the growing generation window; Q8 F16 KV traffic is derived from geometry, not instrumented.",
              "gguf_sha256": digest, "llama_commit": LLAMA_COMMIT, "layers": layers, "kv_heads": kv_heads, "head_dim": head_dim,
              "fp32_bytes_per_cached_token": f32_kv_bytes, "f16_bytes_per_cached_token": f32_kv_bytes // 2,
              "q8_matrix_bytes": matrix_bytes, "q8_auxiliary_bytes": auxiliary_bytes, "q8_embedding_row_bytes": embedding_row_bytes,
              "rows": rows, "tensors": [{"name": t.name, "type": t.tensor_type.name, "elements": t.n_elements, "bytes": t.n_bytes} for t in reader.tensors]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(portable(output), indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
