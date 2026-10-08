"""Compare actual GGUF tensor storage with the engine's measured per-step bytes."""
import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

from tools.download_model import REVISION, file_hash
from tools.portable import portable
from tools.prepare_llama import LLAMA_COMMIT

KV_BYTES = {"f16": 2, "f32": 4}
# (llama artifact, GGUF file suffix, engine label, required matrix type or None to report whatever llama-quantize chose)
PAIRS = [("Q8_0", "q8_0", "q8-f16", "Q8_0"), ("Q4_0", "q4_0", "q4h8-f16", None)]


def engine_elements(traffic: dict, raw: dict) -> tuple[float, float]:
    per_byte = {"q8": 1, "q4": 2}
    return traffic["projection_weights"] * per_byte[raw["weight_format"]], traffic["lm_head"] * per_byte[raw["head_format"]]


def compare(gguf_path: Path, artifact: str, engine_label: str, required: str | None, manifest: dict, args) -> dict:
    expected = manifest["artifacts"].get(artifact)
    if expected is None:
        raise ValueError(f"{args.manifest} has no {artifact} artifact")
    digest = file_hash(gguf_path)
    if digest != expected["sha256"]:
        raise ValueError(f"{gguf_path} does not match the pinned {artifact} preparation hash")
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(gguf_path)
    matrices = [tensor for tensor in reader.tensors if len(tensor.shape) == 2]
    vectors = [tensor for tensor in reader.tensors if len(tensor.shape) == 1]
    if required is not None:
        wrong = sorted({t.tensor_type.name for t in matrices if t.tensor_type != GGMLQuantizationType[required]}
                       | {t.tensor_type.name for t in vectors if t.tensor_type != GGMLQuantizationType.F32})
        if wrong:
            raise ValueError(f"Expected all {required} matrices and F32 vectors in {gguf_path}, found {wrong}")
    matrix_elements = sum(int(t.n_elements) for t in matrices)
    matrix_bytes = sum(int(t.n_bytes) for t in matrices)
    auxiliary_bytes = sum(int(t.n_bytes) for t in vectors)
    embedding = next(t for t in matrices if t.name == "token_embd.weight")
    if any(t.name == "output.weight" for t in matrices):
        raise ValueError(f"{gguf_path} has an untied output.weight; the engine model uses a tied head")
    embedding_row_bytes = int(embedding.n_bytes) // int(embedding.shape[1])
    head_elements = int(embedding.n_elements)
    gguf_weight_bytes = matrix_bytes + auxiliary_bytes + embedding_row_bytes
    by_type = collections.defaultdict(lambda: {"tensors": 0, "elements": 0, "bytes": 0})
    for t in reader.tensors:
        row = by_type[t.tensor_type.name]
        row["tensors"] += 1
        row["elements"] += int(t.n_elements)
        row["bytes"] += int(t.n_bytes)
    types = {name: {**row, "bits_per_weight": 8 * row["bytes"] / row["elements"]} for name, row in sorted(by_type.items())}
    layers = int(reader.fields["qwen2.block_count"].contents())
    heads = int(reader.fields["qwen2.attention.head_count"].contents())
    kv_heads = int(reader.fields["qwen2.attention.head_count_kv"].contents())
    head_dim = int(reader.fields["qwen2.embedding_length"].contents()) // heads
    per_cached_token = {dtype: layers * kv_heads * head_dim * 2 * size for dtype, size in KV_BYTES.items()}
    pattern = re.compile(rf"engine-{re.escape(engine_label)}-t{args.threads}-c(\d+)\.json")
    files = sorted((int(m[1]), p) for p in args.measurements.glob(f"engine-{engine_label}-t{args.threads}-c*.json")
                   if (m := pattern.fullmatch(p.name)))
    if not files:
        raise FileNotFoundError(f"No engine-{engine_label}-t{args.threads}-c*.json in {args.measurements}")
    rows = []
    for context, path in files:
        raw = json.loads(path.read_text())
        traffic = {key: statistics.mean(s["bytes_per_token"][key] for s in raw["samples"]) for key in raw["samples"][0]["bytes_per_token"]}
        projection_elements, engine_head_elements = engine_elements(traffic, raw)
        if projection_elements + engine_head_elements != matrix_elements or engine_head_elements != head_elements:
            raise ValueError(f"Engine ({path}) and GGUF ({gguf_path}) matrix geometry differ")
        if traffic["kv_write"] != per_cached_token[raw["kv_dtype"]]:
            raise ValueError(f"Engine KV write bytes in {path} do not match GGUF geometry for {raw['kv_dtype']}")
        engine_weights = traffic["matrix_weights"] + traffic["scales"] + traffic["norm_bias"] + traffic["embedding"]
        # llama.cpp runs with F16 KV; scale the engine's measured KV bytes by element size.
        llama_kv_read = traffic["kv_read"] * KV_BYTES["f16"] / KV_BYTES[raw["kv_dtype"]]
        projection_scales = traffic["scales"] - traffic["lm_head_scales"]
        rows.append({
            "context": context, "steps": raw["steps"], "engine_kv_dtype": raw["kv_dtype"],
            "engine_weight_bytes": engine_weights, "gguf_weight_bytes": gguf_weight_bytes,
            "engine_kv_read_bytes": traffic["kv_read"], "llama_f16_kv_read_bytes": llama_kv_read,
            "engine_total_min_bytes": traffic["total_min"],
            "gguf_total_min_bytes": gguf_weight_bytes + llama_kv_read + per_cached_token["f16"],
            "engine_matrix_bits_per_weight_including_scales": 8 * (traffic["matrix_weights"] + traffic["scales"]) / matrix_elements,
            "engine_projection_bits_per_weight_including_scales": 8 * (traffic["projection_weights"] + projection_scales) / projection_elements,
            "engine_head_bits_per_weight_including_scales": 8 * (traffic["lm_head"] + traffic["lm_head_scales"]) / engine_head_elements,
            "gguf_matrix_bits_per_weight_including_scales": 8 * matrix_bytes / matrix_elements,
            "gguf_projection_bits_per_weight_including_scales": 8 * (matrix_bytes - int(embedding.n_bytes)) / (matrix_elements - head_elements),
            "gguf_head_bits_per_weight_including_scales": 8 * int(embedding.n_bytes) / head_elements,
            "engine_source": str(path)})
    return {"artifact": artifact, "engine_label": engine_label, "engine_weight_format": raw["weight_format"], "engine_head_format": raw["head_format"],
            "gguf": str(gguf_path), "gguf_sha256": digest, "layers": layers, "kv_heads": kv_heads, "head_dim": head_dim,
            "bytes_per_cached_token": per_cached_token, "gguf_matrix_elements": matrix_elements, "gguf_matrix_bytes": matrix_bytes,
            "gguf_auxiliary_bytes": auxiliary_bytes, "gguf_embedding_row_bytes": embedding_row_bytes,
            "gguf_token_embedding_type": embedding.tensor_type.name, "gguf_tensor_types": types, "rows": rows,
            "tensors": [{"name": t.name, "type": t.tensor_type.name, "elements": int(t.n_elements), "bytes": int(t.n_bytes)} for t in reader.tensors]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", type=Path, default=Path("external/llama.cpp") / LLAMA_COMMIT)
    parser.add_argument("--manifest", type=Path, default=Path("results/llama-preparation.json"))
    parser.add_argument("--measurements", type=Path, default=Path("results/measurements"))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("results/traffic.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["llama_commit"] != LLAMA_COMMIT:
        raise ValueError(f"{args.manifest} was prepared at {manifest['llama_commit']}, not {LLAMA_COMMIT}")
    sys.path.insert(0, str(args.llama_root / "source/gguf-py"))
    comparisons = []
    for artifact, suffix, engine_label, required in PAIRS:
        gguf_path = args.llama_root / f"qwen-{REVISION}-{suffix}.gguf"
        optional = required is None
        if optional and not (gguf_path.exists() and artifact in manifest["artifacts"]
                             and any(args.measurements.glob(f"engine-{engine_label}-t{args.threads}-c*.json"))):
            print(f"skipping {artifact} vs {engine_label}: GGUF, manifest entry or engine measurements absent", file=sys.stderr)
            continue
        comparisons.append(compare(gguf_path, artifact, engine_label, required, manifest, args))
    output = {"definition": "Storage-level lower bounds, not DRAM counters. KV reads are the engine's counted bytes averaged over the growing generation window; "
                            "llama.cpp F16 KV traffic is derived from the same geometry, not instrumented. GGUF weight bytes = all matrices + F32 vectors + one embedding row.",
              "llama_commit": LLAMA_COMMIT, "threads": args.threads, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(portable(output, {args.llama_root.resolve(): "$LLAMA_ROOT"}), indent=2) + "\n")
    print(json.dumps(portable([{key: value for key, value in c.items() if key != "tensors"} for c in comparisons],
                              {args.llama_root.resolve(): "$LLAMA_ROOT"}), indent=2))


if __name__ == "__main__":
    main()
