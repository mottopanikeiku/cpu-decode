# cpu-decode

A from-scratch C++ int8 decoder for pinned Qwen2.5-0.5B-Instruct on a CPU.

**Question:** What can repeated context and softmax invariance buy in a small decoder?

I built lossless greedy [prompt lookup](src/generate.cpp), a layer-major [batched forward](src/model.cpp), and [int8 KV caches](src/kv_cache.cpp). Four-column [projection tiles](src/kernels.cpp) share weight loads while preserving single-token arithmetic. Centered keys use a fixed per-head mean estimated from the first 64 post-RoPE keys; the shared score shift cancels in softmax.

**Result:** [768/768 generated tokens](results/v3/lookup-acceptance.json) match plain greedy. Draft acceptance is **53.9% on copy-heavy prompts**, versus **6.7% on open-ended prompts**. [Mean-centered int8 KV](results/v3/kv-quality.json) saves **45.3%** of F16 cache storage at capacity 4096 and lowers plain-int8 mean KL from **0.030889 to 0.005105 nats**, but remains worse than F16. **Speed is not yet measured.**

## Lossless prompt lookup

I search earlier occurrences of the longest context suffix, draft up to four tokens, and verify them with the model in one batched pass. At the first disagreement I emit the model's greedy prediction and rewind rejected cache entries. This preserves greedy decoding for the chosen kernel/cache; it is not sampling and does not improve answer quality.

The [public prompt suite](configs/lookup-prompts.json) was committed before measuring: copying, editing and summarizing quoted text, alongside explanations, stories and code. [Raw token sequences and counts](results/v3/lookup-cases/) give:

| Prompt group | Prompts | Accepted / proposed drafts | Acceptance | Exact generated tokens |
|---|---:|---:|---:|---:|
| Copy-heavy | 6 | 228 / 423 | 53.9% | 384 / 384 |
| Open-ended | 6 | 35 / 521 | 6.7% | 384 / 384 |
| All | 12 | 263 / 944 | 27.9% | 768 / 768 |

Acceptance counts all proposals, including unused tokens after a mismatch. Lookup can do extra work on open-ended text; acceptance alone is not a speedup.

## KV storage and long-context quality

I reused the unchanged heldout Austen text, concatenated its windows without cache resets, and scored [64 fixed positions near 2k and 4k](results/v3/long-context-inputs.json). The oracle widens the original BF16 weights to FP32 and uses unmodified Transformers layers with causal chunks. All native rows use the same g64f16 weights and signed-int16 activations.

[Measured results](results/v3/kv-quality.json), allocated cache bytes at capacity 4096; MB is decimal:

| KV cache | MB | Mean KL, nats | p99 KL, nats | Oracle top-1 |
|---|---:|---:|---:|---:|
| F32 | 100.66 | 0.000849 | 0.003234 | 64 / 64 |
| F16 | 50.33 | 0.000855 | 0.003254 | 64 / 64 |
| Plain int8 | 26.74 | 0.030889 | 0.370448 | 57 / 64 |
| Mean-centered int8 | 27.54 | 0.005105 | 0.038579 | 64 / 64 |

Centering helps both context bands, but does not recover F16 distributions. The centered cache keeps raw prefix keys until the mean is available; values are int8 from the start. Storage includes padding, FP32 scales, the mean and retained prefix—not model/attention workspaces or peak process memory. **F16 remains the default.**

## Earlier weight-quality comparison

<!-- FINAL_RESULT_START -->
The earlier [2,048-position heldout comparison](results/v2/quality-vnni16-final.json) gives g64f16/int16 mean KL **0.00093877**, versus actual llama.cpp Q8_0 **0.00240027**. Perplexity is slightly worse. This is a different, short-window evaluation; the new KV table does not rerun the upstream baseline.
<!-- FINAL_RESULT_END -->

<!-- FINAL_TABLE_START -->
The earlier full throughput matrix is not published here. New batching, lookup and int8-KV timings are **not measured**.
<!-- FINAL_TABLE_END -->

## Reproduce

Linux x86-64, C++17/OpenMP, CMake and uv; AVX-512 VNNI/BW and F16C for the recorded int16 path. I used a Ryzen AI 5 PRO 340, GCC 16.2.1, at most two compute threads, and **$0 paid compute**. Set `INT8` to the unchanged g64f16 artifact; [methods](docs/MEASUREMENTS.md#repeating-the-new-comparisons) cover pinned downloads, quantization and all numerical commands.

```sh
nice -n 19 uv sync --locked --python 3.12 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
CPU_DECODE_QUANT_MODEL="$INT8" CPU_DECODE_KERNEL=vnni16 nice -n 19 uv run pytest -q tests/test_v3.py && ctest --test-dir build --output-on-failure
nice -n 19 uv run python -m tools.time_v3 --model "$INT8" --case 0 --output results/v3/timing-copy.json
```

Run timings only on an idle machine; matched conditions report separate prefill/decode medians and ranges. [Checks](results/v3/checks.json): five CTest checks and 562 Python tests passed, including real-model bitwise logits and exact greedy tests; four older optional integrations skipped.

## Limitations

- One model, one laptop, twelve prompts and one heldout book.
- Fixed 64-token continuations include tokens after EOS; this is not task accuracy.
- Centering uses a causal prefix estimate and F32-key warmup; their effects are not isolated.
- Greedy exactness is relative to the selected numerical path, not BF16-oracle text.
- No new speed result, batched serving, downstream evaluation or operation-level profiling for batch/int8 KV.

## Prior work

[Qwen weights](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): Apache-2.0. [Transformers](https://github.com/huggingface/transformers): oracle. [Prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding): drafting idea. [My attention-numerics study](https://github.com/mottopanikeiku/attention-numerics): key-centering motivation. [llama.cpp](https://github.com/ggml-org/llama.cpp): earlier baseline. [Further attribution](docs/PRIOR_WORK.md). The from-scratch engine is MIT licensed.

Written with AI coding assistance.
