# cpu-decode

<!-- FINAL_RESULT_START -->
**v2 result:** My pinned VNNI/int16 decoder beat llama.cpp Q8_0 in **all six pre-specified cloud cells**, at **1.090–1.531× paired throughput**. This is **one virtual host with worker-lifetime CPU binding**, not unchanged laptop CLI performance. My selected weights improve KL agreement but slightly worsen perplexity.
<!-- FINAL_RESULT_END -->

I wrote the C++ [decoder](src/model.cpp), [kernels](src/kernels.cpp) and [attention](src/attention.cpp) from scratch.

## v2 quality

I selected g64f16 through [four-format calibration](results/v2/format-calibration.json) and [separate int16 checks](results/v2/quality-vnni16-final.json), against original BF16-storage/FP32-arithmetic weights. Matrix storage is 8.25 versus Q8_0's 8.50 bits/weight; artifacts are 509.73 versus 531.07 MB, including container overhead.

[Heldout](results/v2/corpus.json): 2,048 positions, fresh F16 KV and 512-input windows. Agreement is not task accuracy.

| Path | Mean KL, nats | p99 KL, nats | Top-1 agreement | Perplexity |
|---|---:|---:|---:|---:|
| [g64f16/int16](results/v2/heldout-g64f16-vnni16-final-f16.json) | 0.00093877 | 0.00336880 | 97.75% | 16.60984 |
| [llama.cpp Q8_0](results/v2/heldout-final-q8_0.json) | 0.00240027 | 0.00813769 | 95.90% | 16.60406 |

## v2 cloud timing

The host is **AMD Zen 4 EPYC (family 25, model 17; the model name wasn't exposed)**, identified from family/model, not an observed SKU. The sandbox exposes 24 virtual CPUs, not proven dedicated physical cores. Both engines use F16 KV, matching strict CPU sets and fresh GCC 12.2.0 `-march=native` builds.

| Threads | Initial context | Native tokens/s | llama tokens/s | Paired ratio | Individual 95% CI |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 39.887 | 32.110 | 1.2335 | [1.1649, 1.2970] |
| 2 | 128 | 55.135 | 50.394 | 1.1361 | [1.0631, 1.2054] |
| 4 | 128 | 93.067 | 87.622 | 1.0898 | [1.0183, 1.1318] |
| 1 | 4096 | 30.111 | 21.854 | 1.3436 | [1.3235, 1.4431] |
| 2 | 4096 | 45.312 | 29.774 | 1.5306 | [1.4643, 1.5969] |
| 4 | 4096 | 68.643 | 49.774 | 1.3627 | [1.2594, 1.4882] |

I retained sixteen 128-forward pairs per cell in eight ABBA quartets. The clock includes LM head/argmax and the final consumed token; load, prefill, warmup and rewind are excluded. Paired medians are not ratios of displayed medians. I bootstrap quartets 20,000 times: per-cell, not simultaneous intervals. Runtime and flash-selection pilots are excluded.

Native holds public `CpuBinding` for its worker lifetime; inactive native execution is stopped and the actual GGML pool paused outside clocks. llama.cpp selects flash ON/OFF/AUTO by pilot, uses poll50 and enables repacking; no repacked buffer was selected. [Raw samples](results/v2/cloud-vnni/final/raw.json), [spreads](results/v2/cloud-vnni/final/summary.json) and [methods](docs/MEASUREMENTS.md) retain provenance and failed work. [Estimated total cost](results/v2/cloud-vnni/run-cost.json): **$1.6074**, not an invoice.

My separate [earlier AVX2/FP32 run](results/v2/cloud/summary.json) completed only 2/6 cells and lost both, at 0.4108× and 0.3079×. Different hosts do not establish a causal VNNI speedup.

## v3: exact lookup and smaller KV

I added layer-major batching and four-column projection tiles that share weight loads. [Prompt lookup](src/generate.cpp) drafts earlier context tokens, verifies greedy predictions and rewinds rejected entries. [All 768/768 tokens](results/v3/lookup-acceptance.json) matched plain greedy across twelve prompts: acceptance was **53.9% copy-heavy**, **6.7% open-ended**, **27.9% overall (263/944 proposals)**. Acceptance is not acceleration; **I claim no v3 speedup**.

I center post-RoPE keys using a fixed per-head mean from the first 64 keys; the shared score offset cancels in softmax. [64 heldout positions near 2K/4K](results/v3/kv-quality.json), allocated KV at capacity 4096:

| Cache | MB | Mean KL | p99 KL | Oracle top-1 |
|---|---:|---:|---:|---:|
| F16 | 50.33 | 0.000855 | 0.003254 | 64/64 |
| Plain int8 | 26.74 | 0.030889 | 0.370448 | 57/64 |
| Centered int8 | 27.54 | 0.005105 | 0.038579 | 64/64 |

Centered KV saves **45.3%** versus F16 but remains numerically worse. Storage includes padding, scales, mean and retained prefix, not workspaces or peak process memory. The small probe includes a disclosed first-64-position F32-key warmup confound. **F16 remains default**.

## Laptop development and limits

<!-- FINAL_TABLE_START -->
I did **not** run the full 15-cell, nine-configuration laptop matrix. Cloud v2 results do not measure v3 batching, lookup or int8-KV speed.
<!-- FINAL_TABLE_END -->

Host load, clocks and NUMA are uncontrolled; I have no cloud read-bandwidth ceiling. This is one model, not downstream-task accuracy or batched serving. Lookup exactness is relative to its chosen numerical path. [Earlier short-window measurements](results/v2/int16-shipped-short/summary.json) remain development evidence.

## Reproduce and attribution

Linux x86-64, C++17/OpenMP, CMake and uv. [Methods](docs/MEASUREMENTS.md) cover pinned downloads, numerical checks, cloud execution, v3 reproduction and the [AMD identification source](https://docs.amd.com/api/khub/documents/LZ~6p62H~zRDhkNiPAE9NQ/content).

[Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) weights: Apache-2.0; my decoder: MIT. [Transformers](https://github.com/huggingface/transformers): oracle. [llama.cpp](https://github.com/ggml-org/llama.cpp): baseline. [Prompt-lookup attribution](https://github.com/apoorvumang/prompt-lookup-decoding) and [prior work](docs/PRIOR_WORK.md).

Written with AI coding assistance.
