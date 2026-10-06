# cpu-decode

A small C++ engine for single-stream CPU decoding of one pinned Qwen checkpoint.

**Question:** How close can int8 decoding get to a laptop's read-bandwidth ceiling, and how does it compare with native llama.cpp?

I built the complete forward pass and FP32 KV cache in [model.cpp](src/model.cpp), with scalar and AVX2/AVX-512 matrix-vector kernels in [kernels.cpp](src/kernels.cpp). Weights are memory-mapped; offline quantization uses one FP32 scale per int8 output row. [The reference](tools/reference.py) checks the original BF16 values in FP32 arithmetic against Transformers; a separate reader measures real llama.cpp outputs.

**Result:** Short-context decoding approaches the estimated ceiling, but scalar attention leaves a long-context gap. More threads are not automatically faster.

## Measured result

[All thread counts and generated tables](results/tables.md), with [raw samples](results/measurements), use median tokens/s over three repeats; native ranges are minimum–maximum. Context is the initial cache length, not timed prefill.

| Threads | Context | This engine (range) | Read ceiling reached | llama.cpp Q8_0 | BF16 eager |
|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 69.03 (68.81–69.52) | 83.1% | 61.76 | 17.71 |
| 6 | 128 | 65.55 (64.17–66.18) | 78.0% | 68.53 | 19.31 |
| 6 | 1024 | 51.16 (48.51–52.16) | 63.6% | 61.03 | 17.59 |
| 6 | 4096 | 31.69 (31.66–31.82) | 45.1% | 44.06 | 11.81 |

This is **not an equal-quality comparison**. On the same fixed prompt positions, [native quality](results/quality-summary.json), [Q8_0 quality](results/llama-quality.json) and [actual storage accounting](results/traffic.json) give:

| Numerical path | Matrix bits/weight, including scales | Weight bytes/token | Top-1 agreement | Mean / worst KL, nats |
|---|---:|---:|---:|---:|
| Per-row int8, FP32 KV | 8.03 | 496.07 MB | 57/60 (95.0%) | 0.0288 / 0.7897 |
| Q8_0, F16 KV, flash auto | 8.50 | 525.12 MB | 56/60 (93.3%) | 0.00970 / 0.1827 |

These are distribution checks, not task accuracy; the lower Q8_0 mean KL matters despite its slightly lower top-1 count. The baseline's default F16 cache and flash-attention auto differ from this engine's FP32 cache and scalar attention, conservatively favoring the baseline.

[Unique KV reads](results/traffic.json), averaged across the generation window, are **3.35 / 25.37 / 100.87 MB** at the listed context lengths; F16 halves them. FP32 KV needs **24,576 bytes per cached token**. The weight total includes the tied vocabulary head, not just projections.

[Unquantized checks](results/quality-summary.json) cover 88 positions: maximum absolute logit error **0.000439**, with **32/32** greedy tokens matching. [Ablations](results/tables.md) show int8 alone giving **1.01×**, SIMD256 **5.21×** over scalar, widening **1.10×**, four accumulators **1.01×**, and cached RoPE **1.03×**. Small gains should not be read as stable causal effects. At the longest context, [attention takes 17.22 ms of 31.56 ms/token](results/summary.json), making it the clearest next optimization target.

## Reproduce

Requires Linux x86-64, C++17/OpenMP, CMake and uv; AVX-512 for the recorded kernel. [The recorded hardware/software](results/measurements/environment-engine-t1,2,4,6,12-c128-ksimd512x4-ropecached.json) is a Ryzen AI 5 PRO 340 with GCC 16.2.1. [Model checks](results/checks.json) used a 2000 MB memory cap; this is not a peak-memory measurement. Compute is local CPU with free downloads, no paid compute.

```sh
make build
make prepare
make measure llama-quality traffic
```

[llama.cpp is pinned](results/llama-preparation.json) to `6c73b3e12dc501de35fe5f6979960d06921a2f6c`, built Release with `GGML_NATIVE=ON`, CUDA/Vulkan off; [the preparation script](tools/prepare_llama.py) records the full flags. Models and raw logits stay outside git. [Methods](docs/MEASUREMENTS.md), [baseline differences](docs/BASELINE.md) and [cold review](docs/COLD_REVIEW.md) explain the comparison.

## Limitations

- One model and laptop; clocks/temperatures were not fixed. The ceiling is a storage/read-bandwidth estimate, not measured DRAM utilization.
- Shapes match, but llama-bench uses synthetic tokens and omits sampling; native/eager use greedy trajectories.
- Quality uses four short prompts. Long-context numerical agreement and free-running Q8_0 generation were not measured.
- No batched serving, speculative decoding or activation quantization; attention remains scalar within each head.

## Prior work

[Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) supplies the Apache-2.0 weights; [Transformers](https://github.com/huggingface/transformers) supplies the oracle. [llama.cpp](https://github.com/ggml-org/llama.cpp) is the native baseline. [Prior-work notes](docs/PRIOR_WORK.md) also cover llama2.c, gemma.cpp, llamafile, T-MAC and BitNet. The decoder core is written from scratch; the optional baseline-quality reader links llama.cpp. New code is MIT licensed.

Written with AI coding assistance.
