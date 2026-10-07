# cpu-decode

A small C++ engine for single-stream CPU decoding of pinned Qwen checkpoints.

**Question:** How close can int8 decoding get to a laptop's read-bandwidth ceiling without losing the quality comparison with llama.cpp Q8_0?

I built the memory-mapped forward pass in [model.cpp](src/model.cpp), grouped-int8 kernels in [kernels.cpp](src/kernels.cpp), and shared-GQA attention in [attention.cpp](src/attention.cpp). Signed-int16 activations use AVX-512 integer dots; transposed K, vector softmax, fixed-order merges and persistent workers reduce attention and scheduling work.

<!-- FINAL_RESULT_START -->
**Current result:** The [g64f16/int16 path](results/v2/quality-vnni16-final.json) beats actual Q8_0 on mean/p99 KL and oracle top-1 agreement in calibration and heldout; perplexity is slightly worse. One noisy development window favors it. The full speed matrix is **not run**: no all-context victory or read-ceiling target is claimed.
<!-- FINAL_RESULT_END -->

## Quality and format choice

[Four-format calibration](results/v2/format-calibration.json) compares original BF16-storage/FP32-arithmetic weights on exact teacher-forced tokens. Choose the fewest bytes among formats with strictly better mean/p99 KL and top-1 agreement than Q8_0. Selected g64f16 weights pass the [separate int16 decision](results/v2/quality-vnni16-final.json) on calibration and heldout. Perplexity is disclosure only.

| Calibration path | Matrix bits/weight | Artifact MB | Mean KL, nats | p99 KL, nats | Top-1 agreement | Perplexity |
|---|---:|---:|---:|---:|---:|---:|
| [g64f16, FP32 activations](results/v2/cal-g64f16.json) | 8.25 | 509.73 | 0.00080124 | 0.00262021 | 98.05% | 25.25774 |
| [g64f16, int16 activations](results/v2/cal-g64f16-vnni16-final.json) | 8.25 | 509.73 | 0.00080371 | 0.00262030 | 97.85% | 25.26315 |
| [llama.cpp Q8_0](results/v2/cal-q8_0.json) | 8.50 | 531.07 | 0.00236397 | 0.00697494 | 97.27% | 25.23197 |

MB includes container overhead; bits count matrix weights and scales. [Heldout](results/v2/corpus.json) scores 2,048 positions with fresh F16 caches:

| Heldout path | Mean KL, nats | p99 KL, nats | Top-1 agreement | Perplexity |
|---|---:|---:|---:|---:|
| [g64f16, FP32 activations](results/v2/heldout-final-g64f16-f16.json) | 0.00093785 | 0.00338862 | 97.90% | 16.61045 |
| [g64f16, int16 activations](results/v2/heldout-g64f16-vnni16-final-f16.json) | 0.00093877 | 0.00336880 | 97.75% | 16.60984 |
| [llama.cpp Q8_0](results/v2/heldout-final-q8_0.json) | 0.00240027 | 0.00813769 | 95.90% | 16.60406 |

Int16 slightly lowers agreement versus native FP32, but passes Q8. The [F32-cache control](results/v2/heldout-g64f16-vnni16-final-f32.json) is recorded. Agreement is not task accuracy.

## Final matrix

<!-- FINAL_TABLE_START -->
**Not yet run.** The full 15-cell, nine-configuration comparison will appear here after completion. Every losing and noisy cell remains visible.
<!-- FINAL_TABLE_END -->

## Timing on a cloud CPU

I use a [separate cloud harness](tools/cloud_design.json), not laptop timings.
I avoid expensive per-operator affinity syscalls in the cloud sandbox by
holding v2's public `CpuBinding` for the worker lifetime. The laptop path pins
per operator. Cloud numbers therefore show kernel and threading speed without
that syscall cost, not unchanged command-line performance.

## Development throughput and attention

![Retained development stages, not final throughput claims](results/v2/attention-stages.svg)

[Stage records](results/v2/attention-stages.json): g64f16, six threads, context 4096. Median attention time falls from **5.348 to 2.727 ms/token** after transposition; masked tails and parallel merges measure **2.914 / 2.894**. Singleton claims (**3.204**) were rejected. Several spreads exceed 5%; separately timed stages do not establish causality.

[One ABAB window](results/v2/int16-shipped-short/summary.json), two threads/context 128: **69.58 tokens/s** native versus **65.13** Q8_0, flash-auto and selected-core pinning. Six samples per engine; spreads **5.83% / 3.43%**. Native exceeds the noise flag. This is one configuration, not the strongest-of-nine comparison.

The unchanged [original per-row engine](results/tables.md) approached the short-context ceiling but lost at long contexts.

## Reproduce

Linux x86-64, C++17/OpenMP, CMake and uv; AVX-512 VNNI/BW and F16C for int16. Earlier laptop hardware: Ryzen AI 5 PRO 340, GCC 16.2.1, 2000 MiB process-group cap—not measured peak. Those local results used **$0 paid compute**. Use fresh output directories and exclusive timing access.

```sh
nice -n 19 make prepare
nice -n 19 make quality
nice -n 19 make final
```

`final` uses int16 only after its quality comparison passes, otherwise FP32; runs all configurations sequentially and resumes completed units. CLI `auto` remains FP32. [Methods and numerical bounds](docs/MEASUREMENTS.md) cover scheduling and workload differences. Weights and raw logits stay outside git.

## Limitations

- One laptop and the current 0.5B model; clocks/temperatures are not fixed.
- The read ceiling estimates minimum byte traffic, not measured DRAM utilization.
- Native greedy trajectories and upstream synthetic tokens differ; neither timing includes prompt processing.
- Quality uses 512-input windows, not long-context task evaluation. Heldout cannot choose a replacement format; int8 activations were rejected.
- No batched serving, speculative decoding or downstream task evaluation.

## Prior work

[Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): Apache-2.0 weights. [Transformers](https://github.com/huggingface/transformers): oracle. [llama.cpp](https://github.com/ggml-org/llama.cpp): baseline, [pinned build](results/llama-preparation.json). [Prior work](docs/PRIOR_WORK.md) includes llama2.c, gemma.cpp, llamafile, T-MAC and BitNet. The from-scratch decoder is MIT licensed; its optional baseline-quality reader links upstream libraries.

Written with AI coding assistance.
