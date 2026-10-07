# cpu-decode

<!-- FINAL_RESULT_START -->
**Result:** On **one virtual CPU host, model unknown**, my VNNI/int16 decoder beat pinned llama.cpp Q8_0 in **all six pre-specified cells**, at **1.090–1.531× paired throughput**. These are **worker-lifetime CPU-binding** measurements, not unchanged laptop command-line performance. My g64f16 format has better distribution agreement but slightly worse perplexity than Q8_0.
<!-- FINAL_RESULT_END -->
I wrote the C++ [decoder](src/model.cpp), [kernels](src/kernels.cpp) and [attention](src/attention.cpp).


## Quality and format choice

I selected g64f16 through [four-format calibration](results/v2/format-calibration.json) and [separate int16 checks](results/v2/quality-vnni16-final.json). The oracle uses original BF16-storage/FP32-arithmetic weights.

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

Int16 passes the Q8_0 checks; agreement is not task accuracy. I retain the [F32-cache control](results/v2/heldout-g64f16-vnni16-final-f32.json).

## Cloud VNNI result: complete matrix

I retained every cell, with 16 pairs of 128 full greedy forwards each: eight
ABBA quartets. Each clock includes the LM head and final consumed token.

| Threads | Initial context | Native tokens/s | llama tokens/s | Paired ratio | 95% block CI |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 39.887 | 32.110 | 1.2335 | [1.1649, 1.2970] |
| 2 | 128 | 55.135 | 50.394 | 1.1361 | [1.0631, 1.2054] |
| 4 | 128 | 93.067 | 87.622 | 1.0898 | [1.0183, 1.1318] |
| 1 | 4096 | 30.111 | 21.854 | 1.3436 | [1.3235, 1.4431] |
| 2 | 4096 | 45.312 | 29.774 | 1.5306 | [1.4643, 1.5969] |
| 4 | 4096 | 68.643 | 49.774 | 1.3627 | [1.2594, 1.4882] |

Paired medians are not ratios of displayed medians. I bootstrap quartets
20,000 times: per-cell, not simultaneous intervals.
[Raw samples](results/v2/cloud-vnni/final/raw.json), [spreads](results/v2/cloud-vnni/final/summary.json)
and [fixed design](tools/cloud_vnni_design.json) retain provenance; runtime-pilot pairs are excluded.

Both backends use F16 KV, strict CPU sets and fresh GCC 12.2.0
`-march=native` CPU-only builds. The sandbox exposes 24 CPUs and all required
VNNI flags; native resolves to `vnni16`/int16. llama.cpp selects flash
ON/OFF/AUTO by per-cell pilot, uses poll50 and enables repacking;
no repacked buffer was selected.

I hold v2's public `CpuBinding` for the native worker lifetime to avoid
per-operator affinity syscalls. The inactive native process is stopped;
llama.cpp's real GGML pool is paused outside the clock. The first pilot
stalled; I do not claim a proven cause. [Methods](docs/MEASUREMENTS.md)
document the isolation change and all failed work.

My [earlier AVX2/FP32 run](results/v2/cloud/summary.json) completed only
2/6 cells and lost both, at 0.4108× and 0.3079×. It remains separate:
different hosts do not establish a causal VNNI speedup.
[Total cost accounting](results/v2/cloud-vnni/run-cost.json) includes both
runs, setup, pilots and failures; these are estimates, not invoices.

## Laptop development

<!-- FINAL_TABLE_START -->
I did **not** run the full 15-cell, nine-configuration laptop matrix in this
branch. The complete cloud comparison is not that laptop matrix.
<!-- FINAL_TABLE_END -->

My [earlier short ABAB window](results/v2/int16-shipped-short/summary.json)
exceeded the noise flag: not a final speed claim.
[Attention stages](results/v2/attention-stages.json) and the [original engine](results/tables.md)
remain separate history.

## Reproduce

I use Linux x86-64, C++17/OpenMP, CMake and uv. Int16 requires AVX-512 VNNI/BW and F16C; `auto` uses FP32. I require fresh outputs and exclusive laptop timing.

```sh
nice -n 19 make prepare
nice -n 19 make quality
nice -n 19 make final
```

`final` uses int16 only after its quality comparison passes, otherwise FP32. It runs laptop configurations sequentially and resumes completed units. [Methods and reproduction](docs/MEASUREMENTS.md) include the separate cloud launcher and summarizer. I keep weights and raw logits outside git.

## Limitations

- I have one final virtual host and no cloud read-bandwidth ceiling. Clocks, host load and NUMA placement are uncontrolled; intervals are per-cell, not simultaneous.
- I use the same cloud prefix, but each backend follows its own greedy tokens. Load, prefill, warmup and rewind are excluded.
- My quality tests use 512-input windows, not long-context task accuracy. Heldout cannot select a replacement format.
- This comparison does not evaluate batched serving, speculative decoding or downstream tasks.

## Prior work

[Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) weights: Apache-2.0.
My decoder: MIT. [Transformers](https://github.com/huggingface/transformers)
is the oracle; [llama.cpp](https://github.com/ggml-org/llama.cpp) the baseline.
I document [prior work and upstream linking](docs/PRIOR_WORK.md).

Written with AI coding assistance.
