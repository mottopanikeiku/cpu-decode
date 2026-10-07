# cpu-decode

I built a memory-mapped [Qwen decoder](src/model.cpp), [grouped-int8 kernels](src/kernels.cpp) and [shared-GQA attention](src/attention.cpp) in C++. I compare numerical quality separately from decoding speed.

<!-- FINAL_RESULT_START -->
**Result:** My selected g64f16 weights have better distribution agreement than llama.cpp Q8_0 on calibration and heldout, but slightly worse perplexity. My separate cloud comparison completed **2 of 6 planned cells**: native lost both, at **0.4108× and 0.3079×** paired throughput. The host lacked VNNI, so these are the **AVX2/FP32 activation fallback**, not int16 timings or a laptop victory.
<!-- FINAL_RESULT_END -->

## Quality and format choice

I chose the smallest format passing mean/p99 KL and top-1 checks against Q8_0 in [four-format calibration](results/v2/format-calibration.json), then checked [int16 separately](results/v2/quality-vnni16-final.json). The reference uses original BF16-storage/FP32-arithmetic weights. I disclose perplexity and independent heldout:

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

## Cloud CPU result: partial matrix

I retain every completed cell; four are missing after a 40-minute timeout.

| Threads | Initial context | Native tokens/s | llama tokens/s | Paired ratio | 95% block CI | Outcome |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 128 | 11.062 | 26.762 | 0.4108 | [0.4061, 0.4428] | llama wins |
| 2 | 4096 | 7.042 | 22.777 | 0.3079 | [0.3070, 0.3117] | llama wins |

Missing: (threads, context) **(1,4096), (2,128), (4,128), (4,4096)**.
I use 128 full greedy forwards and 16 pairs per cell: eight ABBA quartets.
Paired medians are not the displayed medians' ratio. I bootstrap quartets
20,000 times. [Raw samples](results/v2/cloud/raw.json), [summary and
spreads](results/v2/cloud/summary.json) and [fixed design](tools/cloud_design.json)
retain the separate flash pilot.

Both backends use F16 KV and matching strict CPU sets. I compile both pinned
sources inside one CPU-only Modal container with `-march=native` and GCC
12.2.0. The sandbox exposes 24 CPUs with model name “unknown”; AVX2/F16C are
present but AVX-512/VNNI are absent. Native resolves to `simd256`, FP32
activations. llama.cpp uses flash on and poll50; repacking is enabled but no
repacked model buffer was selected.

I avoid expensive per-operator affinity syscalls in the cloud sandbox by
holding v2's public `CpuBinding` for the worker lifetime. The laptop path pins
per operator. Cloud numbers therefore show kernel and threading speed without
that syscall cost, not unchanged command-line performance. The
[estimated total cost](results/v2/cloud/run-cost.json) is **$0.8553**, including
two failed attempts, not an invoice.

## Laptop development, not cloud results

<!-- FINAL_TABLE_START -->
The separate full 15-cell, nine-configuration laptop comparison is **not run
in this branch**. My partial cloud matrix does not replace it.
<!-- FINAL_TABLE_END -->

My earlier [short ABAB window](results/v2/int16-shipped-short/summary.json)
measured 69.58 versus 65.13 tokens/s, but native's 5.83% spread exceeded the
noise flag: not a final speed claim. [Attention stages](results/v2/attention-stages.json)
and the [original engine](results/tables.md) remain separate history.

## Reproduce

I use Linux x86-64, C++17/OpenMP, CMake and uv. Int16 requires AVX-512 VNNI/BW and F16C; `auto` uses FP32. I require fresh outputs and exclusive laptop timing.

```sh
nice -n 19 make prepare
nice -n 19 make quality
nice -n 19 make final
```

`final` uses int16 only after its quality comparison passes, otherwise FP32. It runs laptop configurations sequentially and resumes completed units. [Methods and reproduction](docs/MEASUREMENTS.md) include the separate cloud launcher and summarizer. I keep weights and raw logits outside git.

## Limitations

- I have one virtualized cloud environment, two completed cells and no cloud read-bandwidth ceiling. Clocks, host load and NUMA placement are uncontrolled; intervals are per-cell, not simultaneous.
- I use the same cloud prefix, but each backend follows its own greedy tokens. Load, prefill, warmup and rewind are excluded.
- My quality tests use 512-input windows, not long-context task accuracy. Heldout cannot select a replacement format.
- This comparison does not evaluate batched serving, speculative decoding or downstream tasks.

## Prior work

[Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): Apache-2.0 weights. [Transformers](https://github.com/huggingface/transformers): oracle. [llama.cpp](https://github.com/ggml-org/llama.cpp): baseline, [pinned build](results/llama-preparation.json). [Prior work](docs/PRIOR_WORK.md) includes llama2.c, gemma.cpp, llamafile, T-MAC and BitNet. The from-scratch decoder is MIT licensed; its optional baseline-quality reader links upstream libraries.

Written with AI coding assistance.
