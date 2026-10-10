# Measurement protocol

Only single-stream decoding is timed. Model loading, tokenization and filling the context are excluded. The engine fills its context one token at a time; it has no batched prefill, so prefill speed is not measured or claimed.

## Sweep

`make measure` runs, for initial cache lengths 128, 1024 and 4096 and 1, 2, 4, 6 and 12 threads, 16 decoding steps after one untimed warmup step, three repeats each:

| Engine | Matched llama.cpp run |
|---|---|
| `q8-f16`: q8 weights and head, F16 KV | `q8_0-f16`: Q8_0, F16 KV, flash attention auto |
| `q4h8-f16`: q4 projections, q8 head, F16 KV | `q4_0-f16`: llama-quantize's default Q4_0 mix (its tied embedding/head is Q8_0), F16 KV |
| `q8-f32` (6 threads, 128 and 4096) | `q8_0-f32`: F32 KV with flash attention **off** (the pinned llama.cpp casts an F32 cache to F16 when flash attention is on) |

Both engines get the same KV type in every pair. Rates are medians with min–max ranges. llama-bench feeds synthetic tokens and omits sampling; the engine repeats a fixed token sequence and includes greedy argmax. The shapes match; the token streams do not.

Ablations (6 threads, contexts 128 and 4096) change one setting at a time: F32 vs F16 KV, mmap vs huge-page weight copy, unfused vs fused projections, scalar vs SIMD kernel, q4 head vs q8 head, q4 vs q8 projections. `tools/summarize.py` pairs every two runs that differ in exactly one field.

## Threads

Every engine, llama.cpp and bandwidth run gets `OMP_PROC_BIND=close OMP_PLACES=cores` by default (`tools/measure.py --places cores`). `--places fast` builds an explicit place list from the physical cores with the highest `lscpu -e` maximum clock, for chips that mix core types; `--places none` leaves the environment alone. The chosen places and `lscpu -e` are recorded next to each run. llama.cpp is built with OpenMP, so the same variables apply to it.

## Read-bandwidth ceiling

`tools/bandwidth.cpp` reads a 256 MiB array with four independent vector XOR accumulators (AVX2/AVX-512 on x86-64, NEON on AArch64) and reports bytes/s. It is a read-only STREAM-style sweep, not a memory-controller counter.

The ideal tokens/s at a cache length is that bandwidth divided by the bytes a step must touch: every projection's weights and block scales once, the tied vocabulary matrix once as the LM head plus one embedding row, norms and biases, the new K/V row, and each cached K/V row once per KV head. The engine counts these as it runs (`bytes_per_token.total_min`). Activations, cache-line effects and clock changes are not included, so the percentage of the ceiling is an estimate, not measured DRAM utilization.

## Quality

- `make correctness`: Transformers FP32 arithmetic on the pinned BF16 weights is the oracle. The unquantized engine (F32 KV) must match every logit within tolerance and every greedy token. Each quantized model is reported at every teacher-forced position of four fixed prompts as top-1 agreement and KL(reference ‖ candidate), with no pass threshold.
- `make llama-quality`: the same positions through llama.cpp Q8_0 and Q4_0 (F16 KV, flash attention auto, one token per batch).
- `make long-context`: the first 4096 tokens of tinyshakespeare (pinned commit and SHA-256). The oracle runs one FP32 forward pass with SDPA attention; the engine and llama.cpp emit logits at positions 960–1023 and 4032–4095 only. The same metrics are reported per window.

These are distribution checks, not task accuracy or perplexity.

## AVX-512 coverage

The x86 kernels use AVX-512 VNNI (`vpdpbusd`) and are selected at run time. CI also builds them through [SIMDe](https://github.com/simd-everywhere/simde) (`-DCPU_DECODE_EMULATE_AVX512=ON`) on an ARM runner, so their arithmetic is tested on hosts without AVX-512. That checks results, not speed.
