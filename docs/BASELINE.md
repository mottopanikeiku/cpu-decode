# What differs from llama.cpp

## v2 comparison

The v2 timing protocol is in [MEASUREMENTS.md](MEASUREMENTS.md#v2-matched-f16-kv-and-the-best-measured-cpu-baseline). Both engines now use **F16 K/V**, and native kernel, scheduler, attention, group size and scale dtype are frozen before final measurements. The baseline remains the same verified model and llama.cpp commit `6c73b3e12dc501de35fe5f6979960d06921a2f6c`; native group32/F16-scale int8 and upstream Q8_0 each store 8.5 bits per matrix weight before alignment/auxiliary tensors, but their activation quantization, rounding and layouts are not identical. Quality measurements, not bit width, determine their numerical differences.

The reported competitor is **the fastest measured baseline configuration in each thread/context cell**, after trying flash attention on/off/auto and pinned/unpinned affinity. Upstream Q8_0, F16 KV, CPU-only execution and repacking enabled are kept fixed. Poll is 50 unless an expanded poll candidate set was explicitly frozen. Pinned candidates receive a mask derived from the same selected native CPU IDs and `--cpu-strict 1`; unpinned candidates retain upstream default affinity. Winner flags, all candidate samples and verbose logs are stored, so a fast default or unpinned result cannot be hidden behind a weaker pinned/attention-off comparison.

### What the pinned CPU implementation establishes

The prepared build cache has `GGML_NATIVE=ON`, `GGML_BLAS=OFF`, `GGML_OPENMP=ON`, `GGML_CPU_REPACK=ON`; the x86 quantization compilation command uses `-O3 -DNDEBUG -march=native -fopenmp`. Individual cached `GGML_AVX*` toggles being OFF does **not** imply an AVX-disabled binary when native compilation is enabled. v2 freezes the actual build settings and binary/shared-library hashes and retains `llama-bench --verbose` runtime stderr for every invocation.

In pinned source, [the Q8_0 trait](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/ggml-cpu.c#L272-L279) converts activations to Q8_0 and supplies its integer-dot path. The [x86 dot implementation](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/arch/x86/quants.c#L1308-L1374) loads 32 signed weights and activations into 256-bit vectors, combines their FP16 scales and uses an FMA accumulation when compiled with AVX2. Its [integer helper branches](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/arch/x86/quants.c#L105-L133) distinguish AVX-VNNI-INT8, AVX512-VNNI+VL, AVX-VNNI and multiply/add fallback. These are **source-established alternatives**, not a claim that a particular function ran in a measured graph. A CPU-feature banner alone does not trace operator dispatch; BLAS being disabled also does not rule out the separately compiled llamafile SGEMM path.

Repacking deserves the same distinction. `repack:true` means extra CPU buffer types were allowed. [Q8_0's optimal-repack selector](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/repack.cpp#L5105-L5126) in this commit has NEON and RISC-V branches, **no x86 Q8_0 branch**. [Per-tensor debug messages](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/repack.cpp#L4916-L4919) and CPU_REPACK buffer messages in the saved verbose logs show actual tensor repacking when it happens. Do not call enabled repacking “Q8_0 x86 repacking,” or infer a VNNI/BLAS kernel from speed. Exact hot-function attribution is **unknown without an operator trace/profile**; the raw runtime logs establish reported CPU features, buffers and effective attention choices, not an instruction-by-instruction trace.

### Remaining timing differences

Per candidate the engines alternate A B A B, each A/B request producing five repeats of 64 decoded tokens. The comparison uses the ten native samples adjacent to the winning baseline's ten samples. The native process and prefilled KV are reused across candidates; baseline processes reload outside the timed interval and each perform upstream's one-token warmup before depth fill. This reduces repeated native autoregressive prefix work, but is not identical initialization. Raw output records both the native request count (60 samples across six candidates) and per-configuration count (10); final medians never pool native samples from unmatched candidate windows.

Both initial context and decode length are matched, not token IDs or generated trajectories. Native includes greedy argmax and per-operation instrumentation; upstream llama-bench excludes sampling and tokenization. Native's unique-head, actual-format byte profile supplies its bandwidth bound. No speed gap is attributed solely to storage bandwidth, VNNI, attention or scheduling without the fixed-cell ablations and numerical checks.

## Archived v1 comparison

The following implementation and KV differences describe the unchanged original results, not the current v2 configuration.

The baseline is pinned in `results/llama-preparation.json`. It is built from source with `GGML_NATIVE=ON`, OpenMP enabled, CPU only, Release mode, and four build jobs. The same verified BF16 snapshot is converted to GGUF and then Q8_0; no third-party prequantized model is used.

The small engine's int8 path widens signed weights to FP32 in each dot product, keeps FP32 activations, and applies one scale per output channel. llama.cpp's Q8_0 path quantizes the input vector to Q8_0 and computes integer dot products before applying block scales. The pinned sources explicitly contain VNNI-enabled integer dot helpers and an AVX2/FMA Q8_0 kernel:

- [Q8_0 type traits and activation conversion](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/ggml-cpu.c#L272-L279)
- [x86 integer dot helpers](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/arch/x86/quants.c#L105-L135)
- [Q8_0 dot product](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/arch/x86/quants.c#L1308-L1374)

This is a concrete algorithmic difference, not just a compiler switch. It can reduce arithmetic spent widening and multiplying each weight, at the cost of an additional activation quantization. The scales also differ: Q8_0 has 32-weight blocks and FP16 scales; this engine has full-row FP32 scales. The two formats are comparable bit widths, not identical accuracy or storage layouts.

The small engine opens separate static OpenMP row/head regions for individual operations. The baseline's compiled OpenMP path keeps one team across a complete graph evaluation ([source](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/ggml-cpu.c#L3419-L3438)). The small engine's attention is straightforward scalar FP32 inside each head; its SIMD setting only changes matrix-vector products. These differences are plausible explanations for a gap, not experimentally isolated causal contributions.

For timing, the small engine includes greedy argmax and per-operation instrumentation. llama-bench excludes sampling and tokenization. Initial context lengths, generation length, thread counts, warmup step counts and repetitions are matched. The small engine uses FP32 K/V; unchanged upstream llama.cpp uses default F16 K/V and flash attention auto, favoring its cache traffic and available attention kernels. Prompt IDs and generated trajectories are not matched: llama-bench uses synthetic inputs, whereas this engine repeats a fixed token sequence then greedily decodes. The eager Transformers reference uses BF16 weights and KV, so it is a separate naive comparison rather than a numerically matched int8 engine.

## Dependency attribution

The decoder core is written from scratch; it does not link ggml or reuse upstream inference kernels. The separate optional quality reader links the pinned llama.cpp library to measure its real outputs. [nlohmann/json 3.11.3](https://github.com/nlohmann/json/tree/v3.11.3), under the [MIT license](https://github.com/nlohmann/json/blob/v3.11.3/LICENSE.MIT), parses Safetensors headers and configuration JSON. CMake uses an available system package or its pinned release download. The repository's code is MIT licensed; Qwen weights remain under [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE). Baseline builds remain outside the repository.
