# What differs from llama.cpp

The baseline is pinned in `results/llama-preparation.json`. It is built from source with `GGML_NATIVE=ON`, OpenMP enabled, CPU only, Release mode, and four build jobs. The same verified BF16 snapshot is converted to GGUF and then Q8_0; no third-party prequantized model is used.

The small engine's int8 path widens signed weights to FP32 in each dot product, keeps FP32 activations, and applies one scale per output channel. llama.cpp's Q8_0 path quantizes the input vector to Q8_0 and computes integer dot products before applying block scales. The pinned sources explicitly contain VNNI-enabled integer dot helpers and an AVX2/FMA Q8_0 kernel:

- [Q8_0 type traits and activation conversion](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/ggml-cpu.c#L272-L279)
- [x86 integer dot helpers](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/arch/x86/quants.c#L105-L135)
- [Q8_0 dot product](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/arch/x86/quants.c#L1308-L1374)

This is a concrete algorithmic difference, not just a compiler switch. It can reduce arithmetic spent widening and multiplying each weight, at the cost of an additional activation quantization. The scales also differ: Q8_0 has 32-weight blocks and FP16 scales; this engine has full-row FP32 scales. The two formats are comparable bit widths, not identical accuracy or storage layouts.

The small engine opens separate static OpenMP row/head regions for individual operations. The baseline's compiled OpenMP path keeps one team across a complete graph evaluation ([source](https://github.com/ggml-org/llama.cpp/blob/6c73b3e12dc501de35fe5f6979960d06921a2f6c/ggml/src/ggml-cpu/ggml-cpu.c#L3419-L3438)). The small engine's attention is straightforward scalar FP32 inside each head; its SIMD setting only changes matrix-vector products. These differences are plausible explanations for a gap, not experimentally isolated causal contributions.

For timing, the small engine includes greedy argmax and per-operation instrumentation. llama-bench excludes sampling and tokenization. Initial context lengths and thread counts are matched, and both use FP32 K/V caches with llama.cpp flash attention disabled. Prompt IDs and generated trajectories are not matched: llama-bench uses synthetic inputs, whereas this engine repeats a fixed token sequence then greedily decodes. The eager Transformers reference uses BF16 weights and KV, so it is a separate naive comparison rather than a numerically matched int8 engine.

## Dependency attribution

The decoder is written from scratch; it does not link ggml or reuse upstream inference kernels. [nlohmann/json 3.11.3](https://github.com/nlohmann/json/tree/v3.11.3), under the [MIT license](https://github.com/nlohmann/json/blob/v3.11.3/LICENSE.MIT), parses Safetensors headers and configuration JSON. CMake uses an available system package or its pinned release download. The repository's code is MIT licensed; Qwen weights remain under [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE). Baseline builds remain outside the repository in the shared cache.
