# Prior work

Checked before implementation on 2026-10-06.

- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp): established full-model CPU inference, GGUF, Q8_0 and several lower-bit formats, architecture-specific vector kernels and a thread pool. This is the strongest baseline here, not a library used by the new engine.
- [llama2.c](https://github.com/karpathy/llama2.c): a small, readable decoder with a KV cache. Its teaching-oriented organization informs the choice to keep one forward pass easy to inspect. Qwen requires grouped-query attention, split-half RoPE, QKV biases and tied embeddings, which must not be silently treated as Llama 2.
- [gemma.cpp](https://github.com/google/gemma.cpp): vertically integrated CPU inference with portable SIMD, compressed weights and topology-aware threading. It demonstrates that a compact engine is not itself new.
- [llamafile's tinyBLAS](https://github.com/Mozilla-Ocho/llamafile/tree/main/llamafile): CPU matrix kernels and careful scheduling alongside llama.cpp. Its dense matrix multiplication focus is particularly relevant to prefill; this experiment instead measures single-vector decoding.
- [T-MAC](https://github.com/microsoft/T-MAC): lookup-table mixed-precision kernels avoid conventional dequantization for low-bit weights. We do not implement its algorithms; an int8 experiment cannot support a claim about its low-bit results.
- [bitnet.cpp](https://github.com/microsoft/BitNet): kernels for trained ternary models, not a interchangeable post-training quantization baseline for Qwen. No claim that ordinary Qwen can become lossless ternary inference is made.

## What this adds

A deliberately narrow measurement: one pinned Qwen2.5-0.5B-Instruct model on one mobile Zen 5 CPU, a from-scratch decoder checked against Transformers, and a thread/context sweep tied to measured read bandwidth. The useful result is the gap to an estimated bandwidth ceiling and to llama.cpp, including negative optimization results. It is not a new general-purpose inference framework.

No upstream engine code is copied. Safetensors and tokenizer formats follow their published formats; the model remains under its Apache-2.0 license. Any third-party parser is identified separately.
