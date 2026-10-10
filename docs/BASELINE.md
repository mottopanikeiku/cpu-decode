# What differs from llama.cpp

The baseline is pinned in `results/llama-preparation.json`: commit `6c73b3e12dc501de35fe5f6979960d06921a2f6c`, Release, `GGML_NATIVE=ON`, OpenMP, CPU only. The same verified BF16 snapshot is converted to GGUF and quantized with the pinned `llama-quantize` to Q8_0 and Q4_0. No third-party GGUF is used.

## Now the same

- **Weight blocks.** Both use 32-weight blocks with an F16 scale. The engine's q8 is Q8_0's scheme (scale = max|w|/127). Its q4 uses Q4_0's levels (scale = signed extreme / −8, levels 0–15 offset by 8) but packs each 64 weights as one 32-byte group so a 256-bit load expands to two blocks without shuffles.
- **Integer dot products.** Both quantize the activation vector to int8 in 32-element blocks and multiply with integer instructions. On x86 the engine stores int8 weights and flips their sign bit (`w XOR 0x80 = w + 128`) to get the unsigned operand `vpdpbusd` needs; the extra `128 × Σa` per block is precomputed once per input vector and loaded as the accumulator's starting value. llama.cpp instead moves weight signs onto the activations. On AArch64 both use signed `sdot`.
- **KV cache.** F16 by default in both; the F32 pair disables llama.cpp's flash attention, because otherwise the pinned version casts an F32 cache to F16.

## Still different

- **Specialization.** The engine handles one architecture (tied-head Qwen2) and fuses what that allows: Q, K and V share one row range and one pass over the quantized input; so do gate and up. llama.cpp evaluates a general graph.
- **Attention.** The engine scores all seven query heads of a KV group against each K/V row while it is in registers, splits each KV head's history into per-thread chunks, and merges partial softmax states (flash decoding). llama.cpp's CPU flash-attention path has its own tiling.
- **Threads.** Both keep one OpenMP team for a whole token. The engine uses static row ranges with a barrier after each phase.
- **Weight layout at load.** The engine copies weights into 2 MiB-aligned anonymous memory advised for transparent huge pages (`--weights hugepage`; `huge_page_kib` is recorded). llama.cpp memory-maps the GGUF, and on some CPUs repacks Q4_0 into interleaved layouts at load.
- **Head.** llama-quantize's Q4_0 mix keeps the tied embedding/head at Q8_0, so the matched engine run is `q4h8` (q4 projections, q8 head). `q4` (q4 head too) is measured as an ablation.
- **Timing scope.** The engine includes greedy argmax and per-phase timers; llama-bench omits sampling and uses synthetic tokens.

## Dependency attribution

The decoder core is written from scratch; it does not link ggml or reuse upstream kernels. The optional quality reader links the pinned llama.cpp library to measure its real outputs. [nlohmann/json 3.11.3](https://github.com/nlohmann/json/tree/v3.11.3) (MIT) parses Safetensors headers and configuration. [SIMDe 0.8.2](https://github.com/simd-everywhere/simde) (MIT) is fetched only for the optional AVX-512 emulation build. Repository code is MIT licensed; Qwen weights remain under [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE).
