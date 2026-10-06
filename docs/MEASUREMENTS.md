# Measurement protocol

The experiment asks about single-stream CPU decoding, not batched serving or prompt-processing speed. One model, one machine, and one quantization format are measured. Timings exclude model loading, tokenization and prompt prefill. Greedy selection is included in the small engine and eager measurements; llama-bench omits sampling, a small favorable difference for the baseline. llama-bench uses its synthetic tokens; the other paths use a repeated fixed token-ID sequence. These are matched shapes, not identical workloads.

## Machine and scheduling

The raw measurement directories contain CPU/OS/compiler/Python/library versions, invocation arguments, environment variables and each executable command. Native builds use release optimization and native ISA selection. Reported timings must run through `/home/alp/Projects/profile-program/bin/pp-run bench`, with nice priority 19. The wrapper waits for the shared heavy-work slots to empty and blocks new heavy jobs; it does not isolate unrelated desktop processes, fix clocks, fix temperatures or disable boost. Each timing slice must finish in 30 minutes. Builds, model loads and conversions use `pp-run heavy`, capped at at most 2000 MB. No GPU or paid compute is used.

Default sweep: 1, 2, 4, 6 and 12 threads; initial cache lengths 128, 1024 and 4096; 16 measured decoding steps; three repeats. Report medians and minimum–maximum spread, not the reciprocal of mean latency. Any differently sized slice is explicitly recorded. The same thread count is used for each compared engine. The small engine and llama.cpp use FP32 KV caches (`-ctk f32 -ctv f32` for llama.cpp), with llama.cpp flash attention disabled. The eager baseline uses BF16 weights and KV, recorded explicitly; it is a naive reference point, not a numerically matched weight-format comparison. Q8_0 uses 32-weight groups while the new engine uses output-channel scales, so these are both 8-bit weight formats but not numerically identical quantizers.

## Read-bandwidth ceiling

`tools/bandwidth.cpp` reads a 256 MiB array, substantially larger than the last-level cache, repeatedly with four independent XOR vector accumulators. It writes only a final checksum during each timed sweep. Initialization and warmup are excluded. SIMD256 and SIMD512 are measured rather than assuming a wider instruction is faster. Bytes/s is array size × passes / elapsed time. This is a read-only STREAM-style sweep, not STREAM triad and not a memory-controller counter.

For a decoded step at cache position n, the storage lower bound is:

- all projection matrix weights and scales, once;
- the tied vocabulary matrix as the LM head, once, plus one embedding row;
- each distinct layer's K and V cache through n, once, and its new K/V writes;
- norm and bias vectors.

The ideal tokens/s ceiling is sustained read bytes/s divided by these bytes/token. The numerator is read bandwidth even though the denominator includes the much smaller KV write term; this is an approximation. GQA lets several query heads share each KV head. A straightforward attention loop can read the same KV head repeatedly; the engine also records logical KV reads separately where applicable. Cache reuse, activation traffic, write allocation, dequantization, nonlinearities, reductions, synchronization and clock changes are not captured by the ideal ceiling. The reported percentage is achieved tokens/s / ideal tokens/s. It is not a claim to have measured physical DRAM utilization. Cache position grows over the generation window; use the engine's average step byte count when computing percentages.

## Baselines and ablations

llama.cpp is built at the pinned commit in `tools/prepare_llama.py`, using native CPU flags and at most four build jobs. Its GGUF is converted from the exact pinned Hugging Face snapshot, then quantized to Q8_0. `llama-bench -p 0 -n 16 -d CONTEXT` excludes the depth-fill from decoding timing. The eager baseline uses Transformers, with its dtype stated in the raw output.

The ablation changes one supported kernel setting at a time, at fixed thread count and context. Scalar FP32 accumulation is the slow arithmetic reference; SIMD256 and SIMD512 retain float activations and int8 weights. Compare recorded medians, not expectations about Zen 5. Operation timings are collected inside the forward pass and sum to less than whole-step timing because loop dispatch, sampling and timing instrumentation also take time. Interpret a dominant matrix-vector section as a mixture of weight loads and arithmetic, not direct proof of a bandwidth bottleneck.

## Correctness

`tools/reference.py` uses the downloaded BF16 values as the starting weights for a Hugging Face Transformers CPU oracle, with FP32 execution implemented without holding the whole expanded model at once. The comparison records the exact dtype strategy, absolute-logit tolerance and fixed prompt set. The unquantized engine must meet that tolerance and generate identical greedy tokens for the tested steps. Int8 quality is reported at every teacher-forced prompt position as top-1 agreement and KL(reference || quantized), rather than hiding changed tokens behind a text example. This is a small deterministic check, not a perplexity benchmark or a claim about downstream task accuracy.
