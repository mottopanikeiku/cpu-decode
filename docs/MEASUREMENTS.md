# Measurement protocol

The experiment asks about single-stream CPU decoding, not batched serving or prompt-processing speed. One model, one machine, and one quantization format are measured. Timings exclude model loading, tokenization and prompt prefill. Greedy selection is included in the small engine and eager measurements; llama-bench omits sampling, a small favorable difference for the baseline. llama-bench uses its synthetic tokens; the other paths use a repeated fixed token-ID sequence. These are matched shapes, not identical workloads.

## Machine and scheduling

The raw measurement directories contain CPU/OS/compiler/Python/library versions, invocation arguments, environment variables and portable command equivalents. Only directory locations are replaced (`$MODEL`, `$INT8`, `$RAW`, `$CACHE`, `$LLAMA_BENCH`, `$PYTHON`); flags, artifact hashes and numeric observations are preserved. Native builds use release optimization and native ISA selection. Reported timings used nice priority 19 with other scheduled compute paused, but unrelated desktop processes were not isolated. Clocks and temperatures were not fixed, and boost was not disabled. No GPU or paid compute was used.

Default sweep: 1, 2, 4, 6 and 12 threads; initial cache lengths 128, 1024 and 4096; 16 measured decoding steps; three repeats. Each model path warms one untimed decoding step. Upstream llama-bench warms before depth fill, while the other paths warm at the requested context; prefill/depth fill is excluded in every case. Report medians and minimum–maximum spread, not the reciprocal of mean latency. Any differently sized slice is explicitly recorded. The small engine uses FP32 KV, while upstream llama.cpp uses its supported default F16 KV and flash attention auto; this favors the baseline's cache traffic and available attention kernels. The eager baseline uses BF16 weights/KV and eager attention. Q8_0 stores 32 weights and a two-byte scale per block; the new engine stores int8 rows and a four-byte scale per output channel. Both use eight-bit weights, but neither cache precision nor quantization is numerically identical.

## Read-bandwidth ceiling

`tools/bandwidth.cpp` reads a 256 MiB array, substantially larger than the last-level cache, repeatedly with four independent XOR vector accumulators. It writes only a final checksum during each timed sweep. Initialization and warmup are excluded. SIMD256 and SIMD512 are measured rather than assuming a wider instruction is faster. Bytes/s is array size × passes / elapsed time. This is a read-only STREAM-style sweep, not STREAM triad and not a memory-controller counter.

For a decoded step at cache position n, the storage lower bound is:

- all projection matrix weights and scales, once;
- the tied vocabulary matrix as the LM head, once, plus one embedding row;
- each distinct layer's K and V cache through n, once, and its new K/V writes;
- norm and bias vectors.

The ideal tokens/s ceiling is sustained read bytes/s divided by these bytes/token. The numerator is read bandwidth even though the denominator includes the much smaller KV write term; this is an approximation. GQA lets several query heads share each KV head. A straightforward attention loop can read the same KV head repeatedly; the engine also records logical KV reads separately where applicable. Cache reuse, activation traffic, write allocation, dequantization, nonlinearities, reductions, synchronization and clock changes are not captured by the ideal ceiling. The reported percentage is achieved tokens/s / ideal tokens/s. It is not a claim to have measured physical DRAM utilization. Cache position grows over the generation window; use the engine's average step byte count when computing percentages.

`tools/traffic.py` reads actual GGUF tensor sizes and the native profile to compare formats. For the pinned model, FP32 KV stores `24 layers × 2 KV heads × 64 head dimensions × K/V × 4 bytes = 24,576 bytes` per cached token; F16 halves it. The generated traffic table averages unique-head reads over the growing generation window, alongside weight/scales/bias/embedding bytes for both formats. Q8 KV traffic is a geometry estimate, not instrumented DRAM traffic.

## Baselines and ablations

llama.cpp is built at the pinned commit in `tools/prepare_llama.py`, using native CPU flags and at most four build jobs. Its GGUF is converted from the exact pinned Hugging Face snapshot, then quantized to Q8_0. `llama-bench -p 0 -n 16 -d CONTEXT` excludes the depth-fill from decoding timing. The eager baseline uses Transformers, with its dtype stated in the raw output.

The ablations compare BF16 storage with int8 storage at identical FP32 arithmetic, scalar/SIMD256/SIMD512 matrix-vector kernels, one versus four SIMD512 accumulators, and direct versus cached RoPE, at fixed thread count and context. The four-accumulator variant exposes independent additions instead of one serial accumulator; it changes reduction order but not activation or weight quantization. Compare recorded medians, not expectations about Zen 5. Operation timings are collected inside the forward pass; the separate loop/timing-overhead field closes the breakdown to whole-step time. Interpret a dominant matrix-vector section as a mixture of weight loads and arithmetic, not direct proof of a bandwidth bottleneck.

## Correctness

`tools/reference.py` uses the downloaded BF16 values as the starting weights for a Hugging Face Transformers CPU oracle, with FP32 execution implemented without holding the whole expanded model at once. The comparison records the exact dtype strategy, absolute-logit tolerance and fixed prompt set. The unquantized engine must meet that tolerance and generate identical greedy tokens for the tested steps. Int8 quality is reported at every teacher-forced prompt position as top-1 agreement and KL(reference || quantized), rather than hiding changed tokens behind a text example. This is a small deterministic check, not a perplexity benchmark or a claim about downstream task accuracy.

The oracle verifies the pinned snapshot before loading and records its checked file identities. Full-model pytest comparisons require explicit `CPU_DECODE_MODEL`; an absent explicitly requested engine is an error, not a skip. The int8 artifact is also opened with the official Safetensors parser in integration tests. This quantizer requires each matrix element count to be divisible by four, so FP32 row scales remain aligned without invalid unindexed padding. Unsupported shapes are rejected before output creation; the loader rejects gaps and trailing unindexed data.

The optional `llama-quality` target links a small reader to the same pinned upstream library. It uses Q8_0, F16 KV and flash attention auto on every exact oracle teacher-forced token, with one-token batches and fresh contexts. The comparison verifies artifact and oracle hashes, then reports all per-position top-1/KL and separate prompt-only and reference-continuation aggregates. It does not evaluate Q8 free-running greedy generation. Thus the comparison describes the complete numerical paths, not an isolated weight-quantizer ranking.
