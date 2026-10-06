# Measurement protocol

## v2: matched F16 KV and the best measured CPU baseline

The v2 tools write only under `results/v2`; the original results and the archived protocol below are unchanged. This is still single-stream decode on one pinned Qwen2.5-0.5B-Instruct model, not prompt processing, batched serving or a downstream-task benchmark. Native and llama.cpp use F16 K/V and native CPU Release builds. The baseline is the verified Q8_0 GGUF and commit in `results/llama-preparation.json`, converted from the same BF16 snapshot as the native artifact. Formats need not be numerically identical: native group size/scale dtype and quality are recorded separately.

### Fixed settings and sampling

Freeze native kernel, weight artifact, group size, scale dtype, scheduler, attention and RoPE **before** the final run. `freeze` verifies model/config/GGUF hashes, exact source-manifest equality, Release/native build settings, executable hashes and shared-library identities. It saves a digest-addressed `protocol.json`; subsequent windows reject changed artifacts. The full matrix is threads **1, 2, 4, 6, 12** × initial contexts **128, 1024, 4096**. Defaults are **64 measured tokens per repeat, five repeats per run request, two rounds and one untimed native warmup step**. Final aggregation rejects shorter sampling or development protocols.

Each cell tries flash attention **on, off, auto** × **pinned, unpinned**, with repacking enabled and upstream poll=50. `freeze --polls 0,50` can add a poll choice, but the candidate list must be fixed before the final run and every candidate must be sampled. Unpinned means the upstream default mask `0x0`, strict=false, within the externally allowed CPU set; it does not mean escape from a container's affinity. Pinned uses `--cpu-mask HEX --cpu-strict 1`. The mask is built from the first N entries of the identical native CPU order.

`--cpu-order` explicitly preserves the supplied fast-physical-first order; on the development machine that is `0,1,4,2,3,5,6,7,10,8,9,11` (six physical cores, then their SMT siblings). Do not reuse those IDs blindly elsewhere: omit the override to use `cpu-decode cpus`'s `allowed_cpu_ids` and `preferred_cpu_ids` topology/frequency metadata. The runner rejects duplicate, unavailable or insufficient IDs. Native workers use `--cpu-set` with exactly the selected N entries.

### Interleaved windows without repeated native prefill

`window --candidate all` runs one cell per externally scheduled benchmark window. It starts `cpu-decode bench --interactive 1` once, builds the native prefix once, warms once, and waits for `ready:true`. For **each** baseline candidate it requests native A (five repeats), launches baseline B (five repeats), requests A again, then launches B again: **A B A B**. Native rewinds to the original fixed cache length and seed for every repeat; it does not rebuild the prefix per token or copy KV. One cell with six candidates therefore has 12 native requests / **60 native samples**, plus 12 baseline processes / **10 samples per baseline configuration**. Native remains idle while B runs. Loading, depth fill and prefill are outside decode timing.

The baseline winner is the configuration with the **highest median of its ten measured tokens/s samples**, not the upstream default and not a weaker convenient baseline. Ties use candidate ID for deterministic output. The comparison uses **only the ten native samples interleaved with that winning configuration**: equal run-request/process counts and equal sample counts. All other native samples, candidates, commands and logs remain visible, and their total sample count is reported. Selection and reporting use the same samples, not an independent holdout; selection optimism favors the baseline. The two native rounds reuse one warmed process; the two baseline rounds each load their model and run the upstream one-step warmup. That cache-residency asymmetry is disclosed rather than treated as identical initialization.

Each window has a 1,740-second deadline including setup and all candidates, leaving a minute below 30 minutes. Timeout, nonzero exit, malformed output, missing candidate, mismatched settings or incomplete ABAB sequence is retained as unsuccessful evidence, never counted as a successful timing sample. If a shared native process exits unsuccessfully, all its emitted results are invalidated. Files are not silently overwritten, and duplicate candidates are rejected rather than choosing the better rerun. A failed candidate prevents a complete-final-matrix claim. Run heavy/timing commands through your external resource scheduler when needed; public tools do not start or nest a scheduler.

### Commands

Set `$INT8_V2`, `$LLAMA_BENCH` and `$Q8_GGUF` to your prepared artifacts. Use the Python environment from the project lockfile. After choosing the native configuration:

```sh
nice -n 19 python -m tools.measure_v2 freeze --model "$INT8_V2" --model-manifest results/v2/quantized-manifest.json --llama "$LLAMA_BENCH" --gguf "$Q8_GGUF" --kernel vnni
nice -n 19 python -m tools.measure_v2 window --model "$INT8_V2" --llama "$LLAMA_BENCH" --gguf "$Q8_GGUF" --threads 6 --contexts 4096 --candidate all
nice -n 19 python -m tools.measure_v2 bandwidth --threads 6
```

The second and third lines are individual timing windows, not the entire matrix. Invoke the second separately for every matrix cell and the third for every thread count; acquire external exclusivity separately per command, not around a potentially hours-long shell loop. `--candidate ID` can split a cell if a slower machine needs shorter windows; use every frozen ID once. That split loads the native prefix once per candidate and is distinguishable in raw process records.

For development only, freeze with `--development --steps 16 --repeats 3 --output results/v2/development`, then measure cells **2/128** and **6/4096** with the same output option. This cannot be aggregated into final data. To regenerate final artifacts:

```sh
nice -n 19 python -m tools.summarize_v2
nice -n 19 python -m tools.figure_v2
```

For development use `summarize_v2 --input results/v2/development --output results/v2/development/summary.json --allow-partial`, and point `figure_v2 --input`/`--output` there.

### Format-specific read ceiling and ablations

The existing 256 MiB read-only probe is pinned with `taskset -c` to the **same selected core set** as native and pinned baseline. Ordered `OMP_PLACES`, binding and dynamic=false control the probe's OpenMP workers. SIMD256 and SIMD512 are both measured; the greater median bandwidth is used, and both raw results remain. Probe priority, core sets, samples and bytes/elapsed-time rates are checked.

For each compared native sample use its actual `bytes_per_token`, including its selected matrix format and scale storage, norm/bias, embedding, unique-KV-head reads and KV writes. `total_min` must equal their sum; LM-head subfields are already included and must not be counted again. Independently check KV bytes against the frozen model geometry and selected F16/F32 dtype. With geometry L layers, H KV heads and D head dimensions, cache storage per position is `L × H × D × 2 × sizeof(KV)`. The mean read length for context C and S measured steps is `C + (S+1)/2`, including the current position. No v1 FP32-KV or full-row-FP32-scale byte constant is reused. The ceiling is median read GB/s × 1e9 / actual mean minimum bytes/token; percentage is native median / ceiling × 100. This is an **EXT storage/read-bandwidth bound**, not a physical DRAM counter, and ignores activation traffic, write allocation, cache reuse, arithmetic and synchronization.

`ablation` collects two native run requests using the same sampling protocol, restricted to **2/128** and **6/4096**. The fixed ordered labels are `per-row-scalar`, `simd256`, `simd512x4`, `blocked`, `f16-kv`, `pool`, `grouped`, `vnni`: respectively per-row int8 scalar/OpenMP/scalar-attention/F32-KV; SIMD256; SIMD512x4; blocked attention; F16 KV; persistent pool; grouped32/F16 scales; VNNI. Keep unchanged settings identical at adjacent rungs and identify changed weight artifacts. Example rung:

```sh
nice -n 19 python -m tools.measure_v2 ablation --model "$INT8_V2" --llama "$LLAMA_BENCH" --gguf "$Q8_GGUF" --threads 6 --contexts 4096 --label vnni --kernel vnni --kv f16 --attention blocked --scheduler pool
```

Every rung records its weight hash, observed group size/scale dtype, flags, rates and profile bytes. `ablation_ladders` lists measured rungs and missing labels at each fixed cell; `complete_ablation_ladder` is separate from matrix completeness. The summarizer does not invent missing measurements or causal contributions.

### Output definitions and limits

- `protocol.json`: frozen native settings, sampling/candidate list, CPU ordering, model geometry, source/quantized/GGUF identities, binary/library hashes, build cache settings, compiler/platform/CPU observations and environment aliases.
- `window-tT-cC-*.json`: process readiness/exit status, cell, candidates, ordered run requests, portable full commands, parsed raw native/upstream output, and paths to stdout/stderr. `--verbose` baseline stderr retains runtime CPU features, attention and tensor/repack messages; a requested repack flag is not proof that Q8_0 tensors were repacked.
- `bandwidth-tT.json` and `ablation-tT-cC-*.json`: full observations for same-core-set read probes and explicitly labeled fixed-cell rungs.
- `summary.json`: per-cell native and **best measured** baseline median/min/max/sample count/spread, actual winning parameters and flags, all candidates/raw-log references, actual byte counts, read ceiling, percentage, ratios, ablation observations, failed records, missing cells/candidates and noisy cells.
- `complete_final_matrix` requires all 15 cells, every frozen candidate and matched bandwidth with no failures. `targets_all_cells` is false for incomplete/development data. Default target predicates are native/best-baseline ≥1 and ceiling percentage ≥50; optional `--target-ratio` and `--target-ceiling-percent` are written into the summary, not silently changed.
- Spreads are `100 × (max − min) / median`. **Every spread >5% is disclosed**, including losing baseline candidates. Samples are never discarded for noise. `decode.svg` is generated only from summary values, with native/best-baseline/estimated-ceiling curves, min–max bars and noise markers.

Artifacts replace local directories with portable aliases (`$INT8`, `$GGUF`, `$ENGINE`, `$LLAMA_BENCH`, `$LLAMA_ROOT`, `$BANDWIDTH`, `$HOME`, `$PYTHON`). Numeric CPU IDs, flags, hashes and observations are preserved. Nice 19 and exclusive scheduled benchmarking do not isolate the desktop, fix temperature/clocks or disable boost. Matched shapes still are not identical token trajectories: native uses the fixed seed sequence and greedy argmax, upstream uses synthetic tokens and excludes sampling. Operation instrumentation remains included in native timing.

## Archived v1 protocol

Everything below describes the unchanged original runs, including their shorter sampling, FP32 native KV and full-row scale storage; it does not define v2 settings.

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
