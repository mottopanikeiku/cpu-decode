# Measurement protocol

## v2: matched F16 KV and the best measured CPU baseline

The v2 tools write only under `results/v2`; the original results and the archived protocol below are unchanged. This is still single-stream decode on one pinned Qwen2.5-0.5B-Instruct model, not prompt processing, batched serving or a downstream-task benchmark. Native and llama.cpp use F16 K/V and native CPU Release builds. The baseline is the verified Q8_0 GGUF and commit in `results/llama-preparation.json`, converted from the same BF16 snapshot as the native artifact. Formats need not be numerically identical: native group size/scale dtype and quality are recorded separately.

### Development checks against the unchanged original engine

These are short diagnostics, not the final comparison: two rounds, three repeats per invocation, 16 measured tokens, two threads and context 128. Each process exits before the next model loads. The original binary comes from commit `d98ba9c`; native v2 uses the unchanged per-row artifact, F16 KV and blocked attention. Commands, binary/source hashes, affinity, individual samples and operation timings are retained in each directory. This does not establish the calibrated grouped format's speed or quality.

| Change | Original strict median | New pool strict median | New strict spread |
|---|---:|---:|---:|
| Before row-kernel specialization | 67.72 | 57.38 | 3.97% |
| Separate grouped/row kernels | 66.42 | 57.26 | 5.61% |
| Claim eight contiguous fixed tasks | 62.94 | 61.21 | 3.78% |
| Rejected four-row kernel | 67.87 | 41.33 | 5.18% |
| Remove four-row kernel; explicit SIMD FMA | 69.27 | 69.85 | 3.32% |

Raw records: [before](../results/v2/regression-before/summary.json), [specialization](../results/v2/regression-row-specialization/summary.json), [claim-eight](../results/v2/regression-claim8/summary.json), [rejected kernel](../results/v2/regression-row4/summary.json), [FMA](../results/v2/regression-fma/summary.json). Spreads above 5% occur in these development records, including some original/OpenMP arms. Different windows are not a controlled causal decomposition.

In the last same-window comparison, original → new strict median milliseconds/token were gate/up plus SiLU **5.564 → 5.534**, vocabulary head **3.443 → 3.525**, down projection **3.075 → 3.175**, and attention **0.858 → 0.406**. Unpinned new workers achieved **69.56 tokens/s**, spread **2.25%**: unpinning was not required to recover the short-context regression. The original strict spread was **5.01%**, so the small rate difference is not evidence of a stable speedup.

The empty fixed-task [dispatch probe](../tools/pool_dispatch.cpp) records [two-worker hot](../results/v2/dispatch-before/t2-hot.json) and [six-worker hot](../results/v2/dispatch-before/t6-hot.json) medians of **100.8 ns** and **457.2 ns** per dispatch. With a separately excluded 20 µs serial gap, they were [125.6 ns](../results/v2/dispatch-before/t2-gap20us.json) and [1,744.3 ns](../results/v2/dispatch-before/t6-gap20us.json). This synthetic probe includes a checksum, not model work; a 20 µs gap does not guarantee the bounded-spin workers have slept.

### Fixed settings and sampling

Freeze native kernel, weight artifact, group size, scale dtype, scheduler, attention and RoPE **before** the final run. `freeze` verifies model/config/GGUF hashes, exact source-manifest equality, Release/native build settings, executable hashes and shared-library identities. It saves a digest-addressed `protocol.json`; subsequent windows reject changed artifacts. The full matrix is threads **1, 2, 4, 6, 12** × initial contexts **128, 1024, 4096**. Defaults are **64 measured tokens per repeat, five repeats per run request, two rounds and one untimed native warmup step**. Final aggregation rejects shorter sampling or development protocols.

Final `--kernel vnni` additionally requires `--quality results/v2/quality.json` (the default): a held-out **retained** decision bound to the calibrated chosen weight hash, configuration hash, engine binary and execution settings. The runner checks linked report hashes and independently requires mean KL ≤ Q8_0, p99 KL ≤ Q8_0, top-1 agreement ≥ Q8_0 and perplexity ≤ Q8_0. It archives the quality-file hash and eligibility proof in the frozen protocol; windows and the summarizer reject missing, rejected or changed eligibility. Development/ablation can time rejected VNNI paths with an explicit `experimental_vnni` marker, not a final claim. An unapproved auto-selected VNNI path is also rejected. Freeze and timing commands require nice 19; put `nice` inside an external wrapper if that wrapper changes priority.

Each cell tries flash attention **on, off, auto** × **pinned, unpinned, defaults**, with repacking enabled and upstream poll=50. The first two categories are process-restricted to the physical-first selected N CPUs: pinned also uses the matching mask/strict1, while unpinned uses mask0/strict0 inside that selected set. `defaults` uses mask0/strict0 on the full frozen allowed set. All nine enter the conservative best-baseline maximum. Actual process CPU sets and winner matching status are reported against the frozen native affinity, rather than assumed equal. `freeze --polls 0,50` can expand the list; every candidate must be sampled.

Process affinity is necessary even for pinned one-thread tests: at this upstream revision the OpenMP single-thread graph branch does not apply the stored worker mask. Echoed mask/strict flags alone do not establish effective placement. Both selected-set categories therefore receive process-level affinity at every thread count; the separately labeled full-set defaults category preserves an unrestricted baseline comparison.

`--cpu-order` preserves the supplied fast-physical-first order; on the development machine this is `0,1,4,2,3,5,6,7,10,8,9,11`. Elsewhere omit it to use `cpu-decode cpus`'s `allowed_cpu_ids`/`preferred_cpu_ids`. Native `freeze --affinity strict` (default) supplies exactly N selected IDs with `--cpu-set`; `--affinity unpinned` omits that option and inherits the full allowed mask without per-worker binding. Affinity is part of the frozen configuration and native metadata, including the actual full/selected `cpu_set`; it must not change midway through a final run.

### Sequential interleaved windows

Only **one model is resident at a time**. Each candidate uses **A B A B**: run ordinary native `bench` with five repeats, wait for its process/model to exit, run llama with five repeats and wait for exit, then repeat both. Each native invocation builds its prefix once and warms once; its repeats rewind to the original context/seed without rebuilding the prefix per token. No interactive native process stays resident during llama. With nine candidates, one cell has 18 native processes / **90 native samples** and 18 baseline processes / **10 samples per baseline configuration**. Loading, prefill and depth fill remain outside decode timing.

The baseline winner has the **highest median of its ten measured rates**, not a weaker convenient configuration. Ties use candidate ID. Compare **only the ten native samples interleaved with that winner**: equal invocation and sample counts. All other candidates/samples/commands/logs remain visible. Selection and reporting use the same samples, without a holdout; selection optimism favors the baseline. Both engines now use separate processes for each invocation. Native warms at the requested context; upstream warms before depth fill, so warmup placement still differs.

Every externally scheduled window has a 1,740-second deadline including setup, below 30 minutes. Use **one candidate per command** if all nine will not fit; collect every frozen candidate exactly once per cell. Timeout, nonzero exit, malformed output, setting mismatch or incomplete ABAB is visibly retained and never counted as successful sampling. Duplicate candidates are rejected rather than selecting a favorable rerun. Any failed/missing candidate prevents final-matrix acceptance. Public tools do not start or nest a scheduler.

### Commands

Set `$INT8_V2`, `$LLAMA_BENCH` and `$Q8_GGUF` to your prepared artifacts. Use the Python environment from the project lockfile. After choosing the native configuration:

```sh
nice -n 19 python -m tools.measure_v2 freeze --model "$INT8_V2" --model-manifest results/v2/quantized-manifest.json --llama "$LLAMA_BENCH" --gguf "$Q8_GGUF" --kernel simd512x4
nice -n 19 python -m tools.measure_v2 window --model "$INT8_V2" --llama "$LLAMA_BENCH" --gguf "$Q8_GGUF" --threads 6 --contexts 4096 --candidate auto-pinned-poll50
nice -n 19 python -m tools.measure_v2 bandwidth --threads 6
```

The second line is one candidate in one cell, not the entire matrix. Invoke it for all nine frozen candidate IDs in every matrix cell; invoke bandwidth separately for every thread count. Acquire external exclusivity per command, not around a potentially hours-long loop. `--candidate all` bundles candidates only when they fit within the deadline; it preserves the same sequential one-model lifecycle and cannot be combined with separate measurements of the same candidates.

For development only, freeze with `--development --steps 16 --repeats 3 --output results/v2/development`, then measure cells **2/128** and **6/4096** with the same output option. This cannot be aggregated into final data. To regenerate final artifacts:

```sh
nice -n 19 python -m tools.summarize_v2
nice -n 19 python -m tools.figure_v2
```

For development use `summarize_v2 --input results/v2/development --output results/v2/development/summary.json --allow-partial`, and point `figure_v2 --input`/`--output` there.

### Format-specific read ceiling and ablations

The 256 MiB read-only probe uses `taskset -c` with **the native configuration's actual eligible CPU set**. Strict native affinity uses its selected N CPUs plus ordered OpenMP places/binding; unpinned native affinity uses the full frozen allowed mask with OpenMP binding disabled. Dynamic teams are disabled in both. Both SIMD widths are measured, and the greater median supplies the bound; actual sets, bindings, samples and bytes/elapsed-time rates are checked. Baseline candidates keep their independently recorded selected/full-set categories.

For each compared native sample use its actual `bytes_per_token`, including its selected matrix format and scale storage, norm/bias, embedding, unique-KV-head reads and KV writes. `total_min` must equal their sum; LM-head subfields are already included and must not be counted again. Independently check KV bytes against the frozen model geometry and selected F16/F32 dtype. With geometry L layers, H KV heads and D head dimensions, cache storage per position is `L × H × D × 2 × sizeof(KV)`. The mean read length for context C and S measured steps is `C + (S+1)/2`, including the current position. No v1 FP32-KV or full-row-FP32-scale byte constant is reused. The ceiling is median read GB/s × 1e9 / actual mean minimum bytes/token; percentage is native median / ceiling × 100. This is an **EXT storage/read-bandwidth bound**, not a physical DRAM counter, and ignores activation traffic, write allocation, cache reuse, arithmetic and synchronization.

`ablation` collects two separate native invocations with the same sampling protocol, restricted to **2/128** and **6/4096**. Fixed labels are `per-row-scalar`, `simd256`, `simd512x4`, `blocked`, `f16-kv`, `pool`, `grouped`, `vnni`: per-row int8 scalar/OpenMP/scalar-attention/F32-KV; SIMD256; SIMD512x4; blocked attention; F16 KV; persistent pool; grouped32/F16 scales; VNNI. Keep unchanged settings, including affinity, identical at adjacent rungs and identify changed weight artifacts. Example:

```sh
nice -n 19 python -m tools.measure_v2 ablation --model "$INT8_V2" --llama "$LLAMA_BENCH" --gguf "$Q8_GGUF" --threads 6 --contexts 4096 --label vnni --kernel vnni --kv f16 --attention blocked --scheduler pool
```

Every rung records its weight hash, observed group size/scale dtype, flags, rates and profile bytes. `ablation_ladders` lists measured rungs and missing labels at each fixed cell; `complete_ablation_ladder` is separate from matrix completeness. The summarizer does not invent missing measurements or causal contributions.

### Output definitions and limits

- `protocol.json`: frozen native settings, sampling/candidate list, full allowed and selected CPU sets, model geometry, source/quantized/GGUF identities, binary/library hashes, compile flags/compiler/platform observations, portable aliases and final VNNI quality-file hash/eligibility proof when applicable.
- `window-tT-cC-*.json`: cell, candidates, ordered completed subprocess invocations and exit statuses, actual CPU-set eligibility, portable commands, raw native/upstream output and stdout/stderr paths. Verbose baseline logs preserve CPU/attention/tensor/repack messages; a requested repack flag is not proof that Q8_0 tensors were repacked.
- `bandwidth-tT.json` and `ablation-tT-cC-*.json`: full observations for same-core-set read probes and explicitly labeled fixed-cell rungs.
- `summary.json`: per-cell native and **best measured** baseline median/min/max/sample count/spread, actual winning parameters, flags and process CPU sets, explicit `winner_core_sets_matched`, all candidates/raw-log references, byte counts/read ceiling/percentage, ratios, quality eligibility, ablations, failed records, missing cells/candidates and noise.
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
