# Measurement protocol

## v2: matched F16 KV and the best measured CPU baseline

The v2 tools write only under `results/v2`; the original results and the archived protocol below are unchanged. This is still single-stream decode on one pinned Qwen2.5-0.5B-Instruct model, not prompt processing, batched serving or a downstream-task benchmark. Native and llama.cpp use F16 K/V and native CPU Release builds. The baseline is the verified Q8_0 GGUF and commit in `results/llama-preparation.json`, converted from the same BF16 snapshot as the native artifact. Formats need not be numerically identical: native group size/scale dtype and quality are recorded separately.

### VNNI-only cloud comparison

I committed the [unchanged six-cell workload](../tools/cloud_vnni_design.json)
at `21481f3` before collecting new results. I searched twelve fresh CPU
containers: [five reported every required flag](../results/v2/cloud-vnni/probes-cpu.json).
All reported CPU model “unknown”; fresh containers do not prove distinct
physical hosts. I bound the overall search to twenty probes, with GPU-host
fallback only if the CPU-only search fails.

The [timing launcher](../tools/cloud_modal.py) checks its own CPU at startup,
before source checkout, model access or compilation. It requires exactly the
ISA used by `parse_kernel("vnni16")`: `avx512f`, `avx512_vnni`, `avx512bw`,
`avx2`, `f16c`. It retries at most five fresh rejected starts, not failed
accepted benchmarks. The driver must then report `vnni16` and int16
activations. I never relabel an FP32 fallback as this result.

I [prepare pinned assets on CPU](../tools/cloud_assets_modal.py) in the named
`cpu-decode-day-vnni-assets` Modal Volume. Timing reads those models offline,
copies only cached upstream source and compiles both engines and the driver
fresh on its actual host. I never reuse the preparation CPU's native builds.
If I rent a GPU container for its CPU instructions, I record that request and
still disable all GPU layers, backends and operation offload.

A separate full first-cell runtime pilot determines the final booking:
`ceil(pilot function minutes × 6 × 1.3)`. Its pairs never enter the final
six-cell inference. I keep threads 1/2/4, initial contexts 128/4096, 128 full
forwards, sixteen pairs and eight ABBA quartets per cell unchanged. Every
completed cell is streamed immediately; atomic raw writes preserve earlier
cells. I retain the same worker-lifetime `CpuBinding` scope described below,
not unchanged per-operator laptop CLI performance.

The launcher accepts an optional UTC deadline and reserves forty seconds for
process cleanup, including a thirty-five-second termination grace period.
The earlier AVX2 partial run remains separate under `results/v2/cloud`.

### Earlier AVX2 partial comparison

I completed **2/6 planned cells** before the function timed out. Both favor
llama.cpp: native/llama paired ratios are 0.4108 [0.4061, 0.4428] at one
thread/context 128 and 0.3079 [0.3070, 0.3117] at two threads/context 4096.
These are 95% per-cell block-bootstrap intervals. Missing cells are
(1,4096), (2,128), (4,128) and (4,4096); I publish no unfinished cell.
[Raw data](../results/v2/cloud/raw.json), [derived summary](../results/v2/cloud/summary.json)
and [cost accounting](../results/v2/cloud/run-cost.json) preserve the outcome.

The sandbox reports 24 exposed CPUs, one thread per exposed core and CPU
model “unknown”. It reports AVX2 and F16C but no AVX-512/VNNI; native resolves
to `simd256` with FP32 activations. Both paths were compiled with GCC 12.2.0
inside the measured container. llama.cpp resolves flash to on in both cells;
repacking was enabled, but its logs show a mapped rather than repacked model
buffer. F16 KV capacity is 256 for both at the short context; at the long
context native allocates 4224 positions and llama.cpp rounds to 4352.

I keep this comparison separate from the laptop matrix. The
[committed design](../tools/cloud_design.json) fixes the v2 source, model,
g64f16 weights and upstream commit before measurement. I build both paths with
`-march=native` inside one running CPU-only container. I request `vnni16` only
when the reported CPU flags support it; otherwise I record the resolved
FP32 dispatcher fallback. This transfers the format choice, not a new quality
evaluation on the cloud CPU.

I measure threads 1/2/4 at initial contexts 128/4096. Both engines prefill the
same repeated seed IDs and warm all 128 decode steps. Each sample rewinds to
the original prefix and measures 128 complete token forwards, including the
LM head, greedy argmax and final consumed token. Load, prefill, warmup, rewind
and JSON are excluded. Both models stay resident; I confirm the inactive
process group's suspension with `waitpid` before activating the other.
Each engine follows its own greedy trajectory without EOS stopping; I retain
the token IDs and check repeat consistency, not cross-engine equality.

I hold the native caller's public v2 `CpuBinding` for its whole worker lifetime.
This avoids expensive repeated affinity syscalls in the cloud sandbox; the
laptop path binds per operator. The measured cloud scope is therefore kernel
and threading speed without that syscall cost, not unchanged CLI performance.
The pinned engine source and arithmetic stay unchanged.

I choose flash on/off/auto using three separate pilot observations per setting
and cell, breaking median-time ties in that order. Pilot data stays visible
but does not enter final inference. The baseline uses F16 KV, strict matching
CPU sets, poll50 and repacking, with `GGML_OPENMP=OFF` for its real persistent
CPU pool. This is not the strongest-of-nine laptop baseline.

Eight ABBA quartets produce sixteen chronological pairs per cell. I report
the median paired ratio `llama_seconds/native_seconds`, each engine's median
and min/max throughput, and a 95% percentile interval from 20,000 whole-ABBA
bootstrap draws with seed 20261007. An interval above 1 favors native, below 1
favors llama.cpp, and one touching 1 is inconclusive. I retain every cell and
do not remove performance outliers. These are per-cell intervals, not a
simultaneous test; host load, boost and NUMA placement remain uncontrolled.
I do not measure a cloud read-bandwidth ceiling.

My earlier launcher at commit `cf266bf` requested eight cores, 8 GiB, no GPU
and a forty-minute limit. Streaming retained the two completed cells, alongside
preparation manifests, CPU flags/topology, settings and hashes. The current
launcher instead requires VNNI and the CPU-prepared Volume described above.
I keep the original design and data unchanged, rather than mixing these
fallback observations into the VNNI study. The
[summarizer](../tools/cloud_summary.py) rejects an incomplete six-cell matrix
unless `--allow-partial` is explicit. Partial publication lists missing cells;
each published cell still needs all sixteen chronological pairs.

To regenerate this partial summary from the retained raw data:

```sh
uv run python -m tools.cloud_summary --input results/v2/cloud/raw.json \
  --output results/v2/cloud/summary.json --csv results/v2/cloud/table.csv \
  --allow-partial --markdown
```

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

### Transposed K and the shipped g64f16 development stages

The [stage data](../results/v2/attention-stages.json) and
[SVG generator](../tools/figure_attention.py) retain all six samples per stage:
two independent native processes, three repeats, 16 measured steps,
six threads, initial context 4096, strict CPU order and the same g64f16 weights.
These are separately scheduled development windows, not interleaved stages or
a strongest-baseline comparison.

| Stage | Median tokens/s | Min–max | Rate spread | Attention ms/token |
|---|---:|---:|---:|---:|
| Original K layout | 52.470 | 50.674–53.588 | 5.55% | 5.348 |
| Transposed K | 59.032 | 54.855–62.160 | 12.37% | 2.727 |
| Masked SIMD tails | 59.146 | 53.140–61.455 | 14.06% | 2.914 |
| Parallel head merge | 57.909 | 56.964–58.549 | 2.74% | 2.894 |
| Singleton task claims, rejected | 56.467 | 54.370–59.678 | 9.40% | 3.204 |

K storage is `[KV head][allocated block64][dimension][token64]`; V remains
time-major. Allocation pads K's final block and records actual cache bytes.
Scores vectorize over 16 tokens, avoiding horizontal dot-product reductions.
Inactive lanes are masked from stores, maxima and softmax sums, including
poisoned padding. Per-head merges run in parallel but traverse blocks in the
same ascending order at every thread count. Local and merged coefficients
are normalized before weighted averaging to avoid avoidable finite overflow.
The singleton-claim trial is not shipped; the adaptive eight-task policy
remains, and its extra trial API was removed.

The AVX-512 exponential uses nearest-integer base-2 range reduction and a
degree-seven polynomial on the reduced interval. The native dense test checks
160,000 inputs: normal relative error ≤2e-6, subnormal absolute error ≤two
minimum positive float subnormals; zero, negative infinity and NaN have
explicit checks. This is a tested numerical bound, not an exhaustive proof
over all float inputs or rounding modes. Attention checks cover every
16-lane tail width, F16/F32, poisoned inactive keys, oversized K allocations,
rejection of insufficient blocks, scalar-reference tolerance 3e-6, and
bitwise results at 1/2/4/6/12 threads.

### Signed-int16 activation path

The original weight-format decision remains
[g64f16](../results/v2/format-selection.json): the fewest artifact bytes among
calibration formats strictly better than actual Q8_0 on mean KL, p99 KL and
top-1 agreement, evaluated with FP32 activations. A separate
[signed-int16 comparison](../results/v2/quality-vnni16-final.json) keeps those
weights fixed and checks the same three strict inequalities on 512 calibration
and 2048 heldout positions. Perplexity is worse than Q8_0 and is not a selection
criterion. F32 KV is a separately reported required control, not another
format-selection opportunity.

`vnni16` represents inputs in 64-value groups with a finite positive scale,
signed clipping at ±32767 and half-away-from-zero rounding. Quantization happens
once per projection call, shared by fused QKV or gate/up. Weight groups of
32/64/128 and row scales are supported; the final chosen format is g64f16.
Thirty-two weights sign-extend to words; two signed-word VNNI instructions
cover an ordinary 64-value group. Four products per lane, including -128
weights, remain below 2^24, so int32 accumulation and its FP32 conversion are
exact before scaling. Integer sums reset before applying each scale, with one
FP32 vector FMA per ordinary group and one final horizontal reduction per row.
Subnormal or excessively large combined scales use the independent wide
whole-row arithmetic path rather than lose finite results through early
underflow/overflow. The activation-scale FLT_MIN floor is an intentional
precision loss for extremely tiny values, not a universal relative-error bound.

FP32 kernels allocate no integer activation buffers. Int8 activation VNNI
remains governed by its earlier four-metric rule and was rejected. `make final`
chooses signed-int16 only when its recorded comparison passes, otherwise the
configured FP32 kernel; native CLI `auto` remains FP32. Freeze, windows and
summary rehash all five linked signed-int16 quality reports and recompute their
metrics/commands, binding corpus, selection, source, binary, model/config and
actual Q8 artifact. A changed or rejected comparison cannot enter final timings.

The [same-binary provisional FP32/int16 windows](../results/v2/int16-development/summary.json)
show the largest difference at one/two threads; the long-context six-thread
window is noisy and does not establish a speed improvement. Those pre-final
binary records remain development data. The later
[selected-path ABAB window](../results/v2/int16-shipped-short/summary.json)
uses the quality-checked final binary at two threads/context 128: native
69.580 tokens/s (66.044–70.104, spread 5.834%) versus Q8_0 65.131
(63.843–66.078, spread 3.432%). Both have six samples, 16 measured tokens and
three repetitions. This is one candidate, not the full strongest-of-nine
64-token/five-repetition comparison; its native spread exceeds 5%.


### Fixed settings and sampling

Freeze native kernel, weight artifact, group size, scale dtype, scheduler, attention and RoPE **before** the final run. `freeze` verifies model/config/GGUF hashes, exact source-manifest equality, Release/native build settings, executable hashes and shared-library identities. It saves a digest-addressed `protocol.json`; subsequent windows reject changed artifacts. The full matrix is threads **1, 2, 4, 6, 12** × initial contexts **128, 1024, 4096**. Defaults are **64 measured tokens per repeat, five repeats per run request, two rounds and one untimed native warmup step**. Final aggregation rejects shorter sampling or development protocols.

The baseline uses the same resolved-library/build identity reader as the quality measurements: `ldd` under the invocation's inherited loader environment identifies the actual llama/ggml CPU dependencies, not neighboring `*.so` files. Unknown llama/ggml libraries and dynamic backend loading are rejected. The protocol binds resolved locations (with a location hash to distinguish identically built copies), content hashes, hashes of `LD_LIBRARY_PATH`/`LD_PRELOAD`/`LD_AUDIT` values, upstream cache/generated compilation flags, compiler setting and pinned source. Loader strings are not published. Each baseline process is checked immediately before and after execution; changed identities retain raw output but cannot contribute samples. A copied or renamed benchmark executable still binds the build of its resolved dependencies. Replaying `freeze` rejects an existing protocol or any CPU-discovery JSON/stdout/stderr before writing discovery evidence; use a new output directory rather than overwriting old evidence.

Final `--kernel vnni` additionally requires `--quality results/v2/quality.json` (the default): a held-out **retained** decision bound to the calibrated chosen weight hash, configuration hash, engine binary and execution settings. The runner checks linked report hashes and independently requires mean KL ≤ Q8_0, p99 KL ≤ Q8_0, top-1 agreement ≥ Q8_0 and perplexity ≤ Q8_0. It archives the quality-file hash and eligibility proof in the frozen protocol; windows and the summarizer reject missing, rejected or changed eligibility. Development/ablation can time rejected VNNI paths with an explicit `experimental_vnni` marker, not a final claim. An unapproved auto-selected VNNI path is also rejected. Freeze and timing commands require nice 19; put `nice` inside an external wrapper if that wrapper changes priority.


Final `--kernel vnni16` instead requires its separate strictly better
calibration and shipped-F16-heldout mean/p99 KL and top-1 rule described above;
perplexity remains display-only. Both native KV variants and actual Q8 reports
must be present and linked. The int8 activation rule is intentionally unchanged.
Each cell tries flash attention **on, off, auto** × **pinned, unpinned, defaults**, with repacking enabled and upstream poll=50. The first two categories are process-restricted to the physical-first selected N CPUs: pinned also uses the matching mask/strict1, while unpinned uses mask0/strict0 inside that selected set. `defaults` uses mask0/strict0 on the full frozen allowed set. All nine enter the conservative best-baseline maximum. Actual process CPU sets and winner matching status are reported against the frozen native affinity, rather than assumed equal. `freeze --polls 0,50` can expand the list; every candidate must be sampled.

Process affinity is necessary even for pinned one-thread tests: at this upstream revision the OpenMP single-thread graph branch does not apply the stored worker mask. Echoed mask/strict flags alone do not establish effective placement. Both selected-set categories therefore receive process-level affinity at every thread count; the separately labeled full-set defaults category preserves an unrestricted baseline comparison.

`--cpu-order` preserves the supplied fast-physical-first order; on the development machine this is `0,1,4,2,3,5,6,7,10,8,9,11`. Elsewhere omit it to use `cpu-decode cpus`'s `allowed_cpu_ids`/`preferred_cpu_ids`. Native `freeze --affinity strict` (default) supplies exactly N selected IDs with `--cpu-set`; `--affinity unpinned` omits that option and inherits the full allowed mask without per-worker binding. Affinity is part of the frozen configuration and native metadata, including the actual full/selected `cpu_set`; it must not change midway through a final run.

### Sequential interleaved windows

Only **one model is resident at a time**. Each candidate uses **A B A B**: run ordinary native `bench` with five repeats, wait for its process/model to exit, run llama with five repeats and wait for exit, then repeat both. Each native invocation builds its prefix once and warms once; its repeats rewind to the original context/seed without rebuilding the prefix per token. No interactive native process stays resident during llama. With nine candidates, one cell has 18 native processes / **90 native samples** and 18 baseline processes / **10 samples per baseline configuration**. Loading, prefill and depth fill remain outside decode timing.

The baseline winner has the **highest median of its ten measured rates**, not a weaker convenient configuration. Ties use candidate ID. Compare **only the ten native samples interleaved with that winner**: equal invocation and sample counts. All other candidates/samples/commands/logs remain visible. Selection and reporting use the same samples, without a holdout; selection optimism favors the baseline. Both engines now use separate processes for each invocation. Native warms at the requested context; upstream warms before depth fill, so warmup placement still differs.

Every externally scheduled window has a 1,740-second deadline including setup, below 30 minutes. Use **one candidate per command** if all nine will not fit; collect every frozen candidate exactly once per cell. Timeout, nonzero exit, malformed output, setting mismatch or incomplete ABAB is visibly retained and never counted as successful sampling. Duplicate candidates are rejected rather than selecting a favorable rerun. Any failed/missing candidate prevents final-matrix acceptance. The low-level measurement tools do not start or nest a scheduler; the unattended runner below accepts an explicit runtime prefix per window.

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
- `window-tT-cC-*.json`: cell, candidates, ordered completed subprocess invocations and exit statuses, actual CPU-set eligibility, portable commands, raw native/upstream output, baseline resolved identities before/after each invocation and stdout/stderr paths. Verbose baseline logs preserve CPU/attention/tensor/repack messages; a requested repack flag is not proof that Q8_0 tensors were repacked.
- `bandwidth-tT.json` and `ablation-tT-cC-*.json`: full observations for same-core-set read probes and explicitly labeled fixed-cell rungs.
- `summary.json`: per-cell native and **best measured** baseline median/min/max/sample count/spread, actual winning parameters, flags and process CPU sets, explicit `winner_core_sets_matched`, all candidates/raw-log references, byte counts/read ceiling/percentage, ratios, quality eligibility, ablations, failed records, missing cells/candidates and noise.
- `complete_final_matrix` requires all 15 cells, every frozen candidate and matched bandwidth with no failures and final sampling settings. Recorded `thresholds` default to native/best-baseline **≥1 in all 15 cells**, **≥85% of the read ceiling at at least one context-128 cell** (`short_best_percent_of_ceiling`), and **≥75% at every context-4096 cell** (`long_all_percent_of_ceiling`). There is no ceiling-percentage target at context 1024 and no requirement that every context-128 cell reach 85%. `target_predicates` aggregates those three comparisons; all are false and per-cell `targets` are null for incomplete/development data. `numeric_targets_met` combines only these numeric targets, not the unquantified no-collapse goal.
- Explicit overrides are `--target-ratio`, `--target-short-best-ceiling-percent` and `--target-long-all-ceiling-percent`. They preserve all-cell / short-best / long-all scopes and record `threshold_basis: custom` whenever their values differ from the user targets. The obsolete uniform `--target-ceiling-percent` option is removed; a 50% uniform target is not acceptance of the user goal.
- `thread12_scaling` reports the 12-thread native median divided by the 6-thread median and by the strongest measured lower-thread median **at the same context**, along with the lower-thread winner, whether 12 threads are slower, and the nonnegative percentage decrease. These are descriptive MEAS comparisons of the native samples paired with each cell's baseline winner, not an invented user tolerance. No numeric cutoff for “collapse” was specified, so `acceptance` remains null; numeric target success alone must not be described as satisfying the entire no-collapse goal. Incomplete comparisons are omitted and matrix completeness remains visible.
- Spreads are `100 × (max − min) / median`. **Every spread >5% is disclosed**, including losing baseline candidates. Samples are never discarded for noise. `decode.svg` is generated only from summary values, with native/best-baseline/estimated-ceiling curves, min–max bars and noise markers.

Artifacts replace local directories with portable aliases (`$INT8`, `$GGUF`, `$ENGINE`, `$LLAMA_BENCH`, `$LLAMA_ROOT`, `$LLAMA_BUILD`, `$LLAMA_CXX_COMPILER`, `$BANDWIDTH`, `$HOME`, `$PYTHON`). The baseline root aliases come from resolved dependencies, not executable placement. Numeric CPU IDs, flags, hashes and observations are preserved. Nice 19 and exclusive scheduled benchmarking do not isolate the desktop, fix temperature/clocks or disable boost. Matched shapes still are not identical token trajectories: native uses the fixed seed sequence and greedy argmax, upstream uses synthetic tokens and excludes sampling. Operation instrumentation remains included in native timing.

### Calibration-only format choice

The [new decision](../results/v2/format-selection.json) uses all four existing
native calibration reports plus a real [Q8_0 calibration](../results/v2/cal-q8_0.json):
two windows, **512 teacher-forced positions**, the same pinned
Qwen2.5-0.5B-Instruct revision and BF16-storage/FP32-arithmetic oracle. Every
candidate has fresh F16 KV and two threads. Native settings are SIMD512x4,
blocked attention, strict pool workers on CPUs 0,1; all four use the same
recorded native binary. Q8_0 uses the pinned llama.cpp reader with flash
attention auto. This compares complete numerical paths, not quantizers alone.

Require native **mean KL < Q8_0**, **p99 KL < Q8_0**, and **top1 > Q8_0** on
calibration. Among eligible formats choose the fewest weights-artifact bytes,
then mean KL, p99 KL, negative top1, and label. Perplexity is reported but
**is not a selection criterion**. No eligible format is an explicit error;
heldout cannot choose a replacement.

| Format | Matrix bits/weight | Weights artifact bytes | Mean KL (nats) | p99 KL (nats) | Top1 agreement | PPL | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| [g32f16](../results/v2/cal-g32f16.json) | 8.5 | 525,170,936 | 0.000715347 | 0.002411287 | 0.98046875 | 25.223541 | Eligible, larger |
| [g64f32](../results/v2/cal-g64f32.json) | 8.5 | 525,171,728 | 0.000801546 | 0.002527370 | 0.982421875 | 25.220360 | Eligible, larger |
| [g64f16](../results/v2/cal-g64f16.json) | 8.25 | 509,734,624 | 0.000801242 | 0.002620214 | 0.98046875 | 25.257738 | **Selected** |
| [g128f16](../results/v2/cal-g128f16.json) | 8.125 | 502,016,336 | 0.001216109 | 0.004394612 | 0.96875 | 25.270526 | Rejected: top1 below Q8_0 |
| [Q8_0](../results/v2/cal-q8_0.json) | 8.5 | 531,068,416 | 0.002363970 | 0.006974944 | 0.97265625 | 25.231974 | Baseline |

Bits count matrix int8 plus scale storage; artifact bytes include the full
safetensors/GGUF container, non-matrix tensors and metadata, not configuration
or tokenizer files. The selected [g64f16 manifest](../results/v2/quantized-g64f16.json)
binds `model.safetensors` SHA256
`c70c215d96047722ac638b9677e761874e46c7d41b0b01068a94ef33b12fa049`.
Its calibration PPL is slightly worse than Q8_0; this does not disqualify it
under the specified three-metric rule.

The [structured table](../results/v2/format-calibration.json) keeps exact values,
all report hashes, native/Q8 artifact identities, oracle/source identities and
binary/library records. The new decision's top-level `policy` is the rule
above. The old rule remains inside the unchanged corpus/oracle identity and
in historical `selection.json`/`calibration.json`; those are not overwritten.
The calibration reports' legacy generic `scope` sentence says “held-out”;
their `split`, window IDs and all 512 positions establish calibration scope.
The tool now emits split-specific wording.

To reproduce in a **fresh output/raw directory** after generating native
calibration reports with identical settings:

```sh
nice -n 19 python -m tools.quality_v2 evaluate --split calibration --backend llama --model "$Q8_GGUF" --threads 2 --label q8_0-calibration --corpus "$RESULTS/corpus.json" --raw-dir "$RAW" --selection "$RESULTS/format-selection.json" --output "$RESULTS/cal-q8_0.json"
nice -n 19 python -m tools.quality_v2 compare --split calibration --corpus "$RESULTS/corpus.json" --selection "$RESULTS/format-selection.json" --reports "$RESULTS/cal-g32f16.json" "$RESULTS/cal-g64f32.json" "$RESULTS/cal-g64f16.json" "$RESULTS/cal-g128f16.json" "$RESULTS/cal-q8_0.json" --output "$RESULTS/format-calibration.json"
```

Reader and preparation paths are configurable with `--reader` and
`--artifact-manifest`. Outputs are exclusive; supply the exact new
`--selection` path to every subsequent heldout evaluate/compare. The earlier
g32f16 heldout records remain historical and do not measure the selected
g64f16 artifact or the later native K-layout changes. Final-binary heldout
quality and speed must use this same g64f16 artifact without reselection.

### Unattended final run and resume

[`tools/run_final_v2.py`](../tools/run_final_v2.py) collects the complete matrix
in a dedicated `results/v2/final` directory. It needs only the prepared native
and bandwidth binaries, pinned baseline binary/GGUF/preparation manifest,
chosen quantized artifact/manifest, new format selection and final-binary
heldout quality report. It does not build, recalibrate or regenerate quality.
The current format choice is **g64f16**, not the historical g32f16 selection;
the heldout summary must link the exact supplied selection hash and content.
Both engines use F16 KV. Native kernel and affinity are explicit arguments;
the runner never selects a quality format or an automatic kernel.

Set the artifact variables to their prepared locations. `$BENCH_WRAPPER` is
an optional, privately supplied command prefix, including its scheduling mode;
leave it empty for public local reproduction at nice 19. The runner passes
`PP_MEM=2000M` to the wrapper environment. The private wrapper must enforce
that memory budget and exclusive measurement access; the public default
does not implement a scheduler or enforce a memory limit. Do **not** wrap
the entire unattended command in an exclusive benchmark slot: each candidate
and each bandwidth window acquires its own slot.

```sh
nice -n 19 uv run python -m tools.run_final_v2 --model "$INT8_V2" --model-manifest results/v2/quantized-g64f16.json --format-selection results/v2/format-selection.json --engine "$FINAL_ENGINE" --bandwidth "$FINAL_BANDWIDTH" --llama "$LLAMA_BENCH" --gguf "$Q8_GGUF" --preparation results/llama-preparation.json --quality results/v2/quality-vnni16-final.json --kernel vnni16 --affinity strict --wrapper "$BENCH_WRAPPER" --output results/v2/final
```

`--wrapper` is split with shell-style quoting, but never evaluated by a shell.
Its resolved spelling is hashed, not published. Each timing command appends
`timeout --foreground --signal=TERM --kill-after=5s 1770s nice -n 19`
**inside** that prefix: queue waits do not consume the timing deadline.
The low-level 1,740-second window deadline remains; the extra hard limit plus
termination grace is 1,775 seconds, below 30 minutes. The runner remains nice
19 throughout. CPU discovery defaults to the allowed physical-first order;
pass `--cpu-order` explicitly to preserve a different prepared order.
Linux, the locked Python environment and the existing system tools
(`nice`, `taskset`, `timeout`, `ldd`, compiler metadata tools) are required.

The equivalent Make target has no build or calibration prerequisites:

```sh
nice -n 19 make final FINAL_MODEL="$INT8_V2" FINAL_LLAMA_BIN="$LLAMA_BENCH" FINAL_GGUF="$Q8_GGUF" FINAL_WRAPPER="$BENCH_WRAPPER"
```

With the public Make reproduction, `FINAL_FORMAT_SELECTION` defaults to
`$(RESULTS)/format-selection.json` (`RESULTS=results/v2/reproduction`);
`FINAL_LABEL` reads that file and `FINAL_MANIFEST` follows it.
Override `FINAL_MODEL` when artifacts live outside `$CACHE`.
Other overrides are `FINAL_ENGINE`, `FINAL_BANDWIDTH`, `FINAL_KERNEL`,
`FINAL_AFFINITY`, `FINAL_CPU_ORDER`, `FINAL_PREPARATION`, `FINAL_QUALITY`,
and `FINAL_OUTPUT`.
`FINAL_OPTIONS=--plan` or adding `--plan`/`--dry-run` to the Python command
prints the full command list without reading/loading models or starting
subprocesses. The synthetic tests are in
[`tests/test_final_runner.py`](../tests/test_final_runner.py).

**INFERENCE from the command generator, not an executed timing estimate:**
threads 1/2/4/6/12 × contexts 128/1024/4096 × nine candidates gives **135
separate candidate windows**, plus **five bandwidth windows**. Each candidate
retains ABAB, five repetitions per process, 64 measured tokens and ten samples
per engine: **540 sequential model processes**, **1,350 samples and 86,400
measured tokens per engine** across the sweep. Bandwidth has ten processes
(two SIMD widths at each thread count). The sum of 140 low-level deadline
limits is 67 hours 40 minutes; it is not a forecast. Actual wall time is
**unknown**, including model load/prefix work and unbounded queue waits.

Rerun the **identical command** to resume. `runner.json` binds selection,
quality/preparation/manifests, tool hashes, runtime locations, wrapper and
configuration. There is one successful `protocol.json` freeze, binding actual
resolved upstream libraries (including the benchmark/common libraries),
model/config/GGUF, all binaries, builds and CPUs. Resume checks those original
identities rather than creating a new protocol. A changed identity is an error;
retain the existing directory and use a new dedicated output only for a
deliberately different experiment.

Every command uses a fresh `attempts/UNIT/NNNNNN/` directory. Driver logs,
in-flight stdout/stderr and completed invocation records survive interruption.
`attempt.json` records accepted/failed/interrupted disposition, exit status
when available, and its raw-directory link. Interrupt cleanup stops the
driver's process group; resume also checks a recorded process start identity
before stopping a still-running orphan. An incomplete ABAB is excluded as a
whole: no sample stitching and no duplicate counting. A complete valid unit
is atomically checkpointed, then published at the final directory root;
resume restores publication if interrupted between those steps. Accepted
raw files are hash-checked and skipped, never rerun because of a low rate or
spread. Multiple accepted attempts for the same unit are an error.
Failed attempts remain in place, and a later resume can collect a fresh full
unit without overwriting them.

The existing summarizer sees exactly one complete accepted sample set for
each candidate and thread/context cell, with matching bandwidth at all five
thread counts. Its candidate-level spreads, including every losing candidate
above 5%, remain intact. Historical failed/interrupted attempts appear under
`runner_attempts` in the derived summary with their raw links; they are not
fabricated samples or silently erased. If any unit remains unsuccessful,
the launch exits nonzero after retaining a partial summary with aggregate
acceptance disabled. `decode.svg` is published only after all **15 × 9
candidate units plus five bandwidth units** are complete. Derived summary/SVG
versions are retained in their own attempt directories; the original v1,
calibration and earlier heldout results are never overwritten.

### Publishing the completed matrix

After the full 0.5B run has completed, publish its README result/table, SVG
and machine-readable outcomes with one command:

```sh
nice -n 19 uv run python -m tools.finalize_v2
```

[`tools/finalize_v2.py`](../tools/finalize_v2.py) reads
`results/v2/final/summary.json` and its sibling retained `protocol.json`.
It verifies the protocol digest, source and copied quality decision, then
reloads the current comparison and all five linked reports through the
existing quality validator. The actual 15 cells must contain all nine
baseline candidates, two ABAB rounds, ten samples per arm and command flags
for 64 measured tokens/five repeats. Partial, development, inconsistent or
S1-subset inputs cannot update publication files.

The outputs are `decode.svg`, `cell-outcomes.csv` and `cell-outcomes.json`
beside the summary, plus the two marked sections of the README. Every native
and Q8 median win/loss/tie appears, independently of requested target margins;
noise, sample counts, ranges, winning flags, CPU sets and estimated ceilings
remain disclosed. Rendering and validation finish before staged replacements;
each file is replaced atomically and the README is replaced last. This is
not a cross-file transaction. Repeating identical input produces identical
output without replacing unchanged files. The README retains **Not yet run**
until a complete final matrix exists.

### Fixed-format 1.5B transfer, not another calibration

The optional S1 checkpoint is pinned Qwen2.5-1.5B-Instruct at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Its
[`fixed-format.json`](../results/v2/s1/fixed-format.json) transfers g64f16 from
the original 0.5B decision unchanged. No S1 calibration, format reselection
or signed-int16 approval is performed; its timing kernel is FP32 `simd512x4`.

[`streamed_oracle.py`](../tools/streamed_oracle.py) runs the unmodified
Transformers decoder layer with one resident FP32 layer, widened from the
original BF16 snapshot, then projects the tied head in chunks. The real
eight-window/2048-position run completed at the imposed 2000 MiB cap:
[`oracle-summary.json`](../results/v2/s1/oracle-summary.json).
This completion does not measure peak RSS.

The ordinary upstream S1 BF16 converter was terminated with exit 143 at
that cap; the exit alone does not establish an OOM cause. Its failed partial
was retained. [`streamed_gguf.py`](../tools/streamed_gguf.py) instead supplies
bounded row chunks to the **actual pinned upstream** metadata/vocabulary,
name mapping, BF16 codec and GGUF writer. Before accepting that adapter, the
full original 0.5B BF16 GGUF was compared with a fresh streamed conversion:
[`gguf-equivalence-0.5b.json`](../results/v2/s1/gguf-equivalence-0.5b.json)
records exact whole-file equality, 994,157,056 bytes and SHA256
`794f9d6e09be1e0509d0c917e4aff89d837f3e16b852b59453894e61a9816ed2`.
The same comparison also checks every tensor's type/shape/bytes and all
metadata. It reads bounded byte regions, not materialized tensor arrays.
The completed S1 BF16 and **unmodified upstream `llama-quantize` Q8_0**
artifacts are in its
[`preparation record`](../results/v2/s1/llama-preparation.json); no local Q8
encoder or repacker substitutes for upstream. Previously prepared caches
with the old manifest name are migrated only after exact source/commit and
all existing artifact hashes pass; the old manifest is then retired.

The [2048-position heldout comparison](../results/v2/s1/quality-heldout.json)
uses that real Q8 artifact and the streamed FP32 oracle:

| S1 path | Mean KL, nats | p99 KL, nats | Top-1 agreement | Perplexity |
|---|---:|---:|---:|---:|
| Native g64f16, F16 KV | 0.00079655 | 0.00451672 | 98.3887% | 9.47324 |
| Native g64f16, F32 KV | 0.00079326 | 0.00435566 | 98.3398% | 9.47242 |
| Upstream Q8_0, F16 KV | 0.00246597 | 0.01160820 | 96.7773% | 9.50208 |

A [separate native cache check](../results/v2/s1/cache-4096-check.json)
completed context 4096, capacity 4160 and 64 measured steps with two threads
at the same 2000 MiB cap. Its 119,275,520 cache bytes are an allocation count,
not total RSS; one diagnostic repetition is not a final speed comparison.
A separate upstream Q8 diagnostic also completed depth 4096, F16 KV,
two threads and one generated token under that cap:
[`q8-cache-4096-check.json`](../results/v2/s1/q8-cache-4096-check.json).
It is a single memory-fit diagnostic, not the final 64-token comparison.
The optional timing subset is threads `{2,6}` × contexts `{128,4096}`, still
all nine baseline candidates. It does not satisfy the primary 15-cell
matrix or add another target-approval condition.

### Executed baseline and real thread checks

[`hot-kernel-q8.json`](../results/v2/hot-kernel-q8.json) records an actual
debugger hit in the pinned upstream CPU library on the real 0.5B Q8 model.
The debugger stepped over `ggml_vec_dot_q8_0_q8_0+103`:
VEX `vpdpbusd` on YMM registers, followed by `vcvtdq2ps`. Binary/library hashes
and invocation flags are recorded. This proves an optimized integer-dot path
executed; it does not prove a particular repacked layout or report throughput.
The debugger intentionally stopped the inferior immediately afterward.

[`thread-invariance-final.json`](../results/v2/thread-invariance-final.json)
records 20 real processes on the shipped immutable 0.5B ELF: FP32/int16
kernels × F16/F32 KV × threads 1/2/4/6/12. Each scored all vocabulary logits
for 129 protected inputs. Within each kernel/KV pair, the complete binary
logit files were bitwise equal across thread counts. This is neither a
cross-kernel equality claim nor a quality/performance measurement.

### Historical report storage

Current calibration, signed-int16 approval and S1 inputs remain plain JSON.
Superseded pre-wide int16 reports are preserved byte-for-byte in
[`int16-pre-wide.zip`](../results/v2/int16-pre-wide.zip), and eight older
g32/per-row/FP32-control reports in
[`historical-heldout.zip`](../results/v2/historical-heldout.zip).
The [archive index](../results/v2/archive-index.json) records archive and
original member hashes, verified after compression. Original v1 measurements
and historical summary numbers are unchanged. To inspect historical sibling
links at their original filenames, extract the corresponding ZIP into
`results/v2`; these reports cannot authorize the current binary.

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
