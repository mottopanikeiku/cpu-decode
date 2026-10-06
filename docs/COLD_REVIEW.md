# Cold code and methodology review

Scope: the initial from-scratch Qwen decoder and comparison protocol, before throughput measurements. Read all requested build files, public interface, native implementation/tests, Python oracle/tests, measurement/aggregation/preparation tools, fixed prompts, and PRIOR_WORK/MEASUREMENTS/BASELINE. Also traced the installed Transformers 4.56.2 Qwen2 implementation, pinned llama-bench consumers, quantization/download/tokenization helpers, and newly available correctness/quality JSON and its aggregation script. This is source inspection, not an independent execution of the reported tests.

## Findings, ranked by impact

All four findings are P2. No P0/P1 failure was established for the pinned forward pass or default benchmark commands.

### 1. Verify identity before reporting an exact pinned oracle

**Source:** `tools/reference.py:84,109-110`; `tools/download_model.py:43-56` already provides `verify_snapshot`.

The comparison accepts any local tied-head Qwen2 directory but unconditionally labels the oracle with the fixed model ID and revision. A different revision or edited BF16 snapshot supplied through `--model` or `CPU_DECODE_MODEL` is compared to itself by both implementations; the output identity still claims the pinned snapshot. Architecture and numerical agreement do not establish weight identity. Verify the snapshot at the pinned comparison entrypoint and retain the verified manifest in its result. Leave the small `load_oracle` conversion test able to use synthetic weights.

This is a provenance gap, not evidence that the current downloaded weights are wrong: the normal prepare path separately verifies them.

### 2. Make an explicitly requested integration test fail on a missing engine

**Source:** `tests/test_reference.py:98-101`; `Makefile:31-32`.

After `CPU_DECODE_MODEL` opts into integration, a missing native executable still triggers `pytest.skip`. With downloaded weights but no build, `make model-test` can therefore succeed with both full-model comparisons skipped; that target has no build prerequisite. Keep the ordinary opt-out skip when the model variable is absent, but fail if an explicitly requested comparison lacks its executable. Add a regression exercising a nonexistent `CPU_DECODE_ENGINE` with integration enabled.

### 3. Check eager metadata before calling a table row matched

**Source:** `tools/summarize.py:50-56`; `tools/measure.py:63-64,79-82`; `tests/test_tables.py:26`.

llama settings are checked, whereas eager settings are trusted solely from its filename. Filenames omit generation length and token IDs. Running engine with 16 steps and eager with 8 steps into the same measurement directory produces filenames the summary joins without objection, despite different growing-cache windows. Check eager threads, context, steps and seed IDs against the engine metadata before aggregation. Extend the synthetic table test with a mismatch that must fail; its current eager fixture contains only samples, making this missing validation invisible.

### 4. Restrict or repair gapped quantization output

**Source:** `src/model.cpp:367-370,398-400`; `tools/quantize.py:34`.

The quantizer aligns scales by inserting unindexed bytes after an I8 matrix whose element count is not divisible by four. An accepted configuration with hidden size 6, three query heads, one KV head, head dimension 2 and vocabulary 11 has a 66-byte embedding matrix and a two-byte gap before its scales. The [Safetensors specification](https://github.com/huggingface/safetensors#format) forbids holes; standard readers reject such a layout even though the native loader accepts it. Reject these dimensions and state the restriction, or emit a fully indexed layout. Do not describe arbitrary alignment gaps as valid Safetensors.

The pinned Qwen matrix sizes are multiples of four, so this does not invalidate its artifact. The newly added official-parser integration cross-check addresses that artifact, not the unsupported generic layout.

## What the inspection supports

- GQA maps consecutive groups of query heads to each KV head; native K/V strides match the writes. Split-half RoPE, Q/K/V biases, both residuals, SwiGLU and the shared embedding/head follow the installed Qwen2 consumer implementation. No dropped dtype/kernel/profile field was found on the traced paths.
- The FP32 oracle gathers BF16 embedding rows then widens them, widens each linear projection and bias, keeps hidden states/attention FP32, and chunks the tied head. Installed RMSNorm and rotary code are compatible with that dtype strategy. This is not BF16 arithmetic mislabeled FP32.
- Native repeats rewind to the original context, restore the same selected seed, and overwrite new KV positions before reading them. Stale continuation slots are outside the attention length. Eager rebuilds each prefix; pinned llama-bench clears/restores depth outside its timer. Native and llama measure forwards at positions context through context+steps-1.
- Traffic totals include the head exactly once through matrix totals; separate head fields are breakdowns. FP32 copied norms/biases, row scales, one embedding row, distinct GQA KV reads and KV writes are accounted for. Logical repeated-head reads are separate. The ceiling remains an algorithmic estimate, not measured DRAM utilization; write traffic, cache reuse and other omitted costs are already disclosed.

`results/quality-summary.json` records 88 unquantized positions within tolerance, maximum absolute error 0.0003812313, and 32/32 greedy tokens matching. Int8 prompt-only agreement is 57/60, mean KL 0.0287937628 and maximum KL 0.7897130896; including reference continuations gives 85/88 agreement and mean KL 0.0200650213. The aggregation explicitly distinguishes these scopes. All four int8 greedy sequences match, which does not imply every prompt-position top-1 matches.

Int8 quality is deliberately reporting-only, not an accuracy gate; `passed` must not be advertised as an int8 quality guarantee. BF16 eager KV versus FP32 native/llama KV, distinct quantizers, llama synthetic trajectories/no sampling, and native instrumentation are documented tradeoffs, not newly discovered defects.

## Verification and remaining evidence boundary

No builds, tests, benchmarks, linters, formatters, model loads or smoke runs were executed by this reviewer. Parent-reported test outcomes were not rerun. The numerical statements above are read from result files, not independently reproduced. No throughput measurements or final README were reviewed.

Parent verification after fixes should cover explicit integration failure, mismatched eager metadata, official-parser acceptance/rejection for a non-multiple-of-four matrix, and pinned-source identity rejection, alongside the existing native and model checks. Long-context numerical agreement at 128/1024/4096 is not established by the short fixed-prompt oracle files; this is an unmeasured coverage boundary, not an observed numerical defect. No hypothesis is promoted to a finding about performance or long-context accuracy.


## Post-fix source disposition

The four findings above describe the pre-fix state. Re-read the parent changes and regression source; **all four are resolved at source level**:

1. `oracle_worker` calls `verify_snapshot` before NumPy/PyTorch imports and model loading, and saves `verified_source`. The direct synthetic `load_oracle` test is unaffected. The new edited-snapshot regression exercises rejection before loading.
2. Explicit integration now calls `pytest.fail` for a missing executable. Its regression sets both the opt-in model and a nonexistent engine and requires that specific failure; ordinary model opt-out remains unchanged.
3. Aggregation compares eager threads, context, steps and seed IDs against engine metadata before calculating the eager rate. The table fixture supplies those fields, and four mismatch cases require failure with the expected diagnostic.
4. Quantization rejects matrix element counts not divisible by four before creating the output directory. Both header-offset and payload padding have been removed. The loader now requires contiguous coverage from offset zero through the payload end. Native regressions exercise the accepted six-hidden/three-head source fixture and require the intended shape error with no output directory; gap/trailing-data checks also require the intended exceptions. MEASUREMENTS documents the restricted shape support, and pinned-artifact integration uses the official parser.

No additional actionable defect was found in these corrections. This disposition is based on source review only: no builds, tests, linters, formatters, model loads or benchmarks were run or rerun by this reviewer. Parent execution of the updated native checks remains separate from this review. Final throughput tables and README are still outside this review scope.


## Four-accumulator SIMD512 extension

Source-reviewed the subsequent `simd512x4` change, its public enum, native/parser dispatch, 147-column math fixture, measurement defaults and ablation pairing. No tests/builds/model loads were run.

**P1 found during the extension review — resolved below.** `Makefile:29-33` selects `simd512x4` for correctness and model integration, but `tools/reference.py:197` still lists only scalar/simd256/simd512 as argparse choices. Both requested checks fail at argument parsing, before any new-kernel comparison. Add the missing choice and exercise the selection through the Python CLI. Native enum dispatch alone does not cover this consuming parser.

No source-level arithmetic/indexing defect was found in `dot512x4`: each 64-element block loads four disjoint 16-element ranges, the vector remainder advances by 16, and the scalar tail starts at the first unconsumed element. BF16 widening and signed-int8 widening share the existing semantics; the int8 scale is still applied once after reduction. All accumulation remains FP32, with a different addition order rather than bitwise equivalence. The 147-column fixture covers two full blocks, one vector remainder and three scalar elements for F32/BF16/I8 and one/two threads; tiny whole-engine coverage also includes the new enum when supported.

Traced the new enum through `parse_kernel`, `kernel_name`, Engine construction and `matvec` to its explicit `dot512x4` branch. Native CLI help and the text-generation helper admit it. Engine measurement and aggregation default to x4, while the bandwidth stage deliberately retains supported read widths. The ablation schedule includes single-accumulator SIMD512 and x4 at the same int8/cached/thread/context settings, and aggregation pairs those variants; direct/cached RoPE is now paired on x4 in both Make and the summary. No speed claim is established by this inspection.

The earlier numeric results in this review describe the pre-extension artifact. A new x4 oracle result is required before attributing those numbers to the new default. The original four P2 findings remain resolved; this parser integration issue is the only newly established blocker in the extension reviewed here.


### Extension disposition: resolved

Re-read `tools/reference.py:197`: its argparse choices now explicitly include `simd512x4`. Read the refreshed `results/correctness.json`: all eight BF16/int8 prompt cases record `engine_settings.kernel` as `simd512x4`, the oracle includes `verified_source`, both comparison branches and the overall result report passed, and all greedy sequences match. The P1 consuming-parser finding is resolved; none of the five findings remains open.

The refreshed `results/quality-summary.json` records x4 maximum absolute unquantized logit error **0.0004389286**, all 88 positions within tolerance and 32/32 greedy tokens matching. Int8 prompt-only agreement remains 57/60, with mean KL **0.0287933171** and maximum KL **0.7896918149**; including reference continuations gives 85/88 agreement and mean KL **0.0200647228**. These replace the pre-extension numbers for claims about the current default. Int8 quality remains reporting-only.

This is source/result-file inspection, not independent reproduction. Parent-reported native/integration outcomes were not rerun. No reviewer tests, builds or benchmarks were executed. Final timing results and README still require their separate review.


## Publication-location cutover review

Read `tools/portable.py` and its tests, download/reference/quantize/prepare_llama/measure/benchmark_eager, Make, MEASUREMENTS, and the published artifact manifests. No builds/tests/formatters, downloads, model work or benchmarks were run.

**P2 found during the publication cutover — resolved below.** `tools/portable.py:14-16` replaces substrings in every string, while `tools/quantize.py:42-45` adds the literal recorded model/output operands as aliases. A valid pinned snapshot supplied as `--source a` inserts the relative string `a` into the replacement map. Every matching hexadecimal `a` in source/output hashes is then replaced by `$MODEL`; even the recorded `quantize` command word is altered. A relative source named `model` also turns the intended `weights.path` value `model.safetensors` into `$MODEL.safetensors`. This is a provenance/reproduction failure, not cosmetic path normalization, and contradicts the documented preservation of hashes and flags. Canonicalize command paths before constructing aliases, and match only actual path values or path-boundary-aware occurrences. Add a short-relative-name regression requiring hash values and non-path strings to remain identical. Current helper tests cover nested absolute paths and ordinary numeric values, not this case.

The other traced boundary changes are coherent at source level: download defaults delegate to Hugging Face cache settings, Make obtains the same hub-cache constant, large local outputs default to ignored `external/`, oracle raw-logit producers now emit basenames and the comparison consumer joins them with `args.raw_dir`, and publication conversion happens after actual subprocess execution. llama artifact reuse checks actual cache files against hashes rather than attempting to open its published `$CACHE` aliases. Numeric observations retain numeric types, and the previously reviewed source-identity checks remain in place. No further concrete cutover defect was established in this inspection.

Public timing scripts no longer require the workstation-specific wrapper; this is an intentional portability change. This review does not establish what scheduling was actually used for future measurements or interpret unpublished timings. All five earlier findings remain resolved; the unrestricted substitution above is the only new open finding.


### Recorded public-command verification

Re-read the publication-hygiene requirement and kept this review free of host-specific paths and internal tool/project names. `results/checks.json` records native checks 3 passed/0 failed, fast Python checks 13 passed/2 explicitly optional model skips, model-specific checks 11 passed/0 skipped, and the full x4 oracle comparison passed. These are parent-reported, file-recorded outcomes; this reviewer independently executed none of them.

The current portability helper and quantizer still contain the unrestricted relative-substring substitution described above. Passing the recorded default-location checks does not resolve that concrete alternative-directory failure. No new finding is added, and the P2 cutover finding remains open pending a source correction and targeted regression.


### Publication disposition: resolved

Re-read the correction in `tools/portable.py`: every location-map key is resolved to an absolute path, so a relative directory name such as `a` cannot become a global replacement for hexadecimal characters or command words. Re-read `tools/quantize.py`: model/output aliases and the executable label are now assigned only to their known command-list slots; literal relative operands are no longer inserted as substring aliases. `weights.path` remains the intended basename, and hashes are left untouched by these changes.

The new `test_relative_aliases_cannot_rewrite_hashes_or_command_words` checks an all-`a` hexadecimal value, relative command operands, and a real absolute path in the same record; it requires preservation of the first two while transforming the path. This source regression addresses the reported trigger. No additional actionable defect was established in the fix, and all six findings are now resolved at source level.

No tests, builds, formatters, downloads, model loads or benchmarks were run by this reviewer. Updated command outcomes remain the parent responsibility; final throughput/README interpretation has not yet been reviewed.
