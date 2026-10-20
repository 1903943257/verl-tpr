# CP1 long-prefix / short-suffix F/B benchmark (N=8)

Status: experimental runner on a branch from main; **no Ascend measurements yet**.

- Real Qwen3-4B-128K BF16 checkpoint, TP=2, CP=1, DP=1, **2 Ascend NPUs**.
- Fixed **N=8**, retained original actor PPO response tokens and historical old_log_probs, advantages, response_mask, loss_mask, temperature.
- Each trajectory: nominal **P=8,192 tokens** (last prompt tokens), distinct **S=1,024 tokens** (first response tokens); L=9,216. All eight prompts must be token-identical for 8,192 tokens. Actual qwen_turn Root must cover eight leaves and share **at least 8,065 tokens**. With production turn_min_segment_tokens=128, the prompt boundary may be merged with a response turn if the preceding Turn boundary is fewer than 128 tokens away; therefore Root need not equal 8,192. Record **actual_shared_root_tokens** separately from the nominal P:S ratio. Incompatible sources are rejected, not repaired.
- Original session identifiers are saved in metadata. A synthetic shared UID is used for this *performance-only* comparison to allow different sessions with identical prompt tokens to share Prefix KV. Left-truncation and shortened responses **change original conditioning**; stored PPO labels are not recomputed. Do not use for on-policy correctness claims.
- Full: eight independent complete 9,216-token model F/B calls without TPR context. Uses **controlled CP1 square-CANN Full**, not the native Megatron/MindSpeed schedule. Do not mix this baseline with Native Full CP4.
- TPR: same TP2 model and PPO fields, one Root / eight distinct Leaves through the production SegmentExecutor. Recompute OFF. No optimizer step.
- Timer: F/B plus CP1/DP1 no-op gradient finalize; synchronization on every rank. Median of 3 rank-max measured iterations after 1 warmup, plus worst peak allocated/reserved. No full-gradient dumps, no per-layer profiler, no Fixed Tile.

## Run

Synchronize the experiment branch to the Docker worktree and run:

```bash
cd /workspace/verl-tpr-ring-review
bash tests/models/mcore/tpr/parallel/run_tpr_cp1_n8_9k_perf.sh
```

Set TPR_CP1_N8_SOURCE for another source if the default capture lacks a valid eight-way common 8K prompt suffix. Use TPR_CP1_N8_ROOT for a fresh output directory.

### CPU preflight on the Ascend server

Use the repository's isolated CPU harness instead of importing the full
Megatron/MindSpeed stack with direct `python -m pytest`:

```bash
python tests/models/mcore/tpr/run_cpu_regressions.py --only-cp1-n8-perf
```

If both `results/full.json` and `results/tpr.json` already exist, the
performance measurement has already finished. After a comparator-only fix,
do not rerun the NPU benchmark; simply regenerate the missing
`results/comparison.json` via the compare module.

## Experiment artifact layout

All NEW runtime artifacts are below this one directory, not the Git checkout:

```text
/workspace/tpr_experiments/performance/cp1_n8_9k_8to1_v1/
├── data/
│   ├── cp1_n8_p8192_s1024.pt
│   └── cp1_n8_p8192_s1024.audit.json
├── logs/
│   ├── preflight.log
│   ├── prepare.log
│   ├── full.log
│   ├── tpr.log
│   └── compare.log
└── results/
    ├── full.json
    ├── tpr.json
    └── comparison.json
```

The original input remains read-only at
`/workspace/tpr_actor_capture_4b_n8_run2/actor_update_step000003_rank2_capture000002_batch0.pt`
unless overridden with TPR_CP1_N8_SOURCE.

**Failure diagnostics:** Data preparation logs `TPR_N8_SOURCE_LENGTHS`
for every original row (Prompt length, Response length, PPO supervision, and
exact shortfall) and `TPR_N8_ACTUAL_TREE` for the actual compressed segment
topology, before rejecting an ineligible capture. Even if the
preparation fails, its log stays at `logs/prepare.log`; no derived input or
F/B result is created. A row with Response shorter than 1024 is *not*
automatically padded or replaced. Pick another eight-row source whose actual
shared prompt and response lengths meet the defined workload.

Existing older experiments under `/workspace/tpr_experiments/` and
original captures are deliberately NOT moved or deleted by this script. The
former default `/workspace/tpr_experiments/cp1_n8_9k_8to1_v1/` may contain
an empty folder from a failed run; inspect it before manually removing it.

**OOM policy:** report failure; do not automatically reduce P, S or N, and do not enable Recompute mid-comparison. Any shortened case must be separately labeled.

**Baseline limitation:** Final production Native Full vs TPR comparisons still require a separate validated CP1 native Megatron schedule, since the current native Full PPO helper hardcodes CP4. This controlled CP1 square-CANN result measures physical whole-sequence versus reuse, not production Native Full throughput.

## 27K CP4 P:S comparison (Native Full vs TPR)

Use the same audited N8 capture derivation and original fixed-shape CP4
native Megatron/MindSpeed Full-vs-TPR F/B runner. Two independent workloads:

| Ratio | P | S | L | TP x CP x DP |
|---|---:|---:|---:|---|
| 1:1 | 13824 | 13824 | 27648 | 2 x 4 x 1 |
| 8:1 | 24576 | 3072 | 27648 | 2 x 4 x 1 |

Each ratio requires **eight real captured response rows** with at least S
original response tokens each and a **token-identical trailing prompt**
of length P. All eight must have positive original PPO supervision in their
retained response window; no label synthesis or automatic ratio adjustment.

Run for each ratio with its own source capture:

```bash
bash tests/models/mcore/tpr/parallel/run_tpr_cp4_n8_27k_perf.sh 1to1 /path/to/eligible_capture.pt
bash tests/models/mcore/tpr/parallel/run_tpr_cp4_n8_27k_perf.sh 8to1 /path/to/eligible_capture.pt
```

Set TPR_SHAPE_PREFLIGHT_ONLY=1 to prepare/validate *without starting NPUs*.
Default output directories:

- /workspace/tpr_experiments/performance/cp4_n8_27k_1to1_v1/{data,logs,results}
- /workspace/tpr_experiments/performance/cp4_n8_27k_8to1_v1/{data,logs,results}

The original Run2 capture used for the CP1 test has Prompt=17483 for all
rows but responses of 48538,17444,7560,51885,30929,57442,25058,6026.
It **cannot** serve as 27K 1:1: rows 2 and 7 are too short. It also
**cannot** serve as 27K 8:1: 24576-token common prompt is unavailable.
These CP4 benchmarks must remain PENDING until eligible captures are found
or a separately disclosed synthetic-shape workload is explicitly approved.
They are not to be silently replaced with unrelated test cases.

For all three benchmark pairs, use 1 warmup + 3 rank-max timed F/B runs,
Recompute OFF, profiler OFF and no AdamW. NPU OOM is retained as a separate
capacity result; do not infer a speedup from a failed mode.

## Additional 9K/36K N8 shape sweep (October 2026)

The original 9K 8:1 and 27K controls remain unchanged. Three additional,
**independent** workloads are supported; 1 Ki token = 1024 tokens.

| Case | P | S | Length | Parallelism | Full baseline |
|---|---:|---:|---:|---|---|
| CP1 9K 1:1 | 4608 | 4608 | 9216 | TP2 CP1 DP1 | controlled whole-sequence Square-CANN |
| CP4 36K 1:1 | 18432 | 18432 | 36864 | TP2 CP4 DP1 | native Megatron/MindSpeed |
| CP4 36K 8:1 | 32768 | 4096 | 36864 | TP2 CP4 DP1 | native Megatron/MindSpeed |

A single sequential command from the Docker repository is:

```bash
cd /workspace/verl-tpr-ring-review
set -o pipefail
bash tests/models/mcore/tpr/parallel/run_tpr_n8_9k_36k_sweep.sh \
  2>&1 | tee /workspace/tpr_experiments/performance/n8_9k_36k_sweep.log
```

CP1 defaults to the original Run2 8-row capture. CP4 searches read-only under
`/workspace/tpr_actor_capture_4b_n8_run{5,3,2}` for the first eligible original
Actor PPO mini-batch capture; it checks eight distinct response branches, a
token-identical P-token prompt suffix, retained response masks and genuine PPO
labels. Override with `TPR_CP4_N8_SOURCE_18K_18K` and
`TPR_CP4_N8_SOURCE_32K_4K` (or one `TPR_CP4_N8_SOURCE` for both).
Missing eligible source is a **PENDING** case, never a synthetic shape or a
benchmark speedup. The older Run2 source is **ineligible for both new CP4 shapes**.
A CP4 36K Full with recompute OFF may OOM; failure must remain a capacity result.

Outputs are separated under `/workspace/tpr_experiments/performance/`:

- `cp1_n8_9k_1to1_v1/{data,logs,results}`
- `cp4_n8_36k_1to1_v1/{data,logs,results}`
- `cp4_n8_36k_8to1_v1/{data,logs,results}`

The script skips a case only if its `results/comparison.json` exists. All
three are F/B-only: 1 warmup + 3 rank-max repeats, Recompute OFF, optimizer
excluded. Because the CP1 and CP4 Full baselines differ, do not pool their
speedup values as if they shared one production baseline.

## CP2 N=8 controlled 18K P:S sweep

After the CP4 36K attempts yielded only `prompt_shortfall` (360/360
candidate sources), run **two CP2 shapes at identical total length**
using native Megatron/MindSpeed Full vs production TPR Ring:

| Case | P | S | Length | Topology | Ranks |
|---|---:|---:|---:|---|---:|
| CP2 8:1 | 16,384 | 2,048 | 18,432 | TP2 CP2 DP1 | 4 |
| CP2 1:1 | 9,216 | 9,216 | 18,432 | TP2 CP2 DP1 | 4 |

Both use Qwen3-4B BF16, N=8 captured Actor PPO rows, 1 warmup + 3
F/B repetitions, recompute OFF, optimizer excluded and no Fixed Tile.
The same real-source eligibility gates apply: token-identical trailing
prompt, eight distinct real response branches and retained PPO labels.
Searching the available Run2/Run3/Run5 data does **not** guarantee a
capture will pass either gate. In particular Run2 has an original row
with under 9,216 response tokens, so it is ineligible for CP2 1:1.

Docker command:

```bash
cd /workspace/verl-tpr-ring-review
set -o pipefail
bash tests/models/mcore/tpr/parallel/run_tpr_cp2_n8_18k_sweep.sh \
  2>&1 | tee /workspace/tpr_experiments/performance/cp2_n8_18k_sweep.log
```

Optional explicit real source files: `TPR_CP2_N8_SOURCE_16K_2K` and
`TPR_CP2_N8_SOURCE_9K_9K`. The default source finder searches
`/workspace/tpr_actor_capture_4b_n8_run{5,3,2}` without rewriting
or fabricating source values.

Isolated results:
- `/workspace/tpr_experiments/performance/cp2_n8_18k_8to1_v1/results/comparison.json`
- `/workspace/tpr_experiments/performance/cp2_n8_18k_1to1_v1/results/comparison.json`

The prior CP1 9K 1:1 F/B result measured 9.818837s Controlled Full
versus 6.694984s TPR (1.4666x), 41.47494 GiB vs 25.62606 GiB
allocated memory (38.21% saving). CP2's native Full baseline is
a different reference and should not be pooled with CP1's controlled
Square-CANN baseline. New CP2 NPU code must still pass server
preflight and benchmark execution before any measured speedup claim.

## N8 CP2 alternatives when P9216/S9216 source is unavailable

Existing source preflight checked 360 Run2/Run3/Run5 Actor PPO files, with
200 eight-row response shortfalls and 160 eight-row prompt shortfalls for
exact P9216/S9216. The original Run2 sample has eight prompts of 17,483
tokens; per-row response lengths are
[48538, 17444, 7560, 51885, 30929, 57442, 25058, 6026].
It supports **N8 P6016/S6016** (strict 1:1, 12,032 tokens) and
**N8 P12416/S6016** (about 2.06:1, 18,432 tokens) by cropping existing
tokens only. Both require the strict source/PPO/audited-Root preflight and
produce independent Full/TPR NPU comparisons.

Use read-only inventory first:

```bash
cd /workspace/verl-tpr-ring-review
python -m tests.models.mcore.tpr.parallel.scan_tpr_n8_capture_lengths \
  --top 10 \
  --json /workspace/tpr_experiments/performance/n8_capture_lengths.json \
  2>&1 | tee /workspace/tpr_experiments/performance/n8_capture_lengths.log
```

It reports the globally shortest/longest captured response, each capture's
minimum and maximum responses, its length-only maximal balanced N8 shape,
fixed-18K N8 shape, and which rows could support two *N6* controlled
ratios (N6 is a proposed separate experiment, **not** implemented by these
N8 benchmark scripts). Length-only shortlisting cannot replace token
identity and supervision checks.

To execute both implemented fallback modes on the same original Run2 source:

```bash
bash tests/models/mcore/tpr/parallel/run_tpr_cp2_n8_fallback_ratios.sh both
```

Output paths:
- `/workspace/tpr_experiments/performance/cp2_n8_12k_1to1_v1/results/comparison.json`
- `/workspace/tpr_experiments/performance/cp2_n8_18k_2to1_v1/results/comparison.json`

Optional run of matched-source CP2 16K/2K into an independent output path:

```bash
bash tests/models/mcore/tpr/parallel/run_tpr_cp2_n8_fallback_ratios.sh 16k_2k
```

This writes to `cp2_n8_18k_8to1_run2matched_v1`, without overwriting
the prior CP2 16K/2K result. The same-source, fixed-total-length 2.06:1
vs 8:1 comparison isolates the prefix ratio better than comparing 12K vs
18K; the 12K 1:1 case tests the desired exact ratio but changes length.
Neither should be relabeled as an exact 18K 1:1 result. If a full N6 18K
1:1 comparison is needed, derive six-row captures consistently from source
rows (0,1,3,4,5,6) and benchmark **both** N6 8:1 and N6 1:1 cases rather
than comparing N6 to the existing N8 baseline.

## Controlled synthetic-response-window sweep: CP2 9K/9K and CP4 17,440/17,440

For **performance only**, retaining true *token window contents* is enough;
the PPO labels are historical, so this benchmark is explicitly **not
on-policy or PPO correctness evidence**.

A Run2 eight-row Actor update includes 17,483-token identical prompts and
responses of lengths [48538, 17444, 7560, 51885, 30929, 57442, 25058,
6026]. Keep six original long-response rows (0,1,3,4,5,6); replace the
two insufficient rows with **distinct later response windows** from
original long rows 0 and 3. Source responses, masks, old log-probs, advantages
and temperature are copied/sliced from their actual donor rows, never
filled with fake zeros or duplicated identical Response streams. The
result is a *synthetic N8 performance workload* with identical full/TPR
inputs, not eight authentic sampled trajectories. Donor indices and
response offsets are preserved in `*.windows.json`, and downstream
capture audit/benchmark report includes
`synthetic_response_windows=true`.

| Mode | P | S | Whole sequence | NPU | Baseline |
|---|---:|---:|---:|---:|---|
| CP2 1:1 | 9216 | 9216 | 18432 | 4 | Native Megatron/MindSpeed Full |
| CP4 1:1 | 17440 | 17440 | 34880 | 8 | Native Megatron/MindSpeed Full |

Choose 17,440 rather than 17,483 for CP4: the shorter original row1 has
17,444 response tokens, and 34,880 is divisible by TP2/CP4's 16-token
zigzag alignment. This differs by just 43 tokens per segment from the
nominal 17,483/17,483 target.

Run within the Docker checkout after syncing PR12:

```bash
cd /workspace/verl-tpr-ring-review
set -o pipefail
bash tests/models/mcore/tpr/parallel/run_tpr_n8_synthetic_window_perf.sh both \
  2>&1 | tee /workspace/tpr_experiments/performance/n8_synthetic_window_sweep.log
```

Modes `cp2` and `cp4` can be run separately instead of `both`.
The original Run2 path can be overridden by
`TPR_SYNTHETIC_WINDOW_SOURCE`. Every mode has separate immutable staged
source, derived capture, audit, Full/TPR performance JSON and comparison
report below its own versioned folder. Nonempty PPO supervision in each
window and distinct final responses remain mandatory. If the 17K native
Full run OOMs with Recompute OFF, report this as an OOM, not an infinite
speedup or a production-size parity result.

Results:
- `/workspace/tpr_experiments/performance/cp2_n8_18k_1to1_synthetic_windows_v1/results/comparison.json`
- `/workspace/tpr_experiments/performance/cp4_n8_34k_1to1_synthetic_windows_v1/results/comparison.json`

Cross-ratio comparisons to *authentic-window* CP2 16K/2K must clearly
mark that branch-source provenance changes; these are system-performance
stress comparisons, not a controlled on-policy PPO ablation.
