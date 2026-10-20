# CP1 long-prefix / short-suffix F/B benchmark (N=8)

Status: experimental runner on a branch from main; **no Ascend measurements yet**.

- Real Qwen3-4B-128K BF16 checkpoint, TP=2, CP=1, DP=1, **2 Ascend NPUs**.
- Fixed **N=8**, retained original actor PPO response tokens and historical old_log_probs, advantages, response_mask, loss_mask, temperature.
- Each trajectory: common **P=8,192 tokens** (last prompt tokens), distinct **S=1,024 tokens** (first response tokens); L=9,216. All eight prompts must share the exact 8,192-token suffix. Actual qwen_turn Root must cover eight leaves. Incompatible sources are rejected, not repaired.
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

**Failure diagnostics:** Data preparation logs `TPR_CP1_N8_SOURCE_LENGTHS`
for every original row (Prompt length, Response length, PPO supervision, and
exact shortfall), before rejecting an ineligible capture. Even if the
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
