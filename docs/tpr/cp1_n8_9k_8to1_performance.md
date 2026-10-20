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
`/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt`
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
