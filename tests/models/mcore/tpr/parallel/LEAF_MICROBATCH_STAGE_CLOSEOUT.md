# CP1 Leaf Micro-batching: N=8 stage closeout

Status: controlled performance benchmark **measured**, complete PPO gradient
equivalence **NOT passed**, production native VERL microbatch semantics **NOT
implemented**. This is a diagnostic result, not a claim of training E2E speedup.

## Fixed conditions

- Real Qwen3-4B BF16, single Ascend NPU, TP=CP=DP=1
- Eight distinct same-session original captured trajectories, controlled short
  input crop changing conditioning, original retained PPO labels
- Each run covers **eight** same original logical trajectories and **6,478**
  supervised PPO tokens, logical input **15,040** tokens
- Full square-CANN controlled whole-sequence physical MB=2: four model calls
- TPR one shared Root and eight Leaf segments (same exact logical rows)
- Median of 3 measured F/B iterations, no optimizer/recompute/CP
- Actual physical Tree unique tokens: **7,900**, dedup **47.47%**
- Root Push, all Leaf groups, Root Pop, and TPR gradient finalize **included**
  in TPR fair wall time; profiler stage timings measured in separate subprocesses

## Verified N8 results from saved SUMMARY.txt (2026-10-23)

| Metric | Full MB=1 | Full MB=2 | TPR K=1 | TPR K=2 | TPR K=4 |
|---|---:|---:|---:|---:|---:|
| F/B median, sec | **pending supplemental run** | 2.5136 | 2.0150 | 1.7183 | 1.5791 |
| Forward calls | 8 expected; verify | 4 | 10 | 6 | 4 |
| Peak allocated, GiB | pending | 41.629 | 21.732 | 28.279 | 40.399 |
| Peak reserved, GiB | pending | 44.918 | 23.527 | 31.061 | 43.014 |
| Full MB2 / TPR speedup | n/a | 1.0000x | 1.2475x | 1.4628x | 1.5918x |

TPR K2 improves on K1 by **1.1726x**, and preserves **13.350 GiB
allocated / 13.857 GiB reserved** headroom against Full MB2. K4 reaches
1.5918x but only **1.230 GiB allocated / 1.904 GiB reserved** headroom.
K2 is the **capacity-balanced experimental choice**, not a demonstrated
default for arbitrary production forests.

Numerical correctness is an **open blocker**: sampled parameter gradient
relative L2 vs Full MB2 was 0.205387 (K1), 0.280163 (K2), 0.243355
(K4); logits/logprob relative L2 was about 0.029-0.032. Similar PPO loss
does not establish gradient equality.

Stage event numbers K2/K4 sum to about 4.1 sec vs fair 1.72/1.58 sec;
profiling overhead/measurement scope is therefore not calibrated. Use
stage events to see grouping/call distribution, **not** to infer fair time
shares or claim a Root Push/Pop bottleneck.

## One remaining measurement: N=8 physical Full MB=1

Reuse existing captured N8 data and all completed benchmark arms.
Only execute `full_mb1`, yielding **eight** physical `[1,S]` model calls,
with the same original inputs/labels and exactly the same square-CANN Full
reference used in MB2.

**Host:**
```bash
cd /mnt/pipeline-data/j00972288/project/uniagent_linear/verl-tpr-leaf-microbatch
git fetch tpr feat/tpr-leaf-microbatch-cp1-20261022
git merge --ff-only FETCH_HEAD
git status --short --branch
```

**Docker** (updated `/workspace/verl-tpr-leaf-microbatch`):

```bash
cd /workspace/verl-tpr-leaf-microbatch
python -m pytest -q tests/models/mcore/tpr/unit/test_cp1_n8_summary_cpu.py
export TPR_CP1_N8_ROOT=/workspace/tpr_experiments/cp1_n8_full_mb2_vs_leaf_k_v1
export TPR_CP1_N8_WARMUPS=1
export TPR_CP1_N8_REPEATS=3
set -o pipefail
bash tests/models/mcore/tpr/parallel/run_cp1_n8_add_full_mb1.sh \
  2>&1 | tee "${TPR_CP1_N8_ROOT}.full_mb1.console.log"
rc=${PIPESTATUS[0]}
echo "===== Full MB1 supplement completed rc=$rc ====="
cat "$TPR_CP1_N8_ROOT/SUMMARY.txt" 2>/dev/null || \
  tail -n 70 "${TPR_CP1_N8_ROOT}.full_mb1.console.log"
```

**No terminal `exit` statement**: interactive host/Docker shell stays open.
The runner refuses any existing `full_mb1` directory and does not rerun or
modify original `full_mb2/tpr_k1/tpr_k2/tpr_k4` result JSON files. It
rebuilds `SUMMARY.txt` and `summary.json` with the new MB1 column, the
Full MB1 vs MB2 speedup, Full MB1 vs each TPR arm, and Full MB1 vs MB2/K1
numerical diagnostic.

### Scope of this stage

**Complete after Full MB1 run:** CP1 equal-length sibling Leaf grouping
K1/K2/K4 physical execution, eight-row apples-to-apples F/B, Root Push/Pop
included, memory/timing report.

**Not claimed:** native VERL microbatch scheduling compatibility, variable
length/padded/packed batching, production E2E speedup, numerical gradient
equivalence, stable throughput across other datasets, CP>1 Leaf batching.

Do not introduce more branches, production path changes, or larger K in this
stage. Future work requires a separate correctness gate.


## 2026-10-25 repository merge inventory and next PR Draft gates

GitHub REMOTE (not necessarily the server's local worktree) has **18
branches including `main`**. Core pending work:

| Topic | Branch | GitHub comparison to main | Decision |
|---|---|---|---|
| Ring/performance investigation | `feat/tpr-cp1-n8-p8192-s1024-perf-20261019` | ahead 134, behind 9; PR #12 Draft, mergeable=dirty | Rebase/split benchmarks and production Ring change before merging |
| Old Ring/PPO slim | `refactor/tpr-pr8-net-slim-ring3-20261019` | ahead 24, behind 113 | Inspect overlap with Ring branch; avoid blanket merge |
| Leaf MB | `feat/tpr-leaf-microbatch-cp1-20261022` | ahead 77, behind 0 at audit time | Full MB1 pending; review and gate correctness |
| Numerical / QKV audit | `test/tpr-qkv-fp64-grad-buffer-audit-20261022` | ahead 35, behind 0; PR #14 Draft | Distinguish diagnostic tests vs runtime changes; NPU gate |
| Tests organization | `refactor/tpr-test-suite-organization-20261019` | ahead 51, behind 0; 253 changed files, 19,638 deletions | LAST, after reconciling newer tests |
| 4K split one-file branch | `test/tpr-cp1-4k-sequence-split-20261023` | ahead 1, behind 0 | Check duplication with PR #14 before merge |
| Short-fix corpus screening | `feat/tpr-swe-rebench-shortfix-screen-20261023` | ahead 3, behind 0 | Keep independent for later long-prefix/short-suffix task selection |

Historical branches still remotely present: `docs/tpr-learning-guide-20261019`,
`docs/tpr-long-prefix-performance-20261019`,
`feat/tpr-full-recompute-on-pr7-main-20261019`,
`feat/tpr-full-recompute-uniform1-prototype`,
`feat/tpr-full-recompute-uniform1-refactor`,
`feat/tpr-production-dp2-ring-3files-20261019`,
`refactor/tpr-cp-ring-only-qkvmerge-20261015`,
`refactor/tpr-ppo-engine-ring-turn-20261015`,
`refactor/tpr-production-api-state-slim-20261016`,
`refactor/tpr-ring-layout-validation-main-20261016`.
Previous PRs #6-#11 already merged baseline Ring, DP and Recompute
pieces. Do NOT merge stale branches solely because they still exist.

**Correct sequence, contingent on passing tests and code review:**
(1) run only N8 Full MB1, append actual measured result to table above;
(2) disentangle and validate Ring/performance changes (PR #12 dirty);
(3) finish correctness audit PR #14, or merge it explicitly as diagnostics
without claiming gradient parity; (4) Leaf K2 review and verification, rebase
after conflicting `adapter.py` work; (5) test organization last, retaining
newer files added to main; (6) AFTER consolidation, resume SWE-reBench
candidate selection and real UniAgent long-prefix/short-suffix workload
research; (7) prepare one clearly scoped upstream Draft PR on the resulting
main once repository/CI and benchmark provenance are audited.

Known blockers for an upstream production PR: ~20-28% sampled gradient rel-L2
gaps in controlled CP1 comparisons; TPR bypasses native
`prepare_micro_batches` (K is not VERL native MB); CP1 equal-length only
for Leaf grouping; NPU acceptance after branch integration; PR #12 conflict
with main; tests cleanup may delete newer test modules. Upstream claim must
be limited to verified configurations and clearly distinguish offline
model-F/B speed from optimizer/rollout E2E.

MB1 result is STILL PENDING: do not insert invented timing, memory, parity or
speedup numbers. The existing runner
`tests/models/mcore/tpr/parallel/run_cp1_n8_add_full_mb1.sh` is
the sole remaining stage performance measurement. It must use the preserved
old N8 Capture and never overwrite the other four arm result JSON files.
