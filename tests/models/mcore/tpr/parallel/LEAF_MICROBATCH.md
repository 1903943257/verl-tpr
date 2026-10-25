> **Merge scope / correctness caveat (2026-10-25):** TreeBatch K>1 is being integrated as an opt-in CP1 capability; this does not mean full-model numerical equivalence has passed. The default remains `TPR_LEAF_BATCH_SIZE=1`. The isolated two-sibling test passes in FP32 (gradient rel-L2 ~`6.3e-8`) and BF16 (rel-L2 ~`2.53e-3`, within the test's `5e-2` tolerance). However, the real Qwen3-4B N=8 K1-vs-K2 comparison still reports full-gradient rel-L2 ~`0.321` and cosine ~`0.951`; this is unresolved and must not be described as numerical equivalence. The performance numbers are controlled CP1 model F/B, not native VERL microbatch scheduling or rollout/optimizer-inclusive E2E. Further parity work is separate from this feature-integration scope.

# CP1 Leaf Micro-batch experiment (opt-in)

This branch explores memory-constrained leaf grouping without replacing the
existing DFS Prefix Push/Pop, PPO objective, KV anchors, or gradient finalizer.
It is **not** an enabled production optimization.

## Where commands run (host vs Docker)

**Host only — Git / worktree synchronization.** The actual source of this
experiment is the dedicated `verl-tpr-leaf-microbatch` worktree. Do not
switch branches in the Ring review worktree or modify `main`.

```bash
# HOST: run inside your existing leaf-microbatch worktree
cd /mnt/pipeline-data/j00972288/project/uniagent_linear/verl-tpr-leaf-microbatch
git fetch tpr feat/tpr-leaf-microbatch-cp1-20261022
git merge --ff-only FETCH_HEAD
git status --short --branch
```

**Docker only — CPU/NPU Pytest and real Actor PPO Capture.** Confirm the
Docker's VERL tree contains the updated commit before running; host Git
changes are not necessarily visible inside the container. Use the existing
host-to-Docker sync / mount arrangement. All `/workspace/tpr_experiments`
paths below refer to the Docker's filesystem.

```bash
# DOCKER: after synchronizing the updated VERL source
cd /workspace/uni-agent/verl  # adjust to your actual mounted/synced VERL directory
export TPR_CP1_CAPTURE=/workspace/tpr_experiments/leaf_batch_k2_first_run/actor_pair_cap2048_p1024.pt
export TPR_CP1_LEAF_ROOT=/workspace/tpr_experiments/leaf_k1k2_example
export TPR_CP1_LEAF_TRACE=1
mkdir -p "$TPR_CP1_LEAF_ROOT"
set -o pipefail
bash tests/models/mcore/tpr/parallel/run_cp1_shared_leaf_k1_k2.sh \
  2>&1 | tee "$TPR_CP1_LEAF_ROOT/run.log"
rc=${PIPESTATUS[0]}
echo "===== LAST SAVED RESULTS / ERRORS exit=$rc ====="
cat "$TPR_CP1_LEAF_ROOT/SUMMARY.txt" 2>/dev/null || {
  tail -n 80 "$TPR_CP1_LEAF_ROOT/run.log"
}
exit "$rc"
```

The runner now always writes and prints **`SUMMARY.txt`** at exit. It shows
an at-a-glance K=1/K=2 table with forward counts, grouping, F/B time, peak
allocated/reserved, loss, gradient norm, and parity gaps. On OOM/exception
it instead identifies missing result arms and prints the current failed log.
It does not turn gradient mismatch into a "PASS", and preserves the original
exit code. Detailed raw data remain in `k1_vs_k2.json`.

## Fair comparison against physical Full MB=2 (two trajectories)

The original `full` arm called Qwen3-4B **twice** with batch size one.
Comparing its time against TPR K2 would unfairly omit conventional batching.
The four-way benchmark therefore adds the `full_batch2` mode, making
**one** whole-sequence GPT invocation on physical input `[2, S]` and
using the same real captured PPO token labels and global denominator.

```text
Full MB=1   two original full trajectories, 2 full-model forwards
Full MB=2   two original full trajectories, 1 full-model forward with B=2
TPR K=1     same two original trajectories, one reused Root + 2 separate Leafs
TPR K=2     same two original trajectories, one reused Root + 1 two-Leaf group
```

**All four modes process exactly 2 logical trajectories.** TPR K=2 is *not*
a smaller dataset: the common Root's tokens are calculated once and the two
suffixes each receive their own PPO loss/gradient. K controls physical sibling
batching *inside* that one forest, not a VERL mini-batch size.

This experiment's `full_batch2` is a **controlled physical-MB2** model
reference using the same square-CANN attention adapter as the other
Full control. It is not yet a true production native VERL/Megatron
`prepare_micro_batches + forward_backward_func + optimizer.step` run.
Consequently, report it as controlled F/B, **not** training step/E2E speedup.

Normal VERL fixed micro-batch configuration is
`actor_rollout_ref.actor.use_dynamic_bsz=False` plus
`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2`.
The Megatron Engine copies this into `engine_config.micro_batch_size_per_gpu`.
With dynamic batch scheduling enabled, `ppo_max_token_len_per_gpu`
rather than this sample-count cap controls micro-batches. **TPR currently
bypasses `prepare_micro_batches()` and runs the whole supplied mini-batch
as a Forest**; `TPR_LEAF_BATCH_SIZE` remains an independent optional
grouping knob. Do not claim these settings are interchangeable.

### Host — Git (not inside Docker)

```bash
cd /mnt/pipeline-data/j00972288/project/uniagent_linear/verl-tpr-leaf-microbatch
git fetch tpr feat/tpr-leaf-microbatch-cp1-20261022
git merge --ff-only FETCH_HEAD
git status --short --branch
```

Sync/mount the updated leaf Worktree into the **actual** Docker source
folder. In the recent run, `/workspace/verl-tpr-leaf-microbatch` had the
new modules while `/workspace/uni-agent/verl` did not: use the former
unless you explicitly synchronized the latter.

### Docker — model / NPU experiments

```bash
cd /workspace/verl-tpr-leaf-microbatch
export TPR_CP1_CAPTURE=/workspace/tpr_experiments/leaf_batch_k2_first_run/actor_pair_cap2048_p1024.pt
export TPR_CP1_FULL_K2_ROOT=/workspace/tpr_experiments/full_mb2_vs_leaf_k2_v1
export TPR_CP1_BENCH_WARMUPS=1
export TPR_CP1_BENCH_REPEATS=3
export TPR_CP1_MEMORY_MARGIN_GIB=0.5
set -o pipefail
bash tests/models/mcore/tpr/parallel/run_cp1_full_mb2_vs_leaf_k2.sh \
  2>&1 | tee "${TPR_CP1_FULL_K2_ROOT}.console.log"
rc=${PIPESTATUS[0]}
echo "===== FINAL FOUR-WAY RESULTS OR FAILURE exit=$rc ====="
cat "$TPR_CP1_FULL_K2_ROOT/SUMMARY.txt" 2>/dev/null || \
  tail -n 90 "${TPR_CP1_FULL_K2_ROOT}.console.log"
exit "$rc"
```

**Do not create the root before invoking this runner**; it deliberately
requires a fresh root to prevent accidental reuse. If capturing outer logs
with tee, tee into the *parent* directory or select a new leaf path.

The runner installs an EXIT trap: even on NPU OOM or a failed arm it
prints a concise table and saves `SUMMARY.txt` and `summary.json`.
The reported time is a median of repeated F/B iterations after warmup;
peak allocated and reserved use the **maximum** observed measured iteration
(per physical Full/TPR arm). A capacity margin is marked PASS only when
**both** TPR K2 peak allocated and reserved are at least
`TPR_CP1_MEMORY_MARGIN_GIB` below Full MB2.

K2 grouping also obeys `TPR_LEAF_BATCH_MAX_TOKENS`: a cap on *suffix
tokens per physical leaf group*, not an absolute memory budget. The
controlled pair here has exactly two trajectories, but extending to
an 8-trajectory PPO mini-batch changes the Forest, graph lifetimes,
and memory; **do not infer its peak from this pair alone.**

The results are diagnostic until numerics are addressed: identical
weights/capture are enforced, but both Full MB2 and TPR K2 can differ from
Full MB1 in BF16 GEMM rounding. A speedup or memory PASS is not proof of
PPO gradient equivalence.

## CP1 N=8: Same two requirements, fair Full MB=2 and stage costs

This next experiment tests the two requested follow-ups:
(1) holding full physical MB=2 fixed, compare `N=8` shared TPR Leaf
Batch `K=1/2/4`, and (2) separately inspect **Root Push,
Leaf/Leaf Group, Root Pop** with NPU-event stage durations.

**N8 data validity:** The default source is a genuine
`actor_update_mini_batch` capture. The N8 preparation chooses **eight
different captured rows from the same original session UID**, requires
identical retained prompt tokens across them, and keeps their original
recorded PPO labels/masks. It refuses to fabricate eight trajectories
from the existing two-row capture. If this source has fewer than eight
same-session rows, it will print `session_group_sizes` and FAIL; find a
different real Actor capture or explicitly supply `TPR_CP1_N8_SOURCE`.
Crop changes original conditioning; this remains a controlled PPO F/B
comparison, not the original on-policy training objective.

### HOST (Git worktree, NOT Docker)

```bash
cd /mnt/pipeline-data/j00972288/project/uniagent_linear/verl-tpr-leaf-microbatch
git fetch tpr feat/tpr-leaf-microbatch-cp1-20261022
git merge --ff-only FETCH_HEAD
git status --short --branch
```

### DOCKER (use updated leaf worktree, not /workspace/uni-agent/verl)

```bash
cd /workspace/verl-tpr-leaf-microbatch
python -m pytest -q -s tests/models/mcore/tpr/unit/test_cp1_n8_summary_cpu.py

export TPR_CP1_N8_SOURCE=/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt
export TPR_CP1_N8_ROOT=/workspace/tpr_experiments/cp1_n8_full_mb2_vs_leaf_k_v1
export TPR_CP1_N8_CAP=2048
export TPR_CP1_N8_PROMPT_WINDOW=1024
export TPR_CP1_N8_WARMUPS=1
export TPR_CP1_N8_REPEATS=3
export TPR_CP1_N8_MEMORY_MARGIN_GIB=0.5
export TPR_CP1_N8_RUN_STAGES=1

# Do not mkdir ROOT ahead of time. This run rejects reused results.
set -o pipefail
bash tests/models/mcore/tpr/parallel/run_cp1_n8_full_mb2_leaf_k.sh \
  2>&1 | tee "${TPR_CP1_N8_ROOT}.console.log"
rc=${PIPESTATUS[0]}
echo "===== N8 FINAL RESULTS OR ERRORS exit=$rc ====="
cat "$TPR_CP1_N8_ROOT/SUMMARY.txt" 2>/dev/null || \
  tail -n 90 "${TPR_CP1_N8_ROOT}.console.log"
exit "$rc"
```

Optional `TPR_CP1_N8_ROWS=0,1,2,3,4,5,6,7` selects exact indices,
but all eight must still be genuine, distinct and from one session.

**Actual work (same for all four benchmark arms):**

- Full MB2: eight complete trajectories, four physical `[2, S]` GPT
  Forward+Backward microbatches, no Prefix cache
- TPR K1: the same eight trajectories, one shared Turn-tree forest,
  sequential sibling visits
- TPR K2: the same eight trajectories, groups of at most two same-parent,
  equal-length sibling leaves, still one forest
- TPR K4: same forest, at most four siblings per grouped Forward;
  if none eligible, fail instead of reporting a fabricated speedup

All fair F/B benchmark arms independently load the same authentic Qwen3-4B
BF16 checkpoint, use the same input SHA and PPO label positions. Each
gets one warmup + 3 measured iterations, full gradient reset between
iterations and no optimizer step. Outputs record measured timing medians,
max peak allocated/reserved GiB, actual token savings and physical
model calls. The summary checks FullMB2-versus-TPR memory safety with
the fixed 0.5 GiB margin, plus Loss, Logprob and sampled gradient parity.

**Stage profiling is isolated from benchmarking:** after the four fair
arms, a new process profiles K2 and K4 individually using
`TPR_EXEC_PROFILE=1` and `SegmentEventProfiler`.
Outputs include grouped sizes and per-kind event-time sums for
`push`, `visit_leaf`, `visit_leaf_group`, `pop`. The event timings
are diagnostic, not independent components that must add up exactly to
the unprofiled F/B wall time. They should not be used as an E2E
speedup. Stage instrumented runs are intentionally single-iteration.

**Failure behavior:** the runner's EXIT trap *always* tries to write
`SUMMARY.txt` and `summary.json`, then prints the current failing
log's error/last lines. No old error logs get mixed into the summary.
The return code remains nonzero on capture incompatibility, OOM,
parity mismatch assertion, or unexpected absence of actual K2/K4 grouping.

## Supported scope

- **CP=1 only**, dense training, same-parent leaves, equal-length adjacent
  siblings, one shared KV path.
- The default `TPR_LEAF_BATCH_SIZE=1` is the existing segment-at-a-time path.
- `TPR_LEAF_BATCH_SIZE>1` enables batched model Forward/Backward. Leaf model
  inputs use shape `[K, S]`; Megatron's attention sees `[S, K, H]`.
- Prefix KV is stored once at batch size 1, expanded across K siblings inside
  attention; autograd sums dK/dV back to the one shared Prefix anchor.
- CANN `TND` uses independent varlen cumulative Q/KV boundaries for each
  equal-length leaf. Every sibling only attends its own suffix and the common
  Prefix; no cross-leaf attention.
- Ineligible leaves use the original single-segment path. CP>1 explicitly
  rejects K>1 instead of silently using unsupported Ring semantics.
- No real-memory auto-tuning, arbitrary-length leaves, TP2 native SP validation,
  or CP4 Grouped Ring are claimed in this first patch.

## Activation

```bash
# Start with CP1, leave normal runs unchanged.
export TPR_LEAF_BATCH_SIZE=2
export TPR_LEAF_BATCH_MAX_TOKENS=8192
# Optional: test K=4 after passing correctness and memory gates.
```

`TPR_LEAF_BATCH_MAX_TOKENS` limits the **total physical leaf tokens per
group** (K × suffix length), not Prefix length. It is a conservative planning
cap, **not a guaranteed GiB limit**. The Prefix KV stack, activation
checkpointing, PPO logits, and allocator fragmentation influence peak memory.

## Verification

CPU:
```bash
python tests/models/mcore/tpr/run_cpu_regressions.py
# or direct pytest in a correctly configured VERL environment:
pytest -q tests/models/mcore/tpr/unit/test_leaf_microbatch_cpu.py
```

NPU:
```bash
pytest -q -s tests/models/mcore/tpr/equivalence/test_leaf_batch_rectangular_npu.py
```

## Visible output and real-model activation gate

`export TPR_LEAF_BATCH_SIZE=2` does **not** change the CPU test's explicit
K=1/2/4 settings or the NPU Attention-only test's fixed B=2. Both tests can
pass without running any grouped Qwen3 model forward.

```bash
# From the new isolated worktree (not the Ring worktree):
pytest -q -s tests/models/mcore/tpr/unit/test_leaf_microbatch_cpu.py
pytest -q -s tests/models/mcore/tpr/equivalence/test_leaf_batch_rectangular_npu.py
```

Look for `TPR_LEAF_CPU_RESULT` (real executor group count, forward calls,
loss/gradient error) and `TPR_LEAF_NPU_ATTENTION_RESULT` (attention-only
dO/dQ/dK/dV forward/backward errors). The latter does **not** validate
Megatron model batching.

For the full Qwen3-4B CP1 short Actor PPO parity gate, set a **fresh**
output directory because the existing script reuses completed results:

```bash
export TPR_LEAF_BATCH_SIZE=2
export TPR_LEAF_BATCH_MAX_TOKENS=8192
export TPR_LEAF_BATCH_TRACE=1
export TPR_CP1_ROOT=/workspace/tpr_experiments/leaf_batch_k2_first_run
bash tests/models/mcore/tpr/parallel/run_real_actor_cp1_short_full_tpr.sh
```

This script requires the original real Actor PPO source capture, Qwen3-4B
checkpoint, and one NPU; it loads model weights and compares four modes,
so it is substantially more expensive than the two small tests. Look for:

- `TPR_LEAF_BATCH_RUN`: actual production adapter model and grouped calls
- `TPR_CP1_LEAF_BATCH_CHECK`: actual real Qwen3 shared-tree grouping
- `TPR_CP1_SHORT_RESULT`: measured forward/backward time, memory, loss

The shared-mode NPU integration test now **fails** with an explicit message
when K>1 but no grouped leaf model forward occurred. A pass therefore proves
that the real model path was reached, not just that the CANN operator works.
The test still uses a controlled cropped capture and does not constitute
a whole production Actor optimizer-step or a CP4 result.

## Isolate grouping errors from pre-existing Full/Shared differences

The existing `run_real_actor_cp1_short_full_tpr.sh` compares Full,
One-node, Split-no-share and Shared. Its `provisional_gates` are diagnostics,
not acceptance assertions, and Shared-vs-Full gradient differences **cannot**
be attributed specifically to leaf batching.

Use the *same exact Shared tree* twice, changing **only** K=1 to K=2:

```bash
# The capture was already derived by run_real_actor_cp1_short_full_tpr.sh
export TPR_CP1_CAPTURE=/workspace/tpr_experiments/leaf_batch_k2_first_run/actor_pair_cap2048_p1024.pt
export TPR_CP1_LEAF_ROOT=/workspace/tpr_experiments/leaf_batch_k1k2_v1
bash tests/models/mcore/tpr/parallel/run_cp1_shared_leaf_k1_k2.sh
```

The runner installs an EXIT diagnostic handler that always prints key
`TPR_*` results or failure lines after all executions. On failure it includes
the last test log and, where supported, `npu-smi` process occupancy on the
physical NPU (`TPR_NPU_SMI_ID`, default 8). An OOM with ~18 GiB reserved by
the process but only a few MiB free on a 61 GiB card strongly suggests
**other device occupants**, not necessarily a K=2-only regression.

To resume after a K=1 success and K=2 OOM, keep the same
`TPR_CP1_LEAF_ROOT`, free the target NPU's occupied capacity, and set
`TPR_CP1_LEAF_RESUME=1`. The runner verifies/reuses the saved K1 result and
archives the incomplete K2 directory and log before rerunning only K2. It
never kills other NPU processes.

This loads Qwen3-4B twice (one NPU) and does not rerun the two unrelated
Full/One-node or Split-no-share cases. It compares exact input/weights,
number of actual grouped model calls, logprobs, loss, sampled/Norm gradients,
optional saved complete parameter gradients, elapsed F/B and peak memory.
The K=2 run **fails** if no real-model grouping occurs, and the comparison
requires the exact Push+2 Leaves+Pop count change `4 → 3`. Results are
saved in `$TPR_CP1_LEAF_ROOT/k1_vs_k2.json`.

To locate the **first forward-stage divergence**, rerun in a new directory
with trace enabled. This records sampled activation and selected gradient
snapshots at layers 1/2/3/4/8/16/36, then compares K1 and K2 by row and stage.
It also checks grouped Attention against independent rectangular Attention
using the exact same Q/K/V. Tracing adds overhead, so **do not use traced
times as performance measurements**.

```bash
export TPR_CP1_CAPTURE=/workspace/tpr_experiments/leaf_batch_k2_first_run/actor_pair_cap2048_p1024.pt
export TPR_CP1_LEAF_ROOT=/workspace/tpr_experiments/leaf_k1k2_trace_v1
export TPR_CP1_LEAF_TRACE=1
unset TPR_CP1_LEAF_RESUME
set -o pipefail
bash tests/models/mcore/tpr/parallel/run_cp1_shared_leaf_k1_k2.sh 2>&1 | tee "$TPR_CP1_LEAF_ROOT/run.log"
rc=${PIPESTATUS[0]}
echo "===== FINAL LAYER TRACE / ERRORS exit=$rc ====="
grep -hE 'TPR_CP1_LEAF_K1_K2_COMPARISON|TPR_CP1_LEAF_K1_K2_REPORT|TPR_LEAF_BATCH_RUN|TPR_CP1_SHORT_RESULT|Traceback|NPU out of memory|FAILED|ERROR' "$TPR_CP1_LEAF_ROOT"/*.log 2>/dev/null | tail -n 80 || true
exit "$rc"
```

If the two traced `trace.pt` files already exist, inspect amplification
across sampled layers **without another NPU run**. This never changes a
saved PPO result or calls the model:

```bash
ROOT=/workspace/tpr_experiments/leaf_k1k2_trace_v1
set -o pipefail
python -m tests.models.mcore.tpr.parallel.compare_real_actor_cp1_leaf_batch \
  "$ROOT/k1/result.json" "$ROOT/k2/result.json" \
  --output "$ROOT/k1_vs_k2_stage_reanalysis.json" --require-trace \
  2>&1 | tee "$ROOT/stage_reanalysis.log"
rc=${PIPESTATUS[0]}
echo "===== FINAL RESULTS / ERRORS exit=$rc ====="
grep -E 'TPR_CP1_LEAF_K1_K2_COMPARISON|TPR_CP1_LEAF_K1_K2_REPORT|Traceback|ERROR' \
  "$ROOT/stage_reanalysis.log" | tail -n 30 || true
exit "$rc"
```

**Interpretation correction:** the historical `same_qkv_attention`
probe compares rectangular causal Attention with a square-causal replay
using the same K/V and reconstructed Prefix Q. It is *not* the isolated
B=2 vs B=1 rectangular Attention test. New trace runs additionally save
`group_vs_single_rectangular`, and print
`largest_group_vs_single_rectangle_rel_l2`: identical Q/K/V, grouped
TND against independent per-leaf TND. Old trace.pt files lack this new
measurement, so reanalysis can only summarize existing data.

The new summary includes `layer_progression`, with the largest relative L2
among sampled stages in each checked layer. The complete layer-1 sampled
stage ordering is in `layer_trace.layer1_forward_stages` in the output JSON.
A recorded early 1e-5 BF16 QKV difference can propagate through later
layers, but this must be demonstrated by the progression rather than assumed.

To isolate **physical GEMM-M shape**, run K1/K2 again in a fresh trace
root after updating this branch. New K2 trace captures first-layer
`linear_qkv` full inputs/outputs, then after training replays the *same*
K2 inputs with the original QKV module at B=1. The summary includes
`same_input_l1_qkv_replay` per leaf:

- `k2_grouped_vs_same_input_single`: native grouped GEMM vs native
  independent GEMM, with exactly the same input/weights.
- `k1_vs_k2_same_input_single`: K1 original output vs native B1
  replay on K2's input. Near-zero here and nonzero in the first
  comparison would implicate GEMM-M shape directly.

Diagnostic replays happen after the measured model update (no optimizer
step) and are excluded from the timed F/B metrics, but the trace hooks
themselves have overhead. Do not interpret trace timings as benchmarks.

The comparator prints `first_forward_mismatch` and
`largest_same_qkv_attention_rel_l2`; the full ordered stage report is saved
in `k1_vs_k2.json`. Interpret the first differing stage as follows:

- `qkv_out`: physical-B GEMM / projection kernel behavior is the first sampled difference.
- `query_post_rope`, `post_rope_k` or `raw_v`: QKV/RoPE or KV construction.
- `attn_out` plus nontrivial same-QKV Attention error: grouped CANN TND attention path.
- `proj_out` / `fc*_out`: later GEMM/MLP stage.
- Forward snapshots agree but PPO logprobs differ: inspect objective row selection and response-token mapping next.

This isolates any additional error due to K=2 independently from pre-existing
differences between TPR and a Full no-context reference. No NPU outcome is
claimed until this comparison is executed.

Before accepting the feature, rerun the real Actor PPO capture on **CP1** with
K=1/2/4 and compare identical model weights, tree, PPO masks and total
denominator. Check loss, logprob, every parameter gradient, Prefix dK/dV,
`tpr/model_forward_calls`, `tpr/grouped_leaf_calls`, F/B time, and
`torch.npu.max_memory_allocated/reserved`. Use the identical native Full
training case for the memory baseline; require a safety margin.

CP4 Ring and its communication protocol are **unchanged** here. A follow-up
must implement a segmented/grouped Ring backend with proper per-leaf masks,
online softmax, dKV aggregation and CP collective order before enabling
grouping with CP>1.
