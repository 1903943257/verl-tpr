# CP1 Leaf Micro-batch experiment (opt-in)

This branch explores memory-constrained leaf grouping without replacing the
existing DFS Prefix Push/Pop, PPO objective, KV anchors, or gradient finalizer.
It is **not** an enabled production optimization.

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
