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
