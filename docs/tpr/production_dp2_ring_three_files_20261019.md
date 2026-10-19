# Production DP2 + three-file Ring refactor (2026-10-19)

Base: `93b24dff45d959a56b609cd99893ea90c1b94212`. This is a code change awaiting NPU acceptance, not a
claim of completed production training.

## DP scope

- Newly admitted **TP2×CP2×DP2 PPO** (8 NPUs, PP=EP=1). Uses VERL's
  existing equal-cardinality DP mini-batch dispatch, global token-mean
  normalization, Megatron's native gradient finalize and Megatron Optimizer.
- TPR trees and retained Prefix KV are **DP-local**. A Group split across
  replicas cannot share Prefix KV between replicas. The optional DTA/DFS
  planner in `placement.py` remains offline and does NOT modify VERL dispatch.
- Other DP>1 training topologies remain unsupported, including TP2×CP4×DP2.
  Previously supported DP1 and the isolated 10K parity matrix are unchanged.
- For useful prefix reuse, distribute several related siblings per DP-local
  mini-batch; singleton mini-batches defeat the goal.

## Ring ownership

| File | Owns |
|---|---|
| `ring_layout.py` | SequenceShard, zigzag layout, physical Padding, static config, shape checks |
| `ring_attention.py` | RingP2P, FA masks/kernels, online softmax, Prefix KV/Q merge, backward owner dKV |
| `ring.py` | Public entry points, RingCPAttentionBackend/RingCPBackend, CP runtime and loss aggregation |

Public imports and existing private symbols are re-exported by `ring.py`
for compatibility. Internal monkeypatches must target the owning module,
rather than a re-exported alias.

Ring transport/FA scheduling and gradient mathematics are moved unmodified.
`TPR_RING_COALESCE_PREFIX_FULL`,
`TPR_RING_COALESCE_PREFIX_QUERY`, and
`TPR_RING_COALESCE_PREFIX_PADDED` remain opt-in (default 0).

## Suggested CPU verification

```bash
python -m pytest -q \
  tests/models/mcore/tpr/unit/test_production_architecture.py \
  tests/models/mcore/tpr/unit/test_tpr_production_dp2_gate_cpu.py \
  tests/models/mcore/tpr/unit/test_ring_coalescing_schedule.py \
  tests/models/mcore/tpr/unit/test_ring_streaming_transport.py \
  tests/models/mcore/tpr/parallel/test_ring_cp_attention_contract.py \
  tests/models/mcore/tpr/parallel/test_ring_numerics_cpu.py \
  tests/models/mcore/tpr/parallel/test_ordinary_ring_adapter.py
```

## Required NPU follow-up before merging

1. Run TP2×CP2×DP2 **actual VERL ActorWorker→MegatronEngine→optimizer**
   for five continuous PPO updates; verify DP gradient reduction, CP/TP
   parameter replicas, nonzero optimizer state, global denominator, and
   full-step throughput. The previous isolated DP2 offline F/B and one
   hand-coded AdamW update do **not** satisfy this.
2. Run CP2/CP4 Ring attention/oracle and real Qwen PPO parity on the
   relocated implementation, including Padding and Q/KV Merge OFF/ON.
3. Compare unchanged old Main and this branch on one identical 30K/40K
   capture: logprob/loss/full gradients, memory and timing. Measure the
   **full Actor Update** separately from the F/B-only benchmark.
