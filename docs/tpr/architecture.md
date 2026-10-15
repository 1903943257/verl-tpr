# TPR production architecture (Dense PPO)

This document specifies the **intended production execution contract** of the
VERL-TPR implementation. The codebase targets Ascend NPU + Megatron; it is not
an alternate PPO implementation. See the [development baseline](../tpr_development_baseline.md)
for the reproducible imported starting point.

## Execution overview

```text
VERL Trainer / TrainingWorker
    | original real actor PPO mini-batch and native loss callback
MegatronEngine.forward_backward_batch
    | tpr_enabled after batch_num_tokens and dp_size are established
workers.engine.megatron.tpr.adapter
    | Turn-aware forest, CP/TP compatibility gates, one native finalize
    +-- tree.py       : Qwen3 ChatML turn radix, SegmentObjectiveRef ownership
    +-- objective.py  : segment logits -> VERL original PPO loss
    +-- executor.py   : DFS Push / Visit / Pop, recompute, Prefix dKV relay
    +-- state.py      : detached KV path stack and sharded gradient anchors
    +-- ring.py       : Prefix-aware MindSpeed RingP2P, online softmax, CP runtime
    +-- placement.py  : optional DP forest planning (not trainer dispatch)
    |
models.mcore.tpr.attention / context / rectangular_attention
    | native Megatron QKV, GQA, TP, RoPE; CANN rectangular CP1 or Ring CP>1
    +-- new KV consumed by descendants; differentiable anchors during replay
```

The base model code remains under `verl/models/mcore/tpr/`. Tree scheduling,
PPO ownership and distributed execution are Engine concerns and must not be
imported by model-side primitives. The first-party `context.py` transfers
per-forward prefix tensors, RoPE and new per-layer KV between the Executor and
native GPTModel layers using a ContextVar. It is **not** the persistent KV cache.

### Data ownership and gradient flow

1. One original TQ mini-batch is grouped into UID-scoped **Turn-aware trees**;
   only complete ChatML message / captured prompt boundaries can be reused.
2. A physical Segment is computed once per branch; logical PPO tokens retain
   their `(sample_row, response_offset, target_token_id, query_offset)` refs.
   A response token at absolute position `t` is predicted at query `t-1`,
   including when the query physically belongs to its parent Segment.
3. DFS `Push` computes and stores graph-free KV, `Visit` computes a leaf
   objective and propagates dKV to ancestors, and `Pop` releases cached KV,
   recomputes a graph-connected Segment, and relays accumulated descendant dKV.
4. `objective.py` calls the **original VERL PPO loss callback**, reusing the
   native tensor-parallel vocab log-prob and retaining token-mean normalization,
   temperature, old logprob, advantage and response masks. It keeps the PPO
   logprob chunk-checkpoint optimization for long trajectories.
5. Every CP rank must participate in Ring backward, including ranks with no
   locally owned supervised tokens (use a connected zero objective). Forest
   gradients are accumulated under Megatron's no_sync and **finalized once**.

### Parallelism and support boundaries

| Dimension | Formal production scope | Implementation owner |
| --- | --- | --- |
| TP | TP=1/2 | Native Megatron QKV/proj/vocab heads; TPR plan agreement |
| CP | CP=1 or CP=2/4 in tested combinations | CP1 CANN, CP>1 Prefix Ring; MindSpeed P2P |
| SP | Native SP for TP2+CP1 only, with aligned Segment lengths | Megatron; TP2+CP>1+SP is gated off |
| DP | DP=1 | Native VERL DP; DP2 currently **offline parity only** |
| PP, EP | 1 | Other sizes fail explicitly |
| Attention | Dense SelfAttention / GQA | GDN/MoE remain experimental |
| PPO | Vanilla, token-mean actor loss | VERL callback, not a reimplemented loss |
| Recompute/offload | Prefix Push/Pop replay, no native layer recompute | Experimental activation offload isolated |

Ring CP uses the MindSpeed two-range zigzag placement. Logical Segment tokens
are distinct from physically padded tokens (multiple of `2*CP`): a physical
query/KV must not receive an unintended padded loss or attention score.
`ring.py` now owns Ring communication, the physical/logical SequenceShard,
zigzag layout, padding metadata and block validity masks. The historical
`models/mcore/tpr/shard.py` is a module alias, not another implementation.
Padding numerical correctness still needs the NPU CP2/CP4 parity gate.

### Optional experiments

`verl/experimental/tpr/` contains AllGather, Ulysses, Hybrid, GDN and activation
offload experiments. Their old paths are compatibility imports for historical
tests. **Production PPO must never import these experimental backends.** The
existing `models/mcore/tpr/parallel/backend.py` factory is legacy/testing only.
The `ring_attention.py` compatibility import is the **same module object** as
Engine `ring.py`: there is only one maintained Ring implementation.

### Reusing native components

TPR replaces the SelfAttention ModuleSpec only where needed to accept external
Prefix KV. It does not independently implement tensor-parallel all-reduces,
optimizer steps, policy clipping, process-group lifecycle or normal distributed
synchronization. Future support for TP4, TP+SP+CP, non-Qwen turn formats or DP2
needs separate explicit numeric and training acceptance, not feature flags that
silently disable checks.

### Validation gate

Before a production merge, compare **old TPR vs new TPR on identical actor
captures**, with TP1CP1, TP2CP4 and the explicitly gated offline TP2CP2DP2
case where available. Check Turn forests and refs, logits/logprobs, PPO loss,
parameter grad relative-L2, optimizer multi-step update, and peak NPU memory.
Do not mix this architectural migration with the existing Full-vs-TPR BF16
numerical drift investigation or enable Fixed Tile. A passed AST/import smoke
test does **not** mean end-to-end PPO parity passed.

CPU structural check:

```bash
python -m pytest -q tests/models/mcore/tpr/unit/test_production_architecture.py
```

NPU integration tests require a configured Ascend 910 / MindSpeed / Megatron
stack and authentic actor captures. Record every skipped gate explicitly.
