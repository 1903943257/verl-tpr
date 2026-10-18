# TPR production architecture (Dense PPO)

This document specifies the **intended production execution contract** of the
VERL-TPR implementation. The codebase targets Ascend NPU + Megatron; it is not
an alternate PPO implementation. See the [development baseline](../tpr_development_baseline.md)
for the reproducible imported starting point.

The [10/18 local contract fixes](contract_fixes_20261018.md) require new NPU
acceptance; prior accepted captures do not certify this working tree.

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

The model package `verl/models/mcore/tpr/` now physically contains only
`attention.py`, `context.py`, `rectangular_attention.py` and its lazy
`__init__.py`. The Engine owns all Segment plans, Prefix State, Shard/Padding,
ModuleSpec installation and TP group agreement. GDN state lives only in
`experimental/tpr/gdn/`, including its separate Context subclass. Native
non-TPR VERL model compatibility is unchanged. Tree scheduling,
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

Execution skips only branches outside the **global** objective-owner ancestor
closure, never based on CP-local token counts. Model preparation is batch-scoped;
the model structure remains fixed while live dropout settings are rechecked.
Selected BF16/FP16 queries are promoted once to FP32 before temperature and
native PPO processing, including FP32 accumulation across target/temperature
expansions. Native GEMM shapes remain unchanged.

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
zigzag layout, padding metadata and block validity masks. The old model-side shard implementation has been deleted; all imports now use Engine Ring.
Padding numerical correctness still needs the NPU CP2/CP4 parity gate.

### Optional experiments

`verl/experimental/tpr/` contains AllGather, Ulysses, Hybrid, GDN and activation
offload experiments. Historical CE/Exact reference implementations are held only under
`experimental/tpr/legacy/`, and their tests now import them explicitly.
**Production PPO must never import these experimental backends.** The
model-side `parallel/` directory and duplicate Ring/CP factories are deleted;
Engine `ring.py` is the sole maintained Ring kernel.

### Reusing native components

TPR replaces the SelfAttention ModuleSpec only where needed to accept external
Prefix KV. It does not independently implement tensor-parallel all-reduces,
optimizer steps, policy clipping, process-group lifecycle or normal distributed
synchronization. Future support for TP4, TP+SP+CP, non-Qwen turn formats or DP2
needs separate explicit numeric and training acceptance, not feature flags that
silently disable checks.

### Source reduction guarantees

The Engine owns all Dense training state and Segment plans. Model-side code
contains only Attention/Context/CANN. Historical CE/Exact code is kept as
an explicitly experimental reference, not a second production implementation.
The old model-side compatibility files are physically deleted.

The query/KV coalescing optimization (qkvmerge) is intentionally retained.
The Ring implementation still includes _can_coalesce_prefix_query,
_prefix_full_slices and its online-softmax path. It should become the default
only after numerical and throughput parity. Ring layout, validity mask and
logical/physical Padding share one canonical implementation, but padded CP
backward still requires an NPU parity gate.

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
