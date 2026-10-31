# Direct Session Tree consumption and cross-Session TPR construction

This directory validates the training-side consumer for the future TQ field:

```python
fields = TensorDict(
    {"trajectory_tree": NonTensorStack(*session_trees)},
    batch_size=[len(session_trees)],
)
```

## Required boundary

TQ provides **one independent Session Tree per NonTensorStack element**.
A Session Tree may contain:

- A single Chain.
- Multiple Chains / branches within that Session.
- A mixture of terminal Leaves and deeper branches.

TQ does not merge Nodes or share topology across different Sessions. Node IDs
are local to their Session Tree and may repeat in other elements.

TPR consumes all Session Trees in the logical Actor Minibatch and constructs
the cross-Session Forest. In particular, it merges exact shared prefixes
across Session boundaries, while preserving each Session's local branches and
each Leaf's objective ownership.

## Direct consumer

Direct planner entry point:

```python
from verl.workers.engine.megatron.tpr.tree import (
    build_tree_execution_plans_from_tree_fields,
)

forest = build_tree_execution_plans_from_tree_fields(
    fields,
    require_loss_mask_alignment=True,
    tree_policy="qwen_turn",
    tree_builder="lcp_sort",
)
```

The direct builder walks each Leaf's Node path as token chunks. It does not
materialize complete `prompts`, `responses`, or `input_ids` rows before
building the cross-Session topology. It preserves per-Leaf response/loss mask
ownership and maps selected response tokens to `SegmentObjectiveRef` objects.
`sample_row` is assigned in stable order: NonTensorStack Session order, then
each Session's Leaf sequence order. The caller must use the same order when
building Leaf-aligned PPO metadata.

The existing flat-row planner remains available as a test-only reference
oracle. Tests compare its plan digest with the direct Session Tree planner;
the flat-row path is not the consumer path being validated.

## Real TQ source and synthetic topologies

The available source capture is a flat TQ rollout batch:

```text
/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
```

It contains 8 source trajectory rows. The test uses their real prompt,
response, response mask, loss mask, rollout log-probabilities, RM scores and
source keys to synthesize three *input topologies*:

| Case | Session Trees | Chain-only Sessions | Mixed Chain+Tree Sessions | Tree-only Sessions | Leaves |
|---|---:|---:|---:|---:|---:|
| `all_chain` | 8 | 8 | 0 | 0 | 8 |
| `mixed` | 8 | 0 | 8 | 0 | 24 |
| `all_tree` | 8 | 0 | 0 | 8 | 16 |

A tree-only Session is synthesized from one source row without combining it
with another source Session: the original response is split into a shared-prefix
Node and one continuation; a second synthetic continuation drops the first
token after that prefix. Its token-aligned mask, loss-mask, logprob and RM-score
slices are cut at the same position. One shared-prefix supervision position is
also masked in the synthetic Leaf when possible, exercising Leaf-specific
ownership.

A mixed Session contains both a single-Chain Leaf and a separate two-branch
subtree. Both structures are derived from the same source row and remain under
that one Session root. This deliberately fabricates Session-local branching
because the current capture has no such branching. It does **not** claim
UniAgent emitted these synthetic Trees.

Important scope: Branch B's sliced logprobs remain position-aligned with the
synthetic tokens, but dropping a token changes the autoregressive context, so
those values are not valid sampled-policy logprobs for a real PPO update. The
fixture validates payload shape, path ownership, mask alignment and topology;
it is not a numerical PPO correctness or training-step test.

Every source row remains in its own Session Tree. All three cases contain 8
independent Session Trees; Node IDs restart locally in each Tree. No
cross-Session Node is created by the fixture. The 8, 24 or 16 Leaves are then
consumed together by TPR, which performs the cross-Session merge.

The smaller deterministic fixture in `synthetic.py` additionally covers
nested branches, Chain handoff, a shared ancestor with Leaf-specific masks,
and `removed_prefix_token_count` rollback.

## What is checked

- The `NonTensorStack` contains one independent Tree per Session.
- Repeated Node IDs across Sessions do not collide.
- Single-chain, mixed and all-tree inputs are all consumable.
- Internal Session branches remain valid while TPR merges prefixes across
  Sessions.
- Per-Leaf masks, Node log-probabilities and RM scores remain attached to the
  correct payload.
- Every supervised token has exactly one objective reference.
- Direct Session Tree Trie and LCP builders produce identical plans.
- Direct Session Tree construction and the flat-row reference produce
  identical Forest / SegmentPlan / ObjectiveRef digests.

This is an in-memory consumer test only. It does not call a TQ client and does
not test TQ Put/Get, KV storage, serialization, or network transport.

## Run

```bash
export TPR_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt

python -m pytest -q -s \
  tests/models/mcore/tpr/tree_transport/test_tree_transport_cpu.py
```

Without `TPR_TQ_BATCH`, the deterministic in-memory Session Tree tests run and
the real-TQ fixture test is skipped. With it set, the test runs all three
synthetic topologies and prints their Session counts, Leaf counts, global
Forest sizes, ObjectiveRef counts and plan digests.
