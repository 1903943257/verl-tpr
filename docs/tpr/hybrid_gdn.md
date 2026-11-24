# Hybrid TPR: GDN state continuation

This is an opt-in extension to the Dense TPR PPO runtime. It reuses the
production Token-LCP tree, SegmentPlan, DFS executor, native PPO objective,
and Full-Attention KV/Ring path. GDN adds a separate per-Segment continuation
state consisting of causal-convolution and recurrent state.

## Enablement

Set the opt-in before model construction and training:

```bash
export TPR_ENABLE_GDN=1
```

Without this flag, a GatedDeltaNet ModuleSpec is not rewritten to
`TPRGatedDeltaNet`, and the production TPR PPO adapter rejects GDN layers.
Dense-only TPR behavior is unchanged.

## State lifecycle

For each GDN layer, `GDNLayerState` contains:

- `conv_state`: causal-convolution history at the Segment boundary.
- `recurrent_state`: Gated Delta Rule recurrent matrix at the Segment boundary.

The Engine keeps a `GDNPrefixState` per active Segment, alongside the existing
FA `KVStack`:

1. **Push** runs the Segment without an autograd graph and saves graph-free
   Conv/Recurrent final states.
2. **Visit** creates branch-local autograd anchors from the direct parent's
   saved state. Anchors share the immutable saved storage but own independent
   gradients.
3. **Backward** accumulates `dConvState` and `dRecurrentState` into the
   direct parent. Sibling contributions are summed.
4. **Pop** recomputes the Segment, injects the accumulated gradients into its
   final Conv/Recurrent states, and relays the resulting initial-state
   gradients to its parent.

Full Attention continues to use the existing KV anchors and Ring backend.
GDN state is rank-local; the CP2 GDN adapter uses MindSpeed's A2A layout
conversion while Full Attention continues to use Ring.

## Current support boundary

This restoration deliberately keeps the validated surface small:

- Hybrid models must contain at least one Full Attention layer and one or more
  GDN layers.
- CP=1 is the first integration target.
- CP=2 is an opt-in path using GDN A2A plus FA Ring. Every Segment length must
  be divisible by four.
- TP=1 and SP=1 are required for GDN.
- PP, EP, packed/THD execution, grouped leaf batching, and activation
  recomputation are not supported by this GDN integration.
- The normal Dense-only TPR path remains the default and does not import GDN
  modules unless a GDN model is explicitly selected.

## Validation status and known operator dependency

The CPU unit tests cover PrefixState storage, independent anchors, sibling
gradient accumulation, and the production DFS executor's Hybrid state relay.
They use a small synthetic GDN module; they do not certify MindSpeed kernels.

Real Ascend validation must separately cover Stateful Causal Conv1D and
Gated Delta Rule forward/backward, CP2 A2A equivalence, and full Qwen3.5
Hybrid PPO replay.

In particular, the current Ascend 910B2C investigation identified a remaining
Stateful `causal_conv1d` backward issue: `dh0` buffer allocation covers only
the initial-state-relevant tiles, while the kernel store can address all time
tiles, producing UB address out-of-bounds (507035). The independent
`dh0` time-indexing correctness fix does not resolve this store-bound issue.
Do not treat GDN TPR as NPU-accepted until the operator issue is fixed and the
full numerical gates pass.
