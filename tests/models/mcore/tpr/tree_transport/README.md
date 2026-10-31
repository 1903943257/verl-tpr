# Session Tree transport correctness prototype

This directory is a test-side schema and correctness prototype for the proposed
UniAgent Gateway -> TQ -> TPR Session Tree path. It does not change the
production TPR Executor or the production TQ schema.

## Contract under test

- One dictionary represents one Session Tree.
- Node prompt_ids are incremental prompt/context tokens added after the parent
  path; response_ids are tokens generated at that node.
- parent_id preserves the Session-local topology.
- chain_id and removed_prefix_token_count reproduce the current UniAgent
  TreeGatewaySession._linearize_path chain ownership and rollback semantics.
- A Leaf is either a node-path Leaf (kind=node_path) or an already
  materialized Trajectory Leaf (kind=trajectory), matching the Gateway's
  MessageNode | Trajectory leaves union.
- A Leaf stores its Chain identity and its own response/loss masks. The Leaf
  response_mask is authoritative: one generated node token may be masked for
  one Leaf while remaining present in the shared Node payload.
- Node response_logprobs align with that Node response_ids. If a path contains
  missing probabilities, reconstructed rollout logprobs are absent, matching
  the current UniAgent linearizer's all-or-none behavior.

The incremental prompt_ids meaning is an explicit prototype assumption.
Before adopting this schema, UniAgent must confirm the exported Node fields use
exactly these semantics, especially around Chain handoff and truncation.

## TQ payload

make_tq_fields(trees) builds the colleague's proposed TensorDict payload with
trajectory_tree=NonTensorStack(*trees) and batch_size=[len(trees)]. The helper
put_session_trees calls async_kv_batch_put(keys=tree_ids,
partition_id='trajectory_trees', fields=fields).

Tests verify NonTensorStack packing, TensorDict shape, Python pickle
serialization, and the async_kv_batch_put call contract using a fake TQ client.
They do not claim a live TQ server Put/Get roundtrip: the concrete TQ read API
and server endpoint are not configured in this repository test.

## Correctness fixture

The synthetic fixture contains two independent Session Trees, each with one
root Leaf and two sibling branches. Both Sessions share the same exact System
Prefix but have different User messages. One branch remains in the original
Chain; the other starts a new Chain, so the ancestor generation moves from
Response to Prompt for that Leaf.

The tests compare reconstructed Leaf rows against an independently authored
Full-Trajectory oracle, including input_ids, prompt/response split, response
mask, loss mask, and rollout logprobs. They then feed both row sets into the
existing LCP TPR planner and require identical plan digests and ObjectiveRef
counts. This validates data/topology planning; it is not yet a numerical
Full-vs-TPR model forward/backward or optimizer-update acceptance test.

## Run

python -m pytest -q tests/models/mcore/tpr/tree_transport/test_tree_transport_cpu.py

After the CPU contract passes, run the reconstructed batch through an existing
NPU Full-vs-TPR numerical witness. Add a live TQ KV Put/Get test only after the
TQ read API is confirmed.
