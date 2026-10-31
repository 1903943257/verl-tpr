# Session Tree / NonTensorStack consumer prototype

This directory validates the training-side consumer for the proposed future TQ
field:

```python
fields = TensorDict(
    {"trajectory_tree": NonTensorStack(*trees)},
    batch_size=[len(trees)],
)
```

The scope is deliberately narrow:

- Use an existing flat TQ rollout capture as the source of real token and
  response-level tensors.
- Synthesize a Session Tree dictionary in memory from those source rows.
- Pack one or more dictionaries into `NonTensorStack`.
- Read the dictionaries back from that in-memory field.
- Reconstruct each Leaf and compare it with the original TQ trajectory row.
- Feed reconstructed rows and original rows into the existing LCP TPR planner
  and require identical plan digests.

This does **not** call a TQ client and does not test TQ Put/Get, KV storage,
serialization, networking, or TQ implementation internals. The current
`tq_batch.pt` is a flat rollout TensorDict; the test synthesizes the future
`trajectory_tree` field that the training-side consumer is expected to read.

## Real TQ fixture

Default capture used by the current experiment:

```text
/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
```

The inspected capture contains 8 rows. Its payload has top-level
`tensordict`, `tags`, `keys`, and `global_steps`. The row keys are
`<uid>_<session_index>_<trajectory_index>`; all 8 rows share one UID, their
source session indexes are 0 through 7, and all 8 prompts are exactly equal
(17,483 tokens).

For this consumer test only, those 8 real rows are grouped under one synthetic
Session Tree:

- Root Node: the exact shared 17,483-token prompt.
- 8 child Nodes: one original response per branch.
- 8 Leaves: each retains its original response mask, loss mask, rollout
  log-probabilities, reward-model scores, and source key/session index.

The original source session indexes remain attached to their Leaves as
provenance. This is a synthetic one-session topology assembled from existing
same-UID TQ rows; it does not claim the original capture was emitted as one
real UniAgent Session Tree.

## Data preserved

The adapter consumes these actual TQ TensorDict fields:

- `prompts`, `responses`, `input_ids`
- `response_mask`, `loss_mask`
- `rollout_log_probs`, `rm_scores`
- top-level `keys` for stable trajectory identity

The response-level tensors are copied into the synthetic Tree payload without
recomputing their values. The reconstruction test checks tensor values and
dtypes, including the original int64 masks and float32 scores.

## Small synthetic unit fixtures

`synthetic.py` retains small deterministic examples for nested branches,
Chain handoff, Leaf-specific masks, and `removed_prefix_token_count` rollback.
They are unit fixtures for path reconstruction edge cases; the primary
integration fixture is the existing real TQ capture described above.

## Run with the existing TQ capture

```bash
export TPR_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt

python -m pytest -q -s \
  tests/models/mcore/tpr/tree_transport/test_tree_transport_cpu.py
```

Without `TPR_TQ_BATCH`, the small CPU unit tests run and the real-capture test
is skipped. With it set, the test reports source rows, shared prompt length,
logical tokens, node/leaf counts, LCP plan digest, and ObjectiveRef count.
