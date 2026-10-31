"""CPU gates: Session Tree -> NonTensorStack -> Leaf reconstruction -> TPR plan."""
from __future__ import annotations

import asyncio
import pickle

import pytest
import torch
from tensordict import NonTensorStack

from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans, digest_forest_plan

from .reconstruct import reconstruct_session_tree, rows_to_tpr_batch
from .schema import pack_session_trees, unpack_session_trees, validate_session_tree
from .synthetic import make_synthetic_batch
from .tq_roundtrip import put_session_trees


def _assert_tree_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and torch.equal(a, b)
        elif isinstance(a, dict):
            _assert_tree_equal(a, b)
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                if isinstance(x, torch.Tensor):
                    assert torch.equal(x, y)
                elif isinstance(x, dict):
                    _assert_tree_equal(x, y)
                else:
                    assert x == y
        else:
            assert a == b


def _assert_row_equal(actual, expected):
    for field in ("prompts", "responses", "input_ids", "response_mask", "loss_mask", "rollout_log_probs"):
        a, b = actual[field], expected[field]
        assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b)), field
    for field in ("key", "uid", "session_id", "leaf_id", "reward_score"):
        assert actual[field] == expected[field], field


def test_nontensorstack_tensor_dict_and_pickle_roundtrip_preserve_tree_payload():
    trees, _ = make_synthetic_batch()
    fields = pack_session_trees(trees)
    stacked = fields.get("trajectory_tree")
    assert isinstance(stacked, NonTensorStack)
    assert list(fields.batch_size) == [2]
    assert len(stacked) == 2
    restored = unpack_session_trees(pickle.loads(pickle.dumps(fields)))
    assert len(restored) == len(trees)
    for actual, expected in zip(restored, trees, strict=True):
        _assert_tree_equal(actual, expected)


def test_leaf_reconstruction_matches_independent_full_trajectory_oracle():
    trees, expected_rows = make_synthetic_batch()
    received = unpack_session_trees(pickle.loads(pickle.dumps(pack_session_trees(trees))))
    actual_rows = [row for tree in received for row in reconstruct_session_tree(tree)]
    assert len(actual_rows) == len(expected_rows) == 6
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        _assert_row_equal(actual, expected)


def test_reconstructed_trees_build_same_cross_session_lcp_plan_as_full_rows():
    trees, expected_rows = make_synthetic_batch()
    received = unpack_session_trees(pickle.loads(pickle.dumps(pack_session_trees(trees))))
    actual_rows = [row for tree in received for row in reconstruct_session_tree(tree)]
    actual_keys, actual_batch = rows_to_tpr_batch(actual_rows)
    expected_keys, expected_batch = rows_to_tpr_batch(expected_rows)
    options = {"require_loss_mask_alignment": True, "turn_min_segment_tokens": 1, "tree_builder": "lcp_sort"}
    actual = build_tree_execution_plans(actual_keys, actual_batch, **options)
    expected = build_tree_execution_plans(expected_keys, expected_batch, **options)
    assert digest_forest_plan(actual) == digest_forest_plan(expected)
    assert actual.logical_loss_tokens == sum(int(mask.sum()) for mask in actual_batch["response_mask"])
    assert len(actual.trees) == 1
    assert actual.trees[0].member_rows == tuple(range(6))


def test_shared_ancestor_generation_has_leaf_specific_response_ownership():
    trees, _ = make_synthetic_batch()
    rows = [row for tree in trees for row in reconstruct_session_tree(tree)]
    root_leaf, nested_chain_leaf, sibling_chain_leaf = [row for row in rows if row["session_id"] == "s0"]
    root_response = root_leaf["responses"]
    assert torch.equal(root_leaf["responses"], root_response)
    assert torch.equal(nested_chain_leaf["prompts"][-(root_response.numel() + 4):-4], root_response)
    assert torch.equal(sibling_chain_leaf["prompts"][-(root_response.numel() + 4):-4], root_response)
    assert bool(root_leaf["response_mask"].all())
    assert not bool(nested_chain_leaf["response_mask"][0])
    assert nested_chain_leaf["responses"].numel() > sibling_chain_leaf["responses"].numel()
    keys, batch = rows_to_tpr_batch(rows)
    plan = build_tree_execution_plans(keys, batch, require_loss_mask_alignment=True,
                                       turn_min_segment_tokens=1, tree_builder="lcp_sort")
    assert plan.logical_loss_tokens == sum(int(mask.sum()) for mask in batch["response_mask"])
    assert len(plan.trees[0].objective_refs) == plan.logical_loss_tokens


def test_async_tq_put_uses_the_proposed_key_partition_and_fields_contract():
    trees, _ = make_synthetic_batch()

    class FakeTQ:
        def __init__(self):
            self.call = None

        async def async_kv_batch_put(self, *, keys, partition_id, fields):
            self.call = (keys, partition_id, fields)

    tq = FakeTQ()
    fields = asyncio.run(put_session_trees(tq, trees))
    keys, partition_id, received_fields = tq.call
    assert keys == [tree["tree_id"] for tree in trees]
    assert partition_id == "trajectory_trees"
    assert received_fields is fields
    assert list(fields.batch_size) == [len(trees)]
    assert len(unpack_session_trees(fields)) == len(trees)


def test_schema_rejects_missing_parent_and_invalid_root_topology():
    trees, _ = make_synthetic_batch()
    broken = {**trees[0], "nodes": [dict(node) for node in trees[0]["nodes"]]}
    broken["nodes"][1]["parent_id"] = 99
    with pytest.raises(ValueError, match="missing parent"):
        validate_session_tree(broken)
    cycle = {**trees[0], "nodes": [dict(node) for node in trees[0]["nodes"]]}
    cycle["nodes"][0]["parent_id"] = 1
    with pytest.raises(ValueError, match="exactly root"):
        validate_session_tree(cycle)


def test_removed_prefix_count_trims_ancestor_response_and_aligned_metadata():
    trees, expected_rows = make_synthetic_batch()
    tree = trees[0]
    # Node 3 rolls back the final token produced by its parent Chain node.
    tree["nodes"][3]["removed_prefix_token_count"] = 1
    leaf = tree["leaves"][1]
    original_mask = leaf["response_mask"]
    leaf["response_mask"] = torch.cat((original_mask[:4], original_mask[5:]))
    leaf["loss_mask"] = leaf["response_mask"].clone()

    actual = reconstruct_session_tree(tree)[1]
    original = expected_rows[1]
    response_a = tree["nodes"][1]["response_ids"]
    tool_a2 = tree["nodes"][3]["prompt_ids"]
    response_a2 = tree["nodes"][3]["response_ids"]
    expected_response = torch.cat((response_a[:-1], tool_a2, response_a2))
    expected_mask = leaf["response_mask"]
    expected_logprobs = torch.tensor(
        tree["nodes"][1]["response_logprobs"][:-1]
        + [0.0] * tool_a2.numel()
        + tree["nodes"][3]["response_logprobs"]
    )

    assert torch.equal(actual["responses"], expected_response)
    assert torch.equal(actual["response_mask"], expected_mask)
    assert torch.equal(actual["loss_mask"], expected_mask)
    assert torch.equal(actual["rollout_log_probs"], expected_logprobs)
    assert torch.equal(actual["prompts"], original["prompts"])
