"""CPU gates for the in-memory Session Tree / NonTensorStack consumer."""
from __future__ import annotations

import os

import pytest
import torch
from tensordict import NonTensorStack

from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans, digest_forest_plan

from .nontensorstack import consume_tree_fields, make_tree_fields
from .reconstruct import reconstruct_session_tree, rows_to_tpr_batch
from .schema import pack_session_trees, unpack_session_trees, validate_session_tree
from .synthetic import make_synthetic_batch
from .synthetic_from_tq import (
    build_synthetic_session_tree_from_tq,
    load_tq_rollout_payload,
)


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
    tensor_fields = (
        "prompts", "responses", "input_ids", "response_mask", "loss_mask",
        "rollout_log_probs", "rm_scores",
    )
    for field in tensor_fields:
        if field not in expected:
            continue
        a, b = actual.get(field), expected[field]
        assert (a is None and b is None) or (
            a is not None and b is not None and torch.equal(a, b)
        ), field
        if a is not None and b is not None:
            assert a.dtype == b.dtype, field
    for field in ("key", "uid", "session_id", "source_session_id", "leaf_id"):
        if field in expected:
            assert actual.get(field) == expected[field], field


def test_nontensorstack_tensor_dict_in_memory_pack_and_consume():
    trees, _ = make_synthetic_batch()
    fields = pack_session_trees(trees)
    stacked = fields.get("trajectory_tree")
    assert isinstance(stacked, NonTensorStack)
    assert list(fields.batch_size) == [2]
    assert len(stacked) == 2

    # This is intentionally an in-memory consumer test: no pickle, TQ client,
    # KV operation, persistence, or network transport.
    restored = unpack_session_trees(fields)
    assert len(restored) == len(trees)
    for actual, expected in zip(restored, trees, strict=True):
        _assert_tree_equal(actual, expected)


def test_leaf_reconstruction_matches_independent_full_trajectory_oracle():
    trees, expected_rows = make_synthetic_batch()
    received = consume_tree_fields(make_tree_fields(trees))
    actual_rows = [row for tree in received for row in reconstruct_session_tree(tree)]
    assert len(actual_rows) == len(expected_rows) == 6
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        _assert_row_equal(actual, expected)


def test_reconstructed_trees_build_same_lcp_plan_as_full_rows():
    trees, expected_rows = make_synthetic_batch()
    received = consume_tree_fields(make_tree_fields(trees))
    actual_rows = [row for tree in received for row in reconstruct_session_tree(tree)]
    actual_keys, actual_batch = rows_to_tpr_batch(actual_rows)
    expected_keys, expected_batch = rows_to_tpr_batch(expected_rows)
    options = {
        "require_loss_mask_alignment": True,
        "turn_min_segment_tokens": 1,
        "tree_builder": "lcp_sort",
    }
    actual = build_tree_execution_plans(actual_keys, actual_batch, **options)
    expected = build_tree_execution_plans(expected_keys, expected_batch, **options)
    assert digest_forest_plan(actual) == digest_forest_plan(expected)
    assert actual.logical_loss_tokens == sum(
        int(mask.bool().sum()) for mask in actual_batch["response_mask"]
    )
    assert len(actual.trees) == 1
    assert actual.trees[0].member_rows == tuple(range(6))


def test_shared_ancestor_generation_has_leaf_specific_response_ownership():
    trees, _ = make_synthetic_batch()
    rows = [row for tree in trees for row in reconstruct_session_tree(tree)]
    root_leaf, nested_chain_leaf, sibling_chain_leaf = [
        row for row in rows if row["session_id"] == "s0"
    ]
    root_response = root_leaf["responses"]
    assert torch.equal(root_leaf["responses"], root_response)
    assert torch.equal(
        nested_chain_leaf["prompts"][-(root_response.numel() + 4):-4],
        root_response,
    )
    assert torch.equal(
        sibling_chain_leaf["prompts"][-(root_response.numel() + 4):-4],
        root_response,
    )
    assert bool(root_leaf["response_mask"].all())
    assert not bool(nested_chain_leaf["response_mask"][0])
    assert nested_chain_leaf["responses"].numel() > sibling_chain_leaf["responses"].numel()
    keys, batch = rows_to_tpr_batch(rows)
    plan = build_tree_execution_plans(
        keys, batch, require_loss_mask_alignment=True,
        turn_min_segment_tokens=1, tree_builder="lcp_sort",
    )
    assert plan.logical_loss_tokens == sum(
        int(mask.bool().sum()) for mask in batch["response_mask"]
    )
    assert len(plan.trees[0].objective_refs) == plan.logical_loss_tokens


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


def test_existing_tq_batch_is_consumed_as_one_synthetic_session_tree():
    tq_batch_path = os.getenv("TPR_TQ_BATCH")
    if not tq_batch_path:
        pytest.skip("Set TPR_TQ_BATCH to the existing tq_batch.pt capture")

    payload = load_tq_rollout_payload(tq_batch_path)
    tree, expected_rows, summary = build_synthetic_session_tree_from_tq(payload)

    # The source capture is flat rows. We synthesize one Session Tree from
    # those rows, then exercise exactly the TensorDict + NonTensorStack shape
    # the future TQ payload is expected to expose.
    fields = make_tree_fields([tree])
    stacked = fields.get("trajectory_tree")
    assert isinstance(stacked, NonTensorStack)
    assert list(fields.batch_size) == [1]
    assert len(stacked) == 1

    received = consume_tree_fields(fields)
    assert len(received) == 1
    assert len(received[0]["nodes"]) == summary["node_count"]
    assert len(received[0]["leaves"]) == summary["leaf_count"]

    actual_rows = reconstruct_session_tree(received[0])
    assert len(actual_rows) == len(expected_rows) == summary["source_rows"]
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        _assert_row_equal(actual, expected)

    actual_keys, actual_batch = rows_to_tpr_batch(actual_rows)
    expected_keys, expected_batch = rows_to_tpr_batch(expected_rows)
    options = {
        "require_loss_mask_alignment": True,
        "turn_min_segment_tokens": 1,
        "tree_builder": "lcp_sort",
    }
    actual_plan = build_tree_execution_plans(actual_keys, actual_batch, **options)
    expected_plan = build_tree_execution_plans(expected_keys, expected_batch, **options)
    assert digest_forest_plan(actual_plan) == digest_forest_plan(expected_plan)
    assert len(actual_plan.trees) == 1
    assert actual_plan.trees[0].member_rows == tuple(range(summary["source_rows"]))
    assert actual_plan.logical_loss_tokens == summary["logical_loss_tokens"]

    print(
        "REAL_TQ_TO_SYNTHETIC_SESSION_TREE PASS "
        f"source_rows={summary['source_rows']} "
        f"source_uid={summary['source_uid']} "
        f"source_session_ids={summary['source_session_ids']} "
        f"prompt_tokens={summary['prompt_tokens']} "
        f"logical_input_tokens={summary['logical_input_tokens']} "
        f"logical_loss_tokens={summary['logical_loss_tokens']} "
        f"nodes={summary['node_count']} leaves={summary['leaf_count']} "
        f"trees={len(actual_plan.trees)} segments={actual_plan.segment_count} "
        f"plan_digest={digest_forest_plan(actual_plan)}",
        flush=True,
    )
