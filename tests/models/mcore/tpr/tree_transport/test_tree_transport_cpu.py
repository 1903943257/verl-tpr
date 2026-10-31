"""CPU gates for direct consumption of independent Session Trees."""
from __future__ import annotations

import os

import pytest
import torch
from tensordict import NonTensorStack

from verl.workers.engine.megatron.tpr.tree import (
    build_tree_execution_plans,
    build_tree_execution_plans_from_session_trees,
    digest_forest_plan,
)

from .nontensorstack import consume_tree_fields, make_tree_fields
from .reconstruct import reconstruct_session_tree, rows_to_tpr_batch
from .schema import pack_session_trees, unpack_session_trees, validate_session_tree
from .synthetic import make_synthetic_batch
from .synthetic_from_tq import (
    build_synthetic_session_trees_from_tq,
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


def _flat_reference_plan(trees):
    """Test-only oracle: flatten Leaf paths and use the existing row builder."""
    rows = [
        row
        for tree in trees
        for row in reconstruct_session_tree(tree)
    ]
    keys, batch = rows_to_tpr_batch(rows)
    plan = build_tree_execution_plans(
        keys,
        batch,
        require_loss_mask_alignment=True,
        turn_min_segment_tokens=1,
        tree_builder="lcp_sort",
    )
    return rows, batch, plan


def _assert_direct_plan_matches_flat_reference(trees):
    direct = build_tree_execution_plans_from_session_trees(
        trees,
        require_loss_mask_alignment=True,
        turn_min_segment_tokens=1,
        tree_builder="lcp_sort",
    )
    rows, batch, reference = _flat_reference_plan(trees)
    assert digest_forest_plan(direct) == digest_forest_plan(reference)
    assert direct.logical_loss_tokens == reference.logical_loss_tokens
    assert direct.logical_loss_tokens == sum(
        int(mask.bool().sum()) for mask in batch["response_mask"]
    )
    assert sum(len(tree.objective_refs) for tree in direct.trees) == sum(
        int(mask.bool().sum()) for mask in batch["response_mask"]
    )
    return direct, rows, batch


def test_nontensorstack_tensor_dict_in_memory_pack_and_consume():
    trees, _ = make_synthetic_batch()
    fields = pack_session_trees(trees)
    stacked = fields.get("trajectory_tree")
    assert isinstance(stacked, NonTensorStack)
    assert list(fields.batch_size) == [2]
    assert len(stacked) == 2

    restored = unpack_session_trees(fields)
    assert len(restored) == len(trees)
    for actual, expected in zip(restored, trees, strict=True):
        _assert_tree_equal(actual, expected)


def test_direct_session_tree_planner_matches_flat_oracle_for_nested_branches():
    trees, _ = make_synthetic_batch()
    received = consume_tree_fields(make_tree_fields(trees))
    direct, rows, batch = _assert_direct_plan_matches_flat_reference(received)

    assert len(received) == 2
    assert len({tree["session_id"] for tree in received}) == 2
    assert all(tree["root_node_id"] == 0 for tree in received)
    # Node IDs intentionally overlap between Sessions; they are locally scoped.
    assert set(node["node_id"] for node in received[0]["nodes"]) & set(
        node["node_id"] for node in received[1]["nodes"]
    )
    assert len(rows) == 6
    assert direct.logical_loss_tokens == sum(
        int(mask.bool().sum()) for mask in batch["response_mask"]
    )
    # The fixture contains a shared ancestor whose generated token is masked
    # for one Leaf but remains present and trainable for another Leaf.
    nested_rows = [row for row in rows if row["session_id"] == "s0"]
    assert any(not bool(row["response_mask"][0]) for row in nested_rows)


def test_direct_session_tree_planner_handles_removed_prefix_rollback():
    trees, _ = make_synthetic_batch()
    tree = trees[0]
    tree["nodes"][3]["removed_prefix_token_count"] = 1
    leaf = tree["leaves"][1]
    original_mask = leaf["response_mask"]
    leaf["response_mask"] = torch.cat((original_mask[:4], original_mask[5:]))
    leaf["loss_mask"] = leaf["response_mask"].clone()

    _assert_direct_plan_matches_flat_reference(trees)


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


def test_real_tq_batch_three_session_topologies_and_direct_cross_session_merge():
    tq_batch_path = os.getenv("TPR_TQ_BATCH")
    if not tq_batch_path:
        pytest.skip("Set TPR_TQ_BATCH to the existing tq_batch.pt capture")

    payload = load_tq_rollout_payload(tq_batch_path)
    expected_cases = {
        "all_chain": (8, 8, 0),
        "mixed": (6, 4, 2),
        "all_tree": (4, 0, 4),
    }
    for topology, (expected_sessions, expected_chains, expected_forks) in expected_cases.items():
        trees, summary = build_synthetic_session_trees_from_tq(
            payload, topology=topology
        )
        fields = make_tree_fields(trees)
        stacked = fields.get("trajectory_tree")
        assert isinstance(stacked, NonTensorStack)
        assert list(fields.batch_size) == [expected_sessions]
        assert len(stacked) == expected_sessions

        # TQ input is strictly Session-local: one independent Tree per
        # Session, unique Session identity, and no cross-Session Node IDs.
        received = consume_tree_fields(fields)
        assert len({tree["tree_id"] for tree in received}) == expected_sessions
        assert len({
            (tree["uid"], tree["session_id"]) for tree in received
        }) == expected_sessions
        assert all(tree["root_node_id"] == 0 for tree in received)
        assert len({id(tree["nodes"]) for tree in received}) == expected_sessions
        assert summary["chain_session_count"] == expected_chains
        assert summary["fork_session_count"] == expected_forks
        assert summary["leaf_count"] == 8

        direct, rows, batch = _assert_direct_plan_matches_flat_reference(received)
        assert len(rows) == 8
        assert direct.logical_loss_tokens == summary["logical_loss_tokens"]
        assert all(
            len(tree.member_rows) == 8 for tree in direct.trees
        ) if len(summary["source_uids"]) == 1 and summary["exact_common_prompt"] else True

        print(
            "SESSION_TREE_DIRECT_CONSUMER PASS",
            f"topology={topology}",
            f"source_rows={summary['source_rows']}",
            f"session_trees={summary['session_tree_count']}",
            f"chain_sessions={summary['chain_session_count']}",
            f"fork_sessions={summary['fork_session_count']}",
            f"leaves={summary['leaf_count']}",
            f"nodes={summary['node_count']}",
            f"exact_common_prompt={summary['exact_common_prompt']}",
            f"logical_input_tokens={summary['logical_input_tokens']}",
            f"logical_loss_tokens={summary['logical_loss_tokens']}",
            f"global_trees={len(direct.trees)}",
            f"global_segments={direct.segment_count}",
            f"objective_refs={direct.logical_loss_tokens}",
            f"plan_digest={digest_forest_plan(direct)}",
            sep="\\n",
            flush=True,
        )
