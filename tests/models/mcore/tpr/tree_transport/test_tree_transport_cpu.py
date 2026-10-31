"""CPU gates for direct consumption of independent Session Trees."""
from __future__ import annotations

import os

import pytest
import torch
from tensordict import NonTensorStack

from verl.workers.engine.megatron.tpr.tree import (
    build_tree_execution_plans,
    build_tree_execution_plans_from_tree_fields,
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


def _assert_direct_plan_matches_flat_reference(trees, fields=None):
    fields = make_tree_fields(trees) if fields is None else fields
    direct = build_tree_execution_plans_from_tree_fields(
        fields,
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


def test_nontensorstack_rejects_two_trees_for_one_session():
    trees, _ = make_synthetic_batch()
    duplicate = {**trees[0], "tree_id": "duplicate-tree-id"}
    with pytest.raises(ValueError, match="one Tree per Session"):
        pack_session_trees([trees[0], duplicate])


def test_direct_session_tree_planner_matches_flat_oracle_for_nested_branches():
    trees, _ = make_synthetic_batch()
    fields = make_tree_fields(trees)
    received = consume_tree_fields(fields)
    direct, rows, batch = _assert_direct_plan_matches_flat_reference(
        received, fields=fields
    )

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
    # The fixture also covers masked response positions on a nested Leaf path.
    nested_rows = [row for row in rows if row["session_id"] == "s0"]
    assert any(not bool(row["response_mask"][0]) for row in nested_rows)


def _make_shared_prefix_mask_tree():
    prompt = torch.tensor([101, 151645, 102, 151645], dtype=torch.long)
    shared_response = torch.tensor([201, 151645], dtype=torch.long)
    branch_a = torch.tensor([301, 151645], dtype=torch.long)
    branch_b = torch.tensor([401, 151645], dtype=torch.long)
    root_logprobs = torch.tensor([-0.1, -0.2])
    leaves = []
    for index, branch in enumerate((branch_a, branch_b)):
        mask = torch.ones(
            shared_response.numel() + branch.numel(), dtype=torch.bool
        )
        if index == 1:
            mask[0] = False
        leaves.append({
            "leaf_id": f"mask-session-leaf-{index}",
            "leaf_index": index,
            "kind": "node_path",
            "leaf_node_id": index + 1,
            "chain_id": 0,
            "source_key": f"uid_mask_mask-session_{index}",
            "response_mask": mask,
            "loss_mask": mask.clone(),
            "rm_scores": torch.zeros(mask.numel()),
        })
    return {
        "schema_version": 1,
        "tree_id": "mask-session-tree",
        "uid": "uid_mask",
        "session_id": "mask-session",
        "root_node_id": 0,
        "nodes": [
            {
                "node_id": 0,
                "parent_id": None,
                "chain_id": 0,
                "prompt_ids": prompt,
                "response_ids": shared_response,
                "response_logprobs": root_logprobs,
                "removed_prefix_token_count": 0,
            },
            {
                "node_id": 1,
                "parent_id": 0,
                "chain_id": 0,
                "prompt_ids": torch.empty(0, dtype=torch.long),
                "response_ids": branch_a,
                "response_logprobs": torch.tensor([-0.3, -0.4]),
                "removed_prefix_token_count": 0,
            },
            {
                "node_id": 2,
                "parent_id": 0,
                "chain_id": 0,
                "prompt_ids": torch.empty(0, dtype=torch.long),
                "response_ids": branch_b,
                "response_logprobs": torch.tensor([-0.5, -0.6]),
                "removed_prefix_token_count": 0,
            },
        ],
        "leaves": leaves,
    }


def test_shared_prefix_tokens_keep_leaf_specific_objective_ownership():
    tree = _make_shared_prefix_mask_tree()
    fields = make_tree_fields([tree])
    received = consume_tree_fields(fields)
    direct, rows, _ = _assert_direct_plan_matches_flat_reference(
        received, fields=fields
    )

    assert len(rows) == 2
    assert direct.logical_loss_tokens == 7
    refs = direct.trees[0].objective_refs
    # Response offset 0 is in the shared ancestor. Leaf 0 supervises it;
    # Leaf 1 masks it. The shared Node is reused without sharing its mask.
    assert any(ref.sample_row == 0 and ref.response_offset == 0 for ref in refs)
    assert not any(ref.sample_row == 1 and ref.response_offset == 0 for ref in refs)
    assert any(ref.sample_row == 1 and ref.response_offset == 1 for ref in refs)


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
        "all_chain": (8, 8, 0, 0, 8),
        "mixed": (8, 0, 8, 0, 24),
        "all_tree": (8, 0, 0, 8, 16),
    }
    for topology, (
        expected_sessions, expected_chains, expected_mixed,
        expected_forks, expected_leaves,
    ) in expected_cases.items():
        trees, summary = build_synthetic_session_trees_from_tq(
            payload, topology=topology
        )
        fields = make_tree_fields(trees)
        stacked = fields.get("trajectory_tree")
        assert isinstance(stacked, NonTensorStack)
        assert list(fields.batch_size) == [expected_sessions]
        assert len(stacked) == expected_sessions

        # TQ input is strictly Session-local: one independent Tree per
        # Session. Node IDs are local and may repeat in different Trees.
        received = consume_tree_fields(fields)
        for actual, expected in zip(received, trees, strict=True):
            _assert_tree_equal(actual, expected)
        assert len({tree["tree_id"] for tree in received}) == expected_sessions
        assert len({
            (tree["uid"], tree["session_id"]) for tree in received
        }) == expected_sessions
        assert all(tree["root_node_id"] == 0 for tree in received)
        assert len({id(tree["nodes"]) for tree in received}) == expected_sessions
        node_object_ids = [
            id(node) for tree in received for node in tree["nodes"]
        ]
        assert len(set(node_object_ids)) == len(node_object_ids)
        assert summary["chain_session_count"] == expected_chains
        assert summary["mixed_session_count"] == expected_mixed
        assert summary["fork_session_count"] == expected_forks
        assert summary["leaf_count"] == expected_leaves

        direct, rows, _ = _assert_direct_plan_matches_flat_reference(
            received, fields=fields
        )
        assert len(rows) == expected_leaves
        for row in rows:
            response_length = row["responses"].numel()
            assert row["response_mask"].numel() == response_length
            assert row["loss_mask"].numel() == response_length
            assert row["rollout_log_probs"] is not None
            assert row["rollout_log_probs"].numel() == response_length
            assert row["rm_scores"] is not None
            assert row["rm_scores"].numel() == response_length
        assert direct.logical_loss_tokens == summary["logical_loss_tokens"]
        assert sum(int(row["input_ids"].numel()) for row in rows) == (
            summary["logical_input_tokens"]
        )
        all_member_rows = sorted(
            row for tree in direct.trees for row in tree.member_rows
        )
        assert all_member_rows == list(range(expected_leaves))
        if len(summary["source_uids"]) == 1 and summary["exact_common_prompt"]:
            assert len(direct.trees) == 1
            assert direct.trees[0].member_rows == tuple(range(expected_leaves))

        print(
            "SESSION_TREE_DIRECT_CONSUMER PASS",
            f"topology={topology}",
            f"source_rows={summary['source_rows']}",
            f"session_trees={summary['session_tree_count']}",
            f"chain_sessions={summary['chain_session_count']}",
            f"mixed_sessions={summary['mixed_session_count']}",
            f"tree_only_sessions={summary['fork_session_count']}",
            f"leaves={summary['leaf_count']}",
            f"leaf_mask_overrides={summary['leaf_mask_override_count']}",
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
