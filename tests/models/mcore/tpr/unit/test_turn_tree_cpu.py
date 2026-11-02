"""The single production topology preserves historical Turn/PPO ownership."""

import random

import pytest
import torch

from verl.experimental.tpr.legacy.plan import SegmentLossTerm
from verl.experimental.tpr.legacy.tree_plan_builder import build_tree_execution_plans as reference_plan
from verl.workers.engine.megatron.tpr.tree import (
    SegmentPlan,
    SegmentSpec,
    build_tree_execution_plans,
    digest_forest_plan,
    digest_segment_plan,
)


def _records(forest):
    """Compare the same physical schedule and refs across legacy and production."""
    result = []
    for item in forest.trees:
        # Only the reference builder still exposes the obsolete .tree alias.
        tree = getattr(item, "tree", item)
        result.append((
            tree.uid, tree.tree_index, tree.member_rows,
            [(segment.segment_id, segment.parent_id, segment.position_start,
              segment.prefix_length, segment.token_ids.tolist())
             for segment in item.segment_plan.segments.values()],
            [(ref.segment_id, ref.query_offset, ref.target_token_id,
              ref.sample_row, ref.response_offset) for ref in item.objective_refs],
        ))
    return result

@pytest.mark.parametrize("seed", range(48))
@pytest.mark.parametrize("with_prompts", (False, True))
def test_turn_forest_matches_reference_for_branches_duplicates_and_masks(seed, with_prompts):
    rng = random.Random(seed)
    end = 151645
    prompt = [rng.randrange(1, 20) for _ in range(rng.randrange(1, 10))]
    shared = prompt + [7] * rng.randrange(1, 9) + [end]
    sequences = [shared + [8, end], shared + [9, end], shared,
                 shared + [8, end], [99, end], shared + [8, 10, end]]
    inputs = [torch.tensor(tokens) for tokens in sequences]
    prompts = [tokens[:len(prompt)] if row != 4 else tokens[:1]
               for row, tokens in enumerate(inputs)]
    responses = [tokens[len(prefix):] for tokens, prefix in zip(inputs, prompts, strict=True)]
    masks = [torch.tensor([rng.choice((False, True)) for _ in response], dtype=torch.bool)
             for response in responses]
    batch = {"input_ids": inputs, "responses": responses, "response_mask": masks,
             "loss_mask": [mask.clone() for mask in masks]}
    if with_prompts:
        batch["prompts"] = prompts
    keys = [f"{'other' if row == 4 else 'uid'}_tpr_{row}" for row in range(len(inputs))]
    options = {"tree_policy": "qwen_turn", "turn_min_segment_tokens": rng.randrange(1, 12),
               "require_loss_mask_alignment": True}

    expected = reference_plan(keys, batch, **options)
    actual = build_tree_execution_plans(keys, batch, **options)
    lcp = build_tree_execution_plans(keys, batch, tree_builder="lcp_sort", **options)

    assert _records(actual) == _records(expected)
    assert _records(lcp) == _records(expected)
    assert actual.logical_loss_tokens == sum(int(mask.sum()) for mask in masks)
    assert digest_forest_plan(actual) == digest_forest_plan(lcp)
    assert digest_forest_plan(actual) == digest_forest_plan(build_tree_execution_plans(keys, batch, **options))
    # The persisted topology is unique; metadata access does not materialize a second tree.
    assert all(not hasattr(tree, "tree") for tree in actual.trees)
    assert all(not hasattr(tree, "nodes") for tree in actual.trees)


def test_lcp_sort_preserves_prefix_termination_duplicate_rows_and_sibling_order():
    sequences = [
        [1, 2, 151645, 3, 151645],
        [1, 2, 151645],
        [1, 2, 151645, 4, 151645],
        [1, 2, 151645, 3, 151645],
        [9, 151645, 10],
    ]
    batch = {
        "input_ids": [torch.tensor(tokens) for tokens in sequences],
        "response_mask": [torch.ones(len(tokens) - 2, dtype=torch.bool) for tokens in sequences],
    }
    keys = ["same_tpr_0", "same_tpr_1", "same_tpr_2", "same_tpr_3", "other_tpr_4"]
    trie = build_tree_execution_plans(keys, batch, turn_min_segment_tokens=1)
    lcp = build_tree_execution_plans(keys, batch, turn_min_segment_tokens=1, tree_builder="lcp_sort")

    assert _records(lcp) == _records(trie)
    assert digest_forest_plan(lcp) == digest_forest_plan(trie)
    assert [tree.member_rows for tree in lcp.trees] == [(0, 1, 2, 3), (4,)]


def test_parent_query_ownership_and_internal_terminal_are_preserved():
    sequences = [[1, 2, 151645, 3, 151645], [1, 2, 151645, 4, 151645],
                 [1, 2, 151645], [1, 2, 151645, 3, 151645]]
    batch = {"input_ids": [torch.tensor(tokens) for tokens in sequences],
             "response_mask": [torch.ones(len(tokens) - 2, dtype=torch.bool) for tokens in sequences]}
    forest = build_tree_execution_plans([f"u_s_{i}" for i in range(4)], batch, turn_min_segment_tokens=1)
    executable = forest.trees[0]
    branching = [ref for ref in executable.objective_refs if ref.response_offset == 1]
    assert [(ref.sample_row, ref.target_token_id) for ref in branching] == [(0, 3), (1, 4), (3, 3)]
    assert {(ref.segment_id, ref.query_offset) for ref in branching} == {(0, 2)}
    assert [(ref.sample_row, ref.target_token_id) for ref in executable.objective_refs
            if ref.sample_row == 2] == [(2, 151645)]
    assert len(executable.segment_plan.segments) == 3


def test_digest_covers_tokens_schedule_and_logical_refs():
    keys = ["u_s_0", "u_s_1"]
    batch = {"input_ids": [torch.tensor([1, 151645, 3]), torch.tensor([1, 151645, 4])],
             "response_mask": [torch.tensor([True]), torch.tensor([True])]}
    plan = build_tree_execution_plans(keys, batch, turn_min_segment_tokens=1)
    digest = digest_forest_plan(plan)
    assert len(digest) == 64
    assert digest_segment_plan(plan.trees[0].segment_plan) == digest_segment_plan(
        build_tree_execution_plans(keys, batch, turn_min_segment_tokens=1).trees[0].segment_plan)
    batch["response_mask"][0][0] = False
    assert digest_forest_plan(build_tree_execution_plans(keys, batch, turn_min_segment_tokens=1)) != digest
    batch["response_mask"][0][0] = True
    batch["input_ids"][0][-1] = 5
    assert digest_forest_plan(build_tree_execution_plans(keys, batch, turn_min_segment_tokens=1)) != digest


def test_parent_position_validation_rejects_disconnected_cycles():
    segments = [SegmentSpec(0, None, torch.tensor([1]), 0, 0),
                SegmentSpec(1, 2, torch.tensor([2]), 2, 2),
                SegmentSpec(2, 1, torch.tensor([3]), 1, 1)]
    with pytest.raises(ValueError, match="must start at parent"):
        SegmentPlan(segments, root_id=0)


def test_token_storage_isolated_from_mutated_input_and_plan_has_no_ce_metadata():
    tokens = torch.tensor([1, 2, 3])
    forest = build_tree_execution_plans(["u_s_0"], {
        "input_ids": [tokens], "response_mask": [torch.tensor([True])],
    })
    segment = forest.trees[0].segment_plan.get(0)
    tokens.zero_()
    assert segment.token_ids.tolist() == [1, 2, 3]
    assert not hasattr(segment, "loss_terms")
    assert not hasattr(forest.trees[0].segment_plan, "topology_only")


def test_empty_response_and_full_length_loss_masks():
    batch = {"input_ids": [torch.tensor([1, 2]), torch.tensor([3, 4])],
             "responses": [torch.empty(0, dtype=torch.long), torch.tensor([4])],
             "response_mask": [torch.empty(0, dtype=torch.bool), torch.tensor([True])],
             "loss_mask": [torch.tensor([False, False]), torch.tensor([False, True])]}
    forest = build_tree_execution_plans(["u_s_0", "v_s_0"], batch, require_loss_mask_alignment=True)
    assert forest.logical_loss_tokens == 1
    assert forest.segment_count == 2


def test_active_segments_follow_query_owners_not_target_children():
    batch = {
        "input_ids": [torch.tensor([1, 151645, 2, 151645]), torch.tensor([1, 151645, 3, 151645])],
        "response_mask": [torch.tensor([True, False]), torch.tensor([False, False])],
    }
    tree = build_tree_execution_plans(["u_s_0", "u_s_1"], batch, turn_min_segment_tokens=1).trees[0]
    assert len(tree.segment_plan.segments) == 3
    assert tree.objective_refs[0].target_token_id == 2
    assert tree.active_segment_ids == frozenset({0})
    batch["response_mask"][0] = torch.tensor([False, True])
    tree = build_tree_execution_plans(["u_s_0", "u_s_1"], batch, turn_min_segment_tokens=1).trees[0]
    child = tree.objective_refs[0].segment_id
    assert tree.active_segment_ids == frozenset({0, child})
    for mask in batch["response_mask"]:
        mask.zero_()
    tree = build_tree_execution_plans(["u_s_0", "u_s_1"], batch, turn_min_segment_tokens=1).trees[0]
    assert tree.active_segment_ids == frozenset()


def test_compression_handles_deep_terminal_chain_without_recursion():
    from verl.workers.engine.megatron.tpr.tree import _compress, _TurnNode

    depth = 1200
    child = None
    for offset in reversed(range(depth)):
        node = _TurnNode(offset + 1, 0, terminal_rows=[offset])
        if child is not None:
            node.children[(offset,)] = child
        child = node
    segments, paths = [], {}
    _compress(node, None, 0, [torch.arange(depth)], segments, paths, ())
    assert len(segments) == depth
    assert [(segment.segment_id, segment.parent_id, segment.position_start, segment.length)
            for segment in segments] == [(i, i - 1 if i else None, i, 1) for i in range(depth)]
    assert paths[depth - 1] == tuple(range(depth))


@pytest.mark.parametrize("field,value", [("query_offset", True), ("target_token_id", -1)])
def test_legacy_loss_term_validation_remains_in_legacy_only(field, value):
    fields = {"query_offset": 0, "target_token_id": 1, field: value}
    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        SegmentLossTerm(**fields)


def _make_token_lcp_batch(shared_tokens: int, *, rows: int = 2):
    shared = torch.arange(1, shared_tokens + 1, dtype=torch.long)
    inputs, prompts, responses, masks = [], [], [], []
    keys = []
    for row in range(rows):
        prompt_tail = torch.full((300,), 100_000 + row, dtype=torch.long)
        response = torch.full((64,), 200_000 + row, dtype=torch.long)
        tokens = torch.cat((shared, prompt_tail, response))
        inputs.append(tokens)
        prompts.append(tokens[:-response.numel()].clone())
        responses.append(response)
        masks.append(torch.ones(response.numel(), dtype=torch.bool))
        keys.append(f"uid{row}_session_{row}")
    return keys, {
        "input_ids": inputs,
        "prompts": prompts,
        "responses": responses,
        "response_mask": masks,
        "loss_mask": [mask.clone() for mask in masks],
    }


def test_token_lcp_shares_exact_prefix_across_uids_without_model_tokens():
    keys, batch = _make_token_lcp_batch(1200)
    # The token sequences contain no Qwen message-ending IDs and no tokenizer
    # or Chat Template is loaded. The first divergence is inside the prompt.
    trie = build_tree_execution_plans(
        keys, batch, tree_policy="token_lcp",
        min_shared_prefix_tokens=1024, tree_builder="trie",
        require_loss_mask_alignment=True,
    )
    lcp = build_tree_execution_plans(
        keys, batch, tree_policy="token_lcp",
        min_shared_prefix_tokens=1024, tree_builder="lcp_sort",
        require_loss_mask_alignment=True,
    )

    assert len(trie.trees) == 1
    tree = trie.trees[0]
    assert tree.uid == "__minibatch__"
    assert tree.member_uids == ("uid0", "uid1")
    assert tree.member_rows == (0, 1)
    assert digest_forest_plan(trie) == digest_forest_plan(lcp)
    assert trie.logical_loss_tokens == 128
    assert sum(segment.length for segment in tree.segment_plan.segments.values()) == (
        1200 + 364 + 364
    )
    shared_segments = [
        segment for segment in tree.segment_plan.segments.values()
        if segment.parent_id is None
    ]
    assert len(shared_segments) == 1
    assert shared_segments[0].length == 1200
    assert torch.equal(shared_segments[0].token_ids, batch["input_ids"][0][:1200])

    for row in tree.member_rows:
        row_refs = [ref for ref in tree.objective_refs if ref.sample_row == row]
        assert len(row_refs) == 64
        assert all(
            ref.target_token_id == int(batch["input_ids"][row][
                len(batch["prompts"][row]) + ref.response_offset
            ])
            for ref in row_refs
        )
        path = tree.segment_plan.path_to(row_refs[0].segment_id)
        # The first supervised response token's path must reconstruct the
        # complete sequence, including the arbitrary in-prompt branch point.
        path = tree.segment_plan.path_to(path[-1])
        replay = torch.cat([tree.segment_plan.get(i).token_ids for i in path])
        assert torch.equal(replay, batch["input_ids"][row])


def test_token_lcp_minimum_shared_prefix_threshold_is_enforced():
    keys, batch = _make_token_lcp_batch(1023)
    forest = build_tree_execution_plans(
        keys, batch, tree_policy="token_lcp",
        min_shared_prefix_tokens=1024,
    )
    assert len(forest.trees) == 2
    assert sorted(tree.member_rows for tree in forest.trees) == [(0,), (1,)]
    assert all(tree.member_uids in (("uid0",), ("uid1",)) for tree in forest.trees)
    logical_tokens = sum(tokens.numel() for tokens in batch["input_ids"])
    unique_tokens = sum(
        segment.length
        for tree in forest.trees
        for segment in tree.segment_plan.segments.values()
    )
    assert unique_tokens == logical_tokens


def test_token_lcp_accepts_exact_threshold_and_preserves_duplicate_rows():
    shared = torch.arange(1, 1025, dtype=torch.long)
    suffix_a = torch.tensor([90001, 90002, 90003], dtype=torch.long)
    suffix_b = torch.tensor([91001, 91002, 91003], dtype=torch.long)
    sequences = [
        torch.cat((shared, suffix_a)),
        torch.cat((shared, suffix_a)),
        torch.cat((shared, suffix_b)),
    ]
    batch = {
        "input_ids": sequences,
        "prompts": [sequence[:-1].clone() for sequence in sequences],
        "responses": [sequence[-1:].clone() for sequence in sequences],
        "response_mask": [torch.ones(1, dtype=torch.bool) for _ in sequences],
    }
    forest = build_tree_execution_plans(
        ["groupA_session_0", "groupA_session_1", "groupB_session_0"],
        batch,
        tree_policy="token_lcp",
        min_shared_prefix_tokens=1024,
    )
    assert len(forest.trees) == 1
    tree = forest.trees[0]
    assert tree.member_rows == (0, 1, 2)
    assert tree.member_uids == ("groupA", "groupB")
    assert forest.logical_loss_tokens == 3
    assert sum(segment.length for segment in tree.segment_plan.segments.values()) == (
        1024 + len(suffix_a) + len(suffix_b)
    )


@pytest.mark.parametrize("value", [0, -1, True])
def test_token_lcp_rejects_invalid_minimum_shared_prefix(value):
    keys, batch = _make_token_lcp_batch(1024)
    with pytest.raises(ValueError, match="min_shared_prefix_tokens"):
        build_tree_execution_plans(
            keys, batch, tree_policy="token_lcp",
            min_shared_prefix_tokens=value,
        )
