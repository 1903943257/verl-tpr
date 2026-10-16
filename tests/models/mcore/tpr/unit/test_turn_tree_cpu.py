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

    assert _records(actual) == _records(expected)
    assert actual.logical_loss_tokens == sum(int(mask.sum()) for mask in masks)
    assert digest_forest_plan(actual) == digest_forest_plan(build_tree_execution_plans(keys, batch, **options))
    # The persisted topology is unique; metadata access does not materialize a second tree.
    assert all(not hasattr(tree, "tree") for tree in actual.trees)
    assert all(not hasattr(tree, "nodes") for tree in actual.trees)


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


@pytest.mark.parametrize("field,value", [("query_offset", True), ("target_token_id", -1)])
def test_legacy_loss_term_validation_remains_in_legacy_only(field, value):
    fields = {"query_offset": 0, "target_token_id": 1, field: value}
    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        SegmentLossTerm(**fields)
