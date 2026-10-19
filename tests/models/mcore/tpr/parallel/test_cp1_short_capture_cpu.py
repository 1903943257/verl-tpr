"""CPU regressions for bounded same-label single-NPU Actor PPO capture."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from .compare_real_actor_cp1_short_full_tpr import _metrics
from .prepare_tpr_cp1_short_capture import _common_divergence, derive_short_capture


def test_short_sibling_divergence_guard():
    assert _common_divergence(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 5])) == 2
    with pytest.raises(ValueError, match="identical"):
        _common_divergence(torch.tensor([1, 2]), torch.tensor([1, 2]))


def test_comparison_metric_rejects_disjoint_witnesses():
    got = _metrics([1.0, 2.0], [1.0, 2.0])
    assert got["relative_l2"] == 0
    assert got["cosine"] == pytest.approx(1.0)
    with pytest.raises(AssertionError, match="Witness"):
        _metrics([1.0], [2.0, 3.0])


def _nested(items):
    return tu.nested_tensor_from_tensor_list(items, ragged_idx=1)


def test_derived_capture_keeps_original_ppo_labels_and_sibling_root(tmp_path: Path):
    prompt = torch.arange(256, dtype=torch.long) % 127
    base_response = (torch.arange(500, dtype=torch.long) + 200) % 2000
    response_0 = base_response.clone()
    response_1 = base_response.clone()
    response_1[200] = 2013
    assert not torch.equal(response_0, response_1)
    responses = [response_0, response_1]
    masks = [torch.ones(500, dtype=torch.bool) for _ in range(2)]
    original = {
        "capture_stage": "actor_update_mini_batch",
        "keys": ("session_tpr_0", "session_tpr_1"),
        "tensordict": TensorDict({
            "input_ids": _nested([torch.cat((prompt, r)) for r in responses]),
            "prompts": _nested([prompt.clone(), prompt.clone()]),
            "responses": _nested(responses),
            "response_mask": _nested(masks),
            "loss_mask": _nested([x.clone() for x in masks]),
            "old_log_probs": _nested([torch.arange(500).float() * (-0.01)] * 2),
            "advantages": _nested([torch.arange(500).float() * 0.001] * 2),
            "temperature": torch.tensor([0.6, 0.6]),
        }, batch_size=[2]),
    }
    source = tmp_path / "real.pt"
    target = tmp_path / "derived.pt"
    torch.save(original, source)
    audit = derive_short_capture(
        source, target, rows=(0, 1), cap=512, prompt_window=256,
    )
    assert target.exists()
    assert target.with_suffix(".audit.json").exists()
    assert audit["original_response_first_divergence"] == 200
    cropped = torch.load(target, map_location="cpu", weights_only=False)["tensordict"]
    assert torch.equal(cropped["prompts"][0], cropped["prompts"][1])
    assert not torch.equal(cropped["responses"][0], cropped["responses"][1])
    assert max(audit["retained_lengths"]) <= 512
    start = audit["retained_response_start"]
    size = audit["retained_response_length"]
    for row in range(2):
        for name in ("response_mask", "old_log_probs", "advantages", "loss_mask"):
            torch.testing.assert_close(
                cropped[name][row], original["tensordict"][name][row][start:start + size]
            )
    with pytest.raises(FileExistsError):
        derive_short_capture(source, target, rows=(0, 1), cap=512, prompt_window=256)


def test_split_no_share_preserves_each_siblings_exact_query_and_target():
    from verl.workers.engine.megatron.tpr.tree import (
        ForestExecutionPlan, SegmentObjectiveRef, SegmentPlan, SegmentSpec,
        TreeExecutionPlan,
    )
    from .test_real_actor_cp1_short_full_tpr_npu import _make_unshared_split

    row0 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    row1 = torch.tensor([1, 2, 3, 4, 9, 10], dtype=torch.long)
    root = SegmentSpec(0, None, row0[:4], 0, 0)
    leaf0 = SegmentSpec(1, 0, row0[4:], 4, 4)
    leaf1 = SegmentSpec(2, 0, row1[4:], 4, 4)
    source_plan = SegmentPlan((root, leaf0, leaf1), root_id=0)
    source_refs = (
        SegmentObjectiveRef(0, 3, 5, 0, 0),
        SegmentObjectiveRef(1, 0, 6, 0, 1),
        SegmentObjectiveRef(0, 3, 9, 1, 0),
        SegmentObjectiveRef(2, 0, 10, 1, 1),
    )
    shared = ForestExecutionPlan((
        TreeExecutionPlan(
            uid="original-session", tree_index=0,
            member_rows=(0, 1), segment_plan=source_plan,
            objective_refs=source_refs,
        ),
    ), logical_loss_tokens=len(source_refs))
    split = _make_unshared_split(shared, {"input_ids": [row0, row1]})
    assert len(split.trees) == 2
    assert split.segment_count == 4
    assert split.logical_loss_tokens == 4
    for row, tree in enumerate(split.trees):
        assert tree.member_rows == (row,)
        assert len(tree.segment_plan.segments) == 2
        assert torch.equal(tree.segment_plan.get(0).token_ids, row0[:4])
        assert torch.equal(tree.segment_plan.get(1).token_ids, (row0, row1)[row][4:])
        refs = sorted(tree.objective_refs, key=lambda x: x.response_offset)
        assert [(r.segment_id, r.query_offset, r.target_token_id) for r in refs] == [
            (0, 3, 5 if row == 0 else 9),
            (1, 0, 6 if row == 0 else 10),
        ]
    assert sum(
        seg.length for tree in split.trees
        for seg in tree.segment_plan.segments.values()
    ) == row0.numel() + row1.numel()
