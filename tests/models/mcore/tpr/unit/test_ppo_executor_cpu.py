"""Check the production PPO executor's branch gradients against full rows."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl.models.mcore.tpr.context import (
    TPRAttentionContext,
    get_tpr_attention_context,
    use_tpr_attention_context,
)
from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans


class _PrefixModel(nn.Module):
    """A causal model whose outputs depend on both cached K and V."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 3, dtype=torch.float64)
        self.head = nn.Linear(3, 7, dtype=torch.float64)
        self.rotary_pos_emb = lambda length, **_: torch.zeros(length, 1, 1, 2)

    def forward(self, *, input_ids, position_ids=None, attention_mask=None):
        context = get_tpr_attention_context()
        key = self.embedding(input_ids.remainder(16)).transpose(0, 1).unsqueeze(2)
        value = key.square()
        context.set_new_kv(1, key, value)
        hidden = (key + value).cumsum(0)
        past = context.get_past_kv(1)
        if past is not None:
            hidden = hidden + (past[0] + past[1]).sum(0, keepdim=True)
        return self.head(hidden.squeeze(2)).transpose(0, 1)


def _batch():
    # Includes a multi-level branch, duplicate rows, and an internal terminal.
    rows = [
        [1, 151645, 2, 151645, 3, 4],
        [1, 151645, 2, 151645, 5, 6],
        [1, 151645, 7, 8],
        [1, 151645, 2, 151645, 3, 4],
        [1, 151645, 2, 151645],
    ]
    return {
        "input_ids": [torch.tensor(row) for row in rows],
        "prompts": [torch.tensor(row[:2]) for row in rows],
        "responses": [torch.tensor(row[2:]) for row in rows],
        "response_mask": [torch.ones(len(row) - 2, dtype=torch.bool) for row in rows],
    }


@pytest.mark.parametrize("loss_scale", [1.0, 8.0])
@pytest.mark.parametrize("mask_branch_tokens", [False, True])
def test_branch_replay_matches_full_rows(loss_scale, mask_branch_tokens):
    torch.manual_seed(37)
    model = _PrefixModel()
    full_model = copy.deepcopy(model)
    batch = _batch()
    if mask_branch_tokens:
        # Some segments then have no owned objective, but must relay dKV.
        for mask in batch["response_mask"]:
            mask[0] = False
    forest = build_tree_execution_plans(
        tuple(f"sample_tpr_{row}" for row in range(len(batch["input_ids"]))),
        batch,
        turn_min_segment_tokens=1,
    )
    denominator = forest.logical_loss_tokens
    full_losses = []
    for row, tokens in enumerate(batch["input_ids"]):
        with use_tpr_attention_context(TPRAttentionContext(0, len(tokens))):
            logits = full_model(input_ids=tokens.unsqueeze(0))
        for offset, active in enumerate(batch["response_mask"][row]):
            if active:
                full_losses.append(
                    logits[0, offset + 1, int(tokens[offset + 2]) % 7].square() * (row + 1)
                )
    full_loss = torch.stack(full_losses).sum() / denominator
    (full_loss * loss_scale).backward()

    tree_losses = []
    for tree in forest.trees:
        refs = {
            segment_id: tree.refs_for_segment(segment_id)
            for segment_id in tree.segment_plan.segments
        }

        def objective(segment, logits):
            terms = [
                logits[0, ref.query_offset, ref.target_token_id % 7].square() * (ref.sample_row + 1)
                for ref in refs[segment.segment_id]
            ]
            return torch.stack(terms).sum() / denominator, {}

        executor = SegmentExecutor(
            model, tree.segment_plan, expected_layer_numbers=(1,),
            segment_loss_fn=objective,
            segment_loss_term_counts={key: len(value) for key, value in refs.items()},
            loss_scale_func=lambda loss: loss * loss_scale,
        )
        profiled = []
        profiler = SimpleNamespace(
            begin=lambda segment, **fields: (segment.segment_id, fields["kind"]),
            end=profiled.append,
        )
        tree_losses.append(executor.run(event_profiler=profiler))
        pushed = [segment for segment, kind in profiled if kind == "push"]
        popped = [segment for segment, kind in profiled if kind == "pop"]
        leaves = [segment for segment, kind in profiled if kind == "visit_leaf"]
        assert set(pushed) == set(popped)
        assert len(popped) + len(leaves) == len(tree.segment_plan.segments)
        assert not executor.failed
        with pytest.raises(RuntimeError, match="single-use"):
            executor.run()
        executor.kv_stack.assert_empty()
    torch.testing.assert_close(sum(tree_losses), full_loss.detach())
    for actual, expected in zip(model.parameters(), full_model.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)
    assert get_tpr_attention_context() is None


def test_failed_replay_restores_context_and_cannot_resume():
    model = _PrefixModel()
    forest = build_tree_execution_plans(
        tuple(f"sample_tpr_{row}" for row in range(5)), _batch(), turn_min_segment_tokens=1,
    )
    tree = forest.trees[0]

    def fail(*_):
        raise RuntimeError("injected PPO failure")

    executor = SegmentExecutor(
        model, tree.segment_plan, expected_layer_numbers=(1,), segment_loss_fn=fail,
        segment_loss_term_counts={key: len(tree.refs_for_segment(key)) for key in tree.segment_plan.segments},
    )
    with pytest.raises(RuntimeError, match="injected PPO failure"):
        executor.run()
    assert executor.failed
    assert get_tpr_attention_context() is None
    with pytest.raises(RuntimeError, match="failed"):
        executor.push(tree.segment_plan.root_id)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("owned", [False, True])
def test_cp_loss_scaling_and_empty_rank_connected_zero(cp_size, owned):
    executor = object.__new__(SegmentExecutor)
    executor.cp_size = cp_size
    executor.loss_scale_func = lambda loss: loss * 8
    logits = torch.randn(1, 3, 7, requires_grad=True)
    loss = logits.square().sum() / 13
    scaled = executor._prepare_backward_loss(loss, logits=logits, has_owned_loss=owned)
    scaled.backward()
    expected = logits.detach() * (16 * cp_size / 13) if owned else torch.zeros_like(logits)
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.parametrize(
    ("enabled", "phase", "no_grad", "expected"),
    [
        (False, "both", True, False),
        (False, "both", False, False),
        (True, "both", True, True),
        (True, "both", False, True),
        (True, "push", True, True),
        (True, "push", False, False),
        (True, "pop", True, False),
        (True, "pop", False, True),
    ],
)
def test_native_root_ring_phase_gates_push_and_pop(enabled, phase, no_grad, expected):
    from verl.workers.engine.megatron.tpr.executor import _native_root_enabled_for_stage

    assert _native_root_enabled_for_stage(
        enabled=enabled, phase=phase, no_grad=no_grad,
    ) is expected


def test_native_root_ring_phase_rejects_invalid_mode():
    from verl.workers.engine.megatron.tpr.executor import _native_root_enabled_for_stage

    with pytest.raises(ValueError, match="TPR_ROOT_NATIVE_RING_PHASE"):
        _native_root_enabled_for_stage(enabled=True, phase="invalid", no_grad=True)
