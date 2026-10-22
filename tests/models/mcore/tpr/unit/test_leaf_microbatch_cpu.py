"""CPU correctness gates for opt-in CP1 leaf micro-batching."""

import copy

import pytest
import torch

from verl.models.mcore.tpr.context import TPRAttentionContext
from verl.models.mcore.tpr.rectangular_attention import _validate_inputs
from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec
from .test_ppo_executor_cpu import _PrefixModel


def _siblings():
    root = SegmentSpec(0, None, torch.tensor([1, 2, 3]), 0, 0)
    # First four siblings have identical lengths. Last child is different.
    leaves = [
        SegmentSpec(i, 0, torch.tensor([i + 3, i + 4, i + 5]), 3, 3)
        for i in range(1, 5)
    ]
    leaves.append(SegmentSpec(5, 0, torch.tensor([8, 9, 10, 11, 12]), 3, 3))
    return SegmentPlan([root, *leaves], root_id=0)


def _run(model, plan, monkeypatch, *, k, budget, loss_scale):
    monkeypatch.setenv("TPR_LEAF_BATCH_SIZE", str(k))
    monkeypatch.setenv("TPR_LEAF_BATCH_MAX_TOKENS", str(budget))
    seen = []

    def objective(segment, logits):
        # Same global denominator across all physical micro-batches.
        return logits.square().sum() / 37, {"count": 1}

    executor = SegmentExecutor(
        model, plan, expected_layer_numbers=(1,),
        segment_loss_fn=objective,
        segment_loss_term_counts={s: int(s != 0) for s in plan.segments},
        loss_scale_func=lambda loss: loss * loss_scale,
    )
    class Events:
        def begin(self, segment, *, kind, **_):
            seen.append((kind, segment.segment_id))
            return None

        def end(self, _):
            pass

    loss = executor.run(event_profiler=Events())
    executor.kv_stack.assert_empty()
    assert not executor.failed
    return loss, seen


@pytest.mark.parametrize("loss_scale", [1.0, 8.0])
def test_grouped_leaves_match_sequential_loss_parameter_and_prefix_gradients(monkeypatch, loss_scale):
    torch.manual_seed(1234)
    plan = _siblings()
    baseline = _PrefixModel()
    grouped = copy.deepcopy(baseline)

    baseline_loss, baseline_events = _run(baseline, plan, monkeypatch, k=1, budget=12, loss_scale=loss_scale)
    grouped_loss, group_events = _run(grouped, plan, monkeypatch, k=4, budget=12, loss_scale=loss_scale)

    # The 4 equal-length siblings become one physical forward/backward.
    assert [kind for kind, _ in group_events].count("visit_leaf_group") == 1
    assert [kind for kind, _ in group_events].count("visit_leaf") == 1
    assert [kind for kind, _ in baseline_events].count("visit_leaf") == 5
    assert [kind for kind, _ in group_events].count("push") == 1
    assert [kind for kind, _ in group_events].count("pop") == 1

    torch.testing.assert_close(grouped_loss, baseline_loss, rtol=1e-12, atol=1e-12)
    for left, right in zip(grouped.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-11, atol=1e-11)


def test_grouping_respects_maximum_token_budget(monkeypatch):
    plan = _siblings()
    model = _PrefixModel()
    monkeypatch.setenv("TPR_LEAF_BATCH_SIZE", "4")
    monkeypatch.setenv("TPR_LEAF_BATCH_MAX_TOKENS", "6")
    executor = SegmentExecutor(
        model, plan, expected_layer_numbers=(1,),
        segment_loss_fn=lambda _segment, logits: (logits.square().sum(), {}),
        segment_loss_term_counts={i: 1 for i in range(6)},
    )
    assert executor._group_ready_children(plan.children_of(0)) == [(1, 2), (3, 4), 5]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_context_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        TPRAttentionContext(0, 3, batch_size=batch_size)


def test_grouped_context_requires_matching_new_kv_batch_size():
    context = TPRAttentionContext(0, 3, batch_size=2)
    good = torch.randn(3, 2, 1, 8)
    context.set_new_kv(1, good, good.square())
    with pytest.raises(ValueError, match="batch size"):
        context.set_new_kv(2, torch.randn(3, 1, 1, 8), torch.randn(3, 1, 1, 8))


def test_rectangular_attention_contract_accepts_equal_length_batched_layout():
    query = torch.randn(3, 2, 4, 8)
    key = torch.randn(9, 2, 2, 8)
    assert _validate_inputs(query, key, torch.randn_like(key), 0.0) == (3, 9, 4, 8)
