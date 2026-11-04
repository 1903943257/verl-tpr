"""Opt-in delayed dKV precision probe. No production-default behavior change."""

import pytest
import torch

from verl.workers.engine.megatron.tpr.state import KVPrefixState


@pytest.mark.parametrize("consume", [False, True])
@pytest.mark.parametrize("fp32_enabled", [False, True])
def test_small_descendant_grads_are_retained_only_by_fp32_accum(
    monkeypatch, consume, fp32_enabled
):
    monkeypatch.setenv("TPR_EXPERIMENT_DELAYED_DKV_FP32", "1" if fp32_enabled else "0")
    kv = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16)
    state = KVPrefixState(0, 1, {1: (kv.detach(), kv.detach().clone())})

    for weight in [1.0] + [1.0 / 256] * 7:
        anchors = state.make_anchors()
        key, value = anchors.key_values[1]
        (weight * (key + value)).sum().backward()
        state.accumulate_anchor_gradients(anchors, consume=consume)

    key_grad, value_grad = state.consume_gradients()[1]
    expected = 1.02734375 if fp32_enabled else 1.0
    for gradient in (key_grad, value_grad):
        assert gradient.dtype == (torch.float32 if fp32_enabled else torch.bfloat16)
        assert gradient.item() == expected
    assert not state.gradients


@pytest.mark.parametrize("consume", [False, True])
def test_native_fp32_grads_remain_fp32(monkeypatch, consume):
    monkeypatch.setenv("TPR_EXPERIMENT_DELAYED_DKV_FP32", "1")
    kv = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    state = KVPrefixState(0, 1, {1: (kv.detach(), kv.detach().clone())})
    anchors = state.make_anchors()
    key, value = anchors.key_values[1]
    (key + value).sum().backward()
    state.accumulate_anchor_gradients(anchors, consume=consume)
    assert all(g.dtype == torch.float32 for g in state.consume_gradients()[1])
