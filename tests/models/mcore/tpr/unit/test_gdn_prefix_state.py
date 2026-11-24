"""CPU contracts for graph-free GDN PrefixState and branch-local VJPs."""

from __future__ import annotations

import pytest
import torch

from verl.experimental.tpr.gdn.prefix_state import GDNLayerState, GDNPrefixState


def _layer_state(value: float = 1.0) -> GDNLayerState:
    return GDNLayerState(
        conv_state=torch.full((1, 12, 3), value, dtype=torch.float32),
        recurrent_state=torch.full((1, 4, 8, 6), value + 10, dtype=torch.float32),
    )


def test_save_detaches_and_compacts_gdn_states() -> None:
    conv_backing = torch.arange(1 * 12 * 7, dtype=torch.float32).reshape(1, 12, 7).requires_grad_()
    recurrent_backing = torch.arange(1 * 4 * 8 * 10, dtype=torch.float32).reshape(1, 4, 8, 10).requires_grad_()
    graph_state = GDNLayerState(
        conv_state=conv_backing[..., :3] * 2,
        recurrent_state=recurrent_backing[..., :6] * 3,
    )

    state = GDNPrefixState.save(7, 32, {3: graph_state})
    restored = state.restore(3)

    assert state.segment_id == 7
    assert state.sequence_length == 32
    assert state.layer_numbers == (3,)
    for tensor in (restored.conv_state, restored.recurrent_state):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
        assert tensor.is_contiguous()
    assert restored.conv_state.data_ptr() != graph_state.conv_state.data_ptr()
    assert restored.recurrent_state.data_ptr() != graph_state.recurrent_state.data_ptr()


def test_branch_anchors_share_saved_storage_but_own_independent_gradients() -> None:
    state = GDNPrefixState(0, 16, {1: _layer_state()})
    saved = state.restore(1)
    first = state.make_anchors().layer_states[1]
    second = state.make_anchors().layer_states[1]

    assert first.conv_state.data_ptr() == saved.conv_state.data_ptr()
    assert second.conv_state.data_ptr() == saved.conv_state.data_ptr()
    assert first.recurrent_state.data_ptr() == saved.recurrent_state.data_ptr()
    assert second.recurrent_state.data_ptr() == saved.recurrent_state.data_ptr()
    assert first.conv_state is not second.conv_state
    assert first.recurrent_state is not second.recurrent_state
    assert first.conv_state.is_leaf and second.conv_state.is_leaf


def test_sibling_anchor_gradients_accumulate_and_are_consumed_once() -> None:
    state = GDNPrefixState(0, 16, {1: _layer_state()})
    old_anchors = None

    for factor in (2.0, 3.0):
        anchors = state.make_anchors()
        old_anchors = anchors
        layer_state = anchors.layer_states[1]
        loss = factor * (layer_state.conv_state.sum() + layer_state.recurrent_state.sum())
        loss.backward()
        state.accumulate_anchor_gradients(anchors)

    gradients = state.consume_gradients()
    assert tuple(gradients) == (1,)
    gradient = gradients[1]
    torch.testing.assert_close(gradient.conv_state, torch.full_like(gradient.conv_state, 5.0))
    torch.testing.assert_close(gradient.recurrent_state, torch.full_like(gradient.recurrent_state, 5.0))
    assert not state.gradients

    with pytest.raises(RuntimeError, match="stale"):
        state.accumulate_anchor_gradients(old_anchors)
    assert not state.consume_gradients()


def test_missing_branch_gradient_is_rejected() -> None:
    state = GDNPrefixState(0, 4, {1: _layer_state(), 2: _layer_state(2.0)})
    anchors = state.make_anchors()
    anchors.layer_states[1].conv_state.sum().backward()

    with pytest.raises(RuntimeError, match="gradient is missing"):
        state.accumulate_anchor_gradients(anchors)


def test_released_gdn_state_rejects_future_access() -> None:
    state = GDNPrefixState(5, 8, {1: _layer_state()})
    state.release()
    with pytest.raises(RuntimeError, match="released"):
        state.make_anchors()


def test_gdn_checkpoint_collector_preserves_initial_state() -> None:
    from verl.experimental.tpr.gdn.context import GDNTPRAttentionContext

    state = _layer_state()
    context = GDNTPRAttentionContext(
        prefix_length=8,
        suffix_length=4,
        initial_gdn_states={1: state},
    )
    collector = context.new_collector()

    assert collector.initial_gdn_states[1] is state
    assert collector.capture_gdn_final_state
    assert not collector.compact_new_kv
