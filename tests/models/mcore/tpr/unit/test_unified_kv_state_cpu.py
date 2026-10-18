"""Unified local/Ring anchors must relay the same branched prefix gradients."""

import pytest
import torch

from verl.workers.engine.megatron.tpr.ring import SequenceShard, make_ring_sequence_shard
from verl.workers.engine.megatron.tpr.state import KVPrefixState, KVStack
from verl.workers.engine.megatron.tpr.tree import SegmentSpec


def _segment(segment_id, parent, length, offset):
    return SegmentSpec(segment_id, parent, torch.arange(length), offset, offset)


def _kv(length):
    return {layer: tuple(torch.randn(length, 1, 2, 3, dtype=torch.float64) for _ in range(2))
            for layer in (1, 3)}


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_unified_anchors_match_direct_multilevel_branch_gradients(cp_size, cp_rank):
    torch.manual_seed(831)
    stack, direct = KVStack(), []
    for segment_id, length, offset in ((0, 7, 0), (1, 5, 7)):
        shard = (SequenceShard.full(length) if cp_size == 1 else
                 make_ring_sequence_shard(length, cp_size=cp_size, cp_rank=cp_rank))
        key_values = _kv(shard.local_length)
        direct.append({layer: tuple(t.clone().requires_grad_() for t in pair)
                       for layer, pair in key_values.items()})
        stack.push(_segment(segment_id, None if segment_id == 0 else 0, length, offset), key_values, shard=shard)

    for factor in (2.0, 3.0):
        anchors = stack.build_past_anchors()
        assert anchors.prefix_length == anchors.global_prefix_length == 12
        if cp_size == 1:
            observed = anchors.key_values
        else:
            with pytest.raises(RuntimeError, match="sharded KV"):
                _ = anchors.key_values
            observed = {layer: tuple(torch.cat([entry.key_values[layer][i] for entry in anchors.entries])
                                     for i in (0, 1)) for layer in (1, 3)}
        expected = {layer: tuple(torch.cat([entry[layer][i] for entry in direct])
                                 for i in (0, 1)) for layer in (1, 3)}
        for pairs in (observed, expected):
            loss = torch.zeros((), dtype=torch.float64)
            for layer, (key, value) in pairs.items():
                weights = torch.arange(key.numel(), dtype=key.dtype).reshape_as(key) + factor + layer
                loss = loss + ((key + 2 * value).square() * weights).sum()
            loss.backward()
        stack.accumulate_anchor_gradients(anchors)

    for segment_id, originals in enumerate(direct):
        for layer, gradients in stack.get(segment_id).kv.consume_gradients().items():
            for gradient, tensor in zip(gradients, originals[layer], strict=True):
                torch.testing.assert_close(gradient, tensor.grad, rtol=1e-14, atol=1e-14)
    with pytest.raises(RuntimeError, match="stale"):
        stack.accumulate_anchor_gradients(anchors)


def test_anchor_generation_rejects_replaced_state_with_same_segment_id():
    stack = KVStack()
    root = _segment(0, None, 3, 0)
    stack.push(root, _kv(3))
    anchors = stack.build_past_anchors()
    for key, value in anchors.entries[0].key_values.values():
        (key + value).sum().backward()
    stack.pop(0).kv.release()
    stack.push(root, _kv(3))
    with pytest.raises(RuntimeError, match="stale"):
        stack.accumulate_anchor_gradients(anchors)


def test_cached_view_is_compact_graph_free_and_released_with_its_gradients():
    backing = torch.randn(100, 1, 2, 3)
    state = KVPrefixState(0, 3, {1: (backing[10:13], backing[40:43])})
    for tensor in state.key_values[1]:
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        assert tensor.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
        assert not tensor.requires_grad and tensor.grad_fn is None
    anchors = state.make_anchors()
    (anchors.key_values[1][0] + anchors.key_values[1][1]).sum().backward()
    state.accumulate_anchor_gradients(anchors)
    state.release()
    assert state.released and not state._key_values and not state._gradients
    with pytest.raises(RuntimeError, match="released"):
        state.consume_gradients()


def test_failed_path_clear_releases_every_level_and_invalidates_old_anchors():
    stack = KVStack()
    states = [stack.push(_segment(0, None, 3, 0), _kv(3)).kv,
              stack.push(_segment(1, 0, 2, 3), _kv(2)).kv]
    anchors = stack.build_past_anchors()
    for entry in anchors.entries:
        for key, value in entry.key_values.values():
            (key + value).sum().backward()
    stack.accumulate_anchor_gradients(anchors, consume=True)
    assert all(state._key_values and state._gradients for state in states)
    stack.clear()
    stack.clear()  # Exception cleanup may be entered more than once.
    stack.assert_empty()
    assert all(state.released and not state._key_values and not state._gradients for state in states)
    with pytest.raises(RuntimeError, match="stale"):
        stack.accumulate_anchor_gradients(anchors)


def test_executor_consumes_anchor_gradients_without_a_second_full_copy():
    state = KVPrefixState(0, 3, _kv(3))
    anchors = state.make_anchors()
    for key, value in anchors.key_values.values():
        (key.square() + 2 * value.square()).sum().backward()
    owned = {layer: tuple(t.grad for t in pair) for layer, pair in anchors.key_values.items()}
    expected = {layer: tuple(t.clone() for t in pair) for layer, pair in owned.items()}
    state.accumulate_anchor_gradients(anchors, consume=True)
    for layer, pair in anchors.key_values.items():
        for index, tensor in enumerate(pair):
            assert tensor.grad is None
            assert state.gradients[layer][index].data_ptr() == owned[layer][index].data_ptr()

    # Clearing .grad makes another backward allocate independent contributions;
    # its in-place accumulation must not replace or double the stored buffers.
    for key, value in anchors.key_values.values():
        (3 * key + 4 * value).sum().backward()
    for layer, pair in anchors.key_values.items():
        for index, tensor in enumerate(pair):
            assert tensor.grad.data_ptr() != state.gradients[layer][index].data_ptr()
    state.accumulate_anchor_gradients(anchors, consume=True)
    for layer, pair in state.gradients.items():
        for index, gradient in enumerate(pair):
            torch.testing.assert_close(gradient, expected[layer][index] + (3 if index == 0 else 4))


def test_public_anchor_accumulation_keeps_caller_gradient_buffers_independent():
    state = KVPrefixState(0, 3, _kv(3))
    anchors = state.make_anchors()
    for key, value in anchors.key_values.values():
        (key + value).sum().backward()
    state.accumulate_anchor_gradients(anchors)
    for layer, pair in anchors.key_values.items():
        for index, tensor in enumerate(pair):
            assert tensor.grad is not None
            assert tensor.grad.data_ptr() != state.gradients[layer][index].data_ptr()
            tensor.grad.zero_()
            torch.testing.assert_close(state.gradients[layer][index], torch.ones_like(tensor))


@pytest.mark.parametrize("alias_cached_kv", [False, True])
def test_consumed_shared_gradients_get_independent_accumulation_storage(alias_cached_kv):
    state = KVPrefixState(0, 3, _kv(3))
    anchors = state.make_anchors()
    shared = state.key_values[1][0] if alias_cached_kv else torch.ones_like(state.key_values[1][0])
    initial = shared.clone()
    for key, value in anchors.key_values.values():
        key.grad = value.grad = shared
    state.accumulate_anchor_gradients(anchors, consume=True)
    pointers = [tensor.data_ptr() for pair in state.gradients.values() for tensor in pair]
    assert len(set(pointers)) == len(pointers)
    for layer, pair in state.gradients.items():
        state.accumulate_layer_gradients(
            layer, torch.full_like(pair[0], layer), torch.full_like(pair[1], layer + 1),
        )
    for layer, pair in state.gradients.items():
        for index, gradient in enumerate(pair):
            torch.testing.assert_close(gradient, initial + layer + index)
    if alias_cached_kv:
        torch.testing.assert_close(state.key_values[1][0], initial)
