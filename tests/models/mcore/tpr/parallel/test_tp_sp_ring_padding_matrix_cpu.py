# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for CP Ring and CP1 SP-only padding with native Megatron TP+SP.

These tests exercise production Shard, Prefix KV, and preflight code across
many topologies; NPU/CANN/HCCL execution must be checked separately.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest
import torch

# MindSpeed imports through verl.workers.engine can parse pytest argv.
_original_argv = sys.argv[:]
try:
    sys.argv[:] = [sys.argv[0]]
    from verl.models.mcore.tpr.context import TPRAttentionContext
    from verl.workers.engine.megatron.tpr.executor import (
        build_sharded_rotary_pos_emb,
        validate_replay_plan,
    )
    from verl.workers.engine.megatron.tpr.ring import (
        RingCPBackend,
        make_cp1_sp_sequence_shard,
        make_ring_sequence_shard,
    )
    from verl.workers.engine.megatron.tpr.state import KVStack
    from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec
finally:
    sys.argv[:] = _original_argv


def _plan(length: int) -> SegmentPlan:
    return SegmentPlan(
        segments=[
            SegmentSpec(
                segment_id=0,
                parent_id=None,
                token_ids=torch.arange(length),
                position_start=0,
                prefix_length=0,
            )
        ],
        root_id=0,
    )


def _prepared(tp: int, *, sp: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                tensor_model_parallel_size=tp,
                sequence_parallel=sp,
            )
        )
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("length", [1, 2, 3, 7, 9, 12, 13, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129, 257, 8193])
def test_sp_ring_matrix_preserves_tokens_and_aligns_all_local_shards(tp_size, cp_size, length):
    alignment = math.lcm(2 * cp_size, cp_size * tp_size)
    expected_padded = ((length + alignment - 1) // alignment) * alignment
    source = torch.arange(length)
    shards = []

    for cp_rank in range(cp_size):
        backend = RingCPBackend(
            object(),
            parallel_size=cp_size,
            parallel_rank=cp_rank,
            sp_tp_size=tp_size,
        )
        shard = backend.make_sequence_shard(length)
        direct = make_ring_sequence_shard(
            length,
            cp_rank=cp_rank,
            cp_size=cp_size,
            sp_tp_size=tp_size,
        )
        assert shard == direct
        assert shard.padded_length == expected_padded
        assert shard.local_length == expected_padded // cp_size
        assert shard.local_length % tp_size == 0
        assert shard.local_length == 2 * (expected_padded // (2 * cp_size))
        local = shard.select(source)
        assert local.numel() == shard.local_length
        indices = shard.physical_global_indices()
        assert local[indices >= length].eq(0).all()
        assert torch.equal(local[indices < length], indices[indices < length])
        shards.append(shard)

    # Every real token has one CP owner, regardless of zigzag chunk/padding.
    seen = torch.cat([s.physical_global_indices() for s in shards])
    real = seen[seen < length]
    assert torch.equal(real.sort().values, source)
    validate_replay_plan(_plan(length), _prepared(tp_size), cp_size=cp_size)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("length", [1, 9, 129])
def test_padding_has_zero_contribution_to_backward(tp_size, cp_size, length):
    source = torch.arange(1, length + 1, dtype=torch.float32, requires_grad=True)
    objective = torch.zeros((), dtype=torch.float32)
    for rank in range(cp_size):
        shard = make_ring_sequence_shard(
            length,
            cp_rank=rank,
            cp_size=cp_size,
            sp_tp_size=tp_size,
        )
        local = shard.select(source)
        valid = shard.physical_global_indices() < length
        assert torch.count_nonzero(local[~valid]) == 0
        objective = objective + local[valid].sum()
    objective.backward()
    assert torch.equal(source.grad, torch.ones_like(source))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("length", [1, 9, 129, 256])
def test_non_sp_keeps_original_cp_only_padding(cp_size, length):
    expected = ((length + 2 * cp_size - 1) // (2 * cp_size)) * (2 * cp_size)
    for rank in range(cp_size):
        backend = RingCPBackend(
            object(),
            parallel_size=cp_size,
            parallel_rank=rank,
        )
        shard = backend.make_sequence_shard(length)
        assert shard.padded_length == expected
        assert backend.sp_tp_size == 1
    validate_replay_plan(_plan(length), _prepared(8, sp=False), cp_size=cp_size)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("length", [1, 7, 9, 127, 128, 129])
def test_cp1_sp_uses_contiguous_physical_padding(tp_size, length):
    shard = make_cp1_sp_sequence_shard(length, sp_tp_size=tp_size)
    expected_padded = ((length + tp_size - 1) // tp_size) * tp_size
    assert shard.cp_size == 1
    assert shard.cp_rank == 0
    assert shard.ranges == ((0, expected_padded),)
    assert shard.padded_length == expected_padded
    assert shard.local_length == expected_padded
    assert shard.valid_local_length == length
    assert shard.local_length % tp_size == 0

    source = torch.arange(length, dtype=torch.float32, requires_grad=True)
    physical = shard.select(source)
    valid = shard.physical_global_indices() < length
    assert torch.equal(physical[valid], source)
    assert physical[~valid].eq(0).all()
    assert torch.equal(
        shard.global_indices(),
        torch.arange(expected_padded).clamp_max(length - 1),
    )

    physical[valid].sum().backward()
    assert torch.equal(source.grad, torch.ones_like(source))

    def rotary(max_seq_len, offset=0, packed_seq=False, cp_group=None):
        assert not packed_seq
        return torch.arange(offset, offset + max_seq_len, dtype=torch.float32).view(-1, 1, 1, 1)

    rope = build_sharded_rotary_pos_emb(
        rotary, position_start=17, shard=shard,
    )
    assert rope.shape[0] == expected_padded
    expected_rope = torch.arange(17, 17 + length, dtype=torch.float32)
    if expected_padded > length:
        expected_rope = torch.cat((
            expected_rope,
            torch.full((expected_padded - length,), 16 + length),
        ))
    assert torch.equal(rope[:, 0, 0, 0], expected_rope)
    validate_replay_plan(_plan(length), _prepared(tp_size), cp_size=1)


def test_cp1_sp_context_tracks_logical_and_physical_suffix_lengths():
    context = TPRAttentionContext(
        prefix_length=9,
        suffix_length=7,
        local_physical_suffix_length=8,
    )
    assert context.suffix_length == 7
    assert context.local_suffix_length == 8
    key = torch.zeros((8, 1, 1, 2))
    context.set_new_kv(1, key, key)
    collector = context.new_collector()
    assert collector.local_suffix_length == 8
    assert collector.new_key_values == {}


def test_cp1_sp_prefix_kv_trims_padding_and_relays_zero_pad_gradients():
    root_length, child_length, tp_size = 9, 7, 4
    root = SegmentSpec(
        segment_id=0,
        parent_id=None,
        token_ids=torch.arange(root_length),
        position_start=0,
        prefix_length=0,
    )
    child = SegmentSpec(
        segment_id=1,
        parent_id=0,
        token_ids=torch.arange(child_length),
        position_start=root_length,
        prefix_length=root_length,
    )
    root_shard = make_cp1_sp_sequence_shard(root_length, sp_tp_size=tp_size)
    child_shard = make_cp1_sp_sequence_shard(child_length, sp_tp_size=tp_size)
    root_key = torch.arange(root_shard.local_length, dtype=torch.float32).view(-1, 1, 1, 1)
    root_value = root_key + 100
    child_key = (torch.arange(child_shard.local_length, dtype=torch.float32) + 20).view(-1, 1, 1, 1)
    child_value = child_key + 100

    stack = KVStack()
    stack.push(root, {1: (root_key, root_value)}, shard=root_shard)
    single_prefix = stack.build_past_key_values()[1][0]
    assert single_prefix.shape[0] == root_length
    assert torch.equal(single_prefix[:, 0, 0, 0], root_key[:root_length, 0, 0, 0])

    stack.push(child, {1: (child_key, child_value)}, shard=child_shard)
    anchors = stack.build_past_anchors()
    combined_key, combined_value = anchors.key_values[1]
    assert combined_key.shape[0] == root_length + child_length
    assert combined_value.shape[0] == root_length + child_length
    assert torch.equal(
        combined_key[:, 0, 0, 0],
        torch.cat((root_key[:root_length, 0, 0, 0], child_key[:child_length, 0, 0, 0])),
    )

    (combined_key.sum() + combined_value.sum()).backward()
    stack.accumulate_anchor_gradients(anchors, consume=True)
    root_gradient = stack.get_new_kv_gradients(0)[1][0]
    child_gradient = stack.get_new_kv_gradients(1)[1][0]
    assert torch.equal(root_gradient[:root_length], torch.ones_like(root_gradient[:root_length]))
    assert root_gradient[root_length:].eq(0).all()
    assert torch.equal(child_gradient[:child_length], torch.ones_like(child_gradient[:child_length]))
    assert child_gradient[child_length:].eq(0).all()


@pytest.mark.parametrize("bad_tp", [0, -1, True, 2.0])
def test_sp_tp_size_must_be_positive_integer(bad_tp):
    with pytest.raises(ValueError):
        make_ring_sequence_shard(9, cp_rank=0, cp_size=2, sp_tp_size=bad_tp)
    with pytest.raises(ValueError):
        RingCPBackend(object(), parallel_size=2, parallel_rank=0, sp_tp_size=bad_tp)
    with pytest.raises(ValueError):
        make_cp1_sp_sequence_shard(9, sp_tp_size=bad_tp)


def test_explicit_cp_padding_must_satisfy_joint_tp_sp_alignment():
    # Old CP-only padding: logical=9 -> padded=12 -> local=6, which TP4 SP cannot partition.
    with pytest.raises(ValueError, match="divisible by 8"):
        make_ring_sequence_shard(
            9,
            cp_rank=0,
            cp_size=2,
            padded_length=12,
            sp_tp_size=4,
        )
    corrected = make_ring_sequence_shard(9, cp_rank=0, cp_size=2, sp_tp_size=4)
    assert (corrected.padded_length, corrected.local_length) == (16, 8)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("sp_enabled", [False, True])
def test_engine_runtime_uses_model_tp_sp_alignment(monkeypatch, tp_size, cp_size, sp_enabled):
    """Adapter's runtime construction must match the standalone Executor."""
    import torch.distributed as dist
    from megatron.core import parallel_state

    from verl.workers.engine.megatron.tpr.ring import resolve_engine_cp_runtime

    fake_cp_group = object()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: cp_size)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda: fake_cp_group)

    engine = SimpleNamespace(
        engine_config=SimpleNamespace(
            context_parallel_size=cp_size,
            tpr_cp_backend=None,
            dynamic_context_parallel=False,
        )
    )
    config = SimpleNamespace(
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sp_enabled,
    )
    runtime = resolve_engine_cp_runtime(engine, config)
    if cp_size == 1:
        assert runtime.backend is None
        shard = make_cp1_sp_sequence_shard(
            9, sp_tp_size=tp_size if sp_enabled else 1,
        )
    else:
        assert runtime.backend.sp_tp_size == (tp_size if sp_enabled else 1)
        shard = runtime.backend.make_sequence_shard(9)
    if sp_enabled:
        assert shard.local_length % tp_size == 0
    elif cp_size == 1:
        assert shard.padded_length == 9
    else:
        assert shard.padded_length == ((9 + 2 * cp_size - 1) // (2 * cp_size)) * (2 * cp_size)
