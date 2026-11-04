# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Ring CP padding combined with native Megatron TP+SP.

These tests exercise production Shard and preflight code across many
topologies; NPU/CANN/HCCL execution must be checked separately.
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
    from verl.workers.engine.megatron.tpr.executor import validate_replay_plan
    from verl.workers.engine.megatron.tpr.ring import RingCPBackend, make_ring_sequence_shard
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


@pytest.mark.parametrize(("tp_size", "length", "allowed"), [
    (1, 1, True),
    (2, 7, False),
    (2, 8, True),
    (4, 9, False),
    (4, 12, True),
    (8, 16, True),
    (8, 17, False),
])
def test_cp1_sp_keeps_explicit_divisibility_guard(tp_size, length, allowed):
    plan = _plan(length)
    if allowed:
        validate_replay_plan(plan, _prepared(tp_size), cp_size=1)
    else:
        with pytest.raises(ValueError, match="must be divisible"):
            validate_replay_plan(plan, _prepared(tp_size), cp_size=1)


@pytest.mark.parametrize("bad_tp", [0, -1, True, 2.0])
def test_sp_tp_size_must_be_positive_integer(bad_tp):
    with pytest.raises(ValueError):
        make_ring_sequence_shard(9, cp_rank=0, cp_size=2, sp_tp_size=bad_tp)
    with pytest.raises(ValueError):
        RingCPBackend(object(), parallel_size=2, parallel_rank=0, sp_tp_size=bad_tp)


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
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("sp_enabled", [False, True])
def test_engine_runtime_uses_model_tp_sp_alignment(monkeypatch, tp_size, cp_size, sp_enabled):
    """Adapter's runtime construction must match the standalone Executor."""
    from megatron.core import parallel_state
    import torch.distributed as dist

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
    assert runtime.backend.sp_tp_size == (tp_size if sp_enabled else 1)
    shard = runtime.backend.make_sequence_shard(9)
    if sp_enabled:
        assert shard.local_length % tp_size == 0
    else:
        assert shard.padded_length == ((9 + 2 * cp_size - 1) // (2 * cp_size)) * (2 * cp_size)
