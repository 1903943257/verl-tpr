# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU-only preflight for Megatron TP2+SP with TPR Ring CP2/CP4.

These tests check the executable shape contract and opt-in gate. They do not
certify distributed collectives, numerical gradients, or E2E PPO behavior.
"""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine.megatron.tpr.adapter import _validate_tpr_cp_combination
from verl.workers.engine.megatron.tpr.executor import validate_replay_plan
from verl.workers.engine.megatron.tpr.ring import make_ring_sequence_shard
from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec


def _plan(length: int) -> SegmentPlan:
    return SegmentPlan(
        segments=[
            SegmentSpec(
                segment_id=0, parent_id=None, token_ids=torch.arange(length),
                position_start=0, prefix_length=0,
            ),
        ],
        root_id=0,
    )


def _prepared(*, tp: int = 2, sp: bool = True):
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                tensor_model_parallel_size=tp, sequence_parallel=sp,
            ),
        ),
    )


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("length", [1, 3, 7, 127, 128, 129, 257, 8193])
def test_ring_cp_local_length_is_tp2_sp_aligned(cp_size, length):
    plan = _plan(length)
    for cp_rank in range(cp_size):
        shard = make_ring_sequence_shard(
            length, cp_rank=cp_rank, cp_size=cp_size,
        )
        assert shard.local_length % 2 == 0
        assert shard.padded_length % (2 * cp_size) == 0
        assert shard.select(torch.arange(length)).numel() == shard.local_length
    validate_replay_plan(plan, _prepared(), cp_size=cp_size)


def test_sp_cp1_odd_segment_remains_rejected():
    with pytest.raises(ValueError, match="must be divisible"):
        validate_replay_plan(_plan(127), _prepared(), cp_size=1)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_sp_ring_only_opens_with_explicit_acceptance_opt_in(monkeypatch, cp_size):
    runtime = SimpleNamespace(size=cp_size, backend_name="ring")
    config = SimpleNamespace(sequence_parallel=True)
    monkeypatch.delenv("TPR_ENABLE_TP_SP_CP", raising=False)
    with pytest.raises(NotImplementedError, match="TPR_ENABLE_TP_SP_CP"):
        _validate_tpr_cp_combination(2, runtime, config, ppo=True)
    monkeypatch.setenv("TPR_ENABLE_TP_SP_CP", "1")
    _validate_tpr_cp_combination(2, runtime, config, ppo=True, dp_size=1)
    with pytest.raises(NotImplementedError, match="limited to DP1"):
        _validate_tpr_cp_combination(2, runtime, config, ppo=True, dp_size=2)
    with pytest.raises(NotImplementedError, match="Ring backend"):
        _validate_tpr_cp_combination(
            2, SimpleNamespace(size=cp_size, backend_name="allgather"),
            config, ppo=True,
        )
