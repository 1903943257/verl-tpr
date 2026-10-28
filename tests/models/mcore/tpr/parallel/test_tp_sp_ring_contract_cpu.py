# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU-only preflight for Megatron TP2+SP with TPR Ring CP2/CP4.

These tests check physical shape contracts and the supported TP2 x CP2 x DP2
topology. They do not certify distributed collectives, numerical gradients,
or E2E PPO behavior.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

# Importing a submodule under verl.workers.engine executes engine/__init__.py,
# which imports MindSpeed and parses process argv at import time. pytest's
# positional test path / flags can produce an empty MindSpeed config key, later
# breaking Megatron-Bridge TransformerConfig's generated dataclass __init__.
# pytest has already parsed its CLI, so temporarily hide those args only while
# the first engine/MindSpeed import is being performed.
_original_argv = sys.argv[:]
try:
    sys.argv[:] = [sys.argv[0]]
    from verl.workers.engine.megatron.tpr.adapter import (
        _validate_tpr_cp_combination,
        _validate_tpr_parallel_sizes,
    )
    from verl.workers.engine.megatron.tpr.executor import validate_replay_plan
    from verl.workers.engine.megatron.tpr.ring import make_ring_sequence_shard
    from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec
finally:
    sys.argv[:] = _original_argv


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
def test_sp_ring_accepts_without_experimental_opt_in(cp_size):
    runtime = SimpleNamespace(size=cp_size, backend_name="ring")
    config = SimpleNamespace(sequence_parallel=True)
    _validate_tpr_cp_combination(2, runtime, config, ppo=True)
    with pytest.raises(NotImplementedError, match="Ring backend"):
        _validate_tpr_cp_combination(
            2, SimpleNamespace(size=cp_size, backend_name="allgather"),
            config, ppo=True,
        )


def test_tp2_sp_ring_cp2_dp2_is_supported_by_topology_gate():
    sizes = {"TP": 2, "CP": 2, "DP": 2, "PP": 1, "EP": 1}
    assert _validate_tpr_parallel_sizes(sizes, ppo=True) == 2
    _validate_tpr_cp_combination(
        2, SimpleNamespace(size=2, backend_name="ring"),
        SimpleNamespace(sequence_parallel=True), ppo=True,
    )
