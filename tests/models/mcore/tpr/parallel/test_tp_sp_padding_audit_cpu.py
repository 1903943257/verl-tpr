# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Temporary CPU audit of the Ring CP + native TP sequence-parallel shape contract.

This is a layout/preflight witness, NOT a CANN/HCCL/NPU or gradient test.
Keep on the audit branch; do not merge unvalidated production changes.

Run from a VERL environment that has the project and MindSpeed installed:
    python -m pytest -q -s tests/models/mcore/tpr/parallel/test_tp_sp_padding_audit_cpu.py
"""
from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest
import torch

# Importing through verl.workers.engine executes its MindSpeed initialization.
# Match the established test_tp_sp_ring_contract_cpu.py import-time guard.
_original_argv = sys.argv[:]
try:
    sys.argv[:] = [sys.argv[0]]
    from verl.workers.engine.megatron.tpr.executor import validate_replay_plan
    from verl.workers.engine.megatron.tpr.ring import make_ring_sequence_shard
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


def _prepared(tp: int):
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                tensor_model_parallel_size=tp,
                sequence_parallel=True,
            )
        )
    )


@pytest.mark.parametrize(
    ("tp", "cp", "length", "expected_padded", "expected_local", "aligned"),
    [
        (2, 2, 9, 12, 6, True),
        (2, 4, 9, 16, 4, True),
        (2, 2, 1, 4, 2, True),
        (4, 2, 9, 12, 6, False),  # Codex witness: SP guard would reject 6 % 4
        (4, 2, 13, 16, 8, True),
        (4, 2, 1, 4, 2, False),
        (4, 4, 9, 16, 4, True),
        (8, 2, 9, 12, 6, False),
        (8, 4, 9, 16, 4, False),
        (8, 4, 25, 32, 8, True),
    ],
)
def test_current_ring_sp_shape_and_preflight(tp, cp, length, expected_padded, expected_local, aligned):
    shards = [
        make_ring_sequence_shard(length, cp_rank=rank, cp_size=cp)
        for rank in range(cp)
    ]
    assert all(shard.padded_length == expected_padded for shard in shards)
    assert all(shard.local_length == expected_local for shard in shards)
    assert all(shard.padded_length % (2 * cp) == 0 for shard in shards)

    actual_aligned = (expected_local % tp) == 0
    assert actual_aligned is aligned
    if aligned:
        validate_replay_plan(_plan(length), _prepared(tp), cp_size=cp)
    else:
        # verl-tpr/main has an early rejection. The current Recipe preflight
        # checks CP1 only, so it misses these CP>1 shape failures.
        with pytest.raises(ValueError, match="CP-local physical_length=.*must be divisible"):
            validate_replay_plan(_plan(length), _prepared(tp), cp_size=cp)

    print(
        f"TP_SP_WITNESS tp={tp} cp={cp} logical={length} "
        f"padded={expected_padded} local={expected_local} "
        f"sp_aligned={actual_aligned}",
        flush=True,
    )


@pytest.mark.parametrize(("tp", "cp", "length"), [(2, 2, 9), (4, 2, 9), (8, 4, 9), (4, 4, 9)])
def test_proposed_joint_alignment_contract(tp, cp, length):
    """The minimum safe global alignment for a future TP+SP-aware Ring pad."""
    required = math.lcm(2 * cp, cp * tp)
    joint_padded = ((length + required - 1) // required) * required
    assert joint_padded % (2 * cp) == 0
    assert (joint_padded // cp) % tp == 0
    print(
        f"TP_SP_TARGET tp={tp} cp={cp} logical={length} "
        f"current_align={2 * cp} joint_align={required} "
        f"joint_padded={joint_padded}",
        flush=True,
    )


@pytest.mark.parametrize(("tp", "length"), [(2, 9), (4, 9), (2, 10), (4, 12)])
def test_cp1_sequence_parallel_has_no_automatic_padding(tp, length):
    """CP1 uses the unpadded segment and rejects unaligned TP+SP lengths."""
    aligned = length % tp == 0
    if aligned:
        validate_replay_plan(_plan(length), _prepared(tp), cp_size=1)
    else:
        with pytest.raises(ValueError, match="CP-local physical_length=.*must be divisible"):
            validate_replay_plan(_plan(length), _prepared(tp), cp_size=1)
    print(f"TP_SP_CP1 tp={tp} logical={length} aligned={aligned}", flush=True)
