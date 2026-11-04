# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in scalar diagnostics for delayed dKV accumulation and Replay.

Set TPR_DKV_REPLAY_DIAG_PATH to a JSONL path to enable. The helpers only
observe tensors; they never replace or modify gradients.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
from torch import Tensor


def _diagnostic_path() -> str | None:
    path = os.getenv("TPR_DKV_REPLAY_DIAG_PATH")
    return path if path else None


def _tensor_stats(tensor: Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    result: dict[str, Any] = {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "numel": detached.numel(),
        "device": str(detached.device),
    }
    if not detached.numel():
        result.update(norm=0.0, max_abs=0.0, finite=True)
        return result
    values = detached.float()
    result.update(
        norm=float(torch.linalg.vector_norm(values).item()),
        max_abs=float(values.abs().max().item()),
        finite=bool(torch.isfinite(values).all().item()),
    )
    return result


def _difference_stats(expected: Tensor, actual: Tensor) -> dict[str, Any]:
    if expected.shape != actual.shape:
        return {
            "shape_match": False,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
        }
    expected_fp32 = expected.detach().float()
    actual_fp32 = actual.detach().float()
    difference = actual_fp32 - expected_fp32
    expected_norm = float(torch.linalg.vector_norm(expected_fp32).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    return {
        "shape_match": True,
        "relative_l2": difference_norm / max(expected_norm, 1e-30),
        "max_abs": float(difference.abs().max().item()) if difference.numel() else 0.0,
    }


def _write_record(record: dict[str, Any]) -> None:
    path = _diagnostic_path()
    if path is None:
        return
    payload = {
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "rank": os.getenv("RANK", os.getenv("LOCAL_RANK", "0")),
        **record,
    }
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def record_dkv_accumulation(
    *,
    segment_id: object,
    layer: int,
    accumulation_count: int,
    incoming_key: Tensor,
    incoming_value: Tensor,
    buffer_key: Tensor,
    buffer_value: Tensor,
) -> None:
    """Record one descendant contribution and the resulting K/V buffers."""
    if _diagnostic_path() is None:
        return
    for kind, incoming, buffer in (
        ("key", incoming_key, buffer_key),
        ("value", incoming_value, buffer_value),
    ):
        _write_record(
            {
                "event": "dkv_accumulate",
                "segment_id": str(segment_id),
                "layer": layer,
                "kind": kind,
                "accumulation_count": accumulation_count,
                "incoming": _tensor_stats(incoming),
                "buffer": _tensor_stats(buffer),
            }
        )


def observe_dkv_replay_root(
    root: Tensor,
    seed_gradient: Tensor,
    *,
    segment_id: object,
    layer: int,
    kind: str,
) -> None:
    """Record the dKV passed to autograd and observe the gradient at its root."""
    if _diagnostic_path() is None:
        return
    metadata = {
        "replay_segment_id": str(segment_id),
        "layer": layer,
        "kind": kind,
    }
    _write_record(
        {
            "event": "dkv_replay_input",
            **metadata,
            "root": _tensor_stats(root),
            "seed_gradient": _tensor_stats(seed_gradient),
        }
    )
    if not root.requires_grad:
        return

    def _record_actual_gradient(actual_gradient: Tensor) -> None:
        _write_record(
            {
                "event": "dkv_replay_root_hook",
                **metadata,
                "seed_gradient": _tensor_stats(seed_gradient),
                "actual_gradient": _tensor_stats(actual_gradient),
                "seed_to_hook": _difference_stats(seed_gradient, actual_gradient),
            }
        )
        return None

    root.register_hook(_record_actual_gradient)
