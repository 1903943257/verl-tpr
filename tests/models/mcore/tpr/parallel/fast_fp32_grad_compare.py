# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Fast bounded-memory exact FP64 metrics for saved FP32 gradients on CPU.

The tensors are loaded with torch.load(..., map_location="cpu", mmap=True)
by the caller. Compute every norm/difference in FP64 NumPy blocks, without
hundreds of thousands of tiny PyTorch CPU dispatcher calls.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor


def fast_fp32_tensor_difference(
    reference: Tensor,
    candidate: Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    block_elements: int = 4 * 1024 * 1024,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise TypeError("Saved FP32 gradient comparator requires torch.float32 tensors")
    return fast_tensor_difference(
        reference, candidate, atol=atol, rtol=rtol, block_elements=block_elements, progress=progress
    )


def fast_tensor_difference(
    reference: Tensor,
    candidate: Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    block_elements: int = 4 * 1024 * 1024,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """CPU FP64 metrics; BF16 is promoted exactly one bounded block at a time."""
    if reference.shape != candidate.shape:
        raise ValueError(f"Comparison shape mismatch: {reference.shape} / {candidate.shape}")
    if reference.dtype not in (torch.float32, torch.bfloat16) or candidate.dtype != reference.dtype:
        raise TypeError("Gradient comparator requires matching FP32 or BF16 tensors")
    if reference.device.type != "cpu" or candidate.device.type != "cpu":
        raise ValueError("Saved FP32 comparator must run entirely on CPU")
    if block_elements <= 0:
        raise ValueError("block_elements must be positive")

    # These arrays are views of the CPU tensors (including mmap-backed storage),
    # not 4B-parameter floating-point copies.
    a = reference.detach().reshape(-1)
    b = candidate.detach().reshape(-1)
    aa = bb = dd = dot = maximum = 0.0
    close = True
    n = a.numel()
    for start in range(0, n, block_elements):
        end = min(start + block_elements, n)
        # float64 arithmetic preserves the original comparator's precision.
        x = a[start:end].float().numpy().astype(np.float64)
        y = b[start:end].float().numpy().astype(np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise FloatingPointError("Nonfinite attribution tensor")
        delta = y - x
        absolute = np.abs(delta)
        aa += float(np.einsum("i,i->", x, x, optimize=False))
        bb += float(np.einsum("i,i->", y, y, optimize=False))
        dd += float(np.einsum("i,i->", delta, delta, optimize=False))
        dot += float(np.einsum("i,i->", x, y, optimize=False))
        maximum = max(maximum, float(absolute.max()))
        if close:
            close = bool(np.all(absolute <= atol + rtol * np.abs(x)))
        if progress is not None:
            progress(end, n)
    return {
        "elements": int(n),
        "reference_norm": math.sqrt(aa),
        "candidate_norm": math.sqrt(bb),
        "difference_norm": math.sqrt(dd),
        "relative_l2": math.sqrt(dd / aa) if aa else (0.0 if not dd else None),
        "cosine": dot / math.sqrt(aa * bb) if aa and bb else (1.0 if aa == bb else None),
        "max_abs": maximum,
        "within_tolerance": close,
    }
