# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Attention-only BF16 ablations; every surrounding model operation is unchanged."""

import math

import torch

from .split_fp32_attribution import math_attention

ATTENTION_MODES = ("cann", "math_bf16", "math_fp32")


def math_bf16_attention(query, key, value, *, softmax_scale=None, dropout_p=0.0):
    """BF16 QK/PV outputs, FP32 softmax reduction, then BF16 probabilities."""
    from verl.models.mcore.tpr.rectangular_attention import _validate_inputs

    sq, sk, hq, dim = _validate_inputs(query, key, value, dropout_p)
    if query.dtype != torch.bfloat16:
        raise TypeError("BF16 math control requires BF16 Q/K/V")
    groups = hq // key.shape[2]
    q = query.permute(1, 2, 0, 3)
    k = key.permute(1, 2, 0, 3).repeat_interleave(groups, dim=1)
    v = value.permute(1, 2, 0, 3).repeat_interleave(groups, dim=1)
    scale = 1 / math.sqrt(dim) if softmax_scale is None else softmax_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid attention scale")
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    positions = torch.arange(sq, device=query.device) + sk - sq
    mask = torch.arange(sk, device=query.device)[None, :] > positions[:, None]
    probabilities = scores.float().masked_fill(mask, -torch.inf).softmax(dim=-1).to(query.dtype)
    output = torch.matmul(probabilities, v)
    return output.permute(2, 0, 1, 3).reshape(sq, query.shape[1], hq * dim)


def math_fp32_attention(query, key, value, **kwargs):
    """Promote already-rounded BF16 Q/K/V; cast the result back before projection."""
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise TypeError("Local FP32 attention ablation requires BF16 Q/K/V at its boundary")
    return math_attention(query.float(), key.float(), value.float(), **kwargs).to(query.dtype)


def attention_implementation(mode, cann):
    if mode == "cann":
        return cann
    if mode == "math_bf16":
        return math_bf16_attention
    if mode == "math_fp32":
        return math_fp32_attention
    raise ValueError(f"Unknown attention precision mode: {mode}")
