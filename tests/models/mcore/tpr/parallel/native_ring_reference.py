# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Explicit MindSpeed native Ring API bridge, with no TPR attention implementation."""

import math

import torch
import torch.distributed as dist


def native_cp_positions(length, rank, size, *, device):
    """Megatron causal CP's paired first/last chunks, independent of TPR sharding."""
    if length <= 0 or size < 2 or length % (2 * size) or not 0 <= rank < size:
        raise ValueError("Require unpadded paired causal CP chunks")
    chunks = torch.arange(length, device=device).reshape(2 * size, -1)
    return torch.cat((chunks[rank], chunks[2 * size - rank - 1]))


def native_ring_attention(query, key, value, *, cp_group, softmax_scale, implementation=None):
    """Call native ringattn_context_parallel; all streaming/merging/backward is native.

    SBH causal Ring, one inner rank and no cache/communication overlap. This
    deliberately fixes an explicit API configuration, not arbitrary training flags.
    """
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("Expected sequence,batch,heads,dimension Q/K/V")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise TypeError("Native Ring control requires BF16 Q/K/V")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1] or query.shape[2] % key.shape[2]:
        raise ValueError("Invalid native Ring GQA shapes")
    size, rank = dist.get_world_size(cp_group), dist.get_rank(cp_group)
    if size != 2 or query.shape[0] != 32:
        raise ValueError("Native reference is fixed at CP2 and global length 64")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Invalid native attention scale")
    if implementation is None:
        from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import (
            ringattn_context_parallel,
        )

        implementation = ringattn_context_parallel
    ranks = list(dist.get_process_group_ranks(cp_group))
    cp_para = {
        "causal": True,
        "cp_group": cp_group,
        "cp_size": size,
        "rank": rank,
        "cp_global_ranks": ranks,
        "cp_inner_ranks": [dist.get_rank()],
        "cp_outer_ranks": ranks,
        "cp_dkv_outer_ranks": ranks,
        "megatron_cp_in_bnsd": False,
        "cache_policy": None,
        "pse": None,
        "pse_type": 1,
        "cp_group_for_send_recv_overlap": None,
        "cp_group_for_intra_window": None,
        "cp_group_for_intra_window_send_recv_overlap": None,
    }
    q, k, v = (tensor.flatten(2).contiguous() for tensor in (query, key, value))
    output = implementation(q, k, v, query.shape[2], cp_para, softmax_scale=softmax_scale, dropout_p=0.0)
    if output.shape != q.shape or output.dtype != query.dtype:
        raise AssertionError("Unexpected native Ring output layout/dtype")
    return output
