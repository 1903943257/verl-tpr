# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""CANN fused attention primitives and FP32 online softmax for Ring CP."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from verl.models.mcore.tpr.rectangular_attention import _compressed_causal_mask
from .ring_layout import RingBlockKind, _RingRangeSlice

_TND_LAYOUT = "TND"
_RIGHT_DOWN_CAUSAL_MODE = 3
_FULL_ATTENTION_MODE = 0
_MAX_TOKENS = 2**31 - 1

def _range_validity(item: _RingRangeSlice, *, device: torch.device) -> Tensor:
    return torch.arange(item.physical_length, device=device) < item.valid_length


def _physical_block_attention_mask(
    query_item: _RingRangeSlice,
    kv_item: _RingRangeSlice,
    *,
    block_kind: RingBlockKind,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Return a physical block mask plus optional query/KV validity vectors."""

    if block_kind is RingBlockKind.SKIP:
        raise ValueError("SKIP blocks must not enter fused attention")
    query_has_padding = query_item.valid_length != query_item.physical_length
    kv_has_padding = kv_item.valid_length != kv_item.physical_length
    query_validity = (
        _range_validity(query_item, device=query_item.tensor.device)
        if query_has_padding
        else None
    )
    kv_validity = (
        _range_validity(kv_item, device=kv_item.tensor.device)
        if kv_has_padding
        else None
    )
    if block_kind is RingBlockKind.FULL:
        if kv_validity is None:
            return None, query_validity, None
        attention_mask = torch.logical_not(kv_validity).unsqueeze(0).expand(
            query_item.physical_length,
            -1,
        )
        return attention_mask.contiguous(), query_validity, kv_validity

    if not query_has_padding and not kv_has_padding:
        return None, None, None
    query_start, query_end = query_item.logical_range
    query_physical_end = query_item.physical_range[1]
    kv_start, kv_end = kv_item.logical_range
    kv_physical_end = kv_item.physical_range[1]
    query_positions = torch.arange(
        query_start,
        query_physical_end,
        device=query_item.tensor.device,
    ).clamp_max(query_end - 1)
    kv_positions = torch.arange(
        kv_start,
        kv_physical_end,
        device=kv_item.tensor.device,
    ).clamp_max(kv_end - 1)
    if kv_validity is None:
        kv_validity = torch.ones(
            kv_item.physical_length,
            dtype=torch.bool,
            device=kv_item.tensor.device,
        )
    attention_mask = torch.logical_or(
        torch.logical_not(kv_validity).unsqueeze(0),
        kv_positions.unsqueeze(0) > query_positions.unsqueeze(1),
    )
    return attention_mask.contiguous(), query_validity, (
        kv_validity if kv_has_padding else None
    )


def _block_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_heads: int,
    softmax_scale: float,
    block_kind: RingBlockKind,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - requires the server NPU runtime
        raise RuntimeError("torch_npu is required for Ring CP attention") from exc
    causal_fast_path = block_kind is RingBlockKind.CAUSAL and attention_mask is None
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise ValueError("Ring physical attention mask must use torch.bool")
        if tuple(attention_mask.shape) != (query.shape[0], key.shape[0]):
            raise ValueError(
                "Ring physical attention mask shape must match Q/KV, got "
                f"{tuple(attention_mask.shape)} for Q={query.shape[0]}, KV={key.shape[0]}"
            )
        attention_mask = attention_mask.contiguous()
    result = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        query_heads,
        _TND_LAYOUT,
        pse=None,
        padding_mask=None,
        atten_mask=(
            _compressed_causal_mask(query.device)
            if causal_fast_path
            else attention_mask
        ),
        scale=softmax_scale,
        pre_tockens=_MAX_TOKENS,
        next_tockens=0 if causal_fast_path else _MAX_TOKENS,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=(
            _RIGHT_DOWN_CAUSAL_MODE
            if causal_fast_path
            else _FULL_ATTENTION_MODE
        ),
        actual_seq_qlen=[query.shape[0]],
        actual_seq_kvlen=[key.shape[0]],
    )
    return result[0], result[1], result[2]


def _merge_attention(
    previous: tuple[Tensor, Tensor, Tensor] | None,
    current: tuple[Tensor, Tensor, Tensor],
    *,
    query_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    # Keep the online-softmax accumulator FP32, including the first block.
    # Casting after every merge repeatedly rounds the partial context in BF16.
    current = tuple(t.float() for t in current)
    if previous is None:
        return current
    previous_output, previous_max, previous_sum = (t.float() for t in previous)
    current_output, current_max, current_sum = current
    actual_seq_qlen = (query_length,)

    # MindSpeed 376e9cc's TND unflatten_softmax uses view() immediately
    # after transpose(), which fails on current PyTorch because that tensor is
    # non-contiguous. Keep its exact layout semantics locally and use reshape
    # at the non-contiguous boundary instead of patching MindSpeed globally.
    previous_max = _flatten_tnd_softmax(previous_max, actual_seq_qlen)
    previous_sum = _flatten_tnd_softmax(previous_sum, actual_seq_qlen)
    current_max = _flatten_tnd_softmax(current_max, actual_seq_qlen)
    current_sum = _flatten_tnd_softmax(current_sum, actual_seq_qlen)

    merged_max = torch.maximum(previous_max, current_max)
    previous_scale = torch.exp(previous_max - merged_max)
    current_scale = torch.exp(current_max - merged_max)
    previous_sum = previous_sum * previous_scale
    current_sum = current_sum * current_scale
    merged_sum = previous_sum + current_sum

    head_dim = previous_output.shape[-1]
    previous_output_scale = (previous_sum / merged_sum)[..., 0].unsqueeze(2)
    current_output_scale = (current_sum / merged_sum)[..., 0].unsqueeze(2)
    previous_output_scale = previous_output_scale.expand(-1, -1, head_dim)
    current_output_scale = current_output_scale.expand(-1, -1, head_dim)
    merged_output = previous_output * previous_output_scale
    merged_output.add_(current_output * current_output_scale)
    return (
        merged_output,
        _unflatten_tnd_softmax(merged_max, actual_seq_qlen),
        _unflatten_tnd_softmax(merged_sum, actual_seq_qlen),
    )


def _finalize_attention_result(result, *, dtype):
    """One final output cast shared by the caller and fused backward save."""
    output, maximum, total = result
    return output.to(dtype).contiguous(), maximum, total


def _flatten_tnd_softmax(
    tensor: Tensor,
    actual_seq_qlen: Sequence[int],
) -> Tensor:
    original_shape = tensor.shape
    section_lengths = tuple(length * original_shape[1] for length in actual_seq_qlen)
    flattened = tensor.reshape(-1, original_shape[-1])
    if sum(section_lengths) != flattened.shape[0]:
        raise ValueError(
            "softmax statistics do not match actual_seq_qlen: "
            f"shape={tuple(original_shape)}, lengths={tuple(actual_seq_qlen)}"
        )
    sections = flattened.split(section_lengths, dim=0)
    reordered = tuple(
        section.reshape(original_shape[1], -1, original_shape[-1]).transpose(0, 1)
        for section in sections
    )
    return torch.cat(reordered, dim=0)


def _unflatten_tnd_softmax(
    tensor: Tensor,
    actual_seq_qlen: Sequence[int],
) -> Tensor:
    original_shape = tensor.shape
    section_lengths = tuple(length * original_shape[1] for length in actual_seq_qlen)
    flattened = tensor.reshape(-1, original_shape[-1])
    if sum(section_lengths) != flattened.shape[0]:
        raise ValueError(
            "softmax statistics do not match actual_seq_qlen: "
            f"shape={tuple(original_shape)}, lengths={tuple(actual_seq_qlen)}"
        )
    sections = flattened.split(section_lengths, dim=0)
    reordered = tuple(
        section.reshape(-1, original_shape[1], original_shape[-1])
        .transpose(0, 1)
        .reshape(-1, original_shape[-1])
        for section in sections
    )
    return torch.cat(reordered, dim=0).reshape(original_shape)


def _block_attention_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    attention_output: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    query_heads: int,
    softmax_scale: float,
    block_kind: RingBlockKind,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - requires the server NPU runtime
        raise RuntimeError("torch_npu is required for Ring CP attention backward") from exc
    causal_fast_path = block_kind is RingBlockKind.CAUSAL and attention_mask is None
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise ValueError("Ring physical attention mask must use torch.bool")
        if tuple(attention_mask.shape) != (query.shape[0], key.shape[0]):
            raise ValueError(
                "Ring physical attention mask shape must match Q/KV, got "
                f"{tuple(attention_mask.shape)} for Q={query.shape[0]}, KV={key.shape[0]}"
            )
        attention_mask = attention_mask.contiguous()
    result = torch_npu.npu_fusion_attention_grad(
        query,
        key,
        value,
        grad_output,
        query_heads,
        _TND_LAYOUT,
        pse=None,
        padding_mask=None,
        atten_mask=(
            _compressed_causal_mask(query.device)
            if causal_fast_path
            else attention_mask
        ),
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attention_output,
        scale_value=softmax_scale,
        pre_tockens=_MAX_TOKENS,
        next_tockens=0 if causal_fast_path else _MAX_TOKENS,
        keep_prob=1.0,
        sparse_mode=(
            _RIGHT_DOWN_CAUSAL_MODE
            if causal_fast_path
            else _FULL_ATTENTION_MODE
        ),
        actual_seq_qlen=[query.shape[0]],
        actual_seq_kvlen=[key.shape[0]],
    )
    return result[0], result[1], result[2]


