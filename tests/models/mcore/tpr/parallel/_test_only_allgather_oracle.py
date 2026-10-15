"""Test-only differentiable CP reference, not a production CP backend."""
from __future__ import annotations

from typing import Any
import torch
import torch.distributed as dist
from torch import Tensor

def _group_world_size_and_rank(process_group: Any) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        if process_group is not None:
            raise RuntimeError("a CP process group requires torch.distributed to be initialized")
        return 1, 0
    return dist.get_world_size(process_group), dist.get_rank(process_group)


def _all_gather_into_tensor(output: Tensor, input_: Tensor, process_group: Any) -> None:
    dist.all_gather_into_tensor(output, input_, group=process_group)


def _reduce_scatter_tensor(output: Tensor, input_: Tensor, process_group: Any) -> None:
    dist.reduce_scatter_tensor(output, input_, op=dist.ReduceOp.SUM, group=process_group)


class _AllGatherSequence(torch.autograd.Function):
    """AllGather on sequence in forward and SUM ReduceScatter in backward."""

    @staticmethod
    def forward(ctx, local_tensor: Tensor, process_group: Any) -> Tensor:
        world_size, _ = _group_world_size_and_rank(process_group)
        ctx.process_group = process_group
        ctx.world_size = world_size
        if world_size == 1:
            return local_tensor

        output_shape = list(local_tensor.shape)
        output_shape[0] *= world_size
        output = torch.empty(
            output_shape,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            memory_format=torch.contiguous_format,
        )
        _all_gather_into_tensor(output, local_tensor.contiguous(), process_group)
        return output

    @staticmethod
    def backward(ctx, global_gradient: Tensor) -> tuple[Tensor, None]:
        if ctx.world_size == 1:
            return global_gradient, None
        if global_gradient.shape[0] % ctx.world_size != 0:
            raise RuntimeError(
                f"global gradient sequence length {global_gradient.shape[0]} "
                f"is not divisible by CP size {ctx.world_size}"
            )

        output_shape = list(global_gradient.shape)
        output_shape[0] //= ctx.world_size
        local_gradient = torch.empty(
            output_shape,
            dtype=global_gradient.dtype,
            device=global_gradient.device,
            memory_format=torch.contiguous_format,
        )
        _reduce_scatter_tensor(local_gradient, global_gradient.contiguous(), ctx.process_group)
        return local_gradient, None


def all_gather_sequence(local_tensor: Tensor, process_group: Any) -> Tensor:
    """Gather uniform contiguous sequence shards with an autograd inverse."""

    if not isinstance(local_tensor, Tensor):
        raise TypeError(f"local_tensor must be a torch.Tensor, got {type(local_tensor).__name__}")
    if local_tensor.ndim == 0 or local_tensor.shape[0] <= 0:
        raise ValueError(f"local_tensor must have a non-empty sequence dimension, got {local_tensor.shape}")
    return _AllGatherSequence.apply(local_tensor, process_group)
