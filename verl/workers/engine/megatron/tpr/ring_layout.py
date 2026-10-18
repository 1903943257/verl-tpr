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


"""Ring CP logical/physical shards and block classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

SequenceRange = tuple[int, int]

def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _validate_parallel_coordinates(cp_rank: int, cp_size: int) -> None:
    _validate_integer("cp_size", cp_size, minimum=1)
    _validate_integer("cp_rank", cp_rank, minimum=0)
    if cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")


def _normalize_dimension(tensor: Tensor, dim: int) -> int:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"dim must be an integer, got {dim!r}")
    normalized = dim if dim >= 0 else tensor.ndim + dim
    if normalized < 0 or normalized >= tensor.ndim:
        raise IndexError(f"dimension {dim} is out of range for tensor with {tensor.ndim} dimensions")
    return normalized


def _validate_global_offset(global_offset: int, global_length: int) -> None:
    _validate_integer("global_offset", global_offset, minimum=0)
    if global_offset >= global_length:
        raise ValueError(f"global_offset must be in [0, {global_length}), got {global_offset}")


@dataclass(frozen=True, slots=True)
class SequenceShard:
    """A Segment's logical length and ordered physical ranges on one CP rank.

    CP1 and Ring use the same metadata. Tail padding stays in the physical
    ranges so every rank has equal-shaped tensors; logical ranges exclude it.
    """

    global_length: int
    ranges: tuple[SequenceRange, ...]
    cp_rank: int = 0
    cp_size: int = 1
    padded_length: int | None = None

    def __post_init__(self) -> None:
        _validate_integer("global_length", self.global_length, minimum=1)
        _validate_parallel_coordinates(self.cp_rank, self.cp_size)
        padded_length = self.global_length if self.padded_length is None else self.padded_length
        _validate_integer("padded_length", padded_length, minimum=self.global_length)
        ranges = tuple(tuple(item) for item in self.ranges)
        if not ranges:
            raise ValueError("ranges must contain at least one non-empty range")
        previous_end = 0
        for index, item in enumerate(ranges):
            if len(item) != 2:
                raise ValueError(f"ranges[{index}] must be a (start, end) pair, got {item!r}")
            start, end = item
            _validate_integer(f"ranges[{index}].start", start, minimum=0)
            _validate_integer(f"ranges[{index}].end", end, minimum=1)
            if start >= end or end > padded_length:
                raise ValueError(f"ranges[{index}] must be non-empty and within [0, {padded_length})")
            if index and start < previous_end:
                raise ValueError("ranges must be ordered and non-overlapping")
            previous_end = end
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "padded_length", padded_length)

    @classmethod
    def full(cls, global_length: int) -> SequenceShard:
        return cls(global_length, ((0, global_length),))

    @property
    def global_ranges(self) -> tuple[SequenceRange, ...]:
        return tuple((start, min(end, self.global_length)) for start, end in self.ranges if start < self.global_length)

    @property
    def local_length(self) -> int:
        return sum(end - start for start, end in self.ranges)

    @property
    def valid_local_length(self) -> int:
        return sum(end - start for start, end in self.global_ranges)

    @property
    def physical_global_ranges(self) -> tuple[SequenceRange, ...]:
        return self.ranges

    @property
    def is_full(self) -> bool:
        return self.valid_local_length == self.global_length

    def select(self, tensor: Tensor, *, dim: int = 0) -> Tensor:
        dim = _normalize_dimension(tensor, dim)
        if tensor.shape[dim] != self.global_length:
            raise ValueError(
                f"tensor sequence dimension must be {self.global_length}, got {tensor.shape[dim]}"
            )
        chunks = []
        for start, end in self.ranges:
            valid_start, valid_end = min(start, self.global_length), min(end, self.global_length)
            chunk = tensor.narrow(dim, valid_start, valid_end - valid_start)
            padding = end - start - (valid_end - valid_start)
            if padding:
                shape = list(tensor.shape)
                shape[dim] = padding
                # Retain an autograd edge even on ranks that own only padding.
                chunk = torch.cat((chunk, tensor.new_zeros(shape)), dim=dim)
            chunks.append(chunk)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=dim)

    def global_indices(self, *, device: torch.device | str | None = None) -> Tensor:
        return self.physical_global_indices(device=device).clamp_max(self.global_length - 1)

    def physical_global_indices(self, *, device: torch.device | str | None = None) -> Tensor:
        chunks = tuple(torch.arange(start, end, dtype=torch.long, device=device) for start, end in self.ranges)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def owns(self, global_offset: int) -> bool:
        return self.global_to_local(global_offset) is not None

    def global_to_local(self, global_offset: int) -> int | None:
        _validate_global_offset(global_offset, self.global_length)
        local_start = 0
        for start, end in self.ranges:
            if start <= global_offset < end:
                return local_start + global_offset - start
            local_start += end - start
        return None

    def local_to_global(self, local_offset: int) -> int:
        _validate_integer("local_offset", local_offset, minimum=0)
        if local_offset >= self.local_length:
            raise ValueError(f"local_offset must be in [0, {self.local_length}), got {local_offset}")
        for start, end in self.ranges:
            if local_offset < end - start:
                global_offset = start + local_offset
                if global_offset >= self.global_length:
                    raise ValueError(f"local offset {local_offset} refers to a padding token")
                return global_offset
            local_offset -= end - start
        raise AssertionError("validated local offset was not mapped")


def round_up_sequence_length(logical_length: int, alignment: int) -> int:
    """Return the smallest positive multiple of ``alignment`` covering a Segment."""

    _validate_integer("logical_length", logical_length, minimum=1)
    _validate_integer("alignment", alignment, minimum=1)
    return ((logical_length + alignment - 1) // alignment) * alignment


def iter_valid_sequence_slices(
    shard: SequenceShard,
) -> tuple[tuple[SequenceRange, slice], ...]:
    """Map each non-empty logical range to its slice in the local physical tensor."""

    result: list[tuple[SequenceRange, slice]] = []
    local_start = 0
    for physical_start, physical_end in shard.physical_global_ranges:
        physical_length = physical_end - physical_start
        valid_end = min(physical_end, shard.global_length)
        if physical_start < valid_end:
            valid_length = valid_end - physical_start
            result.append(((physical_start, valid_end), slice(local_start, local_start + valid_length)))
        local_start += physical_length
    if sum(item[0][1] - item[0][0] for item in result) != shard.valid_local_length:
        raise RuntimeError("valid sequence slices do not match shard validity metadata")
    return tuple(result)


def resolve_cp_group(group: Any) -> tuple[int, int]:
    """Resolve an existing distributed CP group without depending on AllGather."""
    if not dist.is_available() or not dist.is_initialized():
        if group is not None:
            raise RuntimeError("a CP process group requires torch.distributed initialization")
        return 1, 0
    return dist.get_world_size(group), dist.get_rank(group)


_group_world_size_and_rank = resolve_cp_group

class RingBlockKind(str, Enum):
    """Visibility of one Current query range against one KV range."""

    FULL = "full"
    CAUSAL = "causal"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class RingLocalKVBlock:
    """One Prefix segment's local zigzag KV shard for a single layer."""

    segment_id: int
    shard: SequenceShard
    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.segment_id, int)
            or isinstance(self.segment_id, bool)
            or self.segment_id < 0
        ):
            raise ValueError(
                f"segment_id must be a non-negative integer, got {self.segment_id!r}"
            )


@dataclass(frozen=True, slots=True)
class _RingAttentionConfig:
    cp_group: Any
    global_ranks: tuple[int, ...]
    cp_size: int
    cp_rank: int
    segment_lengths: tuple[int, ...]
    segment_padded_lengths: tuple[int, ...]
    current_shard: SequenceShard
    query_heads: int
    head_dim: int
    softmax_scale: float
    coalesce_prefix_full: bool = False
    coalesce_prefix_query: bool = False


@dataclass(frozen=True, slots=True)
class _RingRangeSlice:
    """One physical Ring chunk plus its clipped logical validity range."""

    logical_range: SequenceRange
    physical_range: SequenceRange
    local_slice: slice
    tensor: Tensor

    @property
    def physical_length(self) -> int:
        return self.physical_range[1] - self.physical_range[0]

    @property
    def valid_length(self) -> int:
        return self.logical_range[1] - self.logical_range[0]


def make_ring_sequence_shard(
    global_length: int,
    *,
    cp_rank: int,
    cp_size: int,
    padded_length: int | None = None,
) -> SequenceShard:
    """Return MindSpeed's symmetric two-range causal Ring shard."""

    if not isinstance(global_length, int) or isinstance(global_length, bool) or global_length <= 0:
        raise ValueError(f"global_length must be a positive integer, got {global_length!r}")
    if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size <= 1:
        raise ValueError(f"Ring CP requires cp_size greater than one, got {cp_size!r}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool) or not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank!r}")
    chunk_count = 2 * cp_size
    if padded_length is None:
        padded_length = round_up_sequence_length(global_length, chunk_count)
    if padded_length < global_length or padded_length % chunk_count != 0:
        raise ValueError(
            f"padded_length must cover {global_length} and be divisible by {chunk_count}, "
            f"got {padded_length}"
        )
    chunk_length = padded_length // chunk_count
    first_start = cp_rank * chunk_length
    second_start = (chunk_count - cp_rank - 1) * chunk_length
    return SequenceShard(
        global_length,
        (
            (first_start, first_start + chunk_length),
            (second_start, second_start + chunk_length),
        ),
        cp_rank=cp_rank,
        cp_size=cp_size,
        padded_length=padded_length,
    )


def classify_ring_block(
    query_range: SequenceRange,
    kv_range: SequenceRange,
    *,
    is_prefix: bool,
) -> RingBlockKind:
    """Classify FULL/CAUSAL/SKIP using logical Segment-local positions."""

    if is_prefix:
        return RingBlockKind.FULL
    query_start, query_end = query_range
    kv_start, kv_end = kv_range
    if kv_end <= query_start:
        return RingBlockKind.FULL
    if kv_start >= query_end:
        return RingBlockKind.SKIP
    if query_range != kv_range:
        raise ValueError(
            "partially overlapping Current query/KV ranges cannot use the aligned Ring schedule: "
            f"Q={query_range}, KV={kv_range}"
        )
    return RingBlockKind.CAUSAL


