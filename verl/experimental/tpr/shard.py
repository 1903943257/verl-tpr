"""Historical shard constructors for experimental CP backends.

All tensor selection and offset semantics live in the production SequenceShard.
These adapters retain the old constructor/type checks without a second runtime.
"""

from verl.workers.engine.megatron.tpr.ring import (
    SequenceShard,
    _validate_integer,
    _validate_parallel_coordinates,
)


class _ContiguousType(type):
    def __instancecheck__(cls, instance):
        return super().__instancecheck__(instance) or (
            type(instance) is SequenceShard and len(instance.ranges) == 1
        )


class _RangeType(type):
    def __instancecheck__(cls, instance):
        return super().__instancecheck__(instance) or (
            type(instance) is SequenceShard and len(instance.ranges) > 1
        )


class _PaddedType(type):
    def __instancecheck__(cls, instance):
        return super().__instancecheck__(instance) or (
            type(instance) is SequenceShard and instance.padded_length > instance.global_length
        )


class PrefixShard(SequenceShard, metaclass=_ContiguousType):
    __slots__ = ()

    def __init__(self, global_length, local_start, local_end, cp_rank=0, cp_size=1):
        _validate_integer("global_length", global_length, minimum=1)
        _validate_parallel_coordinates(cp_rank, cp_size)
        if global_length % cp_size:
            raise ValueError(f"global_length {global_length} must be divisible by cp_size {cp_size}")
        local_length = global_length // cp_size
        if (local_start, local_end) != (cp_rank * local_length, (cp_rank + 1) * local_length):
            raise ValueError("contiguous shard range mismatch")
        super().__init__(global_length, ((local_start, local_end),), cp_rank, cp_size)

    @classmethod
    def full(cls, global_length):
        return cls(global_length, 0, global_length)

    @classmethod
    def contiguous(cls, global_length, *, cp_rank, cp_size):
        _validate_integer("global_length", global_length, minimum=1)
        _validate_parallel_coordinates(cp_rank, cp_size)
        local_length = global_length // cp_size
        return cls(global_length, cp_rank * local_length, (cp_rank + 1) * local_length, cp_rank, cp_size)

    @property
    def local_start(self):
        return self.ranges[0][0]

    @property
    def local_end(self):
        return self.ranges[0][1]


class RangeSequenceShard(SequenceShard, metaclass=_RangeType):
    __slots__ = ()


class PaddedSequenceShard(SequenceShard, metaclass=_PaddedType):
    __slots__ = ("physical_shard",)

    def __init__(self, logical_length, physical_shard):
        if not isinstance(physical_shard, SequenceShard):
            raise TypeError("physical_shard must be a SequenceShard")
        if logical_length >= physical_shard.global_length:
            raise ValueError("PaddedSequenceShard requires logical_length smaller than the physical length")
        super().__init__(
            logical_length, physical_shard.ranges, physical_shard.cp_rank,
            physical_shard.cp_size, physical_shard.global_length,
        )
        object.__setattr__(self, "physical_shard", physical_shard)

    @property
    def logical_length(self):
        return self.global_length

    def __reduce__(self):
        return type(self), (self.global_length, self.physical_shard)


def maybe_pad_sequence_shard(logical_length, physical_shard):
    return physical_shard if logical_length == physical_shard.global_length else PaddedSequenceShard(
        logical_length, physical_shard,
    )


def physical_sequence_shard(shard):
    if hasattr(shard, "physical_shard"):
        return shard.physical_shard
    if shard.global_length == shard.padded_length:
        return shard
    return SequenceShard(shard.padded_length, shard.ranges, shard.cp_rank, shard.cp_size)
