"""Compatibility contracts for heterogeneous experimental prefix states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from verl.workers.engine.megatron.tpr.ring import SequenceShard
from verl.workers.engine.megatron.tpr.state import (
    KVPrefixAnchors,
    KVPrefixState,
    KVStack,
    PastKVAnchors,
    _compact_graph_free_tensor,
    _normalize_local_kv,
    _validate_layer_number,
    _validate_segment_id,
)
from verl.workers.engine.megatron.tpr.tree import SegmentId, SegmentSpec


@runtime_checkable
class PrefixState(Protocol):
    segment_id: SegmentId
    state_kind: str
    shard: SequenceShard
    global_length: int
    local_length: int
    layer_numbers: tuple[int, ...]
    released: bool

    def validate(self) -> None: ...
    def make_anchors(self) -> Any: ...
    def accumulate_anchor_gradients(self, anchors: Any) -> None: ...
    def consume_gradients(self) -> Any: ...
    def release(self) -> None: ...


@dataclass(slots=True)
class PrefixStateEntry:
    segment: SegmentSpec
    state: PrefixState


class PrefixStateStack(KVStack):
    """Reuse path topology operations; only experimental entries are generic."""

    def push_state(self, segment: SegmentSpec, state: PrefixState) -> PrefixStateEntry:
        if not isinstance(state, PrefixState):
            raise TypeError(f"state must implement PrefixState, got {type(state).__name__}")
        state.validate()
        if state.segment_id != segment.segment_id:
            raise ValueError(f"state segment_id {state.segment_id} does not match segment {segment.segment_id}")
        if state.global_length != segment.length:
            raise ValueError(f"segment {segment.segment_id} state global length must be {segment.length}")
        if segment.segment_id in self._entries:
            raise RuntimeError(f"segment {segment.segment_id} is already on the prefix-state stack")
        parent = self.top().segment.segment_id if self._entries else None
        if segment.parent_id != parent:
            raise ValueError(f"cannot push segment {segment.segment_id}: expected parent {parent}")
        if segment.prefix_length != self.prefix_length or segment.position_start != self.prefix_length:
            raise ValueError(f"cannot push segment {segment.segment_id}: stack prefix length is {self.prefix_length}")
        entry = PrefixStateEntry(segment, state)
        self._entries[segment.segment_id] = entry
        return entry

    pop_state = KVStack.pop


@dataclass(frozen=True, slots=True)
class PastKVSlice:
    segment_id: SegmentId
    start: int
    end: int


SegmentKV = KVPrefixState
ShardedPastKVAnchors = PastKVAnchors


def build_sharded_past_anchors(kv_stack: KVStack) -> PastKVAnchors:
    return kv_stack.build_past_anchors()


def accumulate_sharded_past_anchor_gradients(kv_stack: KVStack, anchors: PastKVAnchors) -> None:
    kv_stack.accumulate_anchor_gradients(anchors)
