# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Lower Turn token tries to executable TPR plans and logical PPO token references.

The topology never owns PPO values or normalization factors. A response token
at absolute position t is predicted by the query at t-1, which may belong to
the *parent* segment when t is the first token of a child.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor


SegmentId = int


def _validate_segment_id(segment_id: SegmentId, *, field_name: str = "segment_id") -> None:
    if not isinstance(segment_id, int) or isinstance(segment_id, bool) or segment_id < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {segment_id!r}")


@dataclass(frozen=True, slots=True)
class SegmentLossTerm:
    """One weighted next-token loss owned by a segment query position.

    Multiple terms may use the same ``query_offset``. This represents a
    branching position whose logit predicts the first token of several child
    segments without retaining a separate logit-gradient relay buffer.
    """

    query_offset: int
    target_token_id: int
    weight: float = 1.0
    sample_id: int | None = None

    def __post_init__(self) -> None:
        _validate_segment_id(self.query_offset, field_name="query_offset")
        _validate_segment_id(self.target_token_id, field_name="target_token_id")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool) or not math.isfinite(self.weight):
            raise ValueError(f"weight must be finite, got {self.weight!r}")
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight!r}")
        if self.sample_id is not None:
            _validate_segment_id(self.sample_id, field_name="sample_id")


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """An immutable description of one non-empty contiguous token segment."""

    segment_id: SegmentId
    parent_id: SegmentId | None
    token_ids: Tensor
    position_start: int
    prefix_length: int
    loss_terms: tuple[SegmentLossTerm, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_segment_id(self.segment_id)
        if self.parent_id is not None:
            _validate_segment_id(self.parent_id, field_name="parent_id")
            if self.parent_id == self.segment_id:
                raise ValueError(f"segment {self.segment_id} cannot be its own parent")
        if not isinstance(self.token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if self.token_ids.device.type != "cpu" or self.token_ids.dtype != torch.long or self.token_ids.ndim != 1:
            raise ValueError(
                "token_ids must be a one-dimensional CPU torch.long tensor, "
                f"got shape={tuple(self.token_ids.shape)}, dtype={self.token_ids.dtype}, device={self.token_ids.device}"
            )
        if self.token_ids.numel() == 0:
            raise ValueError("token_ids must contain at least one token")
        if torch.any(self.token_ids < 0).item():
            raise ValueError("token_ids must be non-negative")
        _validate_segment_id(self.position_start, field_name="position_start")
        _validate_segment_id(self.prefix_length, field_name="prefix_length")

        normalized_terms = tuple(self.loss_terms)
        for term in normalized_terms:
            if not isinstance(term, SegmentLossTerm):
                raise TypeError(f"loss_terms must contain SegmentLossTerm objects, got {type(term).__name__}")
            if term.query_offset >= self.length:
                raise ValueError(
                    f"segment {self.segment_id} loss query_offset {term.query_offset} is outside length {self.length}"
                )
        object.__setattr__(self, "loss_terms", normalized_terms)

        # Isolate the plan from later mutation of the caller's tensor. Tensor
        # identity is not part of the scheduler protocol.
        object.__setattr__(self, "token_ids", self.token_ids.detach().clone())

    @property
    def length(self) -> int:
        return self.token_ids.numel()

    @property
    def position_end(self) -> int:
        """Exclusive absolute position at the end of this segment."""

        return self.position_start + self.length


class SegmentEventKind(str, Enum):
    PUSH = "push"
    POP = "pop"


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    kind: SegmentEventKind
    segment_id: SegmentId

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SegmentEventKind):
            raise TypeError(f"kind must be SegmentEventKind, got {type(self.kind).__name__}")
        _validate_segment_id(self.segment_id)


def PushSegment(segment_id: SegmentId) -> SegmentEvent:  # noqa: N802
    """Create an event that enters and forwards a segment."""

    return SegmentEvent(SegmentEventKind.PUSH, segment_id)


def PopSegment(segment_id: SegmentId) -> SegmentEvent:  # noqa: N802
    """Create an event that recomputes/backwards and releases a segment."""

    return SegmentEvent(SegmentEventKind.POP, segment_id)


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """Validated rooted segment tree stored entirely as CPU metadata."""

    segments: Mapping[SegmentId, SegmentSpec] | Sequence[SegmentSpec]
    root_id: SegmentId
    total_loss_weight: float | None = None
    topology_only: bool = False
    _children: Mapping[SegmentId, tuple[SegmentId, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_segment_id(self.root_id, field_name="root_id")
        source: Iterable[SegmentSpec]
        source = self.segments.values() if isinstance(self.segments, Mapping) else self.segments

        normalized: dict[SegmentId, SegmentSpec] = {}
        for segment in source:
            if not isinstance(segment, SegmentSpec):
                raise TypeError(f"segments must contain SegmentSpec objects, got {type(segment).__name__}")
            if segment.segment_id in normalized:
                raise ValueError(f"duplicate segment_id {segment.segment_id}")
            normalized[segment.segment_id] = segment
        if not normalized:
            raise ValueError("segments must not be empty")
        if self.root_id not in normalized:
            raise ValueError(f"root_id {self.root_id} does not exist")

        roots = [segment.segment_id for segment in normalized.values() if segment.parent_id is None]
        if roots != [self.root_id]:
            raise ValueError(f"plan must have exactly root {self.root_id}, got roots={sorted(roots)}")

        children: dict[SegmentId, list[SegmentId]] = {segment_id: [] for segment_id in normalized}
        for segment in normalized.values():
            if segment.parent_id is None:
                if segment.prefix_length != 0 or segment.position_start != 0:
                    raise ValueError("root segment must have prefix_length=0 and position_start=0")
                continue
            if segment.parent_id not in normalized:
                raise ValueError(f"segment {segment.segment_id} has missing parent {segment.parent_id}")
            parent = normalized[segment.parent_id]
            if segment.prefix_length != parent.position_end or segment.position_start != parent.position_end:
                raise ValueError(
                    f"segment {segment.segment_id} must start at parent {parent.segment_id} end {parent.position_end}, "
                    f"got prefix_length={segment.prefix_length}, position_start={segment.position_start}"
                )
            children[segment.parent_id].append(segment.segment_id)

        # Every non-root has an existing parent at a strictly smaller position
        # (segments are non-empty). Thus cycles and disconnected components
        # are impossible once the unique root and positions above validate.
        inferred_weight = sum(term.weight for segment in normalized.values() for term in segment.loss_terms)
        if self.topology_only:
            # PPO owns its objective/denominator outside the topology. Do not
            # invent a fake CE normalizer for purely structural segment plans.
            if any(segment.loss_terms for segment in normalized.values()) or self.total_loss_weight is not None:
                raise ValueError("topology_only plans must not carry CE loss terms or total_loss_weight")
            total_loss_weight = None
        else:
            total_loss_weight = inferred_weight if self.total_loss_weight is None else self.total_loss_weight
            if (
                not isinstance(total_loss_weight, (int, float))
                or isinstance(total_loss_weight, bool)
                or not math.isfinite(total_loss_weight)
                or total_loss_weight <= 0
            ):
                raise ValueError(f"total_loss_weight must be finite and positive, got {total_loss_weight!r}")

        object.__setattr__(self, "segments", MappingProxyType(normalized))
        object.__setattr__(self, "_children", MappingProxyType({key: tuple(value) for key, value in children.items()}))
        object.__setattr__(
            self, "total_loss_weight", None if total_loss_weight is None else float(total_loss_weight)
        )

    def get(self, segment_id: SegmentId) -> SegmentSpec:
        try:
            return self.segments[segment_id]
        except KeyError as exc:
            raise KeyError(f"unknown segment_id {segment_id}") from exc

    def parent_of(self, segment_id: SegmentId) -> SegmentSpec | None:
        segment = self.get(segment_id)
        return None if segment.parent_id is None else self.segments[segment.parent_id]

    def children_of(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        self.get(segment_id)
        return tuple(self.segments[child_id] for child_id in self._children[segment_id])

    def path_to(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        path: list[SegmentSpec] = []
        current: SegmentSpec | None = self.get(segment_id)
        while current is not None:
            path.append(current)
            current = None if current.parent_id is None else self.segments[current.parent_id]
        return tuple(reversed(path))

    def depth_of(self, segment_id: SegmentId) -> int:
        return len(self.path_to(segment_id)) - 1

    def dfs_events(self) -> Iterator[SegmentEvent]:
        """Yield a deterministic child-insertion-order DFS Push/Pop stream."""

        def traverse(segment_id: SegmentId) -> Iterator[SegmentEvent]:
            yield PushSegment(segment_id)
            for child_id in self._children[segment_id]:
                yield from traverse(child_id)
            yield PopSegment(segment_id)

        yield from traverse(self.root_id)

    def validate_events(self, events: Iterable[SegmentEvent]) -> tuple[SegmentEvent, ...]:
        """Validate stack discipline and exactly-once execution for a plan."""

        normalized = tuple(events)
        stack: list[SegmentId] = []
        pushed: set[SegmentId] = set()
        popped: set[SegmentId] = set()
        for event in normalized:
            if not isinstance(event, SegmentEvent):
                raise TypeError(f"events must contain SegmentEvent objects, got {type(event).__name__}")
            segment = self.get(event.segment_id)
            if event.kind is SegmentEventKind.PUSH:
                if event.segment_id in pushed:
                    raise ValueError(f"segment {event.segment_id} is pushed more than once")
                expected_parent = stack[-1] if stack else None
                if segment.parent_id != expected_parent:
                    raise ValueError(
                        f"cannot push segment {event.segment_id}: expected parent on stack {segment.parent_id}, "
                        f"got {expected_parent}"
                    )
                stack.append(event.segment_id)
                pushed.add(event.segment_id)
            else:
                if not stack or stack[-1] != event.segment_id:
                    raise ValueError(f"cannot pop segment {event.segment_id}: current stack is {stack}")
                if any(child_id not in popped for child_id in self._children[event.segment_id]):
                    raise ValueError(f"cannot pop segment {event.segment_id} before all children")
                stack.pop()
                popped.add(event.segment_id)

        expected = set(self.segments)
        if stack or pushed != expected or popped != expected:
            raise ValueError(
                f"incomplete event stream: stack={stack}, unpushed={sorted(expected - pushed)}, "
                f"unpopped={sorted(expected - popped)}"
            )
        return normalized


# Native TP group agreement over complete physical and logical forests.
def _write_record(hasher, record: tuple) -> None:
    hasher.update(json.dumps(record, separators=(",", ":")).encode())


def _write_plan(
    hasher,
    plan: SegmentPlan,
    events: Iterable[SegmentEvent] | None = None,
) -> None:
    # Insertion order is schedule-significant; do not sort segments.
    hasher.update(str((plan.root_id, bool(plan.topology_only))).encode())
    for segment in plan.segments.values():
        # SegmentSpec clones CPU tokens; raw bytes avoid list ambiguity.
        raw = segment.token_ids.contiguous().numpy().tobytes()
        terms = tuple((t.query_offset, t.target_token_id, t.weight, t.sample_id) for t in segment.loss_terms)
        _write_record(hasher, (
            segment.segment_id, segment.parent_id, segment.position_start,
            segment.prefix_length, len(raw), terms,
        ))
        hasher.update(raw)
    if events is None:
        events = plan.dfs_events()
    for event in events:
        hasher.update(f"{event.kind.value}:{event.segment_id};".encode())


def digest_segment_plan(
    plan: SegmentPlan, events: Iterable[SegmentEvent] | None = None
) -> str:
    hasher = hashlib.sha256()
    _write_plan(hasher, plan, events)
    return hasher.hexdigest()


def digest_forest_plan(forest: ForestExecutionPlan) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"loss_tokens={forest.logical_loss_tokens};".encode())
    for executable in forest.trees:
        tree = executable.tree
        _write_record(hasher, (tree.uid, tree.tree_index, list(tree.member_rows), tree.root_id))
        for node in tree.nodes.values():
            _write_record(hasher, (
                node.node_id, node.parent_id, node.segment.row,
                node.segment.start, node.segment.end,
                node.member_rows, node.terminal_rows, node.children
            ))
        _write_plan(hasher, executable.segment_plan)
        # PPO refs are logical and may differ even if physical tree coincides.
        # Identical refs ensure TP ranks enter vocab-parallel collectives with
        # identical query/target/temperature ownership and call counts.
        for ref in executable.objective_refs:
            _write_record(hasher, (
                ref.segment_id, ref.query_offset, ref.target_token_id,
                ref.sample_row, ref.response_offset
            ))
    return hasher.hexdigest()
