# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Lower exact token tries to executable TPR plans and logical PPO token references.

The topology never owns PPO values or normalization factors. A response token
at absolute position t is predicted by the query at t-1, which may belong to
the *parent* segment when t is the first token of a child.
"""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping
from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
import math
from torch import Tensor

import torch
from types import MappingProxyType



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
        if not isinstance(self.query_offset, int) or isinstance(self.query_offset, bool) or self.query_offset < 0:
            raise ValueError(f"query_offset must be a non-negative integer, got {self.query_offset!r}")
        if (
            not isinstance(self.target_token_id, int)
            or isinstance(self.target_token_id, bool)
            or self.target_token_id < 0
        ):
            raise ValueError(f"target_token_id must be a non-negative integer, got {self.target_token_id!r}")
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
        if not isinstance(self.position_start, int) or isinstance(self.position_start, bool) or self.position_start < 0:
            raise ValueError(f"position_start must be a non-negative integer, got {self.position_start!r}")
        if not isinstance(self.prefix_length, int) or isinstance(self.prefix_length, bool) or self.prefix_length < 0:
            raise ValueError(f"prefix_length must be a non-negative integer, got {self.prefix_length!r}")

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

        visited: set[SegmentId] = set()
        active: set[SegmentId] = set()

        def visit(segment_id: SegmentId) -> None:
            if segment_id in active:
                raise ValueError(f"cycle detected at segment {segment_id}")
            if segment_id in visited:
                return
            active.add(segment_id)
            for child_id in children[segment_id]:
                visit(child_id)
            active.remove(segment_id)
            visited.add(segment_id)

        visit(self.root_id)
        if visited != set(normalized):
            raise ValueError(f"segments are not reachable from root: {sorted(set(normalized) - visited)}")

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


@dataclass(frozen=True, slots=True)
class SegmentRef:
    """A zero-copy reference into one row of the original training batch."""

    row: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.row, int) or isinstance(self.row, bool) or self.row < 0:
            raise ValueError(f"row must be a non-negative integer, got {self.row!r}")
        if not isinstance(self.start, int) or isinstance(self.start, bool) or self.start < 0:
            raise ValueError(f"start must be a non-negative integer, got {self.start!r}")
        if not isinstance(self.end, int) or isinstance(self.end, bool) or self.end <= self.start:
            raise ValueError(f"end must be an integer > start, got {self.end!r}")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TrajectoryNode:
    """One compressed token segment in a trajectory radix tree."""

    node_id: int
    parent_id: int | None
    segment: SegmentRef
    member_rows: tuple[int, ...]
    terminal_rows: tuple[int, ...]
    children: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, int) or isinstance(self.node_id, bool) or self.node_id < 0:
            raise ValueError(f"node_id must be a non-negative integer, got {self.node_id!r}")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, int)
            or isinstance(self.parent_id, bool)
            or self.parent_id < 0
            or self.parent_id == self.node_id
        ):
            raise ValueError(f"invalid parent_id {self.parent_id!r} for node {self.node_id}")
        if not self.member_rows:
            raise ValueError(f"node {self.node_id} must contain at least one member row")
        if len(set(self.member_rows)) != len(self.member_rows):
            raise ValueError(f"node {self.node_id} member_rows must be unique")
        if not set(self.terminal_rows).issubset(self.member_rows):
            raise ValueError(f"node {self.node_id} terminal_rows must be a subset of member_rows")


@dataclass(frozen=True, slots=True)
class TrajectoryTree:
    """One executable compressed trie whose real root starts at token position zero."""

    uid: str
    tree_index: int
    root_id: int
    member_rows: tuple[int, ...]
    nodes: Mapping[int, TrajectoryNode]

    def __post_init__(self) -> None:
        normalized = dict(self.nodes)
        if self.root_id not in normalized:
            raise ValueError(f"root_id {self.root_id} does not exist")
        root = normalized[self.root_id]
        if root.parent_id is not None:
            raise ValueError("trajectory tree root must not have a parent")
        if root.segment.start != 0:
            raise ValueError("trajectory tree executable root must start at token position 0")
        if root.member_rows != self.member_rows:
            raise ValueError("trajectory tree member_rows must match root member_rows")

        for node_id, node in normalized.items():
            if node_id != node.node_id:
                raise ValueError(f"node mapping key {node_id} does not match node_id {node.node_id}")
            if node.parent_id is not None and node.parent_id not in normalized:
                raise ValueError(f"node {node_id} has missing parent {node.parent_id}")
            for child_id in node.children:
                if child_id not in normalized:
                    raise ValueError(f"node {node_id} has missing child {child_id}")
                child = normalized[child_id]
                if child.parent_id != node_id:
                    raise ValueError(f"node {node_id} child {child_id} has inconsistent parent")
                if child.segment.start != node.segment.end:
                    raise ValueError(
                        f"node {child_id} must start at parent {node_id} end {node.segment.end}, "
                        f"got {child.segment.start}"
                    )

        object.__setattr__(self, "nodes", MappingProxyType(normalized))

    @property
    def key(self) -> str:
        return f"{self.uid}:{self.tree_index}"

    def get(self, node_id: int) -> TrajectoryNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown trajectory node {node_id}") from exc


def _trajectory_rows(batch: Any, name: str) -> list[Any]:
    try:
        value = batch[name]
    except Exception as exc:
        raise ValueError(f"training batch is missing required field {name!r}") from exc

    if isinstance(value, (list, tuple)):
        return list(value)

    unbind = getattr(value, "unbind", None)
    if callable(unbind):
        return list(unbind())

    raise TypeError(f"field {name!r} does not expose row-wise values: {type(value)!r}")


def _as_1d_input_ids(value: Any, *, row: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"input_ids[{row}] must be a torch.Tensor, got {type(value)!r}")
    if value.ndim != 1:
        raise ValueError(f"input_ids[{row}] must be 1-D, got shape={tuple(value.shape)}")
    if value.numel() == 0:
        raise ValueError(f"input_ids[{row}] must not be empty")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise ValueError(f"input_ids[{row}] must use an integer dtype, got {value.dtype}")
    if torch.any(value < 0).item():
        raise ValueError(f"input_ids[{row}] must contain non-negative token ids")
    return value


def _parse_key(key: str) -> tuple[str, str, int]:
    if not isinstance(key, str):
        raise TypeError(f"trajectory key must be str, got {type(key)!r}")
    fields = key.rsplit("_", 2)
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise ValueError(f"unexpected trajectory key format: {key!r}")
    try:
        trajectory_index = int(fields[2])
    except ValueError as exc:
        raise ValueError(f"trajectory key has non-integer index: {key!r}") from exc
    return fields[0], fields[1], trajectory_index


QWEN3_IM_END_TOKEN_ID = 151645


@dataclass(slots=True)
class _TurnNode:
    end: int
    source_row: int
    member_rows: list[int] = field(default_factory=list)
    terminal_rows: list[int] = field(default_factory=list)
    children: dict[tuple[int, ...], "_TurnNode"] = field(default_factory=dict)


def _turn_units(
    tokens: list[int],
    *,
    prompt_length: int | None,
    min_segment_tokens: int,
    end_token_id: int,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Token-exact units ending at complete ChatML messages / prompt cut.

    End-of-input is always a boundary, but does not automatically introduce a
    shared internal node if a longer sibling has a different unfinished tail.
    """
    length = len(tokens)
    candidates = {i + 1 for i, token in enumerate(tokens)
                  if token == end_token_id}
    if prompt_length is not None and 0 < prompt_length < length:
        candidates.add(prompt_length)
    candidates.add(length)
    previous = 0
    units = []
    for end in sorted(candidates):
        # Only complete message/prompt boundaries can be shared internally.
        # A last short tail is always kept, as terminal rows need every token.
        if end < length and end - previous < min_segment_tokens:
            continue
        if end <= previous:
            raise AssertionError("turn unit boundaries must increase")
        units.append((tuple(tokens[previous:end]), end))
        previous = end
    if previous != length or not units:
        raise AssertionError("turn units must reconstruct entire trajectory")
    return tuple(units)


def _insert(root: _TurnNode, row: int,
            units: tuple[tuple[tuple[int, ...], int], ...]) -> None:
    current = root
    for signature, end in units:
        child = current.children.get(signature)
        if child is None:
            child = _TurnNode(end=end, source_row=row)
            current.children[signature] = child
        elif child.end != end:
            raise AssertionError("identical turn tokens have different absolute ends")
        child.member_rows.append(row)
        current = child
    current.terminal_rows.append(row)


def _compress(node: _TurnNode, parent_id: int | None,
              nodes: dict[int, TrajectoryNode], start: int) -> int:
    current = node
    while not current.terminal_rows and len(current.children) == 1:
        candidate = next(iter(current.children.values()))
        if current.member_rows != candidate.member_rows:
            break
        current = candidate
    node_id = len(nodes)
    segment = SegmentRef(row=node.source_row, start=start, end=current.end)
    nodes[node_id] = TrajectoryNode(
        node_id=node_id, parent_id=parent_id, segment=segment,
        member_rows=tuple(current.member_rows),
        terminal_rows=tuple(current.terminal_rows), children=(),
    )
    child_ids = tuple(
        _compress(child, node_id, nodes, current.end)
        for child in current.children.values()
    )
    nodes[node_id] = TrajectoryNode(
        node_id=node_id, parent_id=parent_id, segment=segment,
        member_rows=tuple(current.member_rows),
        terminal_rows=tuple(current.terminal_rows), children=child_ids,
    )
    return node_id


def build_trajectory_trees_turn(
    keys: list[str] | tuple[str, ...],
    batch: Any,
    *,
    min_segment_tokens: int = 128,
    end_token_id: int = QWEN3_IM_END_TOKEN_ID,
) -> tuple[TrajectoryTree, ...]:
    """Build exact TPR topology sharing only completed Turn units.

    Production Qwen3/ChatML turn-boundary strategy. A captured prompt endpoint may be used
    even when the ChatML prompt ends in an open assistant header. When no
    common valid unit exists, returns separate full-sequence trees rather
    than fabricating any shared prefix.
    """
    if not isinstance(min_segment_tokens, int) or isinstance(min_segment_tokens, bool) or min_segment_tokens < 1:
        raise ValueError("min_segment_tokens must be a positive integer")
    if not isinstance(end_token_id, int) or isinstance(end_token_id, bool) or end_token_id < 0:
        raise ValueError("end_token_id must be a non-negative integer")
    keys = list(keys)
    if not keys:
        return ()
    inputs = _trajectory_rows(batch, "input_ids")
    if len(inputs) != len(keys):
        raise ValueError("turn builder input rows and trajectory keys disagree")
    try:
        prompts = _trajectory_rows(batch, "prompts")
    except (ValueError, KeyError):
        prompts = None
    if prompts is not None and len(prompts) != len(keys):
        raise ValueError("turn builder prompt rows and trajectory keys disagree")

    groups: dict[str, list[int]] = {}
    units_by_row = []
    seen = set()
    for row, key in enumerate(keys):
        if key in seen:
            raise ValueError(f"duplicate trajectory key: {key!r}")
        seen.add(key)
        uid, _, _ = _parse_key(key)
        groups.setdefault(uid, []).append(row)
        input_ids = _as_1d_input_ids(inputs[row], row=row)
        token_ids = input_ids.detach().cpu().tolist()
        prompt_length = None
        if prompts is not None:
            prompt = _as_1d_input_ids(prompts[row], row=row)
            prompt_length = int(prompt.numel())
            if prompt_length > len(token_ids) or not torch.equal(input_ids[:prompt_length].cpu(), prompt.cpu()):
                raise ValueError(f"turn builder prompt is not an exact input prefix for row={row}")
        units_by_row.append(_turn_units(
            token_ids, prompt_length=prompt_length,
            min_segment_tokens=min_segment_tokens, end_token_id=end_token_id,
        ))

    forest: list[TrajectoryTree] = []
    for uid, rows in groups.items():
        dummy = _TurnNode(end=0, source_row=rows[0])
        for row in rows:
            _insert(dummy, row, units_by_row[row])
        for tree_index, child in enumerate(dummy.children.values()):
            nodes: dict[int, TrajectoryNode] = {}
            root_id = _compress(child, None, nodes, 0)
            forest.append(TrajectoryTree(
                uid=uid, tree_index=tree_index, root_id=root_id,
                member_rows=nodes[root_id].member_rows, nodes=nodes,
            ))
    return tuple(forest)


@dataclass(frozen=True, slots=True)
class SegmentObjectiveRef:
    segment_id: int
    query_offset: int
    target_token_id: int
    sample_row: int
    response_offset: int


@dataclass(frozen=True, slots=True)
class TreeExecutionPlan:
    tree: TrajectoryTree
    segment_plan: SegmentPlan
    objective_refs: tuple[SegmentObjectiveRef, ...]

    def refs_for_segment(self, segment_id: int) -> tuple[SegmentObjectiveRef, ...]:
        self.segment_plan.get(segment_id)
        return tuple(ref for ref in self.objective_refs if ref.segment_id == segment_id)

    @property
    def logical_loss_tokens(self) -> int:
        return len(self.objective_refs)


@dataclass(frozen=True, slots=True)
class ForestExecutionPlan:
    trees: tuple[TreeExecutionPlan, ...]
    logical_loss_tokens: int

    def __post_init__(self):
        if sum(tree.logical_loss_tokens for tree in self.trees) != self.logical_loss_tokens:
            raise ValueError("forest logical_loss_tokens disagrees with objective refs")

    @property
    def segment_count(self) -> int:
        return sum(len(tree.segment_plan.segments) for tree in self.trees)


def _rows(batch: Any, key: str) -> list[torch.Tensor]:
    try:
        value = batch[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"batch is missing {key!r}") from exc
    values = list(value) if isinstance(value, (list, tuple)) else list(value.unbind())
    if not all(isinstance(item, torch.Tensor) and item.ndim == 1 for item in values):
        raise ValueError(f"{key!r} must contain one-dimensional tensor rows")
    return values


def build_tree_execution_plans(
    keys: list[str] | tuple[str, ...],
    batch: Any,
    *,
    trees: tuple[TrajectoryTree, ...] | None = None,
    require_loss_mask_alignment: bool = False,
    tree_policy: str = "qwen_turn",
    turn_min_segment_tokens: int = 128,
) -> ForestExecutionPlan:
    """Build a forest of topology-only SegmentPlans and PPO ownership references.

    response_mask is right-aligned with input_ids, matching VERL's RL
    prompt+response layout. Every True response-mask element maps to exactly
    one ref; duplicate branches/samples intentionally retain separate refs.
    Optional loss-mask alignment catches mismatch against the native
    batch_num_tokens denominator before entering the Megatron schedule.
    """
    if tree_policy != "qwen_turn":
        raise ValueError("Production TPR only supports qwen_turn")
    keys = list(keys)
    if trees is None:
        trees = build_trajectory_trees_turn(
            keys, batch, min_segment_tokens=turn_min_segment_tokens,
        )
    input_rows = _rows(batch, "input_ids")
    response_masks = _rows(batch, "response_mask")
    if len(keys) != len(input_rows) or len(keys) != len(response_masks):
        raise ValueError("keys, input_ids and response_mask must have identical row counts")

    try:
        response_rows = _rows(batch, "responses")
    except ValueError:
        response_rows = None

    selected_positions: list[list[int]] = []
    expected_token_count = 0
    for row, (tokens, mask) in enumerate(zip(input_rows, response_masks, strict=True)):
        if mask.numel() > tokens.numel():
            raise ValueError(f"response_mask[{row}] is longer than input_ids[{row}]")
        if response_rows is not None:
            response = response_rows[row]
            if response.numel() != mask.numel():
                raise ValueError(f"responses/response_mask length mismatch on row {row}")
            if not torch.equal(tokens[-response.numel():], response):
                raise ValueError(f"response is not an input_ids suffix on row {row}")
        valid = torch.nonzero(mask.to(torch.bool), as_tuple=False).flatten().tolist()
        prefix_len = tokens.numel() - mask.numel()
        if valid and prefix_len + valid[0] == 0:
            raise ValueError(f"row {row} has no preceding query token for first supervised token")
        selected_positions.append(valid)
        expected_token_count += len(valid)

    if require_loss_mask_alignment:
        loss_masks = _rows(batch, "loss_mask")
        if len(loss_masks) != len(keys):
            raise ValueError("loss_mask rows do not match keys")
        for row, (loss_mask, response_mask) in enumerate(zip(loss_masks, response_masks, strict=True)):
            if loss_mask.numel() == response_mask.numel():
                aligned = loss_mask
            elif loss_mask.numel() == input_rows[row].numel():
                aligned = loss_mask[-response_mask.numel():]
                if bool(loss_mask[:-response_mask.numel()].to(bool).any()):
                    raise ValueError(f"row {row} loss_mask includes prompt tokens")
            else:
                raise ValueError(f"row {row} loss_mask shape is incompatible with response_mask")
            if not torch.equal(aligned.to(bool), response_mask.to(bool)):
                raise ValueError(f"row {row} response_mask and loss_mask disagree")

    consumed_rows: set[int] = set()
    result: list[TreeExecutionPlan] = []
    for tree in trees:
        segments = []
        for node in tree.nodes.values():
            ref = node.segment
            token_ids = input_rows[ref.row][ref.start:ref.end].detach().to(device="cpu", dtype=torch.long)
            segments.append(
                SegmentSpec(
                    segment_id=node.node_id,
                    parent_id=node.parent_id,
                    token_ids=token_ids,
                    position_start=ref.start,
                    prefix_length=ref.start,
                )
            )
        # Physical topology is independent of CE or PPO objective metadata.
        plan = SegmentPlan(segments, root_id=tree.root_id, topology_only=True)
        refs: list[SegmentObjectiveRef] = []
        terminal_to_path = {}

        def visit(node_id: int, path: tuple[int, ...]) -> None:
            node = tree.get(node_id)
            current_path = (*path, node_id)
            for row in node.terminal_rows:
                if row in terminal_to_path:
                    raise ValueError(f"trajectory row {row} has multiple terminal nodes")
                terminal_to_path[row] = current_path
            for child_id in node.children:
                visit(child_id, current_path)

        visit(tree.root_id, ())
        if set(terminal_to_path) != set(tree.member_rows):
            raise ValueError(f"tree {tree.key} terminal rows do not cover members")

        for row in tree.member_rows:
            if row in consumed_rows:
                raise ValueError(f"row {row} appears in more than one tree")
            consumed_rows.add(row)
            input_ids = input_rows[row]
            response_len = response_masks[row].numel()
            response_start = input_ids.numel() - response_len
            path = terminal_to_path[row]
            ends = [tree.get(node_id).segment.end for node_id in path]
            for response_offset in selected_positions[row]:
                target_position = response_start + response_offset
                query_position = target_position - 1
                owner_idx = bisect_right(ends, query_position)
                if owner_idx >= len(path):
                    raise ValueError(f"row {row} query position {query_position} is outside tree")
                node_id = path[owner_idx]
                node = tree.get(node_id)
                if not node.segment.start <= query_position < node.segment.end:
                    raise ValueError(f"row {row} has uncovered query position {query_position}")
                target = int(input_ids[target_position].item())
                refs.append(
                    SegmentObjectiveRef(
                        segment_id=node_id,
                        query_offset=query_position - node.segment.start,
                        target_token_id=target,
                        sample_row=row,
                        response_offset=response_offset,
                    )
                )
        result.append(TreeExecutionPlan(tree=tree, segment_plan=plan, objective_refs=tuple(refs)))

    if consumed_rows != set(range(len(keys))):
        raise ValueError(f"forest does not cover all input rows: {set(range(len(keys))) - consumed_rows}")
    forest = ForestExecutionPlan(tuple(result), logical_loss_tokens=expected_token_count)
    return forest


# Native TP group agreement over complete physical and logical forests.
def _segment_record(segment) -> tuple:
    # CPU token_ids are part of SegmentSpec's immutable metadata (the spec
    # clones the input); hashing raw bytes avoids ambiguous list separators.
    token_bytes = segment.token_ids.contiguous().numpy().tobytes()
    return (
        segment.segment_id,
        segment.parent_id,
        segment.position_start,
        segment.prefix_length,
        len(token_bytes),
        token_bytes,
        tuple(
            (t.query_offset, t.target_token_id, t.weight, t.sample_id)
            for t in segment.loss_terms
        ),
    )


def _write_plan(
    hasher,
    plan: SegmentPlan,
    events: Iterable[SegmentEvent] | None = None,
) -> None:
    # Insertion order is schedule-significant; do not sort segments.
    hasher.update(str((plan.root_id, bool(plan.topology_only))).encode())
    for segment in plan.segments.values():
        seg_id, parent, start, prefix, length, raw, terms = _segment_record(segment)
        hasher.update(json.dumps((seg_id, parent, start, prefix, length, terms),
                                 separators=(",", ":"), ensure_ascii=True).encode())
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
        hasher.update(
            json.dumps((tree.uid, tree.tree_index, list(tree.member_rows), tree.root_id),
                       separators=(",", ":")).encode()
        )
        for node in tree.nodes.values():
            hasher.update(json.dumps((
                node.node_id, node.parent_id, node.segment.row,
                node.segment.start, node.segment.end,
                node.member_rows, node.terminal_rows, node.children
            ), separators=(",", ":")).encode())
        _write_plan(hasher, executable.segment_plan)
        # PPO refs are logical and may differ even if physical tree coincides.
        # Identical refs ensure TP ranks enter vocab-parallel collectives with
        # identical query/target/temperature ownership and call counts.
        for ref in executable.objective_refs:
            hasher.update(json.dumps((
                ref.segment_id, ref.query_offset, ref.target_token_id,
                ref.sample_row, ref.response_offset
            ), separators=(",", ":")).encode())
    return hasher.hexdigest()


def assert_tp_plan_agreement(digest: str, *, tp_size: int) -> None:
    """Collectively reject divergent execution plans on the native TP group.

    Called ONCE per TPR minibatch, *before* model forward or TP collectives.
    Uses only a 32-byte SHA256 signature per rank, not model tensors / KV.
    Every peer receives the same success or mismatch result.
    """
    if tp_size == 1:
        return
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if len(digest) != 64:
        raise ValueError("SHA256 digest must be a 64-character hex string")

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("TPR TP2 requires an initialized distributed process group")
    group = parallel_state.get_tensor_model_parallel_group()
    if group is None:
        raise RuntimeError("TPR TP2 cannot resolve native Megatron TP group")
    if dist.get_world_size(group=group) != tp_size:
        raise RuntimeError("TPR TP group size differs from Engine TP config")
    # HCCL AllGather supports int64, so store SHA256 as eight int64 words
    # (each word is a non-negative 32-bit unsigned value).
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    device = torch.device("npu", torch.npu.current_device()) if dist.get_backend(group) == "hccl" else torch.device("cpu")
    local = torch.tensor(words, dtype=torch.int64, device=device)
    signatures = [torch.empty_like(local) for _ in range(tp_size)]
    dist.all_gather(signatures, local, group=group)
    if any(not torch.equal(value, local) for value in signatures):
        rank = dist.get_rank(group=group)
        raise RuntimeError(
            f"TPR TP schedule mismatch on tp_rank={rank}: forest, SegmentPlan, "
            "DFS events or PPO refs differ across model-parallel peers"
        )

