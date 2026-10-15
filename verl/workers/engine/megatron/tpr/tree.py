# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Lower exact token tries to executable TPR plans and logical PPO token references.

The topology never owns PPO values or normalization factors. A response token
at absolute position t is predicted by the query at t-1, which may belong to
the *parent* segment when t is the first token of a child.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from types import MappingProxyType

from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec


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

    Qwen3-only experimental strategy. A captured prompt endpoint may be used
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
