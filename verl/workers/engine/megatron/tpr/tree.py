# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Turn-boundary token sharing and native PPO query ownership.

The compressed SegmentPlan is the only persistent physical topology. Temporary
Turn trie nodes are discarded after assigning each logical response token to
its predicting query, including child-first tokens predicted by the parent.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor


SegmentId = int
QWEN3_IM_END_TOKEN_ID = 151645


def _validate_segment_id(value: int, *, field_name: str = "segment_id") -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    segment_id: SegmentId
    parent_id: SegmentId | None
    token_ids: Tensor
    position_start: int
    prefix_length: int

    def __post_init__(self) -> None:
        _validate_segment_id(self.segment_id)
        if self.parent_id is not None:
            _validate_segment_id(self.parent_id, field_name="parent_id")
        _validate_segment_id(self.position_start, field_name="position_start")
        _validate_segment_id(self.prefix_length, field_name="prefix_length")
        if self.prefix_length != self.position_start:
            raise ValueError("prefix_length must equal position_start")
        if not isinstance(self.token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if (self.token_ids.device.type != "cpu" or self.token_ids.dtype != torch.long
                or self.token_ids.ndim != 1 or not self.token_ids.numel()):
            raise ValueError("token_ids must be a non-empty one-dimensional CPU torch.long tensor")
        if bool((self.token_ids < 0).any()):
            raise ValueError("token_ids must be non-negative")
        object.__setattr__(self, "token_ids", self.token_ids.detach().clone())

    @property
    def length(self) -> int:
        return self.token_ids.numel()

    @property
    def position_end(self) -> int:
        return self.position_start + self.length


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    segments: Mapping[SegmentId, SegmentSpec] | Sequence[SegmentSpec]
    root_id: SegmentId
    _children: Mapping[SegmentId, tuple[SegmentId, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_segment_id(self.root_id, field_name="root_id")
        source = self.segments.values() if isinstance(self.segments, Mapping) else self.segments
        segments = {}
        for segment in source:
            if not isinstance(segment, SegmentSpec):
                raise TypeError("segments must contain SegmentSpec objects")
            if segment.segment_id in segments:
                raise ValueError(f"duplicate segment_id {segment.segment_id}")
            segments[segment.segment_id] = segment
        roots = [s.segment_id for s in segments.values() if s.parent_id is None]
        if roots != [self.root_id]:
            raise ValueError(f"plan must have exactly root {self.root_id}, got roots={roots}")
        children = {segment_id: [] for segment_id in segments}
        for segment in segments.values():
            if segment.parent_id is None:
                if segment.position_start != 0:
                    raise ValueError("root segment must start at position zero")
                continue
            parent = segments.get(segment.parent_id)
            if parent is None or segment.position_start != parent.position_end:
                raise ValueError(f"segment {segment.segment_id} must start at parent end")
            children[parent.segment_id].append(segment.segment_id)
        # Positive segment lengths and increasing parent positions exclude cycles.
        object.__setattr__(self, "segments", MappingProxyType(segments))
        object.__setattr__(self, "_children", MappingProxyType({k: tuple(v) for k, v in children.items()}))

    def get(self, segment_id: SegmentId) -> SegmentSpec:
        return self.segments[segment_id]

    def children_of(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        return tuple(self.segments[child] for child in self._children[segment_id])

    def path_to(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        path = []
        while segment_id is not None:
            segment = self.get(segment_id)
            path.append(segment)
            segment_id = segment.parent_id
        return tuple(reversed(path))

    def depth_of(self, segment_id: SegmentId) -> int:
        return len(self.path_to(segment_id)) - 1


@dataclass(frozen=True, slots=True)
class SegmentObjectiveRef:
    segment_id: int
    query_offset: int
    target_token_id: int
    sample_row: int
    response_offset: int


@dataclass(frozen=True, slots=True)
class TreeExecutionPlan:
    uid: str
    tree_index: int
    member_rows: tuple[int, ...]
    segment_plan: SegmentPlan
    objective_refs: tuple[SegmentObjectiveRef, ...]
    active_segment_ids: frozenset[SegmentId] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Global query owners and their ancestors, never CP-local counts.
        active = set()
        for ref in self.objective_refs:
            segment_id = ref.segment_id
            while segment_id is not None and segment_id not in active:
                active.add(segment_id)
                segment_id = self.segment_plan.get(segment_id).parent_id
        object.__setattr__(self, "active_segment_ids", frozenset(active))

    @property
    def key(self) -> str:
        return f"{self.uid}:{self.tree_index}"

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


def _rows(batch: Any, key: str) -> list[Tensor]:
    try:
        value = batch[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"batch is missing {key!r}") from exc
    values = list(value) if isinstance(value, (list, tuple)) else list(value.unbind())
    if not all(isinstance(item, Tensor) and item.ndim == 1 for item in values):
        raise ValueError(f"{key!r} must contain one-dimensional tensor rows")
    return values


def _parse_key(key: str) -> tuple[str, str, int]:
    if not isinstance(key, str):
        raise TypeError(f"trajectory key must be str, got {type(key)!r}")
    fields = key.rsplit("_", 2)
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise ValueError(f"unexpected trajectory key format: {key!r}")
    try:
        return fields[0], fields[1], int(fields[2])
    except ValueError as exc:
        raise ValueError(f"trajectory key has non-integer index: {key!r}") from exc


@dataclass(slots=True)
class _TurnNode:
    end: int
    source_row: int
    member_rows: list[int] = field(default_factory=list)
    terminal_rows: list[int] = field(default_factory=list)
    children: dict[tuple[int, ...] | bytes, _TurnNode] = field(default_factory=dict)


def _turn_units(tokens: list[int], prompt_length: int, minimum: int):
    boundaries = {i + 1 for i, token in enumerate(tokens) if token == QWEN3_IM_END_TOKEN_ID}
    if 0 < prompt_length < len(tokens):
        boundaries.add(prompt_length)
    boundaries.add(len(tokens))
    previous = 0
    for end in sorted(boundaries):
        if end < len(tokens) and end - previous < minimum:
            continue
        yield tuple(tokens[previous:end]), end
        previous = end


def _turn_units_bytes(tokens: Tensor, prompt_length: int, minimum: int) -> tuple[tuple[bytes, int], ...]:
    """Extract the same Turn units as _turn_units, using compact exact byte signatures.

    Token IDs are CPU int64 values, so their contiguous byte representation is
    an exact equality/sort key. This avoids materializing one Python int object
    per token in the LCP path and lets sorting/LCP comparisons run in C.
    """
    boundaries = {
        int(position) + 1
        for position in torch.nonzero(tokens == QWEN3_IM_END_TOKEN_ID, as_tuple=False).flatten().tolist()
    }
    if 0 < prompt_length < tokens.numel():
        boundaries.add(prompt_length)
    boundaries.add(tokens.numel())
    previous = 0
    units = []
    for end in sorted(boundaries):
        if end < tokens.numel() and end - previous < minimum:
            continue
        signature = tokens[previous:end].contiguous().numpy().tobytes()
        units.append((signature, end))
        previous = end
    return tuple(units)


def _unit_lcp(left: tuple[tuple[bytes, int], ...], right: tuple[tuple[bytes, int], ...]) -> int:
    """Return the exact common prefix length in complete units, not raw tokens."""
    limit = min(len(left), len(right))
    shared = 0
    while shared < limit and left[shared][0] == right[shared][0]:
        shared += 1
    return shared


def _build_lcp_group(rows: list[tuple[int, tuple[tuple[bytes, int], ...]]]) -> _TurnNode:
    """Build a Turn trie from sorted rows and adjacent LCPs.

    This is deliberately unit-agnostic: the current caller supplies qwen_turn
    spans, while a future message-role extractor can supply message spans
    without changing the sorting/LCP construction.
    """
    if not rows:
        raise ValueError("LCP tree group must contain at least one trajectory")

    # Python's sort is stable; row is included explicitly to make duplicate
    # sequence ordering independent of sort implementation details.
    ordered = sorted(
        rows,
        key=lambda item: (tuple(signature for signature, _ in item[1]), item[0]),
    )
    root = _TurnNode(0, ordered[0][0])
    nodes = [root]
    previous_units = None
    previous_path = [root]

    for row, units in ordered:
        if not units:
            raise ValueError(f"LCP trajectory row {row} has no units")
        shared = 0 if previous_units is None else _unit_lcp(previous_units, units)
        path = previous_path[:shared + 1]
        parent = path[-1]

        # Adjacent-LCP construction guarantees that every suffix unit starts
        # a new child at this parent. A collision indicates a broken ordering
        # or LCP calculation, so fail rather than silently changing topology.
        for signature, end in units[shared:]:
            if signature in parent.children:
                raise RuntimeError("LCP builder encountered an existing suffix child")
            child = _TurnNode(end, row)
            parent.children[signature] = child
            nodes.append(child)
            path.append(child)
            parent = child

        for prefix_node in path[1:]:
            prefix_node.member_rows.append(row)
        parent.terminal_rows.append(row)
        previous_units = units
        previous_path = path

    # Restore the insertion-Trie's externally visible order: original row order
    # for members/terminals and first-seen row order for sibling branches.
    for node in nodes:
        node.member_rows.sort()
        node.terminal_rows.sort()
        if node.member_rows:
            node.source_row = node.member_rows[0]
        node.children = dict(
            sorted(node.children.items(), key=lambda item: item[1].member_rows[0])
        )
    return root


def _compress(node: _TurnNode, parent_id: int | None, start: int, inputs: list[Tensor],
              segments: list[SegmentSpec], paths: dict[int, tuple[int, ...]], path: tuple[int, ...]) -> None:
    # Explicit DFS preserves insertion order without a Python recursion limit
    # on long conversations with terminal rows at many successive Turn nodes.
    pending = [(node, parent_id, start, path)]
    while pending:
        node, parent_id, start, path = pending.pop()
        while not node.terminal_rows and len(node.children) == 1:
            node = next(iter(node.children.values()))
        segment_id = len(segments)
        segments.append(SegmentSpec(segment_id, parent_id, inputs[node.source_row][start:node.end], start, start))
        path = (*path, segment_id)
        paths.update((row, path) for row in node.terminal_rows)
        pending.extend((child, segment_id, node.end, path) for child in reversed(node.children.values()))


def build_tree_execution_plans(
    keys: list[str] | tuple[str, ...], batch: Any, *,
    require_loss_mask_alignment: bool = False,
    tree_policy: str = "qwen_turn", turn_min_segment_tokens: int = 128,
    tree_builder: str = "trie",
) -> ForestExecutionPlan:
    """Build the executable topology and retain one ref per masked response.

    tree_builder="trie" is the insertion-order reference implementation.
    tree_builder="lcp_sort" sorts complete Turn-unit sequences and rebuilds
    the same trie from adjacent longest-common-prefix lengths.
    """
    if tree_policy != "qwen_turn":
        raise ValueError("Production TPR only supports qwen_turn")
    if tree_builder not in ("trie", "lcp_sort"):
        raise ValueError(f"unknown TPR tree_builder: {tree_builder!r}")
    if (not isinstance(turn_min_segment_tokens, int) or isinstance(turn_min_segment_tokens, bool)
            or turn_min_segment_tokens < 1):
        raise ValueError("turn_min_segment_tokens must be a positive integer")
    inputs, masks = _rows(batch, "input_ids"), _rows(batch, "response_mask")
    if len(keys) != len(inputs) or len(keys) != len(masks):
        raise ValueError("keys, input_ids and response_mask must have identical row counts")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate trajectory key")
    prompts = _rows(batch, "prompts") if "prompts" in batch else None
    responses = _rows(batch, "responses") if "responses" in batch else None
    loss_masks = _rows(batch, "loss_mask") if require_loss_mask_alignment else None
    for values in (prompts, responses, loss_masks):
        if values is not None and len(values) != len(keys):
            raise ValueError("optional batch field rows do not match keys")

    groups: dict[str, _TurnNode] = {}
    lcp_rows: dict[str, list[tuple[int, tuple[tuple[bytes, int], ...]]]] = {}
    selected = []
    cpu_inputs = []
    for row, (key, tokens, mask) in enumerate(zip(keys, inputs, masks, strict=True)):
        if (not tokens.numel() or tokens.dtype == torch.bool or torch.is_floating_point(tokens)
                or torch.is_complex(tokens) or bool((tokens < 0).any())):
            raise ValueError(f"input_ids[{row}] must contain non-empty non-negative integer token ids")
        cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long)
        cpu_inputs.append(cpu_tokens)
        if mask.numel() > tokens.numel():
            raise ValueError(f"response_mask[{row}] is longer than input_ids[{row}]")
        response_start = tokens.numel() - mask.numel()
        if responses is not None:
            response = responses[row]
            if response.numel() != mask.numel():
                raise ValueError(f"responses/response_mask length mismatch on row {row}")
            if not torch.equal(tokens[response_start:], response):
                raise ValueError(f"response is not an input_ids suffix on row {row}")
        offsets = torch.nonzero(mask.to(torch.bool), as_tuple=False).flatten().tolist()
        if offsets and response_start + offsets[0] == 0:
            raise ValueError(f"row {row} has no preceding query token for first supervised token")
        selected.append(offsets)
        if loss_masks is not None:
            loss_mask = loss_masks[row]
            if loss_mask.numel() == tokens.numel():
                if bool(loss_mask[:response_start].to(bool).any()):
                    raise ValueError(f"row {row} loss_mask includes prompt tokens")
                loss_mask = loss_mask[response_start:]
            if loss_mask.numel() != mask.numel() or not torch.equal(loss_mask.to(bool), mask.to(bool)):
                raise ValueError(f"row {row} response_mask and loss_mask disagree")
        prompt_length = 0
        if prompts is not None:
            prompt = prompts[row]
            prompt_length = prompt.numel()
            if prompt_length > tokens.numel() or not torch.equal(tokens[:prompt_length].cpu(), prompt.cpu()):
                raise ValueError(f"turn builder prompt is not an exact input prefix for row={row}")
        uid, _, _ = _parse_key(key)
        if tree_builder == "lcp_sort":
            units = _turn_units_bytes(cpu_tokens, prompt_length, turn_min_segment_tokens)
            lcp_rows.setdefault(uid, []).append((row, units))
        else:
            node = groups.setdefault(uid, _TurnNode(0, row))
            for signature, end in _turn_units(cpu_tokens.tolist(), prompt_length, turn_min_segment_tokens):
                node = node.children.setdefault(signature, _TurnNode(end, row))
                node.member_rows.append(row)
            node.terminal_rows.append(row)

    if tree_builder == "lcp_sort":
        groups = {uid: _build_lcp_group(rows) for uid, rows in lcp_rows.items()}

    result = []
    for uid, root in groups.items():
        for tree_index, child in enumerate(root.children.values()):
            segments, paths = [], {}
            _compress(child, None, 0, cpu_inputs, segments, paths, ())
            plan = SegmentPlan(segments, root_id=0)
            refs = []
            for row in child.member_rows:
                response_start = inputs[row].numel() - masks[row].numel()
                path = paths[row]
                ends = [plan.get(segment_id).position_end for segment_id in path]
                for offset in selected[row]:
                    target = response_start + offset
                    segment_id = path[bisect_right(ends, target - 1)]
                    refs.append(SegmentObjectiveRef(
                        segment_id, target - 1 - plan.get(segment_id).position_start,
                        int(cpu_inputs[row][target]), row, offset,
                    ))
            result.append(TreeExecutionPlan(uid, tree_index, tuple(child.member_rows), plan, tuple(refs)))
    return ForestExecutionPlan(tuple(result), sum(map(len, selected)))



@dataclass(frozen=True, slots=True)
class _SessionTreeUnit:
    signature: bytes
    start: int
    end: int
    token_ids: Tensor


def _session_tree_tokens(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a one-dimensional integer sequence") from exc
    if value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if (value.dtype == torch.bool or torch.is_floating_point(value)
            or torch.is_complex(value) or bool((value < 0).any())):
        raise ValueError(f"{name} must contain non-negative integer token IDs")
    return value.detach().to(device="cpu", dtype=torch.long).clone()


def _session_tree_mask(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a one-dimensional 0/1 sequence") from exc
    if value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    result = value.detach().to(device="cpu")
    if not bool(((result == 0) | (result == 1)).all()):
        raise ValueError(f"{name} must contain only 0/1 values")
    return result.to(torch.bool)


def _trim_token_chunks(chunks: list[Tensor], count: int) -> int:
    """Trim count tokens from the right; return any count not present."""
    remaining = count
    while remaining and chunks:
        tail = chunks[-1]
        if tail.numel() <= remaining:
            remaining -= tail.numel()
            chunks.pop()
        else:
            chunks[-1] = tail[:-remaining]
            remaining = 0
    return remaining


def _trim_response_chunks(
    chunks: list[tuple[Tensor, Tensor, Tensor | None]], count: int
) -> int:
    """Trim response tokens and their leaf-independent node metadata together."""
    remaining = count
    while remaining and chunks:
        tokens, mask, logprobs = chunks[-1]
        if tokens.numel() <= remaining:
            remaining -= tokens.numel()
            chunks.pop()
        else:
            keep = tokens.numel() - remaining
            chunks[-1] = (
                tokens[:keep],
                mask[:keep],
                None if logprobs is None else logprobs[:keep],
            )
            remaining = 0
    return remaining


def _session_leaf_chunks(
    tree: Mapping[str, Any], leaf: Mapping[str, Any]
) -> tuple[list[Tensor], list[tuple[Tensor, Tensor, Tensor | None]], Tensor, Tensor, int]:
    """Resolve one Leaf into prompt/response chunk references, without making a full row."""
    kind = leaf.get("kind", "node_path")
    if kind == "trajectory":
        trajectory = leaf.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise ValueError("materialized trajectory Leaf must contain a trajectory mapping")
        prompt = _session_tree_tokens(trajectory.get("prompt_ids"), "trajectory.prompt_ids")
        response = _session_tree_tokens(trajectory.get("response_ids"), "trajectory.response_ids")
        raw_logprobs = trajectory.get("response_logprobs")
        logprobs = None
        if raw_logprobs is not None:
            if not isinstance(raw_logprobs, Tensor):
                raw_logprobs = torch.as_tensor(raw_logprobs)
            if raw_logprobs.ndim != 1 or torch.is_complex(raw_logprobs):
                raise ValueError("trajectory.response_logprobs must be a one-dimensional numeric tensor")
            logprobs = raw_logprobs.detach().to(device="cpu", dtype=torch.float32).clone()
            if logprobs.numel() != response.numel():
                raise ValueError("trajectory.response_logprobs must align with response_ids")
        response_mask = _session_tree_mask(leaf.get("response_mask"), "leaf.response_mask")
        loss_mask = _session_tree_mask(leaf.get("loss_mask", leaf["response_mask"]), "leaf.loss_mask")
        if response_mask.numel() != response.numel() or loss_mask.numel() != response.numel():
            raise ValueError("trajectory Leaf masks must align with response_ids")
        if bool((loss_mask & ~response_mask).any()):
            raise ValueError("Leaf loss_mask cannot select response_mask=0 tokens")
        response_chunk = (
            response,
            torch.ones_like(response, dtype=torch.bool),
            logprobs,
        )
        return [prompt], [response_chunk], response_mask, loss_mask, prompt.numel()

    if kind != "node_path":
        raise ValueError(f"unsupported Session Tree Leaf kind: {kind!r}")
    nodes = {node["node_id"]: node for node in tree["nodes"]}
    path = []
    visited = set()
    current = leaf.get("leaf_node_id")
    while current is not None:
        if current in visited:
            raise ValueError("cycle detected in Session Tree Node path")
        visited.add(current)
        node = nodes.get(current)
        if node is None:
            raise ValueError(f"Leaf references missing node {current!r}")
        path.append(node)
        current = node.get("parent_id")
    path.reverse()
    if not path or path[0]["node_id"] != tree["root_node_id"]:
        raise ValueError("Leaf path does not reach the declared Session Tree root")
    chain_id = leaf.get("chain_id")
    chain_start = next((i for i, node in enumerate(path) if node.get("chain_id") == chain_id), None)
    if chain_start is None:
        raise ValueError(f"Leaf chain_id={chain_id!r} is absent from its Node path")

    prompt_chunks: list[Tensor] = [
        _session_tree_tokens(path[0].get("prompt_ids"), "root.prompt_ids")
    ]
    response_chunks: list[tuple[Tensor, Tensor, Tensor | None]] = []
    for index, node in enumerate(path):
        removed = node.get("removed_prefix_token_count", 0)
        if not isinstance(removed, int) or isinstance(removed, bool) or removed < 0:
            raise ValueError("removed_prefix_token_count must be a non-negative integer")
        trim_response = _trim_response_chunks(response_chunks, removed)
        trim_prompt = _trim_token_chunks(prompt_chunks, trim_response)
        if trim_prompt:
            raise ValueError("removed_prefix_token_count exceeds the accumulated Leaf prefix")

        if index:
            prompt_delta = _session_tree_tokens(node.get("prompt_ids"), "node.prompt_ids")
            if prompt_delta.numel():
                response_chunks.append((
                    prompt_delta,
                    torch.zeros(prompt_delta.numel(), dtype=torch.bool),
                    torch.zeros(prompt_delta.numel(), dtype=torch.float32),
                ))

        if index == chain_start:
            prompt_chunks.extend(tokens for tokens, _, _ in response_chunks)
            response_chunks.clear()

        generated = _session_tree_tokens(node.get("response_ids"), "node.response_ids")
        raw_logprobs = node.get("response_logprobs")
        logprobs = None
        if raw_logprobs is not None:
            if not isinstance(raw_logprobs, Tensor):
                raw_logprobs = torch.as_tensor(raw_logprobs)
            if raw_logprobs.ndim != 1 or torch.is_complex(raw_logprobs):
                raise ValueError("node.response_logprobs must be a one-dimensional numeric tensor")
            logprobs = raw_logprobs.detach().to(device="cpu", dtype=torch.float32).clone()
            if logprobs.numel() != generated.numel():
                raise ValueError("node.response_logprobs must align with response_ids")
        if generated.numel():
            response_chunks.append((
                generated,
                torch.ones(generated.numel(), dtype=torch.bool),
                logprobs,
            ))

    response_mask = _session_tree_mask(leaf.get("response_mask"), "leaf.response_mask")
    loss_mask = _session_tree_mask(leaf.get("loss_mask", leaf["response_mask"]), "leaf.loss_mask")
    response_length = sum(tokens.numel() for tokens, _, _ in response_chunks)
    if response_mask.numel() != response_length or loss_mask.numel() != response_length:
        raise ValueError(
            f"Leaf masks must align with reconstructed response chunks: "
            f"response={response_length}, response_mask={response_mask.numel()}, "
            f"loss_mask={loss_mask.numel()}"
        )
    if bool((loss_mask & ~response_mask).any()):
        raise ValueError("Leaf loss_mask cannot select response_mask=0 tokens")
    prompt_length = sum(chunk.numel() for chunk in prompt_chunks)
    return prompt_chunks, response_chunks, response_mask, loss_mask, prompt_length


def _concat_chunk_range(chunks: list[Tensor], start: int, end: int) -> Tensor:
    parts = []
    offset = 0
    for chunk in chunks:
        chunk_end = offset + chunk.numel()
        left, right = max(start, offset), min(end, chunk_end)
        if left < right:
            parts.append(chunk[left - offset:right - offset])
        offset = chunk_end
        if offset >= end:
            break
    if not parts:
        raise ValueError(f"empty token range [{start}, {end})")
    return parts[0].clone() if len(parts) == 1 else torch.cat(parts)


def _session_tree_units(
    chunks: list[Tensor], prompt_length: int, minimum: int
) -> tuple[_SessionTreeUnit, ...]:
    total = sum(chunk.numel() for chunk in chunks)
    if total == 0:
        raise ValueError("Session Tree Leaf resolves to an empty token sequence")
    boundaries = set()
    cursor = 0
    for chunk in chunks:
        ends = torch.nonzero(chunk == QWEN3_IM_END_TOKEN_ID, as_tuple=False).flatten().tolist()
        boundaries.update(cursor + int(position) + 1 for position in ends)
        cursor += chunk.numel()
    if 0 < prompt_length < total:
        boundaries.add(prompt_length)
    boundaries.add(total)

    result = []
    previous = 0
    for end in sorted(boundaries):
        if end < total and end - previous < minimum:
            continue
        tokens = _concat_chunk_range(chunks, previous, end)
        result.append(_SessionTreeUnit(
            tokens.contiguous().numpy().tobytes(), previous, end, tokens
        ))
        previous = end
    return tuple(result)


def _compress_session_tree_units(
    node: _TurnNode,
    parent_id: SegmentId | None,
    start: int,
    units_by_row: list[tuple[_SessionTreeUnit, ...]],
    segments: list[SegmentSpec],
    paths: dict[int, tuple[int, ...]],
    path: tuple[int, ...],
) -> None:
    pending = [(node, parent_id, start, path)]
    while pending:
        node, parent_id, start, path = pending.pop()
        while not node.terminal_rows and len(node.children) == 1:
            node = next(iter(node.children.values()))
        source_units = units_by_row[node.source_row]
        token_parts = [
            unit.token_ids for unit in source_units
            if unit.start >= start and unit.end <= node.end
        ]
        if not token_parts:
            raise RuntimeError("Session Tree compression produced an empty Segment")
        token_ids = token_parts[0] if len(token_parts) == 1 else torch.cat(token_parts)
        segment_id = len(segments)
        segments.append(SegmentSpec(segment_id, parent_id, token_ids, start, start))
        path = (*path, segment_id)
        paths.update((row, path) for row in node.terminal_rows)
        pending.extend(
            (child, segment_id, node.end, path)
            for child in reversed(node.children.values())
        )


def _session_unit_token_at(units: tuple[_SessionTreeUnit, ...], position: int) -> int:
    for unit in units:
        if unit.start <= position < unit.end:
            return int(unit.token_ids[position - unit.start])
    raise IndexError(f"token position {position} is outside Session Tree Leaf units")


def build_tree_execution_plans_from_session_trees(
    session_trees: Sequence[Mapping[str, Any]],
    *,
    require_loss_mask_alignment: bool = False,
    tree_policy: str = "qwen_turn",
    turn_min_segment_tokens: int = 128,
    tree_builder: str = "lcp_sort",
) -> ForestExecutionPlan:
    """Build a minibatch Forest directly from independent Session Tree payloads.

    Each input mapping is exactly one Session (one root and its local Nodes /
    Leaves). Session-local topology is traversed as token chunks; no complete
    trajectory rows are materialized. Leaves from different Sessions are then
    grouped by UID and their exact qwen_turn units are merged by the selected
    Trie/LCP builder. Node IDs are scoped to their containing Session Tree.
    """
    if not session_trees:
        raise ValueError("cannot build a Forest from an empty Session Tree batch")
    if tree_policy != "qwen_turn":
        raise ValueError("Production TPR only supports qwen_turn")
    if tree_builder not in ("trie", "lcp_sort"):
        raise ValueError(f"unknown TPR tree_builder: {tree_builder!r}")
    if (not isinstance(turn_min_segment_tokens, int) or isinstance(turn_min_segment_tokens, bool)
            or turn_min_segment_tokens < 1):
        raise ValueError("turn_min_segment_tokens must be a positive integer")

    seen_trees, seen_sessions = set(), set()
    grouped_rows: dict[str, list[tuple[int, tuple[_SessionTreeUnit, ...], int, list[int]]]] = {}
    units_by_row: list[tuple[_SessionTreeUnit, ...]] = []
    selected_by_row: list[list[int]] = []
    prompt_lengths: list[int] = []
    row_index = 0

    for tree in session_trees:
        if not isinstance(tree, Mapping):
            raise TypeError("each NonTensorStack element must be a Session Tree mapping")
        tree_id, uid, session_id = tree.get("tree_id"), tree.get("uid"), tree.get("session_id")
        if not all(isinstance(value, str) and value for value in (tree_id, uid, session_id)):
            raise ValueError("each Session Tree needs non-empty tree_id, uid and session_id")
        if tree_id in seen_trees or (uid, session_id) in seen_sessions:
            raise ValueError("NonTensorStack must contain one unique tree per Session")
        seen_trees.add(tree_id)
        seen_sessions.add((uid, session_id))

        nodes = tree.get("nodes")
        leaves = tree.get("leaves")
        if (
            not isinstance(nodes, Sequence)
            or isinstance(nodes, (str, bytes))
            or not nodes
            or not isinstance(leaves, Sequence)
            or isinstance(leaves, (str, bytes))
            or not leaves
        ):
            raise ValueError("each Session Tree must contain Nodes and Leaves")
        node_by_id = {}
        for node in nodes:
            if not isinstance(node, Mapping):
                raise TypeError("Session Tree Nodes must be mappings")
            node_id = node.get("node_id")
            if (
                not isinstance(node_id, int)
                or isinstance(node_id, bool)
                or node_id < 0
                or node_id in node_by_id
            ):
                raise ValueError("Session Tree Node IDs must be unique non-negative integers")
            node_by_id[node_id] = node
        root_id = tree.get("root_node_id")
        roots = [node_id for node_id, node in node_by_id.items() if node.get("parent_id") is None]
        if roots != [root_id]:
            raise ValueError("each Session Tree must have exactly its declared root")
        children_by_parent = {node_id: [] for node_id in node_by_id}
        for node_id, node in node_by_id.items():
            parent_id = node.get("parent_id")
            if parent_id is not None:
                if not isinstance(parent_id, int) or isinstance(parent_id, bool):
                    raise ValueError(f"Session Tree Node {node_id} has an invalid parent_id")
                if parent_id not in node_by_id:
                    raise ValueError(f"Session Tree Node {node_id} references missing parent {parent_id}")
                children_by_parent[parent_id].append(node_id)
        visited_nodes = set()
        pending_nodes = [root_id]
        while pending_nodes:
            node_id = pending_nodes.pop()
            if node_id in visited_nodes:
                raise ValueError("cycle or duplicate child in Session Tree topology")
            visited_nodes.add(node_id)
            pending_nodes.extend(children_by_parent[node_id])
        if visited_nodes != set(node_by_id):
            raise ValueError("Session Tree contains unreachable Nodes")

        leaf_ids = set()
        for leaf in leaves:
            if not isinstance(leaf, Mapping):
                raise TypeError("Session Tree Leaves must be mappings")
            leaf_id = leaf.get("leaf_id")
            if not isinstance(leaf_id, str) or not leaf_id or leaf_id in leaf_ids:
                raise ValueError("Session Tree Leaf IDs must be unique non-empty strings")
            leaf_ids.add(leaf_id)
            prompt_chunks, response_chunks, response_mask, loss_mask, prompt_length = (
                _session_leaf_chunks(tree, leaf)
            )
            if require_loss_mask_alignment and not torch.equal(response_mask, loss_mask):
                raise ValueError("Leaf response_mask and loss_mask disagree")
            chunks = [chunk for chunk in prompt_chunks if chunk.numel()]
            chunks.extend(tokens for tokens, _, _ in response_chunks if tokens.numel())
            total_length = sum(chunk.numel() for chunk in chunks)
            if (
                prompt_length > total_length
                or total_length - prompt_length != response_mask.numel()
            ):
                raise ValueError("Leaf prompt/response boundary does not align with its response_mask")
            units = _session_tree_units(chunks, prompt_length, turn_min_segment_tokens)
            selected = torch.nonzero(response_mask, as_tuple=False).flatten().tolist()
            if selected and prompt_length + selected[0] == 0:
                raise ValueError("Leaf has no preceding query token for its first supervised token")
            grouped_rows.setdefault(uid, []).append((row_index, units, prompt_length, selected))
            units_by_row.append(units)
            selected_by_row.append(selected)
            prompt_lengths.append(prompt_length)
            row_index += 1

    result = []
    for uid, rows in grouped_rows.items():
        if tree_builder == "lcp_sort":
            root = _build_lcp_group([
                (row, tuple((unit.signature, unit.end) for unit in units))
                for row, units, _, _ in rows
            ])
        else:
            root = _TurnNode(0, rows[0][0])
            for row, units, _, _ in rows:
                node = root
                for unit in units:
                    node = node.children.setdefault(
                        unit.signature, _TurnNode(unit.end, row)
                    )
                    node.member_rows.append(row)
                node.terminal_rows.append(row)

        for tree_index, child in enumerate(root.children.values()):
            segments, paths = [], {}
            _compress_session_tree_units(
                child, None, 0, units_by_row, segments, paths, ()
            )
            plan = SegmentPlan(segments, root_id=0)
            refs = []
            for row in child.member_rows:
                path = paths[row]
                ends = [plan.get(segment_id).position_end for segment_id in path]
                for offset in selected_by_row[row]:
                    target = prompt_lengths[row] + offset
                    segment_id = path[bisect_right(ends, target - 1)]
                    refs.append(SegmentObjectiveRef(
                        segment_id,
                        target - 1 - plan.get(segment_id).position_start,
                        _session_unit_token_at(units_by_row[row], target),
                        row,
                        offset,
                    ))
            result.append(TreeExecutionPlan(
                uid, tree_index, tuple(child.member_rows), plan, tuple(refs)
            ))

    return ForestExecutionPlan(tuple(result), sum(map(len, selected_by_row)))

def _write_record(hasher, record) -> None:
    hasher.update(json.dumps(record, separators=(",", ":")).encode())


def _write_plan(hasher, plan: SegmentPlan) -> None:
    _write_record(hasher, plan.root_id)
    # Insertion order determines DFS order and the collective call sequence.
    for segment in plan.segments.values():
        _write_record(hasher, (segment.segment_id, segment.parent_id, segment.position_start, segment.length))
        hasher.update(segment.token_ids.contiguous().numpy().tobytes())


def digest_segment_plan(plan: SegmentPlan) -> str:
    hasher = hashlib.sha256(b"TPR-PPO-turn-v2")
    _write_plan(hasher, plan)
    return hasher.hexdigest()


def digest_forest_plan(forest: ForestExecutionPlan) -> str:
    hasher = hashlib.sha256(b"TPR-PPO-turn-v2")
    _write_record(hasher, forest.logical_loss_tokens)
    for tree in forest.trees:
        _write_record(hasher, (tree.uid, tree.tree_index, tree.member_rows))
        _write_plan(hasher, tree.segment_plan)
        for ref in tree.objective_refs:
            _write_record(hasher, (ref.segment_id, ref.query_offset, ref.target_token_id,
                                   ref.sample_row, ref.response_offset))
    return hasher.hexdigest()


def assert_tp_plan_agreement(digest: str, *, tp_size: int, cp_size: int = 1) -> None:
    """Reject divergent plans on the native TP group (TP×CP when CP is enabled).

    Called ONCE per TPR minibatch, *before* model forward or TP collectives.
    Uses only a 32-byte SHA256 signature per rank, not model tensors / KV.
    Every peer receives the same success or mismatch result.
    """
    if tp_size < 1 or cp_size < 1:
        raise ValueError("tp_size and cp_size must be positive")
    size = tp_size * cp_size
    if size == 1:
        return
    if len(digest) != 64:
        raise ValueError("SHA256 digest must be a 64-character hex string")

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("TPR TP/CP requires an initialized distributed process group")
    axis = "TP×CP" if cp_size > 1 else "TP"
    group = (parallel_state.get_tensor_and_context_parallel_group() if cp_size > 1
             else parallel_state.get_tensor_model_parallel_group())
    if group is None:
        raise RuntimeError(f"TPR cannot resolve native Megatron {axis} group")
    if dist.get_world_size(group=group) != size:
        raise RuntimeError(f"TPR {axis} group size differs from Engine config")
    # HCCL AllGather supports int64, so store SHA256 as eight int64 words
    # (each word is a non-negative 32-bit unsigned value).
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    device = torch.device("cpu")
    if dist.get_backend(group) != "gloo":
        from verl.utils.device import get_device_id, get_device_name

        device = torch.device(get_device_name(), get_device_id())
    local = torch.tensor(words, dtype=torch.int64, device=device)
    signatures = [torch.empty_like(local) for _ in range(size)]
    dist.all_gather(signatures, local, group=group)
    if any(not torch.equal(value, local) for value in signatures):
        rank = dist.get_rank(group=group)
        raise RuntimeError(
            f"TPR {axis} schedule mismatch on rank={rank}: forest, SegmentPlan, "
            "topology or PPO refs differ across model-parallel peers"
        )
