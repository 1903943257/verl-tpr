"""Versioned Session Tree schema used by the TQ transport prototype."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

TREE_SCHEMA_VERSION = 1
TREE_FIELD = "trajectory_tree"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _tokens(value: Any, name: str) -> list[int]:
    if isinstance(value, Tensor):
        if value.ndim != 1 or value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
        values = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
    else:
        raise TypeError(f"{name} must be a one-dimensional tensor or integer sequence")
    for index, token in enumerate(values):
        _integer(token, f"{name}[{index}]")
    return values


def _mask(value: Any, name: str) -> list[bool]:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
    else:
        raise TypeError(f"{name} must be a one-dimensional tensor or sequence")
    result = []
    for index, item in enumerate(values):
        if isinstance(item, bool):
            result.append(item)
        elif isinstance(item, int) and item in (0, 1):
            result.append(bool(item))
        else:
            raise ValueError(f"{name}[{index}] must be bool or 0/1, got {item!r}")
    return result


def validate_session_tree(tree: Mapping[str, Any]) -> None:
    """Validate node topology and payload alignment without mutating the tree."""
    if not isinstance(tree, Mapping):
        raise TypeError("Session Tree must be a mapping")
    if tree.get("schema_version") != TREE_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {tree.get('schema_version')!r}")
    for field in ("tree_id", "uid", "session_id"):
        if not isinstance(tree.get(field), str) or not tree[field]:
            raise ValueError(f"{field} must be a non-empty string")
    nodes, leaves = tree.get("nodes"), tree.get("leaves")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)) or not nodes:
        raise ValueError("nodes must be a non-empty sequence")
    if not isinstance(leaves, Sequence) or isinstance(leaves, (str, bytes)) or not leaves:
        raise ValueError("leaves must be a non-empty sequence")

    by_id: dict[int, Mapping[str, Any]] = {}
    children: dict[int, list[int]] = {}
    roots = []
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise TypeError(f"nodes[{index}] must be a mapping")
        node_id = _integer(node.get("node_id"), f"nodes[{index}].node_id")
        if node_id in by_id:
            raise ValueError(f"duplicate node_id {node_id}")
        parent_id = node.get("parent_id")
        if parent_id is None:
            roots.append(node_id)
        else:
            parent_id = _integer(parent_id, f"nodes[{index}].parent_id")
            children.setdefault(parent_id, []).append(node_id)
        _tokens(node.get("prompt_ids"), f"nodes[{index}].prompt_ids")
        response = _tokens(node.get("response_ids"), f"nodes[{index}].response_ids")
        _integer(node.get("chain_id"), f"nodes[{index}].chain_id")
        _integer(node.get("removed_prefix_token_count", 0), f"nodes[{index}].removed_prefix_token_count")
        logprobs = node.get("response_logprobs")
        if logprobs is not None:
            if isinstance(logprobs, Tensor):
                if logprobs.ndim != 1 or torch.is_complex(logprobs):
                    raise ValueError(f"nodes[{index}].response_logprobs must be a one-dimensional numeric tensor")
                logprob_values = logprobs.detach().cpu().tolist()
            elif isinstance(logprobs, Sequence) and not isinstance(logprobs, (str, bytes)):
                logprob_values = list(logprobs)
            else:
                raise TypeError(f"nodes[{index}].response_logprobs must be a sequence, tensor, or None")
            if len(logprob_values) != len(response):
                raise ValueError(f"nodes[{index}] response_logprobs must align with response_ids")
            if any(value is not None and not isinstance(value, (int, float)) for value in logprob_values):
                raise ValueError(f"nodes[{index}].response_logprobs must contain numbers or None")
        by_id[node_id] = node
        children.setdefault(node_id, [])

    root_id = _integer(tree.get("root_node_id"), "root_node_id")
    if roots != [root_id]:
        raise ValueError(f"tree must have exactly root_node_id={root_id}, got roots={roots}")
    for node_id, node in by_id.items():
        parent_id = node.get("parent_id")
        if parent_id is not None and parent_id not in by_id:
            raise ValueError(f"node {node_id} references missing parent {parent_id}")

    visited: set[int] = set()
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            raise ValueError(f"cycle or duplicate child reference at node {node_id}")
        visited.add(node_id)
        pending.extend(reversed(children[node_id]))
    if visited != set(by_id):
        raise ValueError(f"nodes are unreachable from root (cycle or orphan): {sorted(set(by_id) - visited)}")

    leaf_ids = set()
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, Mapping):
            raise TypeError(f"leaves[{index}] must be a mapping")
        leaf_id = leaf.get("leaf_id")
        if not isinstance(leaf_id, str) or not leaf_id or leaf_id in leaf_ids:
            raise ValueError(f"leaves[{index}].leaf_id must be unique and non-empty")
        leaf_ids.add(leaf_id)
        _integer(leaf.get("leaf_index"), f"leaves[{index}].leaf_index")
        chain_id = _integer(leaf.get("chain_id"), f"leaves[{index}].chain_id")
        kind = leaf.get("kind", "node_path")
        if kind == "node_path":
            leaf_node_id = _integer(leaf.get("leaf_node_id"), f"leaves[{index}].leaf_node_id")
            if leaf_node_id not in by_id:
                raise ValueError(f"leaf {leaf_id!r} references missing node {leaf_node_id}")
            current = leaf_node_id
            path_chain_ids = set()
            while current is not None:
                node = by_id[current]
                path_chain_ids.add(node["chain_id"])
                current = node.get("parent_id")
            if chain_id not in path_chain_ids:
                raise ValueError(f"leaf {leaf_id!r} chain_id={chain_id} is absent from its path")
        elif kind == "trajectory":
            trajectory = leaf.get("trajectory")
            if not isinstance(trajectory, Mapping):
                raise TypeError(f"leaf {leaf_id!r} materialized trajectory must be a mapping")
            prompt = _tokens(trajectory.get("prompt_ids"), f"leaves[{index}].trajectory.prompt_ids")
            response = _tokens(trajectory.get("response_ids"), f"leaves[{index}].trajectory.response_ids")
            logprobs = trajectory.get("response_logprobs")
            if logprobs is not None:
                if isinstance(logprobs, Tensor):
                    if logprobs.ndim != 1 or torch.is_complex(logprobs):
                        raise ValueError(f"leaves[{index}] response_logprobs must be one-dimensional")
                    logprob_values = logprobs.detach().cpu().tolist()
                elif isinstance(logprobs, Sequence) and not isinstance(logprobs, (str, bytes)):
                    logprob_values = list(logprobs)
                else:
                    raise TypeError(f"leaves[{index}] response_logprobs must be a sequence, tensor, or None")
                if len(logprob_values) != len(response):
                    raise ValueError(f"leaf {leaf_id!r} response_logprobs must align with response_ids")
            if len(prompt) + len(response) == 0:
                raise ValueError(f"leaf {leaf_id!r} materialized trajectory cannot be empty")
        else:
            raise ValueError(f"leaf {leaf_id!r} has unsupported kind {kind!r}")
        if "response_mask" not in leaf:
            raise ValueError(f"leaf {leaf_id!r} is missing response_mask")
        _mask(leaf["response_mask"], f"leaves[{index}].response_mask")
        if "loss_mask" in leaf:
            _mask(leaf["loss_mask"], f"leaves[{index}].loss_mask")
        if "rm_scores" in leaf:
            scores = leaf["rm_scores"]
            if isinstance(scores, Tensor):
                if scores.ndim != 1 or torch.is_complex(scores):
                    raise ValueError(f"leaves[{index}].rm_scores must be a one-dimensional numeric tensor")
                score_values = scores.detach().cpu().tolist()
            elif isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
                score_values = list(scores)
            else:
                raise TypeError(f"leaves[{index}].rm_scores must be a sequence, tensor, or None")
            if any(not isinstance(value, (int, float)) for value in score_values):
                raise ValueError(f"leaves[{index}].rm_scores must contain numeric values")


def pack_session_trees(trees: Sequence[Mapping[str, Any]]):
    """Build an in-memory TensorDict with NonTensorStack(*trees), batch_size=[N]."""
    from tensordict import NonTensorStack, TensorDict

    if not trees:
        raise ValueError("cannot pack an empty Session Tree batch")
    for tree in trees:
        validate_session_tree(tree)
    tree_ids = [tree["tree_id"] for tree in trees]
    if len(set(tree_ids)) != len(tree_ids):
        raise ValueError("tree_id values must be unique within a NonTensorStack batch")
    return TensorDict({TREE_FIELD: NonTensorStack(*trees)}, batch_size=[len(trees)])


def unpack_session_trees(fields: Any) -> list[dict[str, Any]]:
    """Consume the in-memory NonTensorStack without flattening nested tensor payloads."""
    value = fields.get(TREE_FIELD)
    if value is None or not hasattr(value, "tolist"):
        raise TypeError(f"TQ fields must contain a NonTensorStack at {TREE_FIELD!r}")
    trees = value.tolist()
    if not isinstance(trees, list):
        raise TypeError("trajectory_tree NonTensorStack.tolist() must return a list")
    for tree in trees:
        validate_session_tree(tree)
    return trees
