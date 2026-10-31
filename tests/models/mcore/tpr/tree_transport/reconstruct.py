"""Leaf reconstruction adapter mirroring UniAgent PR #273 path ownership."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .schema import validate_session_tree


def _long(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", dtype=torch.long).clone()
    return torch.tensor(list(value), dtype=torch.long)


def _bool(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", dtype=torch.bool).clone()
    return torch.tensor(list(value), dtype=torch.bool)


def _path(tree: dict[str, Any], leaf_node_id: int) -> list[dict[str, Any]]:
    nodes = {node["node_id"]: node for node in tree["nodes"]}
    result = []
    current = leaf_node_id
    while current is not None:
        node = nodes[current]
        result.append(node)
        current = node.get("parent_id")
    return list(reversed(result))


def reconstruct_leaf(tree: dict[str, Any], leaf_id: str) -> dict[str, Any]:
    """Materialize one Leaf using the current UniAgent chain-start/rollback rules."""
    validate_session_tree(tree)
    leaves = [leaf for leaf in tree["leaves"] if leaf["leaf_id"] == leaf_id]
    if len(leaves) != 1:
        raise KeyError(f"expected exactly one leaf {leaf_id!r}, found {len(leaves)}")
    leaf = leaves[0]
    if leaf.get("kind", "node_path") == "trajectory":
        trajectory = leaf["trajectory"]
        prompt = _long(trajectory["prompt_ids"])
        response = _long(trajectory["response_ids"])
        response_mask = _bool(leaf["response_mask"])
        loss_mask = _bool(leaf.get("loss_mask", response_mask))
        if response_mask.numel() != response.numel() or loss_mask.numel() != response.numel():
            raise ValueError(f"materialized leaf {leaf_id!r} masks must align with response_ids")
        if bool((loss_mask & ~response_mask).any()):
            raise ValueError(f"materialized leaf {leaf_id!r} loss_mask exceeds response_mask")
        raw_logprobs = trajectory.get("response_logprobs")
        logprobs = None
        if raw_logprobs is not None:
            values = raw_logprobs.detach().cpu().tolist() if isinstance(raw_logprobs, Tensor) else list(raw_logprobs)
            if len(values) != response.numel():
                raise ValueError(f"materialized leaf {leaf_id!r} logprobs do not align with response")
            if all(value is not None for value in values):
                logprobs = torch.tensor(values, dtype=torch.float32)
        return {
            "key": f"{tree['uid']}_{tree['session_id']}_{leaf['leaf_index']}",
            "uid": tree["uid"],
            "session_id": tree["session_id"],
            "tree_id": tree["tree_id"],
            "leaf_id": leaf_id,
            "leaf_node_id": None,
            "chain_id": leaf["chain_id"],
            "prompts": prompt,
            "responses": response,
            "input_ids": torch.cat((prompt, response)),
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "rollout_log_probs": logprobs,
            "reward_score": leaf.get("reward_score"),
            "finished": leaf.get("finished"),
        }
    if leaf.get("kind", "node_path") != "node_path":
        raise ValueError(f"unsupported leaf kind: {leaf.get('kind')!r}")
    path = _path(tree, leaf["leaf_node_id"])
    chain_start = next((i for i, node in enumerate(path) if node["chain_id"] == leaf["chain_id"]), None)
    if chain_start is None:
        raise ValueError(f"leaf {leaf_id!r} has no chain start on its path")

    prompt_ids = _long(path[0]["prompt_ids"]).tolist()
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float | None] | None = [] if path[-1].get("response_logprobs") is not None else None

    for index, node in enumerate(path):
        removed = int(node.get("removed_prefix_token_count", 0))
        if removed:
            trim_response = min(removed, len(response_ids))
            if trim_response:
                del response_ids[-trim_response:]
                del response_mask[-trim_response:]
                if response_logprobs is not None:
                    del response_logprobs[-trim_response:]
            trim_prompt = removed - trim_response
            if trim_prompt:
                del prompt_ids[-trim_prompt:]

        if index:
            prompt_delta = _long(node["prompt_ids"]).tolist()
            response_ids.extend(prompt_delta)
            response_mask.extend([0] * len(prompt_delta))
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(prompt_delta))

        if index == chain_start:
            prompt_ids.extend(response_ids)
            response_ids.clear()
            response_mask.clear()
            if response_logprobs is not None:
                response_logprobs.clear()

        generated = _long(node["response_ids"]).tolist()
        response_ids.extend(generated)
        response_mask.extend([1] * len(generated))
        if response_logprobs is not None:
            node_logprobs = node.get("response_logprobs")
            response_logprobs.extend(
                list(node_logprobs) if node_logprobs is not None else [None] * len(generated)
            )

    derived_mask = torch.tensor(response_mask, dtype=torch.bool)
    # The Leaf mask is authoritative: a generated node token may be masked for
    # one trajectory (e.g. an aborted/error suffix) while remaining present in
    # the shared node payload. Only alignment, not equality to a node-global
    # mask, is required.
    reconstructed_mask = _bool(leaf.get("response_mask", derived_mask))
    if reconstructed_mask.numel() != len(response_ids):
        raise ValueError(f"leaf {leaf_id!r} response_mask length does not match reconstructed response")
    loss_mask = _bool(leaf.get("loss_mask", reconstructed_mask))
    if loss_mask.numel() != reconstructed_mask.numel():
        raise ValueError(f"leaf {leaf_id!r} loss_mask length does not match reconstructed response")
    if bool((loss_mask & ~reconstructed_mask).any()):
        raise ValueError(f"leaf {leaf_id!r} loss_mask supervises tokens excluded by response_mask")

    aligned_logprobs = None
    if response_logprobs is not None and all(value is not None for value in response_logprobs):
        aligned_logprobs = torch.tensor(response_logprobs, dtype=torch.float32)

    prompt = torch.tensor(prompt_ids, dtype=torch.long)
    response = torch.tensor(response_ids, dtype=torch.long)
    return {
        "key": f"{tree['uid']}_{tree['session_id']}_{leaf['leaf_index']}",
        "uid": tree["uid"],
        "session_id": tree["session_id"],
        "tree_id": tree["tree_id"],
        "leaf_id": leaf_id,
        "leaf_node_id": leaf["leaf_node_id"],
        "chain_id": leaf["chain_id"],
        "prompts": prompt,
        "responses": response,
        "input_ids": torch.cat((prompt, response)),
        "response_mask": reconstructed_mask,
        "loss_mask": loss_mask,
        "rollout_log_probs": aligned_logprobs,
        "reward_score": leaf.get("reward_score"),
        "finished": leaf.get("finished"),
    }


def reconstruct_session_tree(tree: dict[str, Any]) -> list[dict[str, Any]]:
    validate_session_tree(tree)
    return [reconstruct_leaf(tree, leaf["leaf_id"]) for leaf in tree["leaves"]]


def rows_to_tpr_batch(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[Tensor]]]:
    """Adapt Leaf rows to the current production TPR planner input."""
    if not rows:
        raise ValueError("cannot build a TPR batch without Leaf rows")
    keys = [row["key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("reconstructed trajectory keys must be unique")
    batch = {
        field: [row[field] for row in rows]
        for field in ("input_ids", "prompts", "responses", "response_mask", "loss_mask")
    }
    return keys, batch
