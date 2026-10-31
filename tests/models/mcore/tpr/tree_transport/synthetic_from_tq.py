"""Synthesize independent Session-local Chain/Tree fixtures from flat TQ rows.

The source TQ dump contains complete trajectory rows but no Session-internal
forks. This helper fabricates several per-Session topologies for consumer tests.
It does not contact TQ or test its persistence/transport implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .schema import TREE_SCHEMA_VERSION, validate_session_tree

_REQUIRED_FIELDS = (
    "input_ids",
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
    "rollout_log_probs",
    "rm_scores",
)


def _rows(value: Any, *, field: str) -> list[Tensor]:
    if isinstance(value, (list, tuple)):
        result = list(value)
    else:
        unbind = getattr(value, "unbind", None)
        if not callable(unbind):
            raise TypeError(f"TQ field {field!r} is not row-addressable: {type(value)!r}")
        try:
            result = list(unbind(0))
        except TypeError:
            result = list(unbind())
    rows = []
    for index, row in enumerate(result):
        if not isinstance(row, Tensor) or row.ndim != 1:
            raise ValueError(
                f"TQ field {field!r} row {index} must be a 1-D Tensor, "
                f"got {type(row)!r} shape={getattr(row, 'shape', None)}"
            )
        rows.append(row.detach().to(device="cpu").clone())
    return rows


def _parse_key(key: str) -> tuple[str, str, int]:
    if not isinstance(key, str):
        raise TypeError(f"TQ trajectory key must be str, got {type(key)!r}")
    parts = key.rsplit("_", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise ValueError(f"unexpected TQ trajectory key format: {key!r}")
    try:
        trajectory_index = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"trajectory key has invalid trajectory index: {key!r}") from exc
    return parts[0], parts[1], trajectory_index


def load_tq_rollout_payload(path: str | Path) -> dict[str, Any]:
    """Load the existing TQ-produced .pt payload (no TQ client involved)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"TQ dump must be a dict, got {type(payload)!r}")
    missing = [name for name in ("tensordict", "keys") if name not in payload]
    if missing:
        raise ValueError(f"TQ dump is missing top-level fields: {missing}")
    return payload


def _make_chain_session(
    *, uid: str, session_id: str, tree_id: str, key: str,
    source_session_id: str, source_trajectory_index: int,
    prompt: Tensor, response: Tensor, response_mask: Tensor,
    loss_mask: Tensor, logprobs: Tensor, rm_scores: Tensor,
) -> dict[str, Any]:
    return {
        "schema_version": TREE_SCHEMA_VERSION,
        "tree_id": tree_id,
        "uid": uid,
        "session_id": session_id,
        "root_node_id": 0,
        "nodes": [{
            "node_id": 0,
            "parent_id": None,
            "chain_id": 0,
            "prompt_ids": prompt.clone(),
            "response_ids": response.clone(),
            "response_logprobs": logprobs.clone(),
            "removed_prefix_token_count": 0,
        }],
        "leaves": [{
            "leaf_id": f"{session_id}-leaf-0",
            "leaf_index": 0,
            "kind": "node_path",
            "leaf_node_id": 0,
            "chain_id": 0,
            "source_key": key,
            "source_session_id": source_session_id,
            "source_trajectory_index": source_trajectory_index,
            "response_mask": response_mask.clone(),
            "loss_mask": loss_mask.clone(),
            "rm_scores": rm_scores.clone(),
        }],
        "synthetic_from": "flat_tq_rollout_batch",
        "synthetic_topology": "single_chain",
        "source_tq_keys": [key],
    }



def _make_fork_session(
    *,
    uid: str,
    session_id: str,
    tree_id: str,
    source_row: dict[str, Any],
    shared_prefix_tokens: int = 16,
) -> dict[str, Any]:
    """Create two synthetic branches inside this one source Session.

    Branch A preserves the source response. Branch B drops one token directly
    after the shared prefix; all aligned per-token metadata is sliced at the
    same position. A Leaf-specific mask override exercises shared-node
    ownership without introducing another source Session into this Tree.
    """
    prompt = source_row["prompts"]
    response = source_row["responses"]
    response_mask = source_row["response_mask"]
    loss_mask = source_row["loss_mask"]
    logprobs = source_row["rollout_log_probs"]
    rm_scores = source_row["rm_scores"]
    if response.numel() < 2:
        raise ValueError("cannot synthesize a fork from a response shorter than two tokens")
    prefix_length = min(shared_prefix_tokens, response.numel() - 1)
    shared_prefix = response[:prefix_length]
    suffix_a = response[prefix_length:]
    suffix_b = response[prefix_length + 1:]

    mask_a = response_mask.clone()
    loss_a = loss_mask.clone()
    scores_a = rm_scores.clone()

    mask_b = torch.cat((
        response_mask[:prefix_length],
        response_mask[prefix_length + 1:],
    )).clone()
    loss_b = torch.cat((
        loss_mask[:prefix_length],
        loss_mask[prefix_length + 1:],
    )).clone()
    scores_b = torch.cat((
        rm_scores[:prefix_length],
        rm_scores[prefix_length + 1:],
    )).clone()

    # The same shared Node can have different supervision ownership in
    # different Leaves. Mask one shared-prefix target in Branch B when possible.
    shared_supervised = torch.nonzero(
        mask_b[:prefix_length].bool() & loss_b[:prefix_length].bool(),
        as_tuple=False,
    ).flatten()
    mask_override_positions = []
    if shared_supervised.numel():
        override = int(shared_supervised[0])
        mask_b[override] = 0
        loss_b[override] = 0
        mask_override_positions.append(override)

    source_key = source_row["key"]
    source_session_id = source_row["source_session_id"]
    source_trajectory_index = source_row["source_trajectory_index"]
    synthetic_branch_key = (
        f"{uid}_{source_session_id}_{source_trajectory_index + 1000000}"
    )
    nodes = [
        {
            "node_id": 0,
            "parent_id": None,
            "chain_id": 0,
            "prompt_ids": prompt.clone(),
            "response_ids": shared_prefix.clone(),
            "response_logprobs": logprobs[:prefix_length].clone(),
            "removed_prefix_token_count": 0,
        },
        {
            "node_id": 1,
            "parent_id": 0,
            "chain_id": 0,
            "prompt_ids": torch.empty(0, dtype=torch.long),
            "response_ids": suffix_a.clone(),
            "response_logprobs": logprobs[prefix_length:].clone(),
            "removed_prefix_token_count": 0,
        },
        {
            "node_id": 2,
            "parent_id": 0,
            "chain_id": 0,
            "prompt_ids": torch.empty(0, dtype=torch.long),
            "response_ids": suffix_b.clone(),
            "response_logprobs": logprobs[prefix_length + 1:].clone(),
            "removed_prefix_token_count": 0,
        },
    ]
    leaves = [
        {
            "leaf_id": f"{session_id}-leaf-0",
            "leaf_index": 0,
            "kind": "node_path",
            "leaf_node_id": 1,
            "chain_id": 0,
            "source_key": source_key,
            "source_tq_key": source_key,
            "source_session_id": source_session_id,
            "source_trajectory_index": source_trajectory_index,
            "response_mask": mask_a,
            "loss_mask": loss_a,
            "rm_scores": scores_a,
        },
        {
            "leaf_id": f"{session_id}-leaf-1",
            "leaf_index": 1,
            "kind": "node_path",
            "leaf_node_id": 2,
            "chain_id": 0,
            "source_key": synthetic_branch_key,
            "source_tq_key": source_key,
            "source_session_id": source_session_id,
            "source_trajectory_index": source_trajectory_index,
            "response_mask": mask_b,
            "loss_mask": loss_b,
            "rm_scores": scores_b,
            "synthetic_branch": "drop_first_suffix_token",
            "mask_override_positions": mask_override_positions,
        },
    ]
    return {
        "schema_version": TREE_SCHEMA_VERSION,
        "tree_id": tree_id,
        "uid": uid,
        "session_id": session_id,
        "root_node_id": 0,
        "nodes": nodes,
        "leaves": leaves,
        "synthetic_from": "one_flat_tq_row",
        "synthetic_topology": "session_fork",
        "source_tq_keys": [source_key],
        "shared_prefix_tokens": prefix_length,
    }


def build_synthetic_session_trees_from_tq(
    payload: dict[str, Any],
    *,
    topology: str = "all_chain",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Synthesize independent per-Session Chain/Tree payloads from flat TQ rows.

    Each source TQ row remains in its own Session. For fork cases, a second
    branch is fabricated from that same row by dropping one suffix token;
    no rows from different Sessions are combined into one Session Tree.

    topology:
      - all_chain: all source Sessions remain single Chains.
      - mixed: first half are single Chains, second half are per-Session Trees.
      - all_tree: every source Session is converted into a two-Leaf Tree.
    """
    batch = payload["tensordict"]
    keys = list(payload["keys"])
    if not keys:
        raise ValueError("TQ payload contains no trajectory keys")
    if len(keys) != int(batch.batch_size[0]):
        raise ValueError(
            f"TQ key/row mismatch: keys={len(keys)} batch_rows={batch.batch_size[0]}"
        )
    if topology not in ("all_chain", "mixed", "all_tree"):
        raise ValueError(f"unsupported synthetic topology: {topology!r}")

    parsed = [_parse_key(key) for key in keys]
    rows_by_field = {}
    for field in _REQUIRED_FIELDS:
        if field not in batch.keys():
            raise ValueError(f"TQ TensorDict is missing required field {field!r}")
        rows_by_field[field] = _rows(batch[field], field=field)
        if len(rows_by_field[field]) != len(keys):
            raise ValueError(
                f"TQ field {field!r} has {len(rows_by_field[field])} rows; "
                f"expected {len(keys)}"
            )

    source_rows = []
    for index, (key, parsed_key) in enumerate(zip(keys, parsed, strict=True)):
        uid, source_session_id, source_trajectory_index = parsed_key
        prompt = rows_by_field["prompts"][index]
        response = rows_by_field["responses"][index]
        input_ids = rows_by_field["input_ids"][index]
        response_mask = rows_by_field["response_mask"][index]
        loss_mask = rows_by_field["loss_mask"][index]
        logprobs = rows_by_field["rollout_log_probs"][index]
        rm_scores = rows_by_field["rm_scores"][index]

        if not torch.equal(input_ids, torch.cat((prompt, response))):
            raise ValueError(f"source row {index}: input_ids != prompts + responses")
        aligned = {
            "response_mask": response_mask.numel(),
            "loss_mask": loss_mask.numel(),
            "rollout_log_probs": logprobs.numel(),
            "rm_scores": rm_scores.numel(),
        }
        if any(length != response.numel() for length in aligned.values()):
            raise ValueError(
                f"source row {index}: response-aligned fields differ; "
                f"response={response.numel()}, fields={aligned}"
            )
        if bool((loss_mask.bool() & ~response_mask.bool()).any()):
            raise ValueError(f"source row {index}: loss_mask selects response_mask=0")

        source_rows.append({
            "key": key,
            "uid": uid,
            "source_session_id": source_session_id,
            "source_trajectory_index": source_trajectory_index,
            "prompts": prompt,
            "responses": response,
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "rollout_log_probs": logprobs,
            "rm_scores": rm_scores,
        })

    if topology == "all_chain":
        fork_start = len(source_rows)
    elif topology == "mixed":
        fork_start = len(source_rows) // 2
    else:
        fork_start = 0

    trees = []
    for index, row in enumerate(source_rows):
        uid = row["uid"]
        session_id = row["source_session_id"]
        tree_id = f"synthetic-tree-{topology}-{index}"
        if index < fork_start:
            tree = _make_chain_session(
                uid=uid,
                session_id=session_id,
                tree_id=tree_id,
                key=row["key"],
                source_session_id=row["source_session_id"],
                source_trajectory_index=row["source_trajectory_index"],
                prompt=row["prompts"],
                response=row["responses"],
                response_mask=row["response_mask"],
                loss_mask=row["loss_mask"],
                logprobs=row["rollout_log_probs"],
                rm_scores=row["rm_scores"],
            )
        else:
            tree = _make_fork_session(
                uid=uid,
                session_id=session_id,
                tree_id=tree_id,
                source_row=row,
            )
        validate_session_tree(tree)
        trees.append(tree)

    all_prompts = [row["prompts"] for row in source_rows]
    exact_common_prompt = bool(all_prompts) and all(
        torch.equal(all_prompts[0], prompt) for prompt in all_prompts[1:]
    )
    logical_input_tokens = 0
    logical_loss_tokens = 0
    leaf_count = 0
    for tree in trees:
        prompt_length = int(tree["nodes"][0]["prompt_ids"].numel())
        for leaf in tree["leaves"]:
            if leaf.get("kind", "node_path") == "trajectory":
                response_length = int(leaf["trajectory"]["response_ids"].numel())
            else:
                path = []
                nodes_by_id = {node["node_id"]: node for node in tree["nodes"]}
                current = leaf["leaf_node_id"]
                while current is not None:
                    path.append(nodes_by_id[current])
                    current = nodes_by_id[current].get("parent_id")
                path.reverse()
                chain_start = next(
                    i for i, node in enumerate(path)
                    if node["chain_id"] == leaf["chain_id"]
                )
                response_length = sum(
                    int(node["response_ids"].numel())
                    for node in path[chain_start:]
                ) + sum(
                    int(node["prompt_ids"].numel())
                    for node in path[chain_start + 1:]
                )
            logical_input_tokens += prompt_length + response_length
            logical_loss_tokens += int(leaf["loss_mask"].bool().sum())
            leaf_count += 1

    summary = {
        "topology": topology,
        "source_rows": len(source_rows),
        "session_tree_count": len(trees),
        "chain_session_count": sum(
            tree["synthetic_topology"] == "single_chain" for tree in trees
        ),
        "fork_session_count": sum(
            tree["synthetic_topology"] == "session_fork" for tree in trees
        ),
        "leaf_count": leaf_count,
        "node_count": sum(len(tree["nodes"]) for tree in trees),
        "source_uids": sorted({row["uid"] for row in source_rows}),
        "source_session_ids": [row["source_session_id"] for row in source_rows],
        "prompt_lengths": [int(row["prompts"].numel()) for row in source_rows],
        "exact_common_prompt": exact_common_prompt,
        "logical_input_tokens": logical_input_tokens,
        "logical_loss_tokens": logical_loss_tokens,
        "leaf_mask_override_count": sum(
            len(leaf.get("mask_override_positions", []))
            for tree in trees for leaf in tree["leaves"]
        ),
    }
    return trees, summary
