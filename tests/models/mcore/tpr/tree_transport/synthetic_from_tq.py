"""Build one synthetic Session Tree from the existing flat TQ rollout batch.

The source TQ dump contains complete trajectory rows. This helper groups the
rows that share one exact prompt into sibling response branches. It does not
contact TQ or test its persistence/transport implementation.
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
    *, uid: str, session_id: str, tree_id: str,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(source_rows) < 2:
        raise ValueError("a synthetic fork Session needs at least two source trajectories")
    prompt = source_rows[0]["prompts"]
    for row in source_rows[1:]:
        if not torch.equal(row["prompts"], prompt):
            raise ValueError("synthetic fork branches must share an exact prompt")

    nodes = [{
        "node_id": 0,
        "parent_id": None,
        "chain_id": 0,
        "prompt_ids": prompt.clone(),
        "response_ids": torch.empty(0, dtype=torch.long),
        "response_logprobs": torch.empty(
            0, dtype=source_rows[0]["rollout_log_probs"].dtype
        ),
        "removed_prefix_token_count": 0,
    }]
    leaves = []
    for index, row in enumerate(source_rows):
        node_id = index + 1
        nodes.append({
            "node_id": node_id,
            "parent_id": 0,
            "chain_id": 0,
            "prompt_ids": torch.empty(0, dtype=torch.long),
            "response_ids": row["responses"].clone(),
            "response_logprobs": row["rollout_log_probs"].clone(),
            "removed_prefix_token_count": 0,
        })
        leaves.append({
            "leaf_id": f"{session_id}-leaf-{index}",
            "leaf_index": index,
            "kind": "node_path",
            "leaf_node_id": node_id,
            "chain_id": 0,
            "source_key": row["key"],
            "source_session_id": row["source_session_id"],
            "source_trajectory_index": row["source_trajectory_index"],
            "response_mask": row["response_mask"].clone(),
            "loss_mask": row["loss_mask"].clone(),
            "rm_scores": row["rm_scores"].clone(),
        })

    return {
        "schema_version": TREE_SCHEMA_VERSION,
        "tree_id": tree_id,
        "uid": uid,
        "session_id": session_id,
        "root_node_id": 0,
        "nodes": nodes,
        "leaves": leaves,
        "synthetic_from": "flat_tq_rollout_batch",
        "synthetic_topology": "session_fork",
        "source_tq_keys": [row["key"] for row in source_rows],
    }


def build_synthetic_session_trees_from_tq(
    payload: dict[str, Any],
    *,
    topology: str = "all_chain",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Synthesize independent per-Session Chain/Tree payloads from flat TQ rows.

    topology:
      - all_chain: one source TQ row becomes one single-chain Session.
      - mixed: first half remain single-chain Sessions; the second half are
        paired into synthetic fork Sessions.
      - all_tree: source rows are paired into independent fork Sessions.

    Source rows are never combined across Session Tree boundaries. Pairing rows
    is only a fixture-generation technique to fabricate an intra-Session fork;
    each emitted Tree still contains exactly one synthetic Session.
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
        groups = [[row] for row in source_rows]
    elif topology == "mixed":
        split = len(source_rows) // 2
        groups = [[row] for row in source_rows[:split]]
        groups.extend(
            source_rows[index:index + 2]
            for index in range(split, len(source_rows), 2)
        )
    else:
        groups = [
            source_rows[index:index + 2]
            for index in range(0, len(source_rows), 2)
        ]

    trees = []
    for session_index, group in enumerate(groups):
        uids = {row["uid"] for row in group}
        if len(uids) != 1:
            raise ValueError("a synthetic Session Tree cannot combine different UIDs")
        uid = next(iter(uids))
        session_id = f"synthetic-session-{session_index}"
        tree_id = f"synthetic-tree-{topology}-{session_index}"
        if len(group) == 1:
            row = group[0]
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
                source_rows=group,
            )
        validate_session_tree(tree)
        trees.append(tree)

    all_prompts = [row["prompts"] for row in source_rows]
    exact_common_prompt = bool(all_prompts) and all(
        torch.equal(all_prompts[0], prompt) for prompt in all_prompts[1:]
    )
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
        "leaf_count": sum(len(tree["leaves"]) for tree in trees),
        "node_count": sum(len(tree["nodes"]) for tree in trees),
        "source_uids": sorted({row["uid"] for row in source_rows}),
        "source_session_ids": [row["source_session_id"] for row in source_rows],
        "prompt_lengths": [int(row["prompts"].numel()) for row in source_rows],
        "exact_common_prompt": exact_common_prompt,
        "logical_input_tokens": sum(
            int(row["prompts"].numel() + row["responses"].numel())
            for row in source_rows
        ),
        "logical_loss_tokens": sum(
            int(row["loss_mask"].bool().sum()) for row in source_rows
        ),
    }
    return trees, summary
