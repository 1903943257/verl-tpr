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


def build_synthetic_session_tree_from_tq(
    payload: dict[str, Any],
    *,
    session_id: str = "synthetic-session-from-tq",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Turn same-prompt rows into one root plus N response branches.

    The current capture's source session indexes (0..7) are retained on each
    Leaf as provenance. They are deliberately grouped into one *synthetic*
    Session Tree for validating the future NonTensorStack consumer contract.
    """
    batch = payload["tensordict"]
    keys = list(payload["keys"])
    if not keys:
        raise ValueError("TQ payload contains no trajectory keys")
    if len(keys) != int(batch.batch_size[0]):
        raise ValueError(
            f"TQ key/row mismatch: keys={len(keys)} batch_rows={batch.batch_size[0]}"
        )
    parsed = [_parse_key(key) for key in keys]
    uids = {item[0] for item in parsed}
    if len(uids) != 1:
        raise ValueError(
            "one synthetic Session Tree requires one source UID; "
            f"the selected TQ rows contain {len(uids)} UIDs"
        )
    uid = next(iter(uids))

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

    prompts = rows_by_field["prompts"]
    base_prompt = prompts[0]
    if base_prompt.numel() == 0:
        raise ValueError("cannot synthesize a prompt-root tree from an empty prompt")
    for row, prompt in enumerate(prompts):
        if not torch.equal(prompt, base_prompt):
            raise ValueError(
                f"source row {row} does not share the exact prompt; "
                "cannot place it under this single synthetic root"
            )

    nodes = [{
        "node_id": 0,
        "parent_id": None,
        "chain_id": 0,
        "prompt_ids": base_prompt.clone(),
        "response_ids": torch.empty(0, dtype=torch.long),
        "response_logprobs": torch.empty(0, dtype=rows_by_field["rollout_log_probs"][0].dtype),
        "removed_prefix_token_count": 0,
    }]
    leaves = []
    expected_rows = []

    for row, (key, (_, source_session_id, source_trajectory_index)) in enumerate(
        zip(keys, parsed, strict=True)
    ):
        prompt = prompts[row]
        response = rows_by_field["responses"][row]
        input_ids = rows_by_field["input_ids"][row]
        response_mask = rows_by_field["response_mask"][row]
        loss_mask = rows_by_field["loss_mask"][row]
        rollout_log_probs = rows_by_field["rollout_log_probs"][row]
        rm_scores = rows_by_field["rm_scores"][row]

        if not torch.equal(input_ids, torch.cat((prompt, response))):
            raise ValueError(f"source row {row}: input_ids != prompts + responses")
        response_len = response.numel()
        aligned = {
            "response_mask": response_mask.numel(),
            "loss_mask": loss_mask.numel(),
            "rollout_log_probs": rollout_log_probs.numel(),
            "rm_scores": rm_scores.numel(),
        }
        if any(length != response_len for length in aligned.values()):
            raise ValueError(
                f"source row {row}: response-aligned field lengths differ; "
                f"response={response_len}, fields={aligned}"
            )
        if bool((loss_mask.bool() & ~response_mask.bool()).any()):
            raise ValueError(f"source row {row}: loss_mask selects a response_mask=0 token")

        node_id = row + 1
        chain_id = row + 1
        nodes.append({
            "node_id": node_id,
            "parent_id": 0,
            "chain_id": chain_id,
            "prompt_ids": torch.empty(0, dtype=torch.long),
            "response_ids": response.clone(),
            "response_logprobs": rollout_log_probs.clone(),
            "removed_prefix_token_count": 0,
        })
        leaf_id = f"leaf-{row}"
        leaves.append({
            "leaf_id": leaf_id,
            "leaf_index": row,
            "kind": "node_path",
            "leaf_node_id": node_id,
            "chain_id": chain_id,
            "source_key": key,
            "source_session_id": source_session_id,
            "source_trajectory_index": source_trajectory_index,
            "response_mask": response_mask.clone(),
            "loss_mask": loss_mask.clone(),
            "rm_scores": rm_scores.clone(),
        })
        expected_rows.append({
            "key": key,
            "uid": uid,
            "session_id": session_id,
            "source_session_id": source_session_id,
            "source_trajectory_index": source_trajectory_index,
            "leaf_id": leaf_id,
            "prompts": prompt.clone(),
            "responses": response.clone(),
            "input_ids": input_ids.clone(),
            "response_mask": response_mask.clone(),
            "loss_mask": loss_mask.clone(),
            "rollout_log_probs": rollout_log_probs.clone(),
            "rm_scores": rm_scores.clone(),
        })

    tree = {
        "schema_version": TREE_SCHEMA_VERSION,
        "tree_id": f"synthetic-tree-{uid}",
        "uid": uid,
        "session_id": session_id,
        "root_node_id": 0,
        "nodes": nodes,
        "leaves": leaves,
        "synthetic_from": "flat_tq_rollout_batch",
        "source_row_count": len(keys),
    }
    validate_session_tree(tree)
    summary = {
        "source_rows": len(keys),
        "source_uid": uid,
        "source_session_ids": [item[1] for item in parsed],
        "source_trajectory_indices": [item[2] for item in parsed],
        "prompt_tokens": int(base_prompt.numel()),
        "logical_input_tokens": sum(int(x.numel()) for x in rows_by_field["input_ids"]),
        "logical_response_tokens": sum(int(x.numel()) for x in rows_by_field["responses"]),
        "logical_loss_tokens": sum(int(x.bool().sum()) for x in rows_by_field["loss_mask"]),
        "node_count": len(nodes),
        "leaf_count": len(leaves),
    }
    return tree, expected_rows, summary
