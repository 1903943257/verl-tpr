"""Produce EIGHT distinct original PPO trajectories sharing a real session UID.

Controlled short-context benchmark only: crops true original prompt/response
tokens and preserves original PPO labels, which changes model conditioning.
Unlike synthetic N8 replication, never duplicates existing captured rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from verl.utils import tensordict_utils as tu
from .prepare_tpr_10k_parity_capture import _retained_indices
from .prepare_tpr_30k_one_step import _FULL_FIELDS, _PROMPT_FIELDS, _RESPONSE_FIELDS

REQUIRED = ("input_ids", "prompts", "responses", "response_mask",
            "loss_mask", "old_log_probs", "advantages", "temperature")


def _uid(key: str) -> str:
    fields = key.rsplit("_", 2)
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise ValueError(f"Unexpected original actor key {key!r}")
    int(fields[-1])
    return fields[0]


def derive_n8(source: Path, target: Path, *, cap: int = 2048,
              prompt_window: int = 1024, rows: tuple[int, ...] | None = None) -> dict:
    if not 128 <= cap <= 4096 or not 1 <= prompt_window < cap:
        raise ValueError("Expected 128<=cap<=4096 and 1<=prompt_window<cap")
    if target.exists():
        raise FileExistsError(target)
    raw = torch.load(source, map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch":
        raise AssertionError("Expected actual Actor PPO mini-batch source")
    data = raw["tensordict"]
    if any(name not in data for name in REQUIRED):
        raise ValueError(f"Missing PPO fields: {[n for n in REQUIRED if n not in data]}")
    keys = tuple(raw["keys"])
    uid_groups = defaultdict(list)
    for i, key in enumerate(keys):
        uid_groups[_uid(key)].append(i)
    if rows is None:
        # No imputed/copy-pasted trajectories: choose one actual same-session group.
        groups = [group for group in uid_groups.values() if len(group) >= 8]
        if not groups:
            counts = sorted((len(group) for group in uid_groups.values()), reverse=True)
            raise ValueError(
                "Source has no 8 genuine same-session trajectories; "
                f"session_group_sizes={counts}. Provide another Actor mini-batch capture "
                "rather than replicating a 2-row source."
            )
        rows = tuple(groups[0][:8])
    if len(rows) != 8 or len(set(rows)) != 8 or min(rows) < 0 or max(rows) >= len(keys):
        raise ValueError("N8 requires eight different existing source row indices")
    selected_keys = tuple(keys[i] for i in rows)
    if len(set(_uid(key) for key in selected_keys)) != 1:
        raise ValueError("All 8 rows must have the same original session UID")

    prompt_list = [data["prompts"][i] for i in rows]
    response_list = [data["responses"][i] for i in rows]
    for i, p, r in zip(rows, prompt_list, response_list):
        if not torch.equal(data["input_ids"][i], torch.cat((p, r))):
            raise AssertionError(f"Captured row={i} original input != prompt + response")
    keep_p = min(prompt_window, *(int(p.numel()) for p in prompt_list))
    if keep_p == 0:
        raise ValueError("No common prompt")
    reference = prompt_list[0][-keep_p:]
    if any(not torch.equal(p[-keep_p:], reference) for p in prompt_list[1:]):
        raise ValueError(
            "Eight selected session rows have different prompt tails; "
            "use a capture with a genuinely shared prompt instead of changing conditioning per row"
        )
    # Retain the genuine response branch near its FIRST cross-row divergence.
    limit = min(int(r.numel()) for r in response_list)
    if limit < 2:
        raise ValueError("One or more genuine responses too short")
    first_diff = next(
        (j for j in range(limit)
         if any(int(r[j]) != int(response_list[0][j]) for r in response_list[1:])),
        None,
    )
    if first_diff is None:
        raise ValueError("All eight original responses identical in the common range")
    response_budget = cap - keep_p
    if response_budget < 2:
        raise ValueError("Insufficient response budget")
    start_r = max(0, first_diff - min(32, response_budget // 8))
    keep_r = min(response_budget, limit - start_r)
    if keep_r < 2:
        raise ValueError("Window could not retain response branch")

    selected = tu.index_select_tensor_dict(data, list(rows))
    cropped = selected.clone(recurse=False)
    for name in selected.keys():
        value = selected[name]
        if isinstance(value, torch.Tensor) and value.is_nested:
            pieces = []
            for j, (p, r) in enumerate(zip(prompt_list, response_list, strict=True)):
                pieces.append(_retained_indices(
                    name, value[j], p=int(p.numel()), r=int(r.numel()),
                    start_p=int(p.numel())-keep_p, start_r=start_r,
                    keep_p=keep_p, keep_r=keep_r,
                ))
            cropped.set(
                name,
                tu.nested_tensor_from_tensor_list(
                    pieces, ragged_idx=getattr(value, "_ragged_idx", value.dim()-1)
                ),
            )
        elif isinstance(value, torch.Tensor) and name in (
            _FULL_FIELDS | _PROMPT_FIELDS | _RESPONSE_FIELDS
        ):
            raise TypeError(f"Expected nested captured sequence: {name}")
    counts = []
    for j, source_row in enumerate(rows):
        ids, p, r = (cropped[n][j] for n in ("input_ids", "prompts", "responses"))
        if len(ids) != keep_p + keep_r or not torch.equal(ids, torch.cat((p, r))):
            raise AssertionError(f"Broken source row {source_row} prompt/response alignment")
        if not torch.equal(p, reference):
            raise AssertionError("Shared prompt changed during capture cropping")
        for field in ("old_log_probs", "advantages", "response_mask", "loss_mask"):
            if not torch.equal(cropped[field][j], data[field][source_row][start_r:start_r + keep_r]):
                raise AssertionError(f"Modified original PPO field {field} row={source_row}")
        owned = int(cropped["response_mask"][j].bool().sum().item())
        if owned < 1:
            raise ValueError(f"Row={source_row} has zero supervised PPO response tokens")
        counts.append(owned)

    payload = dict(raw)
    payload["keys"] = selected_keys
    payload["tensordict"] = cropped
    payload["derived_from_capture"] = str(source.resolve())
    payload["short_original_rows"] = rows
    payload["parity_window_context_changed"] = True
    payload["offline_sequence_cap"] = cap
    digest = hashlib.sha256()
    for i in range(8):
        for field in ("input_ids", "response_mask", "old_log_probs", "advantages"):
            item = cropped[field][i]
            digest.update(item.contiguous().view(torch.uint8).numpy().tobytes())
    audit = {
        "source": str(source.resolve()), "capture": str(target.resolve()),
        "selected_rows": list(rows),
        "session_trajectory_counts": sorted((len(g) for g in uid_groups.values()), reverse=True),
        "cap": cap, "prompt_window": keep_p, "response_window_start": start_r,
        "response_window_length": keep_r, "full_sequence_length": keep_p+keep_r,
        "response_first_divergence": first_diff,
        "logical_trajectories": 8, "ppo_tokens": sum(counts),
        "ppo_tokens_by_row": counts, "sha256": digest.hexdigest(),
        "conditioning_changed": True,
        "note": "Retained actual PPO labels but truncated genuine causal context; not on-policy",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    target.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("--cap", type=int, default=2048)
    p.add_argument("--prompt-window", type=int, default=1024)
    p.add_argument("--rows", default=None, help="Optional comma-separated EIGHT real source indices")
    args = p.parse_args()
    rows = None if args.rows is None else tuple(map(int, args.rows.split(",")))
    print("TPR_CP1_N8_CAPTURE " + json.dumps(
        derive_n8(args.source, args.target, cap=args.cap,
                  prompt_window=args.prompt_window, rows=rows),
        sort_keys=True,
    ), flush=True)


if __name__ == "__main__":
    main()
