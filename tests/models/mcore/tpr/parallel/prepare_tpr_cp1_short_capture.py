"""Derive a two-sibling, <=4K *controlled* Actor PPO capture for CP1 parity.

Keep real captured labels on every retained response token. Both branches use
identical cropped bytes, but truncating the prompt and selecting a response
window changes conditioning: this is NOT the original online PPO objective.
No Fixed Tile or invented old log-probabilities / advantages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from verl.utils import tensordict_utils as tu

from .prepare_tpr_10k_parity_capture import _retained_indices
from .prepare_tpr_30k_one_step import _FULL_FIELDS, _PROMPT_FIELDS, _RESPONSE_FIELDS

REQUIRED = (
    "input_ids", "prompts", "responses", "response_mask",
    "loss_mask", "old_log_probs", "advantages", "temperature",
)


def _common_divergence(a: torch.Tensor, b: torch.Tensor) -> int:
    limit = min(a.numel(), b.numel())
    mismatch = torch.nonzero(a[:limit] != b[:limit], as_tuple=False)
    if mismatch.numel():
        return int(mismatch[0, 0])
    if a.numel() != b.numel():
        return limit
    raise ValueError("Selected siblings have identical complete responses; no branch to test")


def derive_short_capture(
    source: Path,
    target: Path,
    *,
    rows: tuple[int, int] = (4, 6),
    cap: int = 2048,
    prompt_window: int = 1024,
) -> dict:
    if not 128 <= cap <= 4096:
        raise ValueError("Single-card parity cap must be in [128, 4096]")
    if not 1 <= prompt_window < cap:
        raise ValueError("prompt_window must be in [1, cap)")
    if len(rows) != 2 or len(set(rows)) != 2:
        raise ValueError("Select exactly two distinct original sibling rows")
    if target.exists():
        raise FileExistsError(target)
    original = torch.load(source, map_location="cpu", weights_only=False)
    if original.get("capture_stage") != "actor_update_mini_batch":
        raise AssertionError("Expected original actor_update_mini_batch capture")
    data = original["tensordict"]
    if not all(field in data for field in REQUIRED):
        raise ValueError(f"Missing original Actor PPO field: {[k for k in REQUIRED if k not in data]}")
    if min(rows) < 0 or max(rows) >= len(original["keys"]):
        raise ValueError("Original sibling row is out of range")
    original_keys = tuple(original["keys"][i] for i in rows)
    if original_keys[0].rsplit("_", 2)[0] != original_keys[1].rsplit("_", 2)[0]:
        raise ValueError("Selected trajectories must share the same original session UID")
    source_inputs = [data["input_ids"][i] for i in rows]
    prompts = [data["prompts"][i] for i in rows]
    responses = [data["responses"][i] for i in rows]
    for index, (tokens, prompt, response) in enumerate(zip(source_inputs, prompts, responses, strict=True)):
        if not torch.equal(tokens, torch.cat((prompt, response))):
            raise AssertionError(f"Original row {rows[index]} input_ids != prompts + responses")
    keep_p = min(prompt_window, *(int(x.numel()) for x in prompts))
    # Fail instead of silently losing the common root when two prompts differ.
    if not torch.equal(prompts[0][-keep_p:], prompts[1][-keep_p:]):
        raise ValueError("The retained prompt tail is not token-identical across siblings")
    first_diff = _common_divergence(responses[0], responses[1])
    max_response = cap - keep_p
    # Keep a few genuine response tokens before the branch. This avoids
    # requiring the first response token of the truncated window to differ.
    start_r = max(0, first_diff - min(32, max_response // 8))
    remaining = min(int(x.numel()) - start_r for x in responses)
    keep_r = min(max_response, remaining)
    if keep_r < 2 or not start_r <= first_diff < start_r + keep_r:
        raise ValueError("The selected window cannot contain the actual response branch")
    windows = [
        (
            int(p.numel()), int(r.numel()),
            int(p.numel()) - keep_p, start_r, keep_p, keep_r,
        )
        for p, r in zip(prompts, responses, strict=True)
    ]
    selected = tu.index_select_tensor_dict(data, list(rows))
    cropped = selected.clone(recurse=False)
    for name in selected.keys():
        value = selected[name]
        if isinstance(value, torch.Tensor) and value.is_nested:
            parts = [
                _retained_indices(
                    name, value[j],
                    p=p, r=r, start_p=sp, start_r=sr,
                    keep_p=kp, keep_r=kr,
                )
                for j, (p, r, sp, sr, kp, kr) in enumerate(windows)
            ]
            cropped.set(
                name,
                tu.nested_tensor_from_tensor_list(
                    parts, ragged_idx=getattr(value, "_ragged_idx", value.dim() - 1)
                ),
            )
        elif isinstance(value, torch.Tensor) and name in (
            _FULL_FIELDS | _PROMPT_FIELDS | _RESPONSE_FIELDS
        ):
            raise TypeError(f"Expected nested per-trajectory sequence for {name}")
    if set(selected.keys()) != set(cropped.keys()):
        raise AssertionError("Cropping changed captured PPO field names")
    retained_supervised = []
    for j, row in enumerate(rows):
        ids = cropped["input_ids"][j]
        p = cropped["prompts"][j]
        r = cropped["responses"][j]
        if len(ids) > cap or not torch.equal(ids, torch.cat((p, r))):
            raise AssertionError(f"Cropping broke prompt + response alignment for {row}")
        for field in ("old_log_probs", "advantages", "response_mask", "loss_mask"):
            expected = data[field][row][start_r:start_r + keep_r]
            if not torch.equal(cropped[field][j], expected):
                raise AssertionError(f"{field} was modified for row {row}")
        supervised = int(cropped["response_mask"][j].bool().sum().item())
        if supervised == 0:
            raise ValueError(f"Cropped row {row} has no real PPO supervision")
        retained_supervised.append(supervised)
    if not torch.equal(cropped["prompts"][0], cropped["prompts"][1]):
        raise AssertionError("The shared Root prompt was not preserved")
    if torch.equal(cropped["responses"][0], cropped["responses"][1]):
        raise AssertionError("The cropped siblings have no response branch")

    payload = dict(original)
    payload["keys"] = original_keys
    payload["tensordict"] = cropped
    payload["derived_from_capture"] = str(source.resolve())
    payload["short_original_rows"] = rows
    payload["offline_sequence_cap"] = cap
    payload["parity_window_context_changed"] = True
    digest = hashlib.sha256()
    for j in range(2):
        for name in ("input_ids", "response_mask", "old_log_probs", "advantages"):
            digest.update(cropped[name][j].contiguous().view(torch.uint8).numpy().tobytes())
    audit = {
        "source": str(source.resolve()),
        "capture": str(target.resolve()),
        "original_rows": list(rows),
        "cap": cap,
        "prompt_window": keep_p,
        "original_response_first_divergence": first_diff,
        "retained_response_start": start_r,
        "retained_response_length": keep_r,
        "retained_lengths": [int(cropped["input_ids"][i].numel()) for i in range(2)],
        "supervised_tokens_per_row": retained_supervised,
        "retained_input_sha256": digest.hexdigest(),
        "conditioning_changed": True,
        "note": (
            "Controlled numerical replay only: left-truncated prompt and "
            "response window change conditioning. All retained original PPO "
            "labels and temperatures are preserved, not rederived."
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    target.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--rows", default="4,6")
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--prompt-window", type=int, default=1024)
    args = parser.parse_args()
    rows = tuple(int(x) for x in args.rows.split(","))
    audit = derive_short_capture(
        args.source, args.target, rows=rows,
        cap=args.cap, prompt_window=args.prompt_window,
    )
    print("TPR_CP1_SHORT_CAPTURE " + json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
