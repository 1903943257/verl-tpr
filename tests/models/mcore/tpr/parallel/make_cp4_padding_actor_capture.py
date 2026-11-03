#!/usr/bin/env python3
"""Derive an N=8 Actor PPO capture whose shared TPR Prefix needs CP4 padding.

The source PPO responses and all response-side labels remain unchanged. We
shorten the identical prompt by 1..8 tokens, chosen from the measured common
response prefix so that the final shared Prefix length is congruent to 5 mod 8.
CP4 Ring uses 2*CP=8 chunks, so this guarantees three physical right-padding
tokens on the shared Prefix. This is a synthetic shape/path smoke, not a
semantic PPO correctness dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256(value) -> str:
    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    elif hasattr(value, "unbind"):
        for item in value.unbind(0):
            digest.update(_sha256(item).encode())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def _rows(value, name: str) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [N, sequence], got {tuple(value.shape)}")
        result = list(value.unbind(0))
    elif hasattr(value, "unbind"):
        result = list(value.unbind(0))
    else:
        raise TypeError(f"{name} must be a Tensor or row-stack with unbind(0)")
    if not result or not all(isinstance(row, torch.Tensor) and row.ndim == 1 for row in result):
        raise ValueError(f"{name} rows must be non-empty one-dimensional tensors")
    return result


def _rebuild_rows(original, rows: list[torch.Tensor]):
    if isinstance(original, torch.Tensor):
        return torch.stack(rows, dim=0).contiguous()
    from tensordict import NonTensorStack
    return NonTensorStack(*rows)


def _lcp_all(rows: list[torch.Tensor]) -> int:
    if len(rows) < 2:
        raise ValueError("responses must contain at least two rows")
    limit = min(row.numel() for row in rows)
    first = rows[0]
    for row in rows[1:]:
        mismatch = torch.nonzero(first[:limit] != row[:limit], as_tuple=False)
        if mismatch.numel():
            limit = min(limit, int(mismatch[0, 0]))
    return limit


def _sequence_keep_indices(old_prompt: int, drop: int, total: int) -> torch.Tensor:
    new_prompt = old_prompt - drop
    return torch.cat((
        torch.arange(new_prompt, dtype=torch.long),
        torch.arange(old_prompt, total, dtype=torch.long),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=(
            "/workspace/tpr_experiments/performance/"
            "cp4_n8_merge_perf_none_v1/data/cp4_n8_p17440_s17440.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/tpr_experiments/performance/"
            "cp4_n8_merge_padding_synthetic_v1/cp4_n8_padding_synthetic.pt"
        ),
    )
    parser.add_argument("--target-prefix-mod", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite derived capture: {output}")
    if not 1 <= args.target_prefix_mod <= 7:
        raise ValueError("--target-prefix-mod must be in 1..7")

    raw = torch.load(source, map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch":
        raise ValueError("Source must be an original actor_update_mini_batch capture")
    if len(raw.get("keys", ())) != 8:
        raise ValueError(f"Expected exactly 8 source trajectories, got {len(raw.get('keys', ())) }")

    original_td = raw["tensordict"]
    td = original_td.clone()
    required = (
        "input_ids", "prompts", "responses", "response_mask",
        "loss_mask", "advantages", "old_log_probs", "temperature",
    )
    missing = [key for key in required if key not in td]
    if missing:
        raise KeyError(f"Capture is missing required Actor PPO fields: {missing}")

    input_ids_value = td["input_ids"]
    prompts_value = td["prompts"]
    responses_value = td["responses"]
    input_rows = _rows(input_ids_value, "input_ids")
    prompt_rows = _rows(prompts_value, "prompts")
    response_rows = _rows(responses_value, "responses")

    rows = len(raw["keys"])
    if not (len(input_rows) == len(prompt_rows) == len(response_rows) == rows):
        raise ValueError("Capture row counts do not match keys")
    old_prompt = prompt_rows[0].numel()
    response_length = response_rows[0].numel()
    if any(row.numel() != old_prompt for row in prompt_rows):
        raise ValueError("Source prompt rows have different lengths")
    if any(row.numel() != response_length for row in response_rows):
        raise ValueError("Source response rows have different lengths")
    old_total = old_prompt + response_length
    if any(row.numel() != old_total for row in input_rows):
        raise ValueError("input_ids rows do not have prompt+response length")
    if not all(torch.equal(prompt_rows[0], row) for row in prompt_rows[1:]):
        raise ValueError(
            "Source rows do not have an identical shared prompt; refusing to "
            "invent a different session topology"
        )
    for row, prompt, response in zip(input_rows, prompt_rows, response_rows, strict=True):
        if not torch.equal(row, torch.cat((prompt, response))):
            raise ValueError("Source input_ids are not exactly prompt + response")

    response_lcp = _lcp_all(response_rows)
    if response_lcp >= response_length:
        raise ValueError("All response rows are identical; no branching after shared prompt")

    # The shared prefix is prompt + the common response-token prefix. Choose a
    # small prompt trim so the actual trie segment, not just the prompt field,
    # has a guaranteed non-eight-aligned length.
    drop = (response_lcp - args.target_prefix_mod) % 8
    if drop == 0:
        drop = 8
    if drop > min(8, old_prompt - 1):
        raise ValueError(f"Cannot safely trim {drop} prompt tokens from length {old_prompt}")

    new_prompt = old_prompt - drop
    expected_shared_prefix = new_prompt + response_lcp
    if expected_shared_prefix % 8 != args.target_prefix_mod:
        raise AssertionError("Internal error computing non-aligned shared Prefix")
    padded_prefix = ((expected_shared_prefix + 7) // 8) * 8

    new_prompt_rows = [row[:new_prompt].contiguous() for row in prompt_rows]
    new_input_rows = [
        torch.cat((prompt, response)).contiguous()
        for prompt, response in zip(new_prompt_rows, response_rows, strict=True)
    ]
    td.set("prompts", _rebuild_rows(prompts_value, new_prompt_rows))
    td.set("input_ids", _rebuild_rows(input_ids_value, new_input_rows))

    # Update any explicit sequence-position fields. The Actor capture normally
    # stores only input_ids/prompts/responses; if an attention mask or position
    # tensor is present, crop it consistently rather than leaving stale shapes.
    keep = _sequence_keep_indices(old_prompt, drop, old_total)
    updated_sequence_fields = []
    for key in list(td.keys()):
        if key in ("input_ids", "prompts", "responses"):
            continue
        value = td[key]
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim >= 2 and value.shape[0] == rows and value.shape[-1] == old_total:
            if key == "position_ids":
                updated = value.index_select(-1, keep)
                updated[..., new_prompt:] -= drop
                td.set(key, updated.contiguous())
                updated_sequence_fields.append(key)
            elif key == "attention_mask":
                updated = value.index_select(-1, keep)
                if value.ndim >= 3 and value.shape[-2] == old_total:
                    updated = updated.index_select(-2, keep)
                td.set(key, updated.contiguous())
                updated_sequence_fields.append(key)
            else:
                raise ValueError(
                    f"Unknown full-sequence tensor field {key!r} has shape "
                    f"{tuple(value.shape)}; refusing to leave it misaligned"
                )

    # Verify all original response-side PPO labels are unchanged.
    immutable_fields = (
        "responses", "response_mask", "loss_mask", "advantages",
        "old_log_probs", "temperature",
    )
    label_hashes = {key: _sha256(original_td[key]) for key in immutable_fields}
    for key, digest in label_hashes.items():
        if _sha256(td[key]) != digest:
            raise AssertionError(f"Original PPO field changed unexpectedly: {key}")

    derived_prompts = _rows(td["prompts"], "derived prompts")
    derived_responses = _rows(td["responses"], "derived responses")
    derived_inputs = _rows(td["input_ids"], "derived input_ids")
    if any(
        not torch.equal(ids, torch.cat((prompt, response)))
        for ids, prompt, response in zip(
            derived_inputs, derived_prompts, derived_responses, strict=True
        )
    ):
        raise AssertionError("Derived input_ids do not equal derived prompt + original response")
    if any(prompt.numel() != new_prompt for prompt in derived_prompts):
        raise AssertionError("Derived prompt length mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    derived = dict(raw)
    derived["tensordict"] = td
    torch.save(derived, output)

    report = {
        "protocol": "synthetic_cp4_ring_padding_actor_capture_v1",
        "source_capture": str(source),
        "output_capture": str(output),
        "rows": rows,
        "source_prompt_length": old_prompt,
        "derived_prompt_length": new_prompt,
        "response_length": response_length,
        "common_response_prefix_length": response_lcp,
        "expected_shared_prefix_length": expected_shared_prefix,
        "cp4_ring_chunk_count": 8,
        "expected_padded_prefix_length": padded_prefix,
        "expected_prefix_padding_tokens": padded_prefix - expected_shared_prefix,
        "prompt_tokens_removed_from_each_row": drop,
        "response_side_fields_unchanged": list(immutable_fields),
        "response_side_sha256": label_hashes,
        "updated_full_sequence_fields": updated_sequence_fields,
        "warning": (
            "Synthetic shape/path smoke only. The last prompt tokens were "
            "trimmed; this is not a semantic PPO correctness or training dataset."
        ),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
