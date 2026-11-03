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


def _sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _lcp_all(rows: torch.Tensor) -> int:
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError("responses must be a 2-D tensor with at least two rows")
    first = rows[0]
    limit = rows.shape[1]
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

    input_ids = td["input_ids"]
    prompts = td["prompts"]
    responses = td["responses"]
    if not all(isinstance(x, torch.Tensor) for x in (input_ids, prompts, responses)):
        raise TypeError("input_ids/prompts/responses must be dense tensors")
    if input_ids.ndim != 2 or prompts.ndim != 2 or responses.ndim != 2:
        raise ValueError(
            "Expected dense [N, sequence] tensors; got "
            f"input_ids={tuple(input_ids.shape)}, prompts={tuple(prompts.shape)}, "
            f"responses={tuple(responses.shape)}"
        )

    rows = len(raw["keys"])
    if input_ids.shape[0] != rows or prompts.shape[0] != rows or responses.shape[0] != rows:
        raise ValueError("Capture row counts do not match keys")
    old_prompt = prompts.shape[1]
    response_length = responses.shape[1]
    old_total = old_prompt + response_length
    if input_ids.shape[1] != old_total:
        raise ValueError(
            f"input_ids length {input_ids.shape[1]} != prompt+response {old_total}"
        )
    if not torch.equal(prompts, prompts[:1].expand_as(prompts)):
        raise ValueError(
            "Source rows do not have an identical shared prompt; refusing to "
            "invent a different session topology"
        )
    if not torch.equal(input_ids, torch.cat((prompts, responses), dim=-1)):
        raise ValueError("Source input_ids are not exactly prompt + response")

    response_lcp = _lcp_all(responses)
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

    new_prompts = prompts[:, :new_prompt].contiguous()
    new_input_ids = torch.cat((new_prompts, responses), dim=-1).contiguous()
    td.set("prompts", new_prompts)
    td.set("input_ids", new_input_ids)

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

    if not torch.equal(td["input_ids"], torch.cat((td["prompts"], td["responses"]), dim=-1)):
        raise AssertionError("Derived input_ids do not equal derived prompt + original response")
    if td["prompts"].shape[1] != new_prompt:
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
