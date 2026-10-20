"""Prepare N=8 performance-only response windows from one real Actor capture.

Keep six original long response prefixes and replace only the two short
response rows by later, distinct windows from long source rows. Real
response tokens/labels are sliced together; conditioning and sampling
statistics no longer represent an on-policy PPO batch. Never use this
derived source for correctness or production replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from verl.utils import tensordict_utils as tu
from .prepare_tpr_10k_parity_capture import _retained_indices

DONOR_ROWS = (0, 1, 3, 4, 5, 6, 0, 3)
SUPPORTED = ((9216, 9216), (17440, 17440))
NEEDED = ("input_ids", "prompts", "responses", "response_mask",
          "loss_mask", "old_log_probs", "advantages", "temperature")


def prepare_window_source(source: Path, target: Path, *, prefix: int, suffix: int) -> dict:
    if (prefix, suffix) not in SUPPORTED:
        raise ValueError(f"Unsupported synthetic window shape {(prefix, suffix)}")
    if target.exists() or target.with_suffix(".windows.json").exists():
        raise FileExistsError(f"Refuse to overwrite synthetic source: {target}")
    raw = torch.load(source, map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch":
        raise ValueError("Require an original actor_update_mini_batch source")
    original = raw["tensordict"]
    if tuple(original.batch_size) != (8,) or len(raw["keys"]) != 8:
        raise ValueError("Require exactly eight original Actor PPO rows")
    if any(key not in original for key in NEEDED):
        raise ValueError("Required original PPO fields missing")
    prompts = [original["prompts"][i] for i in range(8)]
    responses = [original["responses"][i] for i in range(8)]
    if any(p.numel() < prefix for p in prompts):
        raise ValueError("Insufficient prompt length")
    common = prompts[0][-prefix:]
    if any(not torch.equal(p[-prefix:], common) for p in prompts[1:]):
        raise ValueError("Original prompts do not share identical P-token suffix")

    # Output branches 0..5 use the six sufficiently long original rows;
    # the two additional branches use non-overlapping later response windows.
    row_windows = []
    for output_row, donor in enumerate(DONOR_ROWS):
        start = suffix if output_row >= 6 else 0
        rlen = int(responses[donor].numel())
        plen = int(prompts[donor].numel())
        if rlen < start + suffix:
            raise ValueError(
                f"Donor row {donor} insufficient for window "
                f"[{start}:{start+suffix}]: response={rlen}"
            )
        full = original["input_ids"][donor]
        if not torch.equal(full, torch.cat((prompts[donor], responses[donor]))):
            raise AssertionError(f"Source row {donor} input_ids mismatch")
        for name in ("response_mask", "loss_mask", "old_log_probs", "advantages"):
            if original[name][donor].numel() != rlen:
                raise AssertionError(f"{name} misaligned in donor {donor}")
        if not original["response_mask"][donor][start:start+suffix].bool().any().item():
            raise ValueError(f"Donor {donor} window has no supervised tokens")
        row_windows.append({
            "output_row": output_row, "donor_source_row": donor,
            "prompt_start": plen-prefix, "prompt_tokens": prefix,
            "response_start": start, "response_tokens": suffix,
            "donor_response_tokens": rlen,
            "donor_original_key": str(raw["keys"][donor]),
        })

    # Index-select duplicates donor row-level (including temperature), then
    # re-slice *all nested fields* with their original source alignments.
    # This protects the original PPO labels and all recorded ragged masks.
    selected = tu.index_select_tensor_dict(original, list(DONOR_ROWS))
    result = selected.clone(recurse=False)
    for name in selected.keys():
        val = selected[name]
        if isinstance(val, torch.Tensor) and val.is_nested:
            pieces = []
            for i, info in enumerate(row_windows):
                donor = info["donor_source_row"]
                pieces.append(_retained_indices(
                    name, val[i],
                    p=int(prompts[donor].numel()),
                    r=int(responses[donor].numel()),
                    start_p=info["prompt_start"],
                    start_r=info["response_start"],
                    keep_p=prefix, keep_r=suffix,
                ))
            result.set(name, tu.nested_tensor_from_tensor_list(
                pieces, ragged_idx=getattr(val, "_ragged_idx", val.dim()-1),
            ))
        elif isinstance(val, torch.Tensor) and name in (
            "input_ids", "prompts", "responses", "response_mask",
            "loss_mask", "old_log_probs", "advantages",
        ):
            raise TypeError(f"Captured ragged field {name} is not nested")

    branches = [result["responses"][i] for i in range(8)]
    if len({b.contiguous().numpy().tobytes() for b in branches}) != 8:
        raise AssertionError("Response windows produced duplicate branches")
    for i in range(8):
        if not torch.equal(result["input_ids"][i], torch.cat((
            result["prompts"][i], result["responses"][i]
        ))):
            raise AssertionError(f"Synthetic source row {i} input mismatch")
        donor = row_windows[i]["donor_source_row"]
        start = row_windows[i]["response_start"]
        for name in ("responses", "response_mask", "loss_mask",
                     "old_log_probs", "advantages"):
            if not torch.equal(result[name][i],
                               original[name][donor][start:start+suffix]):
                raise AssertionError(f"Synthetic source {name} mutation row {i}")
    keys = tuple(f"window_n8_performance_tpr_{i}" for i in range(8))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "capture_stage": "actor_update_mini_batch",
        "keys": keys,
        "tensordict": result,
        "performance_synthetic_response_windows": True,
        "performance_window_original_capture": str(source.resolve()),
        "performance_row_windows": row_windows,
    }
    provenance = {
        "performance_only": True,
        "synthetic_response_windows": True,
        "on_policy_valid": False,
        "gradient_parity_claim": False,
        "original_capture": str(source.resolve()),
        "source_rows": row_windows,
        "prefix": prefix, "suffix": suffix, "n": 8,
        "no_padding": True,
        "responses_distinct": True,
        "response_tokens_and_ppo_labels_copied_from_same_original_windows": True,
    }
    torch.save(payload, target)
    target.with_suffix(".windows.json").write_text(json.dumps(provenance, indent=2))
    print("TPR_SYNTHETIC_WINDOW_SOURCE " + json.dumps(provenance, sort_keys=True), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--suffix", type=int, required=True)
    args = parser.parse_args()
    prepare_window_source(args.source, args.target,
                          prefix=args.prefix, suffix=args.suffix)


if __name__ == "__main__":
    main()
