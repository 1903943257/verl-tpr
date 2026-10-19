"""Controlled CP1 N=8 long-P/short-S Qwen3-4B Actor PPO performance capture.

Preserve real response tokens and their historical PPO labels, but retain only
the last 8192 prompt tokens and first 1024 response tokens. This deliberately
changes conditioning: it is a PERFORMANCE SHAPE TEST, not on-policy PPO parity.
A synthetic shared UID allows identical original prompts from distinct sessions
to share physical KV; original source identifiers remain in audit metadata.
Never pad, fabricate labels, silently shorten rows or change production tree policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans
from .prepare_tpr_10k_parity_capture import _retained_indices

PREFIX = 8192
SUFFIX = 1024
BRANCHES = 8
SYNTHETIC_KEYS = tuple(f"cp1_performance_n8_tpr_{i}" for i in range(BRANCHES))
REQUIRED = (
    "input_ids", "prompts", "responses", "response_mask", "loss_mask",
    "old_log_probs", "advantages", "temperature",
)


def derive_capture(source: Path, target: Path) -> dict:
    if target.exists():
        raise FileExistsError(f"Never overwrite a derived performance capture: {target}")
    raw = torch.load(source, map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch":
        raise AssertionError("Use an original actor_update_mini_batch capture")
    batch = raw["tensordict"]
    if len(raw["keys"]) != BRANCHES or tuple(batch.batch_size) != (BRANCHES,):
        raise ValueError("Require exactly eight original trajectories")
    if any(field not in batch for field in REQUIRED):
        raise ValueError("Missing captured original PPO fields")
    if any(field not in batch.keys() for field in REQUIRED):
        raise ValueError("Missing required PPO field")

    prompts = [batch["prompts"][i] for i in range(BRANCHES)]
    responses = [batch["responses"][i] for i in range(BRANCHES)]
    ids = [batch["input_ids"][i] for i in range(BRANCHES)]
    for i in range(BRANCHES):
        if (prompts[i].numel() < PREFIX or responses[i].numel() < SUFFIX
                or ids[i].numel() != prompts[i].numel() + responses[i].numel()):
            raise ValueError(f"row {i} cannot supply {PREFIX}+{SUFFIX} real tokens")
        if not torch.equal(ids[i], torch.cat((prompts[i], responses[i]))):
            raise AssertionError(f"Capture row {i} input_ids != prompts+responses")
        if any(batch[k][i].numel() != responses[i].numel()
               for k in ("response_mask", "loss_mask", "old_log_probs", "advantages")):
            raise AssertionError(f"Original PPO response fields misaligned on row {i}")
        if not bool(batch["response_mask"][i][:SUFFIX].bool().any()):
            raise ValueError(f"Row {i} has no supervised PPO tokens in the first {SUFFIX}")
    common = prompts[0][-PREFIX:]
    if any(not torch.equal(common, p[-PREFIX:]) for p in prompts[1:]):
        raise ValueError(
            "Original eight prompts lack a token-identical 8192-token suffix. "
            "Choose another true Actor capture: do not synthesize shared tokens."
        )
    # Diverse branches are essential: eight copies of one response are not N=8.
    distinct = {x[:SUFFIX].contiguous().numpy().tobytes() for x in responses}
    if len(distinct) != BRANCHES:
        raise ValueError("Need eight distinct original response suffixes")

    windows = [
        (int(prompts[i].numel()), int(responses[i].numel()),
         int(prompts[i].numel()) - PREFIX, 0, PREFIX, SUFFIX)
        for i in range(BRANCHES)
    ]
    # Only change the genuine sequence-aligned fields, preserving metadata and
    # scalar/batched TensorDict fields without reconstructing their semantics.
    derived = batch.clone(recurse=False)
    for name in batch.keys():
        value = batch[name]
        if isinstance(value, torch.Tensor) and value.is_nested:
            rows = [
                _retained_indices(name, value[i], p=p, r=resp,
                                  start_p=sp, start_r=sr, keep_p=kp, keep_r=kr)
                for i, (p, resp, sp, sr, kp, kr) in enumerate(windows)
            ]
            derived.set(
                name,
                tu.nested_tensor_from_tensor_list(
                    rows, ragged_idx=getattr(value, "_ragged_idx", value.dim() - 1),
                ),
            )
        elif isinstance(value, torch.Tensor) and name in (
            "input_ids", "prompts", "responses", "response_mask",
            "loss_mask", "old_log_probs", "advantages",
        ):
            raise TypeError(f"Expected captured ragged field {name} to be nested")

    supervised = []
    for i in range(BRANCHES):
        p, s, full = derived["prompts"][i], derived["responses"][i], derived["input_ids"][i]
        if p.numel() != PREFIX or s.numel() != SUFFIX or full.numel() != PREFIX+SUFFIX:
            raise AssertionError(f"Unexpected physical benchmark shape row={i}")
        if not torch.equal(full, torch.cat((p, s))):
            raise AssertionError(f"Derived prompt/response corruption row={i}")
        for field in ("response_mask", "loss_mask", "old_log_probs", "advantages"):
            if not torch.equal(derived[field][i], batch[field][i][:SUFFIX]):
                raise AssertionError(f"Historical PPO labels changed: {field} row={i}")
        count = int(derived["response_mask"][i].bool().sum())
        if count == 0:
            raise AssertionError("No retained PPO targets")
        supervised.append(count)
    if any(not torch.equal(derived["prompts"][0], derived["prompts"][i])
           for i in range(1, BRANCHES)):
        raise AssertionError("Derived eight-way shared root is not identical")

    # Verify the *actual* production turn-only tree, rather than reporting
    # nominal 8:1 when the builder did not share all eight roots.
    forest = build_tree_execution_plans(
        SYNTHETIC_KEYS, derived, tree_policy="qwen_turn", turn_min_segment_tokens=128,
    )
    if (len(forest.trees) != 1
            or set(forest.trees[0].member_rows) != set(range(BRANCHES))
            or forest.trees[0].segment_plan.get(0).length != PREFIX):
        raise AssertionError(
            "Actual qwen_turn tree is not exactly one shared 8192-token root"
        )
    if forest.segment_count < 1 + BRANCHES:
        raise AssertionError("Production tree failed to retain distinct branches")
    if forest.logical_loss_tokens != sum(supervised):
        raise AssertionError("Production tree lost original PPO targets")

    payload = dict(raw)
    payload["tensordict"] = derived
    payload["keys"] = SYNTHETIC_KEYS
    payload["performance_original_keys"] = tuple(raw["keys"])
    payload["performance_derived_from"] = str(source.resolve())
    payload["performance_changed_conditioning"] = True
    payload["performance_synthetic_uid"] = True
    digest = hashlib.sha256()
    # Match _load_real_pair's canonical serialization: field-major, row-minor.
    for key in ("input_ids", "response_mask", "old_log_probs", "advantages"):
        for i in range(BRANCHES):
            val = derived[key][i].contiguous()
            digest.update(val.view(torch.uint8).numpy().tobytes())
    audit = {
        "capture": str(target.resolve()),
        "source": str(source.resolve()),
        "source_original_keys": list(raw["keys"]),
        "new_keys": list(SYNTHETIC_KEYS),
        "n": BRANCHES,
        "prefix": PREFIX, "suffix": SUFFIX, "length": PREFIX+SUFFIX,
        "ratio": "8:1", "ppo_tokens": sum(supervised),
        "ppo_tokens_per_row": supervised,
        "forest_trees": len(forest.trees),
        "forest_segments": forest.segment_count,
        "unique_tree_tokens": sum(
            segment.length for tree in forest.trees
            for segment in tree.segment_plan.segments.values()
        ),
        "input_sha256": digest.hexdigest(),
        "changed_conditioning": True,
        "synthetic_shared_uid": True,
        "note": ("Real original response tokens and PPO labels; last prompt 8192 "
                 "and first response 1024 retained, but original conditioning "
                 "has changed and different sessions may share a synthetic UID. "
                 "PERFORMANCE ONLY, not on-policy PPO correctness."),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    target.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print("TPR_CP1_N8_CAPTURE "+json.dumps(audit, sort_keys=True), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    derive_capture(args.source, args.target)


if __name__ == "__main__":
    main()
