"""Controlled CP1 N=8 long-P/short-S Qwen3-4B Actor PPO performance capture.

Preserve real response tokens and their historical PPO labels. Retain the last
P prompt tokens and first S response tokens for an explicit shape. This deliberately
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


def derive_capture(
    source: Path, target: Path, *, prefix: int = PREFIX, suffix: int = SUFFIX,
) -> dict:
    """Build a strictly token-exact eight-branch PPO performance shape."""
    allowed = ((8192, 1024), (4608, 4608), (16384, 2048),
               (9216, 9216), (6016, 6016), (12416, 6016),
               (13824, 13824), (24576, 3072),
               (18432, 18432), (32768, 4096), (17440, 17440))
    if (prefix, suffix) not in allowed:
        raise ValueError(f"Unsupported measured workload {(prefix, suffix)}; allowed={allowed}")
    cp = (1 if (prefix, suffix) in ((8192, 1024), (4608, 4608))
          else 2 if (prefix, suffix) in ((16384, 2048), (9216, 9216),
                                         (6016, 6016), (12416, 6016)) else 4)
    keys = (
        SYNTHETIC_KEYS if (prefix, suffix) == (PREFIX, SUFFIX)
        else tuple(f"cp{cp}_p{prefix}_s{suffix}_tpr_{i}" for i in range(BRANCHES))
    )
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
    length_audit = []
    for i in range(BRANCHES):
        p_len = int(prompts[i].numel())
        r_len = int(responses[i].numel())
        input_len = int(ids[i].numel())
        length_audit.append({
            "row": i,
            "prompt_tokens": p_len,
            "response_tokens": r_len,
            "input_tokens": input_len,
            "prompt_shortfall": max(0, prefix - p_len),
            "response_shortfall": max(0, suffix - r_len),
            "input_length_aligned": input_len == p_len + r_len,
            "supervised_in_suffix": int(
                batch["response_mask"][i][:suffix].bool().sum().item()
            ),
        })
    # Always report all eight original lengths *before* the first rejection;
    # truncated captures must never silently masquerade as a P:S=8:1 case.
    print(
        "TPR_N8_SOURCE_LENGTHS "
        + json.dumps({"source": str(source), "rows": length_audit},
                     sort_keys=True),
        flush=True,
    )
    for i, stat in enumerate(length_audit):
        reasons = []
        if stat["prompt_shortfall"]:
            reasons.append(f"prompt_shortfall={stat['prompt_shortfall']}")
        if stat["response_shortfall"]:
            reasons.append(f"response_shortfall={stat['response_shortfall']}")
        if not stat["input_length_aligned"]:
            reasons.append("input_length_not_prompt_plus_response")
        if reasons:
            raise ValueError(
                f"row {i} cannot supply exact P={prefix}, S={suffix}: "
                f"prompt={stat['prompt_tokens']}, "
                f"response={stat['response_tokens']}, "
                f"input={stat['input_tokens']}, reasons={','.join(reasons)}. "
                "See TPR_N8_SOURCE_LENGTHS for all 8 rows. "
                "Choose a different real capture; no padding or fabricated PPO labels."
            )
        if not torch.equal(ids[i], torch.cat((prompts[i], responses[i]))):
            raise AssertionError(f"Capture row {i} input_ids != prompts+responses")
        if any(batch[k][i].numel() != responses[i].numel()
               for k in ("response_mask", "loss_mask", "old_log_probs", "advantages")):
            raise AssertionError(f"Original PPO response fields misaligned on row {i}")
        if not bool(batch["response_mask"][i][:suffix].bool().any()):
            raise ValueError(f"Row {i} has no supervised PPO tokens in the first {suffix}")
    common = prompts[0][-prefix:]
    if any(not torch.equal(common, p[-prefix:]) for p in prompts[1:]):
        raise ValueError(
            f"Original eight prompts lack a token-identical {prefix}-token suffix. "
            "Choose another true Actor capture: do not synthesize shared tokens."
        )
    # Diverse branches are essential: eight copies of one response are not N=8.
    distinct = {x[:suffix].contiguous().numpy().tobytes() for x in responses}
    if len(distinct) != BRANCHES:
        raise ValueError("Need eight distinct original response suffixes")

    windows = [
        (int(prompts[i].numel()), int(responses[i].numel()),
         int(prompts[i].numel()) - prefix, 0, prefix, suffix)
        for i in range(BRANCHES)
    ]
    # The synthetic window source is Actor F/B-only. Its original PPO capture
    # may also include critic fields such as 'returns'/'values'. When P == S,
    # these unknown ragged fields are ambiguous to the generic shape-inference
    # cropper, and none are consumed by the actor PPO objective. Retain only
    # the actual actor inputs, masks/labels and optional actor loss inputs.
    # Keep the strict unfiltered behavior for every real-capture experiment.
    dropped_auxiliary_fields = []
    if raw.get("performance_synthetic_response_windows", False):
        actor_fields = set(REQUIRED) | {"ref_log_prob", "rollout_is_weights"}
        dropped_auxiliary_fields = [
            name for name in batch.keys() if name not in actor_fields
        ]
        batch = batch.select(*[name for name in batch.keys() if name in actor_fields])
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
        if p.numel() != prefix or s.numel() != suffix or full.numel() != prefix+suffix:
            raise AssertionError(f"Unexpected physical benchmark shape row={i}")
        if not torch.equal(full, torch.cat((p, s))):
            raise AssertionError(f"Derived prompt/response corruption row={i}")
        for field in ("response_mask", "loss_mask", "old_log_probs", "advantages"):
            if not torch.equal(derived[field][i], batch[field][i][:suffix]):
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
        keys, derived, tree_policy="qwen_turn", turn_min_segment_tokens=128,
    )
    # qwen_turn only splits at complete message boundaries (or a prompt
    # boundary which survived min_segment_tokens=128). If the new prompt end
    # falls <128 tokens after the previous turn boundary, that boundary is
    # merged with a subsequent response turn and the actual shared root can
    # end UP TO 127 tokens before the nominal prefix boundary.
    # Do not change production tree policy or require a non-existent split.
    root_tokens = (
        forest.trees[0].segment_plan.get(0).length if len(forest.trees) == 1
        else None
    )
    leaf_shapes = (
        [(seg.segment_id, seg.parent_id, seg.length)
         for seg in forest.trees[0].segment_plan.segments.values()]
        if len(forest.trees) == 1 else []
    )
    minimum_shared = prefix - 127
    print(
        "TPR_N8_ACTUAL_TREE "
        + json.dumps({
            "nominal_prefix": prefix,
            "actual_shared_root": root_tokens,
            "minimum_acceptable_shared_root": minimum_shared,
            "forest_trees": len(forest.trees),
            "forest_segments": forest.segment_count,
            "member_rows": [list(t.member_rows) for t in forest.trees],
            "segment_layout": leaf_shapes,
        }, sort_keys=True),
        flush=True,
    )
    if (len(forest.trees) != 1
            or set(forest.trees[0].member_rows) != set(range(BRANCHES))
            or root_tokens is None
            or not minimum_shared <= root_tokens < prefix + suffix):
        raise AssertionError(
            f"Invalid real qwen_turn eight-way reuse: nominal_prefix={prefix}, "
            f"actual_root={root_tokens}, minimum={minimum_shared}, "
            f"trees={len(forest.trees)}; inspect TPR_N8_ACTUAL_TREE"
        )
    if not torch.equal(forest.trees[0].segment_plan.get(0).token_ids,
                       derived["input_ids"][0][:root_tokens]):
        raise AssertionError("Shared root is not the exact source tokens")
    if any(
        not torch.equal(derived["input_ids"][i][:root_tokens],
                        derived["input_ids"][0][:root_tokens])
        for i in range(1, BRANCHES)
    ):
        raise AssertionError("Shared root would reuse mismatched token sequences")
    if forest.segment_count < 1 + BRANCHES:
        raise AssertionError("Production tree failed to retain distinct branches")
    if forest.logical_loss_tokens != sum(supervised):
        raise AssertionError("Production tree lost original PPO targets")

    payload = dict(raw)
    payload["tensordict"] = derived
    payload["keys"] = keys
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
        "new_keys": list(keys),
        "n": BRANCHES,
        "prefix": prefix, "suffix": suffix, "length": prefix+suffix,
        "ratio": f"{prefix//suffix}:1", "ppo_tokens": sum(supervised),
        "ppo_tokens_per_row": supervised,
        "forest_trees": len(forest.trees),
        "forest_segments": forest.segment_count,
        "actual_shared_root_tokens": root_tokens,
        "effective_root_to_suffix_ratio": root_tokens / suffix,
        "nominal_prefix_minus_actual_root": prefix - root_tokens,
        "min_turn_segment_tokens": 128,
        "unique_tree_tokens": sum(
            segment.length for tree in forest.trees
            for segment in tree.segment_plan.segments.values()
        ),
        "input_sha256": digest.hexdigest(),
        "changed_conditioning": True,
        "synthetic_shared_uid": True,
        "note": (
            f"Original real PPO response tokens and labels; last {prefix} prompt "
            f"tokens and first {suffix} response tokens retained. Historical "
            "conditioning changed and sessions share synthetic UID. "
            "PERFORMANCE ONLY, not on-policy PPO correctness."
        ),
    }
    if raw.get("performance_synthetic_response_windows", False):
        audit["dropped_actor_unused_fields"] = dropped_auxiliary_fields
        audit["synthetic_response_windows"] = True
        audit["original_actor_capture"] = raw["performance_window_original_capture"]
        audit["response_window_sources"] = raw["performance_row_windows"]
        audit["note"] += (
            " Synthetic N8 workload: two short source rows replaced by "
            "distinct later windows from original long replies, preserving "
            "aligned original token slices and PPO labels; NOT on-policy."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    target.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print("TPR_N8_CAPTURE "+json.dumps(audit, sort_keys=True), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--prefix", type=int, default=PREFIX)
    parser.add_argument("--suffix", type=int, default=SUFFIX)
    args = parser.parse_args()
    derive_capture(args.source, args.target, prefix=args.prefix, suffix=args.suffix)


if __name__ == "__main__":
    main()
