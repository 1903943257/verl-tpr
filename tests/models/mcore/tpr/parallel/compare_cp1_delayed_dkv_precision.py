"""Compare exact all-parameter TP1/CP1 BF16 gradients from three fixed-input PPO arms.

Full is a controlled whole-sequence square-CANN reference, not the native VERL
trainer schedule.  The two TPR arms differ only in delayed dKV accumulation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("full", "tpr_old", "tpr_fp32")


def _pair_metrics(left: dict, right: dict) -> dict:
    import torch

    a_dir, b_dir = left.get("full_grad_dir"), right.get("full_grad_dir")
    if not a_dir or not b_dir:
        raise AssertionError("Every arm must export ALL parameter gradients")
    a_manifest = torch.load(Path(a_dir) / "manifest.pt", map_location="cpu", weights_only=False)
    b_manifest = torch.load(Path(b_dir) / "manifest.pt", map_location="cpu", weights_only=False)
    if a_manifest != b_manifest or not a_manifest:
        raise AssertionError("Gradient manifests are empty or differ between arms")

    norm_a_sq = norm_b_sq = diff_sq = dot = abs_sum = 0.0
    count, max_abs = 0, 0.0
    per_parameter = []
    for entry in a_manifest:
        a = torch.load(Path(a_dir) / entry["filename"], map_location="cpu", weights_only=True)
        b = torch.load(Path(b_dir) / entry["filename"], map_location="cpu", weights_only=True)
        if a.shape != b.shape or a.dtype != b.dtype or a.dtype != torch.bfloat16:
            raise AssertionError(f"Expected matching BF16 parameter gradients: {entry['name']}")
        a, b = a.reshape(-1), b.reshape(-1)
        item_sq, item_max = 0.0, 0.0
        for start in range(0, a.numel(), 1_048_576):
            x = a[start:start + 1_048_576].double()
            y = b[start:start + 1_048_576].double()
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise AssertionError(f"Nonfinite gradients in {entry['name']}")
            delta = y - x
            norm_a_sq += float((x * x).sum())
            norm_b_sq += float((y * y).sum())
            item_sq_chunk = float((delta * delta).sum())
            item_sq += item_sq_chunk
            diff_sq += item_sq_chunk
            dot += float((x * y).sum())
            abs_sum += float(delta.abs().sum())
            item_max = max(item_max, float(delta.abs().max()))
        count += a.numel()
        max_abs = max(max_abs, item_max)
        per_parameter.append({
            "name": entry["name"],
            "l2_difference": math.sqrt(item_sq),
            "max_abs": item_max,
        })
        del a, b

    if not count or norm_a_sq == 0 or norm_b_sq == 0:
        raise AssertionError("Empty or zero full gradient vector")
    norm_a, norm_b = math.sqrt(norm_a_sq), math.sqrt(norm_b_sq)
    return {
        "elements": count,
        "reference_norm": norm_a,
        "candidate_norm": norm_b,
        "relative_l2": math.sqrt(diff_sq / norm_a_sq),
        "cosine": dot / (norm_a * norm_b),
        "max_abs": max_abs,
        "mean_abs": abs_sum / count,
        "worst_parameters_by_l2": sorted(
            per_parameter, key=lambda p: p["l2_difference"], reverse=True
        )[:10],
    }


def compare(root: Path) -> dict:
    arms = {}
    for arm in ARMS:
        file = root / arm / "result.json"
        if not file.is_file():
            raise FileNotFoundError(file)
        arms[arm] = json.loads(file.read_text())

    expected_mode = {"full": "full", "tpr_old": "shared", "tpr_fp32": "shared"}
    expected_fp32 = {"full": False, "tpr_old": False, "tpr_fp32": True}
    for arm, result in arms.items():
        if result["mode"] != expected_mode[arm]:
            raise AssertionError(f"{arm} ran the wrong model mode")
        if result.get("experimental_delayed_dkv_accum_fp32") is not expected_fp32[arm]:
            raise AssertionError(f"{arm} delayed dKV precision gate mismatch")
        if result["topology"] != {"tp": 1, "cp": 1, "dp": 1, "world": 1}:
            raise AssertionError(f"{arm} must be truly TP1 CP1 DP1")
        if result["optimizer_step"]:
            raise AssertionError(f"{arm} updated parameters")
        if not result.get("full_grad_dir") or result.get("full_gradient_scope") != "ALL_PARAMETERS_BF16_TP1_CP1":
            raise AssertionError(f"{arm} is missing complete BF16 gradients")

    baseline = arms["full"]
    for arm in ("tpr_old", "tpr_fp32"):
        candidate = arms[arm]
        for key in (
            "source_capture", "input_sha256", "checkpoint",
            "initial_weights_witness_sha256", "ppo_tokens", "original_keys",
            "logical_input_tokens", "tree_policy",
        ):
            if candidate[key] != baseline[key]:
                raise AssertionError(f"{arm}: mismatched input/initial state field {key}")
    for key in ("forest_trees", "forest_segments", "unique_tree_tokens", "reuse_ratio"):
        if arms["tpr_old"][key] != arms["tpr_fp32"][key]:
            raise AssertionError(f"TPR forest changed with delayed dKV dtype: {key}")
    old_logps = arms["tpr_old"]["logprobs"]
    fp32_logps = arms["tpr_fp32"]["logprobs"]
    if [(p[0], p[1]) for p in old_logps] != [(p[0], p[1]) for p in fp32_logps]:
        raise AssertionError("TPR-old and TPR-FP32 supervise different logprob coordinates")
    logprob_max_abs_gap = max(
        (abs(a[2] - b[2]) for a, b in zip(old_logps, fp32_logps, strict=True)),
        default=0.0,
    )
    loss_abs_gap = abs(arms["tpr_old"]["loss"] - arms["tpr_fp32"]["loss"])

    pairs = {
        "full_vs_tpr_old": _pair_metrics(baseline, arms["tpr_old"]),
        "full_vs_tpr_fp32": _pair_metrics(baseline, arms["tpr_fp32"]),
        "tpr_old_vs_tpr_fp32": _pair_metrics(arms["tpr_old"], arms["tpr_fp32"]),
    }
    old_err = pairs["full_vs_tpr_old"]["relative_l2"]
    new_err = pairs["full_vs_tpr_fp32"]["relative_l2"]
    report = {
        "scope": "Qwen3-4B BF16 one NPU TP1 CP1 DP1, fixed cropped two-row Actor PPO",
        "full_reference": "controlled whole-sequence square-CANN; not native VERL trainer",
        "optimizer_step": False,
        "branch_relay_only_changed": True,
        "source_capture": baseline["source_capture"],
        "checkpoint": baseline["checkpoint"],
        "input_sha256": baseline["input_sha256"],
        "initial_weights_witness_sha256": baseline["initial_weights_witness_sha256"],
        "ppo_tokens": baseline["ppo_tokens"],
        "tpr_reuse_ratio": arms["tpr_old"]["reuse_ratio"],
        "pairs": pairs,
        "fp32_relative_l2_over_old": new_err / max(old_err, 1e-30),
        "fp32_reduces_full_gradient_error": new_err < old_err,
        "tpr_old_vs_fp32_forward_loss_abs_gap": loss_abs_gap,
        "tpr_old_vs_fp32_forward_logprob_max_abs_gap": logprob_max_abs_gap,
        "arm_peak_alloc_gib": {arm: result["peak_allocated_gib"] for arm, result in arms.items()},
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = compare(args.root)
    destination = args.root / "comparison_delayed_dkv_fp32.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, indent=2))
    for name, pair in report["pairs"].items():
        print(
            f"{name}: rel_l2={pair['relative_l2']:.9g} "
            f"cosine={pair['cosine']:.12g} max_abs={pair['max_abs']:.9g} "
            f"elements={pair['elements']}",
            flush=True,
        )
    print(
        f"TPR_DKV_FP32_CP1_COMPARISON saved={destination} "
        f"improved={report['fp32_reduces_full_gradient_error']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
