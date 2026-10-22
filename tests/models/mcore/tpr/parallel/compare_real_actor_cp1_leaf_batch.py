"""Isolate the incremental effect of K=2 on ONE identical shared CP1 PPO tree.

This is not a Full vs TPR comparison. Both inputs are produced by
test_real_actor_cp1_short_full_tpr_npu.py with TPR_CP1_MODE=shared.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .compare_real_actor_cp1_short_full_tpr import (
    _complete_cp1_parameter_gradient_metrics,
    _metrics,
)


def compare(k1_result: Path, k2_result: Path) -> dict:
    a = json.loads(k1_result.read_text())
    b = json.loads(k2_result.read_text())
    required_same = (
        "source_capture", "input_sha256", "checkpoint",
        "initial_weights_witness_sha256", "topology", "original_keys",
        "ppo_tokens", "logical_input_tokens", "unique_tree_tokens",
        "tree_policy", "forest_trees", "forest_segments", "optimizer_step",
    )
    if a.get("mode") != "shared" or b.get("mode") != "shared":
        raise AssertionError("Both cases must run the SAME shared-tree mode")
    for key in required_same:
        if a[key] != b[key]:
            raise AssertionError(f"K1/K2 input or topology mismatch: {key}")
    if a["topology"] != {"tp": 1, "cp": 1, "dp": 1, "world": 1}:
        raise AssertionError("This experiment is CP1/TP1 single-NPU only")
    if a["optimizer_step"]:
        raise AssertionError("This benchmark must not update parameters")
    if (a["forest_trees"], a["forest_segments"]) != (1, 3):
        raise AssertionError("Expect one shared Prefix Root with two sibling leaves")
    if (a.get("leaf_batch_configured_k"), b.get("leaf_batch_configured_k")) != (1, 2):
        raise AssertionError("Results must explicitly record K1 and K2 configurations")

    k1_stats, k2_stats = a.get("leaf_batch_stats"), b.get("leaf_batch_stats")
    if not k1_stats or not k2_stats:
        raise AssertionError("Missing real executor leaf batch stats (stale capture output?)")
    if k1_stats["grouped_leaf_calls"] != 0 or k1_stats["grouped_leaf_segments"] != 0:
        raise AssertionError("K1 performed a grouped call unexpectedly")
    if k2_stats["grouped_leaf_calls"] != 1 or k2_stats["grouped_leaf_segments"] != 2:
        raise AssertionError(
            "K2 must group both sibling leaves into exactly one model Forward/Backward"
        )
    if k1_stats["model_forward_calls"] != 4 or k2_stats["model_forward_calls"] != 3:
        raise AssertionError("Unexpected Push/Pop/Visit model-forward counts for this exact tree")

    def logprobs(item):
        result = {(int(row), int(offset)): float(value)
                  for row, offset, value in item["logprobs"]}
        if len(result) != item["ppo_tokens"]:
            raise AssertionError("Duplicate or missing PPO token positions")
        return result

    left, right = logprobs(a), logprobs(b)
    if left.keys() != right.keys():
        raise AssertionError("K1/K2 supervise different logical PPO tokens")
    positions = sorted(left)
    ga, gb = a["gradient_witness"], b["gradient_witness"]
    for key in ("names", "canonical_shapes", "samples_per_name", "samples_per_parameter"):
        if ga[key] != gb[key]:
            raise AssertionError(f"Parameter-gradient witness layout mismatch: {key}")
    ca, cb = ga["complete_norm_gradients"], gb["complete_norm_gradients"]
    if ca["names"] != cb["names"] or ca["shapes"] != cb["shapes"]:
        raise AssertionError("Complete Norm gradient witness layout mismatch")
    full_grads = _complete_cp1_parameter_gradient_metrics(a, b)
    if a.get("full_grad_dir") != b.get("full_grad_dir") and (
        bool(a.get("full_grad_dir")) != bool(b.get("full_grad_dir"))
    ):
        raise AssertionError("Only one run saved full parameter gradients")
    result = {
        "comparison": "identical_shared_tree_K1_vs_K2",
        "input_sha256": a["input_sha256"],
        "initial_weights_witness_sha256": a["initial_weights_witness_sha256"],
        "ppo_tokens": a["ppo_tokens"],
        "k1_stats": k1_stats,
        "k2_stats": k2_stats,
        "loss_abs_gap": abs(a["loss"] - b["loss"]),
        "logprob": _metrics([left[key] for key in positions], [right[key] for key in positions]),
        "sampled_gradient": _metrics(ga["values"], gb["values"]),
        "complete_norm_gradient": _metrics(ca["values"], cb["values"]),
        "full_parameter_gradient": full_grads,
        "global_grad_norm_relative_gap": (
            abs(a["global_grad_norm"] - b["global_grad_norm"])
            / max(abs(a["global_grad_norm"]), 1e-30)
        ),
        "k1_fwd_bwd_seconds": a["fwd_bwd_seconds"],
        "k2_fwd_bwd_seconds": b["fwd_bwd_seconds"],
        "speedup": a["fwd_bwd_seconds"] / b["fwd_bwd_seconds"],
        "k1_peak_allocated_gib": a["peak_allocated_gib"],
        "k2_peak_allocated_gib": b["peak_allocated_gib"],
        "k1_peak_reserved_gib": a["peak_reserved_gib"],
        "k2_peak_reserved_gib": b["peak_reserved_gib"],
        "caveat": (
            "Same shared-tree Qwen3-4B controlled cropped real-label PPO: "
            "isolates physical leaf grouping, not baseline Full-vs-TPR drift. "
            "Single uncached repetition is not a performance benchmark. "
            "Sampled gradient relative L2 is not full-parameter relative L2."
        ),
    }
    for key in ("loss_abs_gap", "speedup"):
        if not math.isfinite(result[key]):
            raise AssertionError(f"Non-finite {key}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k1_result", type=Path)
    parser.add_argument("k2_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.k1_result, args.k2_result)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("TPR_CP1_LEAF_K1_K2_COMPARISON", json.dumps({
        "loss_gap": report["loss_abs_gap"],
        "logprob_rel_l2": report["logprob"]["relative_l2"],
        "sampled_grad_rel_l2": report["sampled_gradient"]["relative_l2"],
        "complete_norm_grad_rel_l2": report["complete_norm_gradient"]["relative_l2"],
        "full_parameter_grad_rel_l2": (
            None if report["full_parameter_gradient"] is None
            else report["full_parameter_gradient"]["relative_l2"]
        ),
        "model_forward_calls": [
            report["k1_stats"]["model_forward_calls"],
            report["k2_stats"]["model_forward_calls"],
        ],
        "grouped_leaf_calls": report["k2_stats"]["grouped_leaf_calls"],
        "speedup_one_run": report["speedup"],
        "peak_alloc_gib": [
            report["k1_peak_allocated_gib"], report["k2_peak_allocated_gib"],
        ],
    }, sort_keys=True), flush=True)
    print(f"TPR_CP1_LEAF_K1_K2_REPORT saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
