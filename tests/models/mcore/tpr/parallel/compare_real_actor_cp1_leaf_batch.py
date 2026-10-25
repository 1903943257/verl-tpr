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


def _compare_leaf_traces(k1_result: Path, k2_result: Path, *, required: bool) -> dict | None:
    import torch

    trace_paths = [
        result.parent / "trace" / "shared" / "trace.pt"
        for result in (k1_result, k2_result)
    ]
    present = [path.is_file() for path in trace_paths]
    if not all(present):
        if required:
            raise FileNotFoundError(f"Required K1/K2 layer traces missing: {trace_paths}")
        return None

    left, right = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in trace_paths
    ]
    if left.get("mode") != "shared" or right.get("mode") != "shared":
        raise AssertionError("K1/K2 trace modes must both be shared")

    def normalize(key: str) -> str:
        return key.replace("|leaf_group|", "|leaf|")

    left_records = {normalize(key): value for key, value in left["records"].items()}
    right_records = {normalize(key): value for key, value in right["records"].items()}
    left_keys = {
        key for key in left_records
        if "|leaf|" in key and not key.endswith("_grad_out")
    }
    right_keys = {
        key for key in right_records
        if "|leaf|" in key and not key.endswith("_grad_out")
    }
    if left_keys != right_keys:
        raise AssertionError(
            "K1/K2 layer trace coordinate mismatch: "
            f"missing_k1={sorted(right_keys-left_keys)[:8]}, "
            f"missing_k2={sorted(left_keys-right_keys)[:8]}"
        )

    stage_order = {
        "attn_in": 0, "qkv_in": 1, "qkv_out": 2, "query_post_rope": 3,
        "post_rope_k": 4, "raw_v": 5, "attn_out": 6, "proj_in": 7, "proj_out": 8,
        "fc1_in": 9, "fc1_out": 10, "fc2_in": 11, "fc2_out": 12,
    }
    metrics = []
    for key in left_keys:
        a = left_records[key]["values"].float()
        b = right_records[key]["values"].float()
        if a.shape != b.shape:
            raise AssertionError(f"Trace tensor shape mismatch for {key}: {a.shape} vs {b.shape}")
        diff = b - a
        norm_a = float(a.norm().item())
        norm_b = float(b.norm().item())
        max_abs = float(diff.abs().max().item())
        mean_abs = float(diff.abs().mean().item())
        rel_l2 = float(diff.norm().item() / max(norm_a, 1e-30))
        exact = bool(torch.equal(left_records[key]["values"], right_records[key]["values"]))
        fields = key.split("|")
        row = int(fields[0][1:])
        layer = int(fields[2][1:])
        stage = fields[3]
        metrics.append({
            "key": key, "row": row, "layer": layer, "stage": stage,
            "stage_order": stage_order.get(stage, 99),
            "relative_l2": rel_l2, "mean_abs": mean_abs, "max_abs": max_abs,
            "exact": exact, "count": diff.numel(),
        })
    metrics.sort(key=lambda item: (item["layer"], item["stage_order"], item["row"]))
    layers = sorted({item["layer"] for item in metrics})
    layer_progression = [
        {
            "layer": layer,
            "max_diff": max(
                (item for item in metrics if item["layer"] == layer),
                key=lambda item: item["relative_l2"],
            ),
            "last_stage_by_row": [
                next(
                    (item for item in reversed(metrics)
                     if item["layer"] == layer and item["row"] == row),
                    None,
                )
                for row in (0, 1)
            ],
        }
        for layer in layers
    ]
    first_changed = next((item for item in metrics if not item["exact"]), None)
    first_material = next((item for item in metrics if item["relative_l2"] > 1e-6), None)

    left_grad_keys = {key for key in left_records if key.endswith("_grad_out")}
    right_grad_keys = {key for key in right_records if key.endswith("_grad_out")}
    if left_grad_keys != right_grad_keys:
        raise AssertionError(
            "K1/K2 backward trace coordinate mismatch: "
            f"missing_k1={sorted(right_grad_keys-left_grad_keys)[:8]}, "
            f"missing_k2={sorted(left_grad_keys-right_grad_keys)[:8]}"
        )
    grad_metrics = []
    for key in left_grad_keys:
        a = left_records[key]["values"].float()
        b = right_records[key]["values"].float()
        if a.shape != b.shape:
            raise AssertionError(f"Gradient trace tensor shape mismatch for {key}: {a.shape} vs {b.shape}")
        diff = b - a
        norm_a = float(a.norm().item())
        fields = key.split("|")
        row = int(fields[0][1:])
        layer = int(fields[2][1:])
        stage = fields[3].removesuffix("_grad_out")
        grad_metrics.append({
            "key": key, "row": row, "layer": layer, "stage": stage,
            "stage_order": stage_order.get(stage, 99),
            "relative_l2": float(diff.norm().item() / max(norm_a, 1e-30)),
            "mean_abs": float(diff.abs().mean().item()),
            "max_abs": float(diff.abs().max().item()),
            "exact": bool(torch.equal(left_records[key]["values"], right_records[key]["values"])),
            "count": diff.numel(),
        })
    grad_metrics.sort(key=lambda item: (item["layer"], item["stage_order"], item["row"]))
    first_grad_change = next((item for item in grad_metrics if not item["exact"]), None)
    first_grad_material = next((item for item in grad_metrics if item["relative_l2"] > 1e-6), None)

    # The grouped run replays native L1 linear_qkv on its *own* captured
    # input with physical B=1 and unchanged checkpoint weights. This isolates
    # the GEMM-M effect without including Prefix segmentation differences.
    grouped_qkv = right.get("grouped_qkv_replay", {})
    k1_qkv = left.get("first_layer_leaf_qkv_operands", {})
    k2_qkv = right.get("first_layer_leaf_qkv_operands", {})
    qkv_replay = {}
    if grouped_qkv:
        for row in (0, 1):
            item = grouped_qkv.get(row)
            if item is None:
                raise AssertionError(f"Missing isolated grouped QKV replay row={row}")
            recorded = left_records[f"r{row}|leaf|L1|qkv_out"]
            sample = item["sampled_single_output"]
            if recorded["positions"] != sample["positions"]:
                raise AssertionError(f"K1 vs native K2 single-GEMM replay coords differ on row={row}")
            reference = recorded["values"].float()
            replay = sample["values"].float()
            if reference.shape != replay.shape:
                raise AssertionError(f"K1 vs replay QKV values differ in shape row={row}")
            delta = replay - reference
            full_first_layer = {}
            if k1_qkv and k2_qkv:
                for stage in ("qkv_in", "qkv_out"):
                    key = (row, stage)
                    if key not in k1_qkv or key not in k2_qkv:
                        raise AssertionError(f"Missing full first-layer operands {key}")
                    x = k1_qkv[key].float()
                    y = k2_qkv[key].float()
                    if x.shape != y.shape:
                        raise AssertionError(f"K1/K2 full first-layer shape mismatch {key}")
                    difference = y - x
                    full_first_layer[stage] = {
                        "relative_l2": float(difference.norm().item() / max(x.norm().item(), 1e-30)),
                        "max_abs": float(difference.abs().max().item()),
                        "mean_abs": float(difference.abs().mean().item()),
                        "exact": bool(torch.equal(k1_qkv[key], k2_qkv[key])),
                        "count": difference.numel(),
                    }
                independent_full = item["single_output"].float()
                reference_full = k1_qkv[row, "qkv_out"].float()
                if independent_full.shape != reference_full.shape:
                    raise AssertionError(f"Isolated native QKV result full shape mismatch row={row}")
                delta_full = independent_full - reference_full
                full_first_layer["k1_vs_single_replay"] = {
                    "relative_l2": float(delta_full.norm().item() / max(reference_full.norm().item(), 1e-30)),
                    "max_abs": float(delta_full.abs().max().item()),
                    "mean_abs": float(delta_full.abs().mean().item()),
                    "exact": bool(torch.equal(independent_full, reference_full)),
                    "count": delta_full.numel(),
                }
            qkv_replay[row] = {
                "k1_vs_k2_full_first_layer": full_first_layer,
                "k2_grouped_vs_same_input_single": item["same_input_grouped_vs_single"],
                "k1_vs_k2_same_input_single": {
                    "relative_l2": float(delta.norm().item() / max(reference.norm().item(), 1e-30)),
                    "mean_abs": float(delta.abs().mean().item()),
                    "max_abs": float(delta.abs().max().item()),
                    "exact": bool(torch.equal(recorded["values"], sample["values"])),
                    "count": delta.numel(),
                },
                "grouped_input_shape": item["grouped_input_shape"],
            }

    attention = right.get("same_qkv_attention", {})
    attention_summary = sorted(
        (
            {
                "key": key,
                "relative_l2": value["rel_l2"],
                "mean_abs": value["mean_abs"],
                "max_abs": value["max_abs"],
                "exact": value["exact"],
            }
            for key, value in attention.items()
        ),
        key=lambda item: item["relative_l2"],
        reverse=True,
    )
    independent_rect = right.get("group_vs_single_rectangular", {})
    independent_rect_summary = sorted(
        (
            {
                "key": key,
                "relative_l2": value["rel_l2"],
                "max_abs": value["max_abs"],
                "exact": value["exact"],
            }
            for key, value in independent_rect.items()
        ),
        key=lambda item: item["relative_l2"], reverse=True,
    )
    return {
        "k2_grouped_vs_single_rectangular": independent_rect_summary,
        "trace_paths": [str(path) for path in trace_paths],
        "leaf_forward_tensor_count": len(metrics),
        "layer_progression": layer_progression,
        "layer1_forward_stages": [item for item in metrics if item["layer"] == 1],
        "first_forward_value_change": first_changed,
        "first_forward_rel_l2_gt_1e-6": first_material,
        "largest_forward_differences": sorted(
            metrics, key=lambda item: item["relative_l2"], reverse=True,
        )[:12],
        "first_gradient_value_change": first_grad_change,
        "first_gradient_rel_l2_gt_1e-6": first_grad_material,
        "largest_gradient_differences": sorted(
            grad_metrics, key=lambda item: item["relative_l2"], reverse=True,
        )[:12],
        "k2_same_input_qkv_single_replay": qkv_replay,
        "k2_same_qkv_rectangular_attention": attention_summary,
    }


def compare(k1_result: Path, k2_result: Path, *, require_trace: bool = False) -> dict:
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
    layer_trace = _compare_leaf_traces(k1_result, k2_result, required=require_trace)
    logprob_deltas = sorted(
        (
            {
                "row": key[0], "response_offset": key[1],
                "k1": left[key], "k2": right[key],
                "abs_diff": abs(left[key] - right[key]),
            }
            for key in positions
        ),
        key=lambda item: item["abs_diff"], reverse=True,
    )[:12]
    result = {
        "top_logprob_differences": logprob_deltas,
        "layer_trace": layer_trace,
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
    parser.add_argument("--require-trace", action="store_true")
    args = parser.parse_args()
    report = compare(args.k1_result, args.k2_result, require_trace=args.require_trace)
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
        "first_forward_mismatch": (
            None if report["layer_trace"] is None
            else report["layer_trace"]["first_forward_rel_l2_gt_1e-6"]
        ),
        "largest_same_qkv_attention_rel_l2": (
            None if not report["layer_trace"]
            or not report["layer_trace"]["k2_same_qkv_rectangular_attention"]
            else report["layer_trace"]["k2_same_qkv_rectangular_attention"][0]["relative_l2"]
        ),
        "largest_group_vs_single_rectangle_rel_l2": (
            None if not report["layer_trace"]
            or not report["layer_trace"]["k2_grouped_vs_single_rectangular"]
            else report["layer_trace"]["k2_grouped_vs_single_rectangular"][0]["relative_l2"]
        ),
        "first_gradient_mismatch": (
            None if report["layer_trace"] is None
            else report["layer_trace"]["first_gradient_rel_l2_gt_1e-6"]
        ),
        "layer_progression": (
            None if report["layer_trace"] is None
            else [
                {
                    "layer": item["layer"],
                    "largest_rel_l2": item["max_diff"]["relative_l2"],
                    "largest_stage": item["max_diff"]["stage"],
                    "largest_row": item["max_diff"]["row"],
                }
                for item in report["layer_trace"]["layer_progression"]
            ]
        ),
        "top_logprob_differences": report["top_logprob_differences"][:5],
        "same_input_l1_qkv_replay": (
            None if report["layer_trace"] is None
            else report["layer_trace"]["k2_same_input_qkv_single_replay"]
        ),
    }, sort_keys=True), flush=True)
    print(f"TPR_CP1_LEAF_K1_K2_REPORT saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
