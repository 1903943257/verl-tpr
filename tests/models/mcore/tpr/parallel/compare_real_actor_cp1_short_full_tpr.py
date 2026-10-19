"""Compare four CP1 Actor PPO arms in canonical TP1 parameter coordinates.

Full means whole-sequence GPT with NO TPR context, using the same CP1
square-CANN oracle core as the TPR-capable model. The derivative capture
changes original causal conditioning: numerical parity only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODES = ("full", "one_node", "split_no_share", "shared")


def _complete_cp1_parameter_gradient_metrics(a: dict, b: dict) -> dict | None:
    """Stream ALL TP1/CP1 gradients for a single Qwen3-4B PPO update.

    The prior canonical witness samples parameter values. The new optional
    manifests contain every original BF16 parameter gradient. Accumulate
    exact vector sums in FP64 on CPU, one tensor/chunk at a time.
    """
    left, right = a.get("full_grad_dir"), b.get("full_grad_dir")
    if left is None and right is None:
        return None
    if not left or not right:
        raise AssertionError("Only one CP1 mode has full gradients")
    import torch
    ma = torch.load(Path(left) / "manifest.pt", map_location="cpu", weights_only=False)
    mb = torch.load(Path(right) / "manifest.pt", map_location="cpu", weights_only=False)
    if ma != mb or not ma:
        raise AssertionError("CP1 full gradient manifests differ")
    sq_a = sq_b = diff_sq = dot = 0.0
    count = 0
    layers = []
    for entry in ma:
        xa = torch.load(Path(left) / entry["filename"], map_location="cpu", weights_only=True)
        xb = torch.load(Path(right) / entry["filename"], map_location="cpu", weights_only=True)
        if xa.shape != xb.shape or xa.dtype != xb.dtype or xa.dtype != torch.bfloat16:
            raise AssertionError(f"CP1 full gradient dtype/shape differs: {entry['name']}")
        va, vb = xa.flatten(), xb.flatten()
        param_diff_sq = 0.0
        for start in range(0, va.numel(), 1_048_576):
            x = va[start:start + 1_048_576].double()
            y = vb[start:start + 1_048_576].double()
            d = y - x
            sq_a += float((x*x).sum())
            sq_b += float((y*y).sum())
            diff_sq += float((d*d).sum())
            param_diff_sq += float((d*d).sum())
            dot += float((x*y).sum())
        layers.append((entry["name"], math.sqrt(param_diff_sq)))
        count += va.numel()
        del xa, xb, va, vb
    norm_a, norm_b = math.sqrt(sq_a), math.sqrt(sq_b)
    return {
        "complete_parameter_elements": count,
        "full_norm": norm_a,
        "candidate_norm": norm_b,
        "relative_l2": math.sqrt(diff_sq/max(sq_a, 1e-30)),
        "cosine": dot/max(norm_a*norm_b, 1e-30),
        "norm_relative_gap": abs(norm_a-norm_b)/max(norm_a, 1e-30),
        "top_absolute_gradient_errors": sorted(layers, key=lambda item: item[1], reverse=True)[:12],
    }


def _metrics(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b) or not a:
        raise AssertionError("Witness coordinates mismatch or empty")
    aa = bb = dot = diff = 0.0
    max_abs = total_abs = 0.0
    for x, y in zip(a, b, strict=True):
        if not math.isfinite(x) or not math.isfinite(y):
            raise AssertionError("Non-finite comparison witness")
        delta = x - y
        aa += x * x
        bb += y * y
        dot += x * y
        diff += delta * delta
        max_abs = max(max_abs, abs(delta))
        total_abs += abs(delta)
    return {
        "relative_l2": math.sqrt(diff / max(aa, 1e-30)),
        "cosine": dot / max(math.sqrt(aa * bb), 1e-30),
        "mean_abs": total_abs / len(a),
        "max_abs": max_abs,
        "count": len(a),
    }


def compare(directory: Path) -> dict:
    results = {}
    for mode in MODES:
        path = directory / mode / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        results[mode] = json.loads(path.read_text())
        if results[mode]["mode"] != mode:
            raise AssertionError(f"{mode}: recorded mode mismatch")
    full = results["full"]
    common = (
        "source_capture", "input_sha256", "checkpoint",
        "initial_weights_witness_sha256", "ppo_tokens",
        "original_keys", "logical_input_tokens", "tree_policy", "topology",
    )
    for mode, sample in results.items():
        for key in common:
            if sample[key] != full[key]:
                raise AssertionError(f"Mismatch in {key}: full vs {mode}")
        if sample["topology"] != {"tp": 1, "cp": 1, "dp": 1, "world": 1}:
            raise AssertionError("Expected exactly CP1 single-NPU layout")
        if sample["optimizer_step"]:
            raise AssertionError("Numerical parity must not update parameters")
    if (full["forest_trees"], full["forest_segments"]) != (2, 2):
        raise AssertionError("Full did not use two independent whole trajectories")
    if (results["one_node"]["forest_trees"], results["one_node"]["forest_segments"]) != (2, 2):
        raise AssertionError("One-node path is not exactly two independent roots")
    split = results["split_no_share"]
    if (split["forest_trees"], split["forest_segments"]) != (2, 4):
        raise AssertionError("Split-no-share must be two Root/Leaf chains")
    if split["unique_tree_tokens"] != full["logical_input_tokens"] or split["reuse_ratio"] != 1:
        raise AssertionError("Split-no-share accidentally reused physical Root tokens")
    shared = results["shared"]
    if (shared["forest_trees"], shared["forest_segments"]) != (1, 3):
        raise AssertionError("Shared path is not Root + two sibling Leafs")
    if not shared["reuse_ratio"] > 1 or not full["reuse_ratio"] == 1:
        raise AssertionError("No actual Prefix reuse in controlled shared mode")
    logps = {}
    for mode, item in results.items():
        pairs = {(int(row), int(offset)): float(value) for row, offset, value in item["logprobs"]}
        if len(pairs) != item["ppo_tokens"] or len(pairs) != len(item["logprobs"]):
            raise AssertionError(f"{mode}: duplicate or missing logical PPO tokens")
        logps[mode] = pairs
    if any(logps[mode].keys() != logps["full"].keys() for mode in MODES):
        raise AssertionError("Full, one-node and shared supervise different tokens")
    keys = sorted(logps["full"])
    outcomes = {}
    for name, reference, candidate in (
        ("full_vs_one_node", "full", "one_node"),
        ("one_node_vs_split_no_share", "one_node", "split_no_share"),
        ("split_no_share_vs_shared", "split_no_share", "shared"),
        ("one_node_vs_shared", "one_node", "shared"),
        ("full_vs_shared", "full", "shared"),
    ):
        a, b = results[reference], results[candidate]
        ga = a["gradient_witness"]
        gb = b["gradient_witness"]
        for field in ("names", "canonical_shapes", "samples_per_name", "samples_per_parameter"):
            if ga[field] != gb[field]:
                raise AssertionError(f"{name}: canonical gradient coordinates differ: {field}")
        na, nb = ga["complete_norm_gradients"], gb["complete_norm_gradients"]
        if na["names"] != nb["names"] or na["shapes"] != nb["shapes"]:
            raise AssertionError(f"{name}: complete Norm gradient layout mismatch")
        one = {
            "loss_abs_gap": abs(a["loss"] - b["loss"]),
            "logprob": _metrics(
                [logps[reference][key] for key in keys],
                [logps[candidate][key] for key in keys],
            ),
            "canonical_sampled_grad": _metrics(ga["values"], gb["values"]),
            "complete_norm_grad": _metrics(na["values"], nb["values"]),
            "complete_parameter_grad": _complete_cp1_parameter_gradient_metrics(a, b),
            "global_grad_norm_reference": a["global_grad_norm"],
            "global_grad_norm_candidate": b["global_grad_norm"],
            "global_grad_norm_relative_gap": (
                abs(a["global_grad_norm"] - b["global_grad_norm"])
                / max(abs(a["global_grad_norm"]), 1e-30)
            ),
            "single_run_seconds_reference": a["fwd_bwd_seconds"],
            "single_run_seconds_candidate": b["fwd_bwd_seconds"],
            "peak_alloc_gib_reference": a["peak_allocated_gib"],
            "peak_alloc_gib_candidate": b["peak_allocated_gib"],
        }
        # Diagnostic targets, NOT assertions of acceptance. Preserve the full
        # values if some gates fail so kernel errors remain visible.
        one["provisional_gates"] = {
            "loss_lt_0.02": one["loss_abs_gap"] < 0.02,
            "logprob_rel_l2_lt_0.05": one["logprob"]["relative_l2"] < 0.05,
            "sampled_grad_rel_l2_lt_0.05": one["canonical_sampled_grad"]["relative_l2"] < 0.05,
            "complete_norm_grad_rel_l2_lt_0.05": one["complete_norm_grad"]["relative_l2"] < 0.05,
        }
        if a.get("full_grad_dir") or b.get("full_grad_dir"):
            if one["complete_parameter_grad"] is None:
                raise AssertionError("Requested full-parameter gradients missing")
            witness_norm = one["complete_parameter_grad"]["full_norm"]
            actual_norm = float(a["global_grad_norm"])
            if abs(witness_norm - actual_norm) / max(actual_norm, 1e-30) > 2e-3:
                raise AssertionError(
                    f"CP1 exact full-gradient norm disagrees with recorded global norm: {name}"
                )
        outcomes[name] = one
    report = {
        "full_parameter_gradients_saved": all(
            results[mode].get("full_grad_dir") for mode in MODES
        ),
        "source_capture": full["source_capture"],
        "input_sha256": full["input_sha256"],
        "initial_weights_witness_sha256": full["initial_weights_witness_sha256"],
        "ppo_tokens": full["ppo_tokens"],
        "full_logical_tokens": full["logical_input_tokens"],
        "shared_unique_tree_tokens": shared["unique_tree_tokens"],
        "split_no_share_unique_tree_tokens": split["unique_tree_tokens"],
        "shared_prefix_reuse_ratio": shared["reuse_ratio"],
        "comparisons": outcomes,
        "caveat": (
            "Full is the same model's no-context square-CANN path, NOT native "
            "unmodified VERL/MindSpeed Engine. Short capture changes conditioning. "
            "Canonical sampled gradients are NOT complete all-parameter L2. "
            "complete_parameter_grad is exact ALL-parameter BF16 L2 only when saved."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = compare(args.directory)
    # Keep the original three-arm comparison.json immutable.
    output = args.directory / "comparison_split_no_share.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2))
    print("TPR_CP1_SHORT_SPLIT_COMPARISON", json.dumps({
        "ppo_tokens": result["ppo_tokens"],
        "reuse": result["shared_prefix_reuse_ratio"],
        "comparisons": {
            name: {
                "loss_gap": metrics["loss_abs_gap"],
                "logprob_rel_l2": metrics["logprob"]["relative_l2"],
                "sampled_grad_rel_l2": metrics["canonical_sampled_grad"]["relative_l2"],
                "complete_norm_rel_l2": metrics["complete_norm_grad"]["relative_l2"],
            "full_grad_rel_l2": (
                metrics["complete_parameter_grad"]["relative_l2"]
                if metrics["complete_parameter_grad"] is not None else None
            ),
            "full_grad_cos": (
                metrics["complete_parameter_grad"]["cosine"]
                if metrics["complete_parameter_grad"] is not None else None
            ),
                "grad_cos": metrics["canonical_sampled_grad"]["cosine"],
                "gates": metrics["provisional_gates"],
            }
            for name, metrics in result["comparisons"].items()
        },
    }, sort_keys=True), flush=True)
    print(f"TPR_CP1_SHORT_REPORT saved={output}", flush=True)


if __name__ == "__main__":
    main()
