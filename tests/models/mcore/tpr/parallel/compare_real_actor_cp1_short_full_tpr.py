"""Compare three CP1 Actor PPO arms in canonical TP1 parameter coordinates.

Full means whole-sequence GPT with NO TPR context, using the same CP1
square-CANN oracle core as the TPR-capable model. The derivative capture
changes original causal conditioning: numerical parity only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODES = ("full", "one_node", "shared")


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
        outcomes[name] = one
    report = {
        "source_capture": full["source_capture"],
        "input_sha256": full["input_sha256"],
        "initial_weights_witness_sha256": full["initial_weights_witness_sha256"],
        "ppo_tokens": full["ppo_tokens"],
        "full_logical_tokens": full["logical_input_tokens"],
        "shared_unique_tree_tokens": shared["unique_tree_tokens"],
        "shared_prefix_reuse_ratio": shared["reuse_ratio"],
        "comparisons": outcomes,
        "caveat": (
            "Full is the same model's no-context square-CANN path, NOT native "
            "unmodified VERL/MindSpeed Engine. Short capture changes conditioning. "
            "Canonical sampled gradients are NOT complete all-parameter L2."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = compare(args.directory)
    output = args.directory / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2))
    print("TPR_CP1_SHORT_COMPARISON", json.dumps({
        "ppo_tokens": result["ppo_tokens"],
        "reuse": result["shared_prefix_reuse_ratio"],
        "comparisons": {
            name: {
                "loss_gap": metrics["loss_abs_gap"],
                "logprob_rel_l2": metrics["logprob"]["relative_l2"],
                "sampled_grad_rel_l2": metrics["canonical_sampled_grad"]["relative_l2"],
                "complete_norm_rel_l2": metrics["complete_norm_grad"]["relative_l2"],
                "grad_cos": metrics["canonical_sampled_grad"]["cosine"],
                "gates": metrics["provisional_gates"],
            }
            for name, metrics in result["comparisons"].items()
        },
    }, sort_keys=True), flush=True)
    print(f"TPR_CP1_SHORT_REPORT saved={output}", flush=True)


if __name__ == "__main__":
    main()
