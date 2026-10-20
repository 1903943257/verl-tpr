"""Compare two immutable TP2 CP1 N8 P8192/S1024 CPU-side performance JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(output_dir: Path, *, prefix: int = 8192, suffix: int = 1024) -> dict:
    if (prefix, suffix) not in ((8192, 1024), (4608, 4608)):
        raise ValueError(f"Unsupported CP1 workload P={prefix} S={suffix}")
    full = json.loads((output_dir / "full.json").read_text())
    tpr = json.loads((output_dir / "tpr.json").read_text())
    common = (
        "model", "checkpoint", "topology", "branches", "prefix", "suffix",
        "length", "source_capture", "source_input_sha256", "ppo_tokens",
        "condition_changed", "synthetic_shared_uid", "recompute",
        "optimizer_step", "warmups", "repeats",
    )
    for key in common:
        if full[key] != tpr[key]:
            raise AssertionError(f"Full/TPR workload mismatch: {key}")
    if full["baseline_kind"] != "CONTROLLED_WHOLE_SQUARE_CANN_NO_TPR_CONTEXT":
        raise AssertionError("Do not label this CP1 control Native Megatron Full")
    if tpr["baseline_kind"] != "PRODUCTION_TPR_SEGMENT_EXECUTOR_CP1":
        raise AssertionError("Missing TPR CP1 SegmentExecutor mode")
    if ((full["topology"]["tp"], full["topology"]["cp"], full["topology"]["dp"])
            != (2, 1, 1) or full["branches"] != 8
            or (full["prefix"], full["suffix"], full["length"]) != (prefix, suffix, prefix+suffix)):
        raise AssertionError("Unexpected TP2 CP1 8-way shape")
    if (full["forest_trees"], full["forest_segments"]) != (8, 8):
        raise AssertionError("Unshared control did not execute eight whole sequences")
    if tpr["forest_trees"] != 1 or tpr["forest_segments"] < 9:
        raise AssertionError("TPR did not share one root across eight leaves")
    shared_root_tokens = tpr["actual_shared_root_tokens"]
    if (not prefix - 127 <= shared_root_tokens < prefix + suffix
            or full["actual_shared_root_tokens"] != shared_root_tokens
            or full["unique_tree_tokens"] != tpr["unique_tree_tokens"]):
        raise AssertionError("Actual physical qwen_turn prefix differs across modes")
    if full["recompute"] or full["optimizer_step"]:
        raise AssertionError("Wrong timed phase")
    if not full["condition_changed"] or not full["synthetic_shared_uid"]:
        raise AssertionError("This capture must be declared performance-only")
    for mode, data in (("full", full), ("tpr", tpr)):
        if not data["runs"] or sum(not item["warmup"] for item in data["runs"]) != data["repeats"]:
            raise AssertionError(f"{mode}: missing measured iterations")
        if any(item["critical_fwd_bwd_finalize_s"] <= 0 or item["peak_allocated_gib"] <= 0
               for item in data["runs"]):
            raise AssertionError(f"{mode}: impossible timing or memory")
    f, t = full["median_fwd_bwd_s"], tpr["median_fwd_bwd_s"]
    if min(f, t) <= 0:
        raise AssertionError("Cannot compute speedup from nonpositive durations")
    report = {
        "experiment": f"QWEN3_4B_CP1_N8_P{prefix}_S{suffix}_PPO_FWD_BWD",
        "baseline": "CONTROLLED_SQUARE_CANN_NOT_NATIVE_MEGATRON",
        "full_median_fwd_bwd_s": f,
        "tpr_median_fwd_bwd_s": t,
        "speedup_full_over_tpr": f/t,
        "full_peak_allocated_gib": full["worst_peak_allocated_gib"],
        "tpr_peak_allocated_gib": tpr["worst_peak_allocated_gib"],
        "memory_saving_fraction": (
            1-tpr["worst_peak_allocated_gib"]/full["worst_peak_allocated_gib"]
        ),
        "ppo_tokens": full["ppo_tokens"],
        "nominal_prefix_tokens": prefix,
        "actual_shared_root_tokens": shared_root_tokens,
        "effective_root_to_suffix_ratio": shared_root_tokens / suffix,
        "unique_tree_tokens": tpr["unique_tree_tokens"],
        "source_input_sha256": full["source_input_sha256"],
        "notes": (
            "Performance-only derived old-PPO data with changed conditioning "
            "and synthetic cross-session UID; Recompute OFF, optimizer excluded. "
            "Full here is a controlled whole-sequence square-CANN reference, "
            "not Megatron native F/B schedule."
        ),
    }
    target = output_dir / "comparison.json"
    if target.exists():
        raise FileExistsError(f"Refuse overwrite of benchmark comparison: {target}")
    target.write_text(json.dumps(report, indent=2))
    print("TPR_CP1_N8_COMPARE "+json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--prefix", type=int, default=8192)
    parser.add_argument("--suffix", type=int, default=1024)
    args = parser.parse_args()
    compare(args.output_dir, prefix=args.prefix, suffix=args.suffix)


if __name__ == "__main__":
    main()
