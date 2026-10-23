"""Four-way controlled 2-trajectory PPO/physical-MB comparison.

All paths must contain results from the same capture/checkpoint. TPR K and
Full MB refer to distinct *physical* groupings of the same 2 logical rows.
This is a model F/B benchmark with a square-CANN Full reference; it is NOT
the production VERL/Megatron micro-batch scheduler or a full optimizer step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("full_mb1", "full_mb2", "tpr_k1", "tpr_k2")
MODES = ("full", "full_batch2", "shared", "shared")


def _load(path: Path):
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")
    return data


def _format(value, decimals=3):
    return "—" if value is None else f"{value:.{decimals}f}"


def summarize(root: Path, *, memory_margin_gib: float = 0.5) -> str:
    data = {arm: _load(root / arm / "result.json") for arm in ARMS}
    lines = [
        "=" * 83,
        "TPR CP1 FAIR PHYSICAL-BATCH COMPARISON | 2 logical trajectories",
        f"root={root}",
        "=" * 83,
        "Full MB=2 means one physical [2,S] square-CANN GPT forward; TPR K=2",
        "means 1 shared Prefix plus 2 suffix Leafs grouped, NOT 2 more trajectories.",
        "Scope: model F/B, controlled captured PPO; no native VERL microbatch",
        "scheduler, rollout, production optimizer step, or CP4.",
        "",
        f"{'Metric':<31} {'Full MB1':>12} {'Full MB2':>12} {'TPR K1':>12} {'TPR K2':>12}",
        "-" * 83,
    ]
    expected_count = 2
    for arm, mode in zip(ARMS, MODES, strict=True):
        item = data[arm]
        if item is None:
            continue
        if item.get("mode") != mode:
            raise AssertionError(f"{arm} expected mode {mode}, got {item.get('mode')}")
        if item.get("logical_trajectories") != expected_count:
            raise AssertionError(f"{arm} did not use 2 logical trajectories")
        if int(item.get("bench_repeats", 0)) < 1 or not item.get("bench_times_s"):
            raise AssertionError(f"{arm} has no measured iterations")
        if len(item["bench_times_s"]) != item["bench_repeats"]:
            raise AssertionError(f"{arm} lost measured iterations")
        if item["fwd_bwd_seconds"] <= 0:
            raise AssertionError(f"{arm} has invalid measured median time")
        if arm == "full_mb1" and item.get("physical_full_micro_batch") != 1:
            raise AssertionError("Full MB1 reference is not one row per model invocation")
        if arm == "full_mb2" and item.get("physical_full_micro_batch") != 2:
            raise AssertionError("Full MB2 run never executed a physical batch of 2")
        if arm == "tpr_k2" and (
            item.get("leaf_batch_configured_k") != 2
            or (item.get("leaf_batch_stats") or {}).get("grouped_leaf_segments") != 2
        ):
            raise AssertionError("TPR K2 did not group 2 sibling leaves")
        if arm == "tpr_k1" and (
            item.get("leaf_batch_configured_k") != 1
            or (item.get("leaf_batch_stats") or {}).get("grouped_leaf_calls") != 0
        ):
            raise AssertionError("TPR K1 unexpectedly grouped")
    present = [item for item in data.values() if item is not None]
    if present:
        witness_keys = (
            "input_sha256", "initial_weights_witness_sha256", "ppo_tokens",
            "source_capture", "checkpoint", "original_keys",
        )
        for key in witness_keys:
            values = {json.dumps(item.get(key), sort_keys=True) for item in present}
            if len(values) != 1:
                raise AssertionError(f"Not an identical two-trajectory comparison: {key}")
        lines.append(
            "Common input SHA: " + present[0]["input_sha256"][:18]
            + f" | PPO tokens: {present[0]['ppo_tokens']}"
        )

    def value(item, key):
        return None if item is None else item.get(key)

    rows = (
        ("Logical trajectories", "logical_trajectories", 0),
        ("Logical input tokens", "logical_input_tokens", 0),
        ("Unique tree tokens", "unique_tree_tokens", 0),
        ("Model forward calls", "physical_model_forward_calls", 0),
        ("Supervised PPO tokens", "ppo_tokens", 0),
        ("Measured iterations", "bench_repeats", 0),
        ("Median F/B (seconds)", "fwd_bwd_seconds", 3),
        ("Peak allocated MAX (GiB)", "peak_allocated_gib", 3),
        ("Peak reserved MAX (GiB)", "peak_reserved_gib", 3),
        ("PPO Loss", "loss", 6),
        ("Global BF16 grad norm", "global_grad_norm", 4),
    )
    for label, key, decimals in rows:
        values = [value(data[arm], key) for arm in ARMS]
        lines.append(
            f"{label:<31} " + " ".join(f"{_format(v, decimals):>12}" for v in values)
        )
    lines.extend(["-" * 83, ""])
    report = {
        "scope": "controlled_CP1_physical_microbatch_no_native_scheduler",
        "benchmark_acceptance": "DIAGNOSTIC_ONLY_UNTIL_GRADIENT_EQUIVALENCE",
        "logical_trajectories_per_arm": expected_count,
        "memory_margin_gib": memory_margin_gib,
        "arms": {
            arm: {
                key: value(item, key) for key in (
                    "fwd_bwd_seconds", "peak_allocated_gib", "peak_reserved_gib",
                    "bench_warmups", "bench_repeats", "bench_times_s",
                    "loss", "global_grad_norm", "logical_trajectories",
                    "physical_full_micro_batch", "physical_model_forward_calls",
                    "logical_input_tokens", "unique_tree_tokens", "ppo_tokens", "leaf_batch_stats",
                )
            }
            for arm, item in data.items()
        },
    }
    full = data["full_mb2"]
    tpr = data["tpr_k2"]
    if full is not None and tpr is not None:
        full_time = float(full["fwd_bwd_seconds"])
        tpr_time = float(tpr["fwd_bwd_seconds"])
        ratio = full_time / tpr_time
        alloc_margin = float(full["peak_allocated_gib"]) - float(tpr["peak_allocated_gib"])
        reserved_margin = float(full["peak_reserved_gib"]) - float(tpr["peak_reserved_gib"])
        margin_pass = alloc_margin >= memory_margin_gib and reserved_margin >= memory_margin_gib
        lines.extend([
            f"Full physical MB2 / TPR K2 MEDIAN F/B speedup: {ratio:.4f}x",
            f"TPR K2 vs Full MB2 allocated headroom: {alloc_margin:+.3f} GiB",
            f"TPR K2 vs Full MB2 reserved headroom:  {reserved_margin:+.3f} GiB",
            f"Required headroom: {memory_margin_gib:.3f} GiB on BOTH metrics",
            f"Memory margin gate: {'PASS' if margin_pass else 'FAIL'}",
        ])
        report["full_mb2_vs_tpr_k2"] = {
            "median_speedup": ratio,
            "allocated_headroom_gib": alloc_margin,
            "reserved_headroom_gib": reserved_margin,
            "memory_margin_pass": margin_pass,
        }
        # Numerical diagnostic; different physical batch and Prefix execution
        # may alter BF16 rounding, so speedup alone is not a correctness gate.
        from .compare_real_actor_cp1_short_full_tpr import _metrics

        def paired_logprobs(item):
            return {(int(r), int(i)): float(v) for r, i, v in item["logprobs"]}
        fa, tb = paired_logprobs(full), paired_logprobs(tpr)
        if fa.keys() != tb.keys() or len(fa) != full["ppo_tokens"]:
            raise AssertionError("Full MB2/TPR K2 logical PPO tokens do not align")
        ordered = sorted(fa)
        lp = _metrics([fa[k] for k in ordered], [tb[k] for k in ordered])
        wg0, wg1 = full["gradient_witness"], tpr["gradient_witness"]
        if wg0["names"] != wg1["names"] or wg0["canonical_shapes"] != wg1["canonical_shapes"]:
            raise AssertionError("Gradient witness coordinate mismatch")
        grad = _metrics(wg0["values"], wg1["values"])
        lines.extend([
            f"Full MB2/TPR K2 PPO loss absolute gap: {abs(float(full['loss']) - float(tpr['loss'])):.7f}",
            f"Full MB2/TPR K2 logprob relative L2: {lp['relative_l2']:.6f}",
            f"Full MB2/TPR K2 sampled grad relative L2: {grad['relative_l2']:.6f}",
        ])
        report["full_mb2_vs_tpr_k2"].update({
            "loss_abs_gap": abs(float(full["loss"]) - float(tpr["loss"])),
            "logprob_relative_l2": lp["relative_l2"],
            "sampled_gradient_relative_l2": grad["relative_l2"],
        })
    else:
        missing = [arm for arm in ("full_mb2", "tpr_k2") if data[arm] is None]
        lines.append(f"NO Full MB2/TPR K2 result: missing {missing}. Check current failure log.")

    for arm in ARMS:
        if data[arm] is None:
            lines.append(f"MISSING {arm}: {root / arm / 'run.log'}")
    lines.extend([
        "Caveat: F/B medians are NOT end-to-end Actor PPO update or rollout timings.",
        "Even with a memory-margin PASS at K=2, K=4 or larger forests may OOM.",
        f"Summary file: {root / 'SUMMARY.txt'}",
        "=" * 83,
    ])
    root.mkdir(parents=True, exist_ok=True)
    (root / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--memory-margin-gib", type=float, default=0.5)
    args = parser.parse_args()
    if args.memory_margin_gib < 0:
        raise ValueError("memory-margin-gib must be nonnegative")
    print(summarize(args.root, memory_margin_gib=args.memory_margin_gib), flush=True)


if __name__ == "__main__":
    main()
