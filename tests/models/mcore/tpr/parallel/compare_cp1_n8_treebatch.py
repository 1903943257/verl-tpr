"""Direct K=1 versus K=2 TreeBatch correctness gate for the N=8 CP1 capture.

This compares two real TPR runs over identical model weights and trajectories.
It deliberately does NOT compare either run against Full: sequence-splitting
differences are tracked by the separate correctness-audit branch.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _metric(a, b):
    if len(a) != len(b):
        raise AssertionError(f"Metric vector lengths differ: {len(a)} vs {len(b)}")
    aa = torch.as_tensor(a, dtype=torch.float64)
    bb = torch.as_tensor(b, dtype=torch.float64)
    diff = aa - bb
    norm_a = torch.linalg.vector_norm(aa).item()
    norm_b = torch.linalg.vector_norm(bb).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    denom = max(norm_a, 1e-30)
    cosine_denom = max(norm_a * norm_b, 1e-30)
    return {
        "relative_l2": diff_norm / denom,
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "cosine": float(torch.dot(aa, bb).item() / cosine_denom),
        "norm_reference": norm_a,
        "norm_test": norm_b,
    }


def _read_logprobs(item, arm):
    values = {}
    for row, offset, value in item["logprobs"]:
        key = (int(row), int(offset))
        if key in values:
            raise AssertionError(f"{arm} contains duplicate logprob coordinate {key}")
        values[key] = float(value)
    return values


def _compare_complete_grads(k1_dir: Path, k2_dir: Path):
    manifest1 = torch.load(k1_dir / "manifest.pt", map_location="cpu", weights_only=False)
    manifest2 = torch.load(k2_dir / "manifest.pt", map_location="cpu", weights_only=False)
    by_name1 = {item["name"]: item for item in manifest1}
    by_name2 = {item["name"]: item for item in manifest2}
    if by_name1.keys() != by_name2.keys():
        missing1 = sorted(by_name2.keys() - by_name1.keys())
        missing2 = sorted(by_name1.keys() - by_name2.keys())
        raise AssertionError(
            f"Full gradient parameter names differ: missing K1={missing1[:8]}, "
            f"missing K2={missing2[:8]}"
        )

    total_diff_sq = 0.0
    total_k1_sq = 0.0
    total_k2_sq = 0.0
    total_dot = 0.0
    top = []
    for name in sorted(by_name1):
        left, right = by_name1[name], by_name2[name]
        if tuple(left["shape"]) != tuple(right["shape"]):
            raise AssertionError(
                f"Full gradient shape mismatch for {name}: "
                f"{left['shape']} vs {right['shape']}"
            )
        g1 = torch.load(k1_dir / left["filename"], map_location="cpu", weights_only=True)
        g2 = torch.load(k2_dir / right["filename"], map_location="cpu", weights_only=True)
        if tuple(g1.shape) != tuple(g2.shape):
            raise AssertionError(f"Gradient tensor shape mismatch for {name}")
        # Keep the source dtype on disk; promote only the current parameter pair.
        a = g1.detach().to(torch.float32)
        b = g2.detach().to(torch.float32)
        d = a - b
        diff_sq = float(torch.sum(d * d, dtype=torch.float64).item())
        a_sq = float(torch.sum(a * a, dtype=torch.float64).item())
        b_sq = float(torch.sum(b * b, dtype=torch.float64).item())
        dot = float(torch.sum(a * b, dtype=torch.float64).item())
        total_diff_sq += diff_sq
        total_k1_sq += a_sq
        total_k2_sq += b_sq
        total_dot += dot
        top.append({
            "name": name,
            "absolute_l2": math.sqrt(diff_sq),
            "relative_l2_to_k1": math.sqrt(diff_sq) / max(math.sqrt(a_sq), 1e-30),
            "k1_norm": math.sqrt(a_sq),
            "k2_norm": math.sqrt(b_sq),
        })
        del g1, g2, a, b, d

    diff_norm = math.sqrt(total_diff_sq)
    k1_norm = math.sqrt(total_k1_sq)
    k2_norm = math.sqrt(total_k2_sq)
    return {
        "parameter_count": len(by_name1),
        "relative_l2": diff_norm / max(k1_norm, 1e-30),
        "cosine": total_dot / max(k1_norm * k2_norm, 1e-30),
        "k1_norm": k1_norm,
        "k2_norm": k2_norm,
        "top_absolute_l2": sorted(top, key=lambda x: x["absolute_l2"], reverse=True)[:15],
    }


def compare(
    root: Path,
    *,
    max_loss_gap: float = 0.02,
    max_logprob_rel_l2: float = 0.05,
    max_sampled_grad_rel_l2: float = 0.05,
    max_full_grad_rel_l2: float = 0.05,
    min_full_grad_cosine: float = 0.995,
):
    p1 = root / "tpr_k1" / "result.json"
    p2 = root / "tpr_k2" / "result.json"
    if not p1.is_file() or not p2.is_file():
        raise FileNotFoundError(f"Need both result files: {p1}, {p2}")
    k1, k2 = json.loads(p1.read_text()), json.loads(p2.read_text())

    for key in (
        "input_sha256", "source_capture", "checkpoint", "original_keys",
        "initial_weights_witness_sha256", "ppo_tokens", "logical_trajectories",
        "logical_input_tokens", "input_lengths", "forest_trees", "forest_segments",
        "unique_tree_tokens",
    ):
        if k1.get(key) != k2.get(key):
            raise AssertionError(f"K1/K2 input or topology mismatch for {key}: "
                                 f"{k1.get(key)!r} vs {k2.get(key)!r}")
    if k1["mode"] != "tpr" or k1["configured_k"] != 1:
        raise AssertionError("tpr_k1 result is not a genuine TPR K=1 run")
    if k2["mode"] != "tpr" or k2["configured_k"] != 2:
        raise AssertionError("tpr_k2 result is not a genuine TPR K=2 run")

    stats1, stats2 = k1.get("leaf_batch_stats"), k2.get("leaf_batch_stats")
    if not isinstance(stats1, dict) or not isinstance(stats2, dict):
        raise AssertionError("Missing real TPR leaf-batching counters")
    if stats1.get("grouped_leaf_calls") != 0:
        raise AssertionError(f"K1 unexpectedly grouped leaves: {stats1}")
    if stats2.get("grouped_leaf_calls", 0) < 1 or stats2.get("grouped_leaf_segments", 0) < 2:
        raise AssertionError(f"K2 did not execute a real grouped-leaf call: {stats2}")
    if k2["physical_model_forward_calls"] >= k1["physical_model_forward_calls"]:
        raise AssertionError(
            "K2 reduced neither model forward-call count nor execution work: "
            f"K1={k1['physical_model_forward_calls']} K2={k2['physical_model_forward_calls']}"
        )

    lp1, lp2 = _read_logprobs(k1, "K1"), _read_logprobs(k2, "K2")
    if lp1.keys() != lp2.keys() or len(lp1) != k1["ppo_tokens"]:
        raise AssertionError("K1/K2 supervised PPO token coordinates differ")
    coords = sorted(lp1)
    logprob = _metric([lp1[x] for x in coords], [lp2[x] for x in coords])
    sampled_grad = _metric(
        k1["gradient_witness"]["values"], k2["gradient_witness"]["values"]
    )
    loss_gap = abs(float(k1["loss"]) - float(k2["loss"]))

    raw_dirs = (k1.get("full_parameter_grad_dir"), k2.get("full_parameter_grad_dir"))
    if not all(raw_dirs):
        raise AssertionError(
            "Complete parameter gradients were not saved. Rerun both arms with "
            "TPR_CP1_N8_SAVE_FULL_GRADS=1."
        )
    dirs = tuple(Path(p).resolve() for p in raw_dirs)
    for directory in dirs:
        if not (directory / "manifest.pt").is_file():
            raise FileNotFoundError(f"Missing full-gradient manifest: {directory}")
    full_grad = _compare_complete_grads(*dirs)

    gates = {
        "loss_abs_gap": loss_gap <= max_loss_gap,
        "logprob_rel_l2": logprob["relative_l2"] <= max_logprob_rel_l2,
        "sampled_grad_rel_l2": sampled_grad["relative_l2"] <= max_sampled_grad_rel_l2,
        "full_grad_rel_l2": full_grad["relative_l2"] <= max_full_grad_rel_l2,
        "full_grad_cosine": full_grad["cosine"] >= min_full_grad_cosine,
        "actual_grouping": stats2["grouped_leaf_calls"] >= 1,
        "fewer_model_forward_calls": (
            k2["physical_model_forward_calls"] < k1["physical_model_forward_calls"]
        ),
    }
    report = {
        "root": str(root.resolve()),
        "input_sha256": k1["input_sha256"],
        "ppo_tokens": k1["ppo_tokens"],
        "logical_trajectories": k1["logical_trajectories"],
        "k1": {
            "forward_calls": k1["physical_model_forward_calls"],
            "leaf_batch_stats": stats1,
            "fwd_bwd_seconds": k1["fwd_bwd_seconds"],
            "peak_allocated_gib": k1["peak_allocated_gib"],
            "peak_reserved_gib": k1["peak_reserved_gib"],
        },
        "k2": {
            "forward_calls": k2["physical_model_forward_calls"],
            "leaf_batch_stats": stats2,
            "fwd_bwd_seconds": k2["fwd_bwd_seconds"],
            "peak_allocated_gib": k2["peak_allocated_gib"],
            "peak_reserved_gib": k2["peak_reserved_gib"],
        },
        "speedup_k1_over_k2": float(k1["fwd_bwd_seconds"]) / float(k2["fwd_bwd_seconds"]),
        "loss_abs_gap": loss_gap,
        "logprob": logprob,
        "sampled_gradient": sampled_grad,
        "full_parameter_gradient": full_grad,
        "thresholds": {
            "max_loss_gap": max_loss_gap,
            "max_logprob_rel_l2": max_logprob_rel_l2,
            "max_sampled_grad_rel_l2": max_sampled_grad_rel_l2,
            "max_full_grad_rel_l2": max_full_grad_rel_l2,
            "min_full_grad_cosine": min_full_grad_cosine,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    (root / "treebatch_k1_vs_k2.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "===== TPR TREEBATCH K1/K2 DIRECT PARITY =====",
        f"root={root.resolve()}",
        f"input_sha256={k1['input_sha256']}",
        f"trajectories={k1['logical_trajectories']} ppo_tokens={k1['ppo_tokens']}",
        f"forward_calls: K1={k1['physical_model_forward_calls']} K2={k2['physical_model_forward_calls']}",
        f"grouped_leaf_calls: K1={stats1['grouped_leaf_calls']} K2={stats2['grouped_leaf_calls']}",
        f"F/B median: K1={k1['fwd_bwd_seconds']:.4f}s K2={k2['fwd_bwd_seconds']:.4f}s "
        f"speedup={report['speedup_k1_over_k2']:.4f}x",
        f"peak allocated GiB: K1={k1['peak_allocated_gib']:.3f} K2={k2['peak_allocated_gib']:.3f}",
        f"loss_abs_gap={loss_gap:.8g}",
        f"logprob_rel_l2={logprob['relative_l2']:.8g}",
        f"sampled_grad_rel_l2={sampled_grad['relative_l2']:.8g}",
        f"FULL_GRAD_REL_L2={full_grad['relative_l2']:.8g}",
        f"FULL_GRAD_COSINE={full_grad['cosine']:.9f}",
        "Top full-gradient absolute differences:",
    ]
    for item in full_grad["top_absolute_l2"]:
        lines.append(
            f"  {item['absolute_l2']:.8g} rel={item['relative_l2_to_k1']:.8g} {item['name']}"
        )
    lines.append("GATES:")
    lines.extend(f"  {name}={'PASS' if ok else 'FAIL'}" for name, ok in gates.items())
    lines.append(f"TREEBATCH_PARITY={'PASS' if report['passed'] else 'FAIL'}")
    lines.append(f"report={root / 'treebatch_k1_vs_k2.json'}")
    print("\n".join(lines), flush=True)
    if not report["passed"]:
        failed = [name for name, ok in gates.items() if not ok]
        raise AssertionError(f"TreeBatch K1/K2 correctness gates failed: {failed}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-loss-gap", type=float, default=0.02)
    parser.add_argument("--max-logprob-rel-l2", type=float, default=0.05)
    parser.add_argument("--max-sampled-grad-rel-l2", type=float, default=0.05)
    parser.add_argument("--max-full-grad-rel-l2", type=float, default=0.05)
    parser.add_argument("--min-full-grad-cosine", type=float, default=0.995)
    args = parser.parse_args()
    compare(
        args.root,
        max_loss_gap=args.max_loss_gap,
        max_logprob_rel_l2=args.max_logprob_rel_l2,
        max_sampled_grad_rel_l2=args.max_sampled_grad_rel_l2,
        max_full_grad_rel_l2=args.max_full_grad_rel_l2,
        min_full_grad_cosine=args.min_full_grad_cosine,
    )


if __name__ == "__main__":
    main()
