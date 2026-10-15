"""Compare OLD vs NEW TPR (not Full) on the same real TP2CP4 Actor PPO capture."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from compare_real_actor_tp2cp4_offline_ppo import (
    _complete_gradient_metrics, _l2_pair, _logprobs, _weight_diff,
)

FIELDS = (
    "mode", "tp", "cp", "dp", "world", "source_capture", "source_rows",
    "source_keys", "data_sha256", "checkpoint", "objective", "optimizer",
    "logical_ppo_tokens", "forest_trees", "forest_segments",
)


def load(root: Path):
    items = []
    for rank in range(8):
        path = root / f"tpr_rank{rank}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item["rank"] != rank or item["mode"] != "tpr":
            raise AssertionError(f"Not a rank {rank} TPR capture: {path}")
        if (item["tp"], item["cp"], item["dp"], item["world"]) != (2, 4, 1, 8):
            raise AssertionError(f"Not TP2CP4DP1: {path}")
        items.append(item)
    return items


def compare(old_root: Path, new_root: Path, *, require_full_grads: bool,
            loss_abs_max: float, logprob_abs_max: float,
            grad_rel_max: float, param_rel_max: float):
    old, new = load(old_root), load(new_root)
    for rank, (a, b) in enumerate(zip(old, new, strict=True)):
        for key in FIELDS:
            if a[key] != b[key]:
                raise AssertionError(f"Rank {rank} invariant {key} differs: {a[key]!r} != {b[key]!r}")
        if (a["tp_rank"], a["cp_rank"]) != (b["tp_rank"], b["cp_rank"]):
            raise AssertionError(f"Rank {rank} TP/CP rank layout mismatch")
    for label, rows in (("old", old), ("new", new)):
        if len({x["data_sha256"] for x in rows}) != 1:
            raise AssertionError(f"{label} has inconsistent Actor input SHA")
        if max(x["loss"] for x in rows) - min(x["loss"] for x in rows) > 1e-4:
            raise AssertionError(f"{label} has CP rank loss inconsistency")
    initial = _weight_diff(old, new, "param_before")
    if initial != 0:
        raise AssertionError(f"Initial parameters differ: sampled relative L2={initial}")
    left, right = _logprobs(old), _logprobs(new)
    if not left or left.keys() != right.keys():
        raise AssertionError("Logical PPO logprob keys differ or missing")
    if len(left) != old[0]["logical_ppo_tokens"]:
        raise AssertionError("Incomplete logical PPO supervision captured")
    keys = sorted(left)
    a = torch.tensor([left[k] for k in keys], dtype=torch.float64)
    b = torch.tensor([right[k] for k in keys], dtype=torch.float64)
    lp_abs = float((a - b).abs().max())
    lp_rel = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-30))
    loss_abs = abs(float(old[0]["loss"]) - float(new[0]["loss"]))
    grad_sample = _l2_pair(old, new)
    full_grads = _complete_gradient_metrics(old, new)
    if require_full_grads and full_grads is None:
        raise AssertionError("Complete BF16 gradients missing; set TPR_OFFLINE_SAVE_FULL_GRADS=1")
    after = _weight_diff(old, new, "param_after")
    g_old, g_new = old[0]["global_bf16_grad_norm"], new[0]["global_bf16_grad_norm"]
    grad_norm_gap = abs(g_old - g_new) / max(abs(g_old), 1e-30)
    gates = {
        "loss": loss_abs <= loss_abs_max,
        "logprob": lp_abs <= logprob_abs_max,
        "sampled_gradient": grad_sample <= grad_rel_max,
        "complete_gradient": full_grads is None or full_grads["relative_l2"] <= grad_rel_max,
        "grad_norm": grad_norm_gap <= grad_rel_max,
        "post_sgd_parameter": after <= param_rel_max,
    }
    report = {
        "capture_sha256": old[0]["data_sha256"],
        "rows": list(old[0]["source_rows"]),
        "logical_ppo_tokens": len(left),
        "forest_trees": old[0]["forest_trees"],
        "forest_segments": old[0]["forest_segments"],
        "initial_weights_exact": True,
        "loss_old": float(old[0]["loss"]), "loss_new": float(new[0]["loss"]),
        "loss_abs_gap": loss_abs, "logprob_abs_max": lp_abs, "logprob_rel_l2": lp_rel,
        "sampled_grad_rel_l2": grad_sample, "complete_grad": full_grads,
        "grad_norm_rel_gap": grad_norm_gap, "post_sgd_sampled_weight_rel_l2": after,
        "time_old_s": max(x["time_forward_backward_finalize_s"] for x in old),
        "time_new_s": max(x["time_forward_backward_finalize_s"] for x in new),
        "peak_old_gib": max(x["peak_allocated_bytes"] for x in old) / 2**30,
        "peak_new_gib": max(x["peak_allocated_bytes"] for x in new) / 2**30,
        "gates": gates, "passed": all(gates.values()),
    }
    for key, value in (("loss", loss_abs), ("logprob", lp_abs),
                       ("grad_sample", grad_sample), ("post_sgd", after)):
        if not math.isfinite(value):
            raise AssertionError(f"Nonfinite {key}: {value}")
    print("TPR_OLD_NEW_TP2CP4_COMPARISON " + json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    if not report["passed"]:
        raise AssertionError(f"Old/new TPR PPO drift fails gate: {gates}")
    print("TPR_OLD_NEW_TP2CP4_PASS", flush=True)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("old", type=Path)
    cli.add_argument("new", type=Path)
    cli.add_argument("--require-full-grads", action="store_true")
    cli.add_argument("--loss-abs-max", type=float, default=1e-3)
    cli.add_argument("--logprob-abs-max", type=float, default=1e-2)
    cli.add_argument("--grad-rel-max", type=float, default=1e-2)
    cli.add_argument("--param-rel-max", type=float, default=1e-3)
    args = cli.parse_args()
    compare(args.old, args.new, require_full_grads=args.require_full_grads,
            loss_abs_max=args.loss_abs_max, logprob_abs_max=args.logprob_abs_max,
            grad_rel_max=args.grad_rel_max, param_rel_max=args.param_rel_max)
