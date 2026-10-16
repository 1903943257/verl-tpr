"""Compare eight-rank recorded actor TPR: same tree with recompute OFF vs ON.

The comparison is CPU-only, uses the existing real-PPO snapshot format, and
checks the original minibatch and exact initial parameter witnesses before
reporting BF16 gradient differences. Complete gradient L2 requires that BOTH
replays were run with TPR_OFFLINE_SAVE_FULL_GRADS=1.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from compare_real_actor_tp2cp4_offline_ppo import (
    _complete_gradient_metrics,
    _l2_pair,
    _logprobs,
    _weight_diff,
)


def compare(off_dir: Path, on_dir: Path, *, require_full_grads: bool = False) -> dict:
    off, on = [], []
    for rank in range(8):
        a, b = off_dir / f"tpr_rank{rank}.pt", on_dir / f"tpr_rank{rank}.pt"
        if not a.is_file() or not b.is_file():
            raise FileNotFoundError(f"missing OFF/ON rank {rank}: {a}, {b}")
        off.append(torch.load(a, map_location="cpu", weights_only=False))
        on.append(torch.load(b, map_location="cpu", weights_only=False))

    invariants = (
        "rank", "tp_rank", "cp_rank", "mode", "source_capture",
        "source_rows", "source_keys", "data_sha256", "checkpoint",
        "objective", "optimizer", "logical_ppo_tokens", "tp", "cp", "dp",
        "forest_trees", "forest_segments",
    )
    for rank, (a, b) in enumerate(zip(off, on, strict=True)):
        if a["rank"] != rank or b["rank"] != rank or a["mode"] != "tpr":
            raise AssertionError(f"rank {rank}: expected matching TPR snapshots")
        for field in invariants:
            if a[field] != b[field]:
                raise AssertionError(f"rank {rank}: OFF/ON input mismatch: {field}")
        if a.get("recompute_enabled", False) or b.get("recompute_enabled") is not True:
            raise AssertionError(f"rank {rank}: OFF/ON recompute metadata invalid")
        if b.get("recompute_granularity") != "full" or b.get("recompute_method") != "uniform":
            raise AssertionError(f"rank {rank}: ON must be full/uniform")
        if b.get("recompute_num_layers") != 1 or b.get("recompute_installed_blocks", 0) < 1:
            raise AssertionError(f"rank {rank}: ON adapter not installed")

    if len({x["data_sha256"] for x in off + on}) != 1:
        raise AssertionError("different source PPO inputs across ranks or modes")
    initial_rel = _weight_diff(off, on, "param_before")
    if initial_rel != 0:
        raise AssertionError(f"initial weights differ: sampled rel L2={initial_rel}")

    left_lp, right_lp = _logprobs(off), _logprobs(on)
    if left_lp.keys() != right_lp.keys():
        raise AssertionError("logical token logprob identities differ")
    if len(left_lp) != off[0]["logical_ppo_tokens"]:
        raise AssertionError("missing logical PPO logprobs")
    keys = sorted(left_lp)
    a = torch.tensor([left_lp[k] for k in keys], dtype=torch.float64)
    b = torch.tensor([right_lp[k] for k in keys], dtype=torch.float64)
    lp_rel = float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-30))
    lp_max = float((a-b).abs().max())

    for mode, snapshots in (("off", off), ("on", on)):
        losses = [x["loss"] for x in snapshots]
        norms = [x["global_bf16_grad_norm"] for x in snapshots]
        if any(not math.isfinite(v) for v in losses + norms):
            raise AssertionError(f"{mode}: nonfinite loss or grad norm")
        if max(losses)-min(losses) > 1e-4 or max(norms)-min(norms) > 1e-4:
            raise AssertionError(f"{mode}: rank results inconsistent")

    sampled_grad_rel = _l2_pair(off, on)
    complete = _complete_gradient_metrics(off, on)
    if require_full_grads and complete is None:
        raise AssertionError(
            "full BF16 gradients not saved in both runs; rerun both with "
            "TPR_OFFLINE_SAVE_FULL_GRADS=1 to measure exact relative L2"
        )
    before_step_rel = _weight_diff(off, on, "param_after")
    time_off = max(x["time_forward_backward_finalize_s"] for x in off)
    time_on = max(x["time_forward_backward_finalize_s"] for x in on)
    alloc_off = max(x["peak_allocated_bytes"] for x in off) / 2**30
    alloc_on = max(x["peak_allocated_bytes"] for x in on) / 2**30
    reserved_off = max(x["peak_reserved_bytes"] for x in off) / 2**30
    reserved_on = max(x["peak_reserved_bytes"] for x in on) / 2**30

    result = {
        "rows": off[0]["source_rows"],
        "ppo_tokens": len(keys),
        "initial_weight_sample_rel_l2": initial_rel,
        "loss_off": off[0]["loss"],
        "loss_on": on[0]["loss"],
        "loss_gap": abs(off[0]["loss"] - on[0]["loss"]),
        "logprob_rel_l2": lp_rel,
        "logprob_max_abs": lp_max,
        "global_grad_norm_off": off[0]["global_bf16_grad_norm"],
        "global_grad_norm_on": on[0]["global_bf16_grad_norm"],
        "sampled_grad_rel_l2": sampled_grad_rel,
        "complete_grad": complete,
        "post_sgd_weight_sample_rel_l2": before_step_rel,
        "fwd_bwd_s_off": time_off,
        "fwd_bwd_s_on": time_on,
        "on_vs_off_runtime_ratio": time_on / max(time_off, 1e-30),
        "peak_alloc_gib_off": alloc_off,
        "peak_alloc_gib_on": alloc_on,
        "peak_reserved_gib_off": reserved_off,
        "peak_reserved_gib_on": reserved_on,
    }
    print(f"TPR_RECOMPUTE_OFF_ON_COMPARISON {result}", flush=True)
    if complete is None:
        print(
            "TPR_RECOMPUTE_GRAD_WARNING only sampled parameter gradients "
            "are available; this is NOT a complete gradient parity proof. "
            "Use TPR_OFFLINE_SAVE_FULL_GRADS=1 for both runs.",
            flush=True,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("off_dir", type=Path)
    parser.add_argument("on_dir", type=Path)
    parser.add_argument("--require-full-grads", action="store_true")
    args = parser.parse_args()
    compare(args.off_dir, args.on_dir, require_full_grads=args.require_full_grads)
