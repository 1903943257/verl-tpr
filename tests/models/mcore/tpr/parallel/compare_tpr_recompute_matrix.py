"""CPU-only four-way matched PPO comparison: Full/TPR x recompute OFF/ON.

'Full' is unshared trajectories through the SAME TPR SegmentExecutor/Ring
backend; it is NOT the native Megatron Full training implementation.
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


def _load(folder: Path, *, mode: str, recompute: bool) -> list[dict]:
    files = [folder / f"{mode}_rank{rank}.pt" for rank in range(8)]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {mode} recompute={recompute}: {missing}")
    snapshots = [torch.load(file, map_location="cpu", weights_only=False) for file in files]
    for rank, item in enumerate(snapshots):
        if item["rank"] != rank or item["mode"] != mode:
            raise AssertionError(f"{folder}: wrong rank/mode at rank={rank}")
        if bool(item.get("recompute_enabled", False)) != recompute:
            raise AssertionError(f"{folder}: incorrect recompute state at rank={rank}")
        if recompute and (
            item.get("recompute_granularity") != "full"
            or item.get("recompute_method") != "uniform"
            or item.get("recompute_num_layers") != 1
            or item.get("recompute_installed_blocks", 0) < 1
        ):
            raise AssertionError(f"{folder}: full/uniform/1 adapter not installed at rank={rank}")
        if not recompute and item.get("recompute_installed_blocks", 0):
            raise AssertionError(f"{folder}: OFF run has installed checkpoint adapter at rank={rank}")
    return snapshots


def _summary(snapshots: list[dict]) -> dict:
    reference = snapshots[0]
    losses = [x["loss"] for x in snapshots]
    norms = [x["global_bf16_grad_norm"] for x in snapshots]
    if any(not math.isfinite(v) for v in losses + norms):
        raise AssertionError("nonfinite loss or gradient norm")
    if max(losses) - min(losses) > 1e-4 or max(norms) - min(norms) > 1e-4:
        raise AssertionError("loss/norm differs across eight ranks")
    return {
        "loss": reference["loss"],
        "global_grad_norm": reference["global_bf16_grad_norm"],
        "time_fwd_bwd_s": max(x["time_forward_backward_finalize_s"] for x in snapshots),
        "peak_allocated_gib": max(x["peak_allocated_bytes"] for x in snapshots) / 2**30,
        "peak_reserved_gib": max(x["peak_reserved_bytes"] for x in snapshots) / 2**30,
        "forest_trees": reference["forest_trees"],
        "forest_segments": reference["forest_segments"],
    }


def _pair(a: list[dict], b: list[dict], *, full_gradients: bool) -> dict:
    # Input/control invariants were already verified globally in compare().
    x = _summary(a)
    y = _summary(b)
    a_lp, b_lp = _logprobs(a), _logprobs(b)
    if a_lp.keys() != b_lp.keys():
        raise AssertionError("logical PPO token owners differ across pair")
    keys = sorted(a_lp)
    left = torch.tensor([a_lp[k] for k in keys], dtype=torch.float64)
    right = torch.tensor([b_lp[k] for k in keys], dtype=torch.float64)
    output = {
        "loss_gap": abs(x["loss"] - y["loss"]),
        "logprob_rel_l2": float(
            torch.linalg.vector_norm(left - right)
            / torch.linalg.vector_norm(left).clamp_min(1e-30)
        ),
        "sampled_grad_rel_l2": _l2_pair(a, b),
        "grad_norm_relative_gap": abs(x["global_grad_norm"] - y["global_grad_norm"])
        / max(x["global_grad_norm"], 1e-30),
        "time_ratio_B_over_A": y["time_fwd_bwd_s"] / max(x["time_fwd_bwd_s"], 1e-30),
        "allocated_saved_pct_A_to_B": 100 * (
            1 - y["peak_allocated_gib"] / max(x["peak_allocated_gib"], 1e-30)
        ),
        "reserved_saved_pct_A_to_B": 100 * (
            1 - y["peak_reserved_gib"] / max(x["peak_reserved_gib"], 1e-30)
        ),
    }
    if full_gradients:
        output["complete_grad"] = _complete_gradient_metrics(a, b)
        if output["complete_grad"] is None:
            raise AssertionError(
                "all four cases must save BF16 gradients with TPR_OFFLINE_SAVE_FULL_GRADS=1"
            )
    return output


def compare(
    full_off_dir: Path,
    full_on_dir: Path,
    tpr_off_dir: Path,
    tpr_on_dir: Path,
    *,
    full_gradients: bool = False,
) -> dict:
    cases = {
        "full_off": _load(full_off_dir, mode="full", recompute=False),
        "full_on": _load(full_on_dir, mode="full", recompute=True),
        "tpr_off": _load(tpr_off_dir, mode="tpr", recompute=False),
        "tpr_on": _load(tpr_on_dir, mode="tpr", recompute=True),
    }
    baseline = cases["full_off"]
    invariants = (
        "rank", "tp_rank", "cp_rank", "source_capture", "source_rows",
        "source_keys", "data_sha256", "checkpoint", "objective", "optimizer",
        "logical_ppo_tokens", "tp", "cp", "dp",
    )
    for name, snapshots in cases.items():
        for rank, (a, b) in enumerate(zip(baseline, snapshots, strict=True)):
            for field in invariants:
                if a[field] != b[field]:
                    raise AssertionError(f"{name} rank {rank}: differs on {field}")
        if _weight_diff(baseline, snapshots, "param_before") != 0:
            raise AssertionError(f"{name}: initial sampled model weights do not match")
    for mode in ("full", "tpr"):
        before, after = cases[f"{mode}_off"], cases[f"{mode}_on"]
        for rank, (a, b) in enumerate(zip(before, after, strict=True)):
            for field in ("forest_trees", "forest_segments", "ring_attention_impl"):
                if a[field] != b[field]:
                    raise AssertionError(f"{mode} rank {rank}: differs on {field}")

    reports = {key: _summary(value) for key, value in cases.items()}
    comparisons = {
        "full_recompute_off_to_on": _pair(
            cases["full_off"], cases["full_on"], full_gradients=full_gradients
        ),
        "tpr_recompute_off_to_on": _pair(
            cases["tpr_off"], cases["tpr_on"], full_gradients=full_gradients
        ),
        "full_vs_tpr_recompute_on": _pair(
            cases["full_on"], cases["tpr_on"], full_gradients=full_gradients
        ),
        "full_vs_tpr_recompute_off": _pair(
            cases["full_off"], cases["tpr_off"], full_gradients=full_gradients
        ),
    }
    print("TPR_RECOMPUTE_MATRIX_CASES", reports, flush=True)
    print("TPR_RECOMPUTE_MATRIX_COMPARISONS", comparisons, flush=True)
    print(
        "TPR_RECOMPUTE_MATRIX_SCOPE full=unshared_SegmentExecutor_Ring "
        "tpr=shared_SegmentExecutor_Ring native_Megatron_Full=False",
        flush=True,
    )
    return {"cases": reports, "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("full_off", type=Path)
    parser.add_argument("full_on", type=Path)
    parser.add_argument("tpr_off", type=Path)
    parser.add_argument("tpr_on", type=Path)
    parser.add_argument("--full-gradients", action="store_true")
    args = parser.parse_args()
    compare(
        args.full_off, args.full_on, args.tpr_off, args.tpr_on,
        full_gradients=args.full_gradients,
    )
