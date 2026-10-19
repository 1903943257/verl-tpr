"""Read-only CPU audit of per-CP parameter-gradient replicas in saved PPO.

Unlike all-reduced global Grad Norm, per-parameter witness values can expose
a CP replica synchronization failure. This test does not load 4B models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .compare_real_actor_tp2cp4_multistep import _rank_replica_audit


def audit(root: Path, steps: tuple[int, ...]) -> dict:
    report = {
        "scope": "SAMPLED_PER_CP_PARAMETER_GRADIENT_REPLICA_AUDIT",
        "note": (
            "Each TP-shard compared across CP0/1/2/3 using saved sampled "
            "gradient witnesses. This is not a full-tensor replica audit; "
            "global_grad_norm rank equality alone is not a replica gate."
        ),
        "steps": [],
    }
    for step in steps:
        modes = {}
        for mode in ("full", "tpr"):
            snapshots = [
                torch.load(root / f"{mode}_step{step}_rank{rank}.pt",
                           map_location="cpu", weights_only=False)
                for rank in range(8)
            ]
            result = _rank_replica_audit(snapshots, mode, step)
            modes[mode] = {
                "sampled_cp_gradient_replica": result["sampled_cp_gradient_replica"],
                "global_grad_norm_rank_spread_is_not_replica_parity": True,
                "loss_global_spread": result["loss_global_spread"],
                "tp_logprob_max_abs": result["tp_logprob_max_abs"],
            }
        report["steps"].append({"step": step, "modes": modes})
    target = root / "saved_cp_gradient_replica_audit.json"
    if target.exists():
        raise FileExistsError(
            f"Original saved CP gradient audit exists; refusing overwrite: {target}"
        )
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"TPR_SAVED_CP_GRAD_REPLICA_AUDIT_REPORT={target}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", default="1,2,5")
    args = parser.parse_args()
    audit(args.root, tuple(int(x) for x in args.steps.split(",") if x))
