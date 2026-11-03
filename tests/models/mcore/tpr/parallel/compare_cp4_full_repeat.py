#!/usr/bin/env python3
"""Compare two independent Native Full replays of the same fixed PPO capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compare_real_actor_tp2cp4_multistep import (
    _lp_compare,
    _master_update,
    _rank_replica_audit,
)
from compare_real_actor_tp2cp4_offline_ppo import _l2_pair, _weight_diff


def _load_arm(directory: Path, step: int) -> list[dict]:
    paths = [directory / f"full_step{step}_rank{rank}.pt" for rank in range(8)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Full repeat snapshots missing: {missing}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def _check_same_protocol(left: list[dict], right: list[dict], step: int) -> None:
    fields = (
        "step", "steps", "source_capture", "source_rows", "source_keys",
        "data_sha256", "checkpoint", "optimizer", "lr", "clip_norm",
        "tokens", "run_kind", "rank", "tp_rank", "cp_rank",
        "source_online_global_step", "source_capture_index",
    )
    for a, b in zip(left, right, strict=True):
        for field in fields:
            if a.get(field) != b.get(field):
                raise AssertionError(
                    f"Full repeat step={step} protocol mismatch in {field}: "
                    f"{a.get(field)!r} != {b.get(field)!r}"
                )
        if a.get("execution_backend") != "native_megatron_schedule":
            raise AssertionError("Both arms must use native Megatron Full")
        if b.get("execution_backend") != "native_megatron_schedule":
            raise AssertionError("Both arms must use native Megatron Full")
        if a.get("native_loss_scaling") != "megatron_cp_once_global_token_mean_v2":
            raise AssertionError("Full-A has obsolete/unknown CP loss scaling")
        if b.get("native_loss_scaling") != "megatron_cp_once_global_token_mean_v2":
            raise AssertionError("Full-B has obsolete/unknown CP loss scaling")
        if a.get("native_recompute_layers") != 0 or b.get("native_recompute_layers") != 0:
            raise AssertionError("Full repeat requires native recompute OFF")
    if left[0]["steps"] != right[0]["steps"]:
        raise AssertionError("Full repeat total step counts differ")


def compare(left_dir: Path, right_dir: Path, steps: int, output: Path) -> dict:
    milestones = tuple(sorted({n for n in (1, 2, 5, 10, 20, 30, 40, 50) if n <= steps} | {steps}))
    rows = []
    for step in milestones:
        a = _load_arm(left_dir, step)
        b = _load_arm(right_dir, step)
        _check_same_protocol(a, b, step)

        audit_a = _rank_replica_audit(a, "full_a", step)
        audit_b = _rank_replica_audit(b, "full_b", step)
        lp = _lp_compare(a, b)
        grad = _l2_pair(a, b)
        param_after = _weight_diff(a, b, "param_after")
        param_initial = _weight_diff(a, b, "param_initial")
        master = _master_update(a, b)
        loss_gap = float(b[0]["loss"]) - float(a[0]["loss"])
        norm_gap = abs(
            float(b[0]["global_grad_norm_preclip"]) -
            float(a[0]["global_grad_norm_preclip"])
        ) / max(abs(float(a[0]["global_grad_norm_preclip"])), 1e-30)

        rows.append({
            "step": step,
            "full_a_loss": float(a[0]["loss"]),
            "full_b_loss": float(b[0]["loss"]),
            "loss_gap_b_minus_a": loss_gap,
            "loss_abs_gap": abs(loss_gap),
            "full_a_grad_norm": float(a[0]["global_grad_norm_preclip"]),
            "full_b_grad_norm": float(b[0]["global_grad_norm_preclip"]),
            "grad_norm_relative_gap": norm_gap,
            "logprob": lp,
            "sampled_grad_relative_l2": grad,
            "param_initial_rel_l2": param_initial,
            "param_after_rel_l2": param_after,
            "master_update": master,
            "full_a_replica_audit": audit_a,
            "full_b_replica_audit": audit_b,
            "replica_numerical_gate_ok": all(
                audit[key]
                for audit in (audit_a, audit_b)
                for key in ("loss_gate_ok", "norm_gate_ok", "tp_logprob_gate_ok")
            ),
        })

    result = {
        "status": "FULL_REPEAT_NOISE_BASELINE",
        "protocol": "same_synthetic_capture_same_checkpoint_same_seed_native_full_twice",
        "steps_completed": steps,
        "milestones": rows,
        "interpretation": (
            "Full-A and Full-B are independent Native Megatron Full jobs initialized "
            "from the same checkpoint and fixed synthetic Actor capture. Their "
            "differences estimate run-to-run numerical variation for this setup; "
            "they are not an algorithmic Full-vs-TPR difference."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    header = (
        "step\tloss_abs_gap\tlp_rel_l2\tlp_mae\tgradnorm_relative_gap\t"
        "sampled_grad_rel_l2\tparam_after_rel_l2\tmaster_update_rel_l2\t"
        "master_update_cosine\treplica_ok"
    )
    lines = [header]
    for row in rows:
        lines.append("\t".join(map(str, (
            row["step"], row["loss_abs_gap"], row["logprob"]["rel_l2"],
            row["logprob"]["mean_abs"], row["grad_norm_relative_gap"],
            row["sampled_grad_relative_l2"], row["param_after_rel_l2"],
            row["master_update"]["master_update_rel_l2"],
            row["master_update"]["master_update_cosine"],
            row["replica_numerical_gate_ok"],
        ))))
    output.with_suffix(".tsv").write_text("\n".join(lines) + "\n")
    print(f"FULL_REPEAT_SUMMARY saved={output}", flush=True)
    print("\n".join(lines), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("full_a", type=Path)
    parser.add_argument("full_b", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.full_a, args.full_b, args.steps, args.output)
