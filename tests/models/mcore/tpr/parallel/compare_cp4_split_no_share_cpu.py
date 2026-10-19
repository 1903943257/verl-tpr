"""CPU-only controlled CP4 attribution: whole vs split no-share vs shared.

All three arms use the SAME model, PPO tokens, TPR Ring Backend, SegmentExecutor,
and identical initial weights. Only physical QKV/GEMM segment boundaries
and actual inter-sibling Root KV sharing vary. Historical captured PPO labels
are unchanged. The comparator DOES NOT equate 5% gates with proven correctness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .compare_real_actor_tp2cp4_offline_ppo import (
    _complete_gradient_metrics, _l2_pair, _logprobs, _weight_diff,
)


ARMS = ("full", "split_no_share", "tpr")
PAIRINGS = (
    ("full_vs_split_no_share", "full", "split_no_share"),
    ("split_no_share_vs_shared", "split_no_share", "tpr"),
    ("full_vs_shared", "full", "tpr"),
)


def analyze(arms: dict[str, list[dict]], *, read_full_grads: bool = True) -> dict:
    for mode in ARMS:
        if mode not in arms or len(arms[mode]) != 8:
            raise AssertionError(f"Need eight saved snapshots for {mode}")
        rows = arms[mode]
        for i, rec in enumerate(rows):
            if rec["rank"] != i or rec["mode"] != mode:
                raise AssertionError(f"Invalid world rank/arm identity: {mode} rank={i}")
    ref = arms["full"]
    provenance = (
        "source_capture", "source_rows", "source_keys", "data_sha256",
        "checkpoint", "objective", "logical_ppo_tokens", "tp", "cp", "dp",
        "optimizer",
    )
    for mode in ("split_no_share", "tpr"):
        cand = arms[mode]
        for i in range(8):
            for key in provenance:
                if cand[i][key] != ref[i][key]:
                    raise AssertionError(
                        f"Input/optimizer mismatch full vs {mode} "
                        f"rank={i} field={key}"
                    )
        if _weight_diff(ref, cand, "param_before") != 0.0:
            raise AssertionError(f"Initial weights changed: {mode}")
    topology = {
        mode: {
            "trees": arms[mode][0]["forest_trees"],
            "segments": arms[mode][0]["forest_segments"],
            "ring": arms[mode][0]["ring_attention_impl"],
        }
        for mode in ARMS
    }
    expected = {"full": (2, 2), "split_no_share": (2, 4), "tpr": (1, 3)}
    for mode, (trees, segments) in expected.items():
        if (topology[mode]["trees"], topology[mode]["segments"]) != (
            trees, segments
        ):
            raise AssertionError(f"Bad physical topology for {mode}: {topology[mode]}")
        ring_backends = {row["ring_attention_impl"] for row in arms[mode]}
        if ring_backends != {"tpr_custom_ring"}:
            raise AssertionError(
                f"Not same custom Ring backend for {mode}: {ring_backends}"
            )
        expected_physical = (trees, segments)
        if any(
            (row["forest_trees"], row["forest_segments"]) != expected_physical
            for row in arms[mode]
        ):
            raise AssertionError(f"Rank-disagreeing physical tree layout: {mode}")
        if any(row["optimizer"] != "diagnostic_bf16_sgd_lr_0.0"
               for row in arms[mode]):
            raise AssertionError("Numerical attribution requires SGD=0")
    pairings = {}
    for label, a, b in PAIRINGS:
        left, right = arms[a], arms[b]
        xs, ys = _logprobs(left), _logprobs(right)
        if xs.keys() != ys.keys() or len(xs) != ref[0]["logical_ppo_tokens"]:
            raise AssertionError(f"{label}: incomplete logical PPO token ownership")
        order = sorted(xs)
        x = torch.tensor([xs[k] for k in order], dtype=torch.float64)
        y = torch.tensor([ys[k] for k in order], dtype=torch.float64)
        difference = x-y
        full_grad = (
            _complete_gradient_metrics(left, right)
            if read_full_grads else None
        )
        norm_a = float(left[0]["global_bf16_grad_norm"])
        norm_b = float(right[0]["global_bf16_grad_norm"])
        stat = {
            "reference_arm": a, "candidate_arm": b,
            "tokens": len(order),
            "logprob_rel_l2": float(
                torch.linalg.vector_norm(difference) /
                torch.linalg.vector_norm(x).clamp_min(1e-30)
            ),
            "logprob_mean_abs": float(difference.abs().mean()),
            "logprob_max_abs": float(difference.abs().max()),
            "loss_gap": float(right[0]["loss"] - left[0]["loss"]),
            "sampled_grad_rel_l2": _l2_pair(left, right),
            "complete_grad": full_grad,
            "gradient_norm_ref": norm_a,
            "gradient_norm_candidate": norm_b,
            "gradient_norm_relative_gap": abs(norm_a-norm_b)/max(norm_a, 1e-30),
            "fwd_bwd_seconds_ref": max(x["time_forward_backward_finalize_s"] for x in left),
            "fwd_bwd_seconds_candidate": max(x["time_forward_backward_finalize_s"] for x in right),
            "peak_gib_ref": max(x["peak_allocated_bytes"] for x in left)/2**30,
            "peak_gib_candidate": max(x["peak_allocated_bytes"] for x in right)/2**30,
        }
        pairings[label] = stat
        print(
            f"TPR_CP4_THREE_ARM {label} lp_rel={stat['logprob_rel_l2']:.8g} "
            f"lp_mae={stat['logprob_mean_abs']:.8g} "
            f"loss_gap={stat['loss_gap']:.8g} "
            f"grad_sample_rel={stat['sampled_grad_rel_l2']:.8g} "
            f"grad_full_rel={full_grad['relative_l2'] if full_grad else 'not_saved'} "
            f"grad_full_cos={full_grad['cosine'] if full_grad else 'not_saved'} "
            f"grad_norm_gap={stat['gradient_norm_relative_gap']:.8g}",
            flush=True,
        )
    return {
        "status": "DIAGNOSTIC_CP4_SAME_RING_THREE_ARM_PPO_PARITY",
        "data_protocol": "IDENTICAL_FIXED_ACTOR_PPO_STEP1_NO_OPTIMIZER",
        "source_capture": ref[0]["source_capture"],
        "source_rows": list(ref[0]["source_rows"]),
        "topologies": topology,
        "comparisons": pairings,
        "interpretation": (
            "Full vs Split isolates whole-sequence vs identical Root/Leaf "
            "physical segment shapes without sharing. Split vs Shared isolates "
            "actual inter-sibling Root KV reuse with the same per-segment "
            "GEMM shapes. Gradient differences reflect all path-dependent "
            "BF16/CP effects, not just a single kernel."
        ),
    }


def compare(directory: Path, *, read_full_grads: bool = True):
    modes = {
        mode: [
            torch.load(directory/f"{mode}_rank{rank}.pt",
                       map_location="cpu", weights_only=False)
            for rank in range(8)
        ]
        for mode in ARMS
    }
    result = analyze(modes, read_full_grads=read_full_grads)
    target = directory/"cp4_same_ring_three_arm_comparison.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {target}")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"TPR_CP4_THREE_ARM_SAVED={target}", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path)
    p.add_argument("--skip-complete-grad", action="store_true")
    args = p.parse_args()
    compare(args.output_dir, read_full_grads=not args.skip_complete_grad)
