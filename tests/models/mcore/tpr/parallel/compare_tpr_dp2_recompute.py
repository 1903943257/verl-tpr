"""Compare matched TP2×CP2×DP2 OFF/ON replays, including ALL BF16 grads.

Requires two fresh jobs on identical original PPO captures and initialization.
The gradient checker streams individual saved parameters; it never flattens an
entire 4B model at once. Samples alone are not a full-gradient gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _load_manifest(directory: Path):
    with (directory / "manifest.pt").open("rb") as handle:
        items = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(items, list) or not items:
        raise AssertionError(f"Empty saved-gradient manifest: {directory}")
    return items


def _all_parameter_gradients(left: dict, right: dict, *, threshold: float) -> dict:
    lhs, rhs = left.get("full_gradient_paths"), right.get("full_gradient_paths")
    if not isinstance(lhs, list) or not isinstance(rhs, list) or len(lhs) != len(rhs) or len(lhs) != 2:
        raise AssertionError("Both jobs must save complete BF16 gradients for BOTH TP shards")
    diff_sq, base_sq, max_abs, tensors, elements = 0.0, 0.0, 0.0, 0, 0
    worst = ("", 0.0)
    for tp in range(2):
        ld, rd = Path(lhs[tp]), Path(rhs[tp])
        lm, rm = _load_manifest(ld), _load_manifest(rd)
        if len(lm) != len(rm):
            raise AssertionError(f"TP{tp} parameter count differs")
        for l, r in zip(lm, rm, strict=True):
            for key in ("name", "shape", "tp_sharded"):
                if l[key] != r[key]:
                    raise AssertionError(f"TP{tp} gradient manifest mismatch: {key}")
            a = torch.load(ld / l["filename"], map_location="cpu", weights_only=True)
            b = torch.load(rd / r["filename"], map_location="cpu", weights_only=True)
            if tuple(a.shape) != tuple(b.shape) or tuple(a.shape) != tuple(l["shape"]):
                raise AssertionError(f"TP{tp} inconsistent gradient shape: {l['name']}")
            if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
                raise AssertionError(f"TP{tp} requires BF16 original finalized gradients")
            fa, fb = a.float(), b.float()
            if not torch.isfinite(fa).all() or not torch.isfinite(fb).all():
                raise AssertionError(f"TP{tp} non-finite finalized gradient: {l['name']}")
            difference = fa.double() - fb.double()
            partial_diff = float((difference * difference).sum().item())
            partial_ref = float((fa.double() ** 2).sum().item())
            absolute = float(difference.abs().max().item())
            diff_sq += partial_diff
            base_sq += partial_ref
            elements += a.numel()
            tensors += 1
            if absolute > max_abs:
                max_abs = absolute
                worst = (f"tp{tp}:{l['name']}", absolute)
            del a, b, fa, fb, difference
    relative = (math.sqrt(diff_sq / base_sq) if base_sq else
                (0.0 if diff_sq == 0.0 else math.inf))
    if relative > threshold:
        raise AssertionError(
            f"FULL gradient parity gate FAILED: rel_l2={relative:.9g} "
            f"> {threshold:.9g}; worst={worst}"
        )
    return {
        "full_parameter_gradient_rel_l2": relative,
        "full_parameter_gradient_max_abs": max_abs,
        "worst_parameter": worst[0],
        "parameter_tensors_checked": tensors,
        "parameter_elements_checked": elements,
        "gradient_gate_threshold": threshold,
        "full_gradient_gate_passed": True,
    }


def _updated_samples(left: dict, right: dict, *, max_abs_threshold: float) -> dict:
    lo, ro = left.get("optimizer_step"), right.get("optimizer_step")
    if not isinstance(lo, dict) or not isinstance(ro, dict):
        raise AssertionError("Full optimizer gate requires one AdamW update in both jobs")
    for key in ("optimizer", "lr", "clip_norm"):
        if lo[key] != ro[key]:
            raise AssertionError(f"Different offline AdamW configuration: {key}")
    if lo["updated_replica_sample_max_abs"] != 0 or ro["updated_replica_sample_max_abs"] != 0:
        raise AssertionError("Updated CP/DP replicas differ")
    maximum, count, changed = 0.0, 0, 0
    for tp in range(2):
        a = torch.load(lo["updated_weights_samples"][tp], map_location="cpu", weights_only=True)
        b = torch.load(ro["updated_weights_samples"][tp], map_location="cpu", weights_only=True)
        for field in ("initial", "updated"):
            if a[field].keys() != b[field].keys():
                raise AssertionError(f"Different TP{tp} {field} sampled parameter set")
        for name in a["initial"]:
            ai, bi = a["initial"][name], b["initial"][name]
            if ai["shape"] != bi["shape"] or not torch.equal(ai["indices"], bi["indices"]):
                raise AssertionError(f"Different TP{tp} initial sample layout: {name}")
            if not torch.equal(ai["values"], bi["values"]):
                raise AssertionError(f"Different TP{tp} initial weights: {name}")
            au, bu = a["updated"][name], b["updated"][name]
            if au["shape"] != bu["shape"] or not torch.equal(au["indices"], bu["indices"]):
                raise AssertionError(f"Different TP{tp} updated sample layout: {name}")
            gap = float((au["values"] - bu["values"]).abs().max().item())
            maximum = max(maximum, gap)
            count += au["values"].numel()
            changed += int((au["values"] != ai["values"]).sum().item())
    if maximum > max_abs_threshold:
        raise AssertionError(f"AdamW updated-weight sample gap {maximum:.9g} exceeds {max_abs_threshold:.9g}")
    if not math.isclose(lo["global_grad_norm_preclip"], ro["global_grad_norm_preclip"],
                        rel_tol=0.05, abs_tol=1e-6):
        raise AssertionError("AdamW gradnorm differs by over 5%")
    return {
        "updated_weight_sample_max_abs": maximum,
        "sampled_updated_elements": count,
        "off_changed_bf16_weight_samples": changed,
        "updated_weight_max_abs_threshold": max_abs_threshold,
        "optimizer_update_gate_passed": True,
        "off_adamw_seconds": lo["critical_optimizer_s"],
        "on_adamw_seconds": ro["critical_optimizer_s"],
    }


def compare(off_file: Path, on_file: Path, *, max_grad_rel_l2: float = 0.05,
            max_loss_abs: float = 1e-3, max_weight_abs: float = 0.02) -> dict:
    off, on = (json.loads(p.read_text()) for p in (off_file, on_file))
    for name, record, state in (("OFF", off, False), ("ON", on, True)):
        if record.get("topology") != {"tp": 2, "cp": 2, "dp": 2, "world": 8}:
            raise AssertionError(f"{name}: wrong TP/CP/DP topology")
        if record.get("mode") != "tpr" or record.get("recompute") is not state:
            raise AssertionError(f"{name}: wrong TPR recompute state")
        if record.get("updated_weights") is not True:
            raise AssertionError(f"{name}: optimizer step missing")
        if record.get("gradient_replica_validation_passed") is not True:
            raise AssertionError(f"{name}: strict CP/DP sampled gradient replica gate missing")
        if len(record["runs"]) != record["warmups"] + record["repeats"]:
            raise AssertionError(f"{name}: incomplete run repetitions")
        if any(x["recompute"] is not state for x in record["runs"]):
            raise AssertionError(f"{name}: mixed checkpoint modes")
    if (on.get("recompute_granularity"), on.get("recompute_method"),
            on.get("recompute_num_layers")) != ("full", "uniform", 1):
        raise AssertionError("ON is not full/uniform/1")
    if on.get("recompute_installed_tpr_blocks", 0) < 1:
        raise AssertionError("ON did not install TPR-aware checkpoint blocks")
    for key in (
        "source_step", "source_capture", "input_sha256", "checkpoint",
        "effective_sequence_cap", "global_ppo_tokens", "dp_partitions",
        "dp_tree_token_costs", "duplicated_tree_tokens", "warmups", "repeats",
        "capture_logprobs", "gradient_sync_reduce_dtype",
        "ppo_logprob_chunk_tokens", "placement_policy",
    ):
        if off.get(key) != on.get(key):
            raise AssertionError(f"OFF/ON mismatch: {key}")
    if off.get("hccl_stats_reduce_dtype") != "float32" or on.get("hccl_stats_reduce_dtype") != "float32":
        raise AssertionError("HCCL FP32 metric reduce required")
    gaps = [abs(a["loss"] - b["loss"]) for a, b in zip(off["runs"], on["runs"], strict=True)]
    if max(gaps) > max_loss_abs:
        raise AssertionError(f"OFF/ON global PPO loss gap {max(gaps):.9g} > {max_loss_abs:.9g}")
    for record in (off, on):
        if any(not math.isfinite(x["critical_fwd_bwd_s"]) or
               not math.isfinite(x["peak_allocated_gib"]) for x in record["runs"]):
            raise AssertionError("Non-finite F/B duration or peak allocated")
    grads = _all_parameter_gradients(off, on, threshold=max_grad_rel_l2)
    weights = _updated_samples(off, on, max_abs_threshold=max_weight_abs)
    ratio = off["median_fwd_bwd_s"] / on["median_fwd_bwd_s"]
    result = {
        "source_step": on["source_step"],
        "topology": on["topology"],
        "input_sha256": on["input_sha256"],
        "effective_sequence_cap": on["effective_sequence_cap"],
        "dp_partitions": on["dp_partitions"],
        "global_ppo_tokens": on["global_ppo_tokens"],
        "loss_max_absolute_gap": max(gaps),
        "loss_threshold": max_loss_abs,
        "off_median_fwd_bwd_s": off["median_fwd_bwd_s"],
        "on_median_fwd_bwd_s": on["median_fwd_bwd_s"],
        "recompute_fwd_bwd_ratio_off_over_on": ratio,
        "off_peak_allocated_gib": max(r["peak_allocated_gib"] for r in off["runs"]),
        "on_peak_allocated_gib": max(r["peak_allocated_gib"] for r in on["runs"]),
        **grads,
        **weights,
        "correctness_passed": True,
        "optimizer_is_offline_cpu_fp32_master": True,
        "online_training_throughput_claimed": False,
    }
    print(
        f"TPR_DP2_RECOMPUTE_COMPARE step={result['source_step']} "
        f"loss_gap={result['loss_max_absolute_gap']:.9g} "
        f"full_grad_rel_l2={result['full_parameter_gradient_rel_l2']:.9g} "
        f"update_max_abs={result['updated_weight_sample_max_abs']:.9g} "
        f"fwd_bwd_off_s={result['off_median_fwd_bwd_s']:.6f} "
        f"fwd_bwd_on_s={result['on_median_fwd_bwd_s']:.6f} "
        f"peak_off_gib={result['off_peak_allocated_gib']:.3f} "
        f"peak_on_gib={result['on_peak_allocated_gib']:.3f} "
        "status=PASS", flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("off_json", type=Path)
    parser.add_argument("on_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--max-grad-rel-l2", type=float, default=0.05)
    parser.add_argument("--max-loss-abs", type=float, default=1e-3)
    parser.add_argument("--max-weight-abs", type=float, default=0.02)
    args = parser.parse_args()
    for value in (args.max_grad_rel_l2, args.max_loss_abs, args.max_weight_abs):
        if not math.isfinite(value) or value < 0:
            parser.error("Comparison thresholds must be nonnegative finite numbers")
    result = compare(
        args.off_json, args.on_json, max_grad_rel_l2=args.max_grad_rel_l2,
        max_loss_abs=args.max_loss_abs, max_weight_abs=args.max_weight_abs,
    )
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(f"TPR_DP2_RECOMPUTE_REPORT={args.output_json}", flush=True)


if __name__ == "__main__":
    main()
