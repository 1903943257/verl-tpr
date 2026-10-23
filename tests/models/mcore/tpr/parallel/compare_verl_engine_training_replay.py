"""CPU-only comparison of independently initialized real VERL Full / TPR updates.

Compared checkpoints and Actor minibatches must match exactly. Numerics
are diagnostic, NOT a fabricated 'bitwise PPO correctness PASS': measured
gradient witnesses are bounded samples, not full model gradient tensors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _pairs(witness: dict) -> dict:
    return {tuple(key): float(value) for key, value in witness.items()}


def _stats(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        raise AssertionError("Empty or mismatched numerical witness")
    full = 0.0
    cand = 0.0
    diff = 0.0
    dot = 0.0
    abs_max = 0.0
    abs_sum = 0.0
    for x, y in zip(left, right, strict=True):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise AssertionError("Non-finite Full/TPR witness")
        d = y - x
        full += x * x
        cand += y * y
        diff += d * d
        dot += x * y
        abs_max = max(abs_max, abs(d))
        abs_sum += abs(d)
    return {
        "count": len(left),
        "relative_l2": math.sqrt(diff / max(full, 1e-30)),
        "cosine": dot / max(math.sqrt(full * cand), 1e-30),
        "mean_abs": abs_sum / len(left),
        "max_abs": abs_max,
    }


def _flatten_samples(first: dict, second: dict) -> tuple[list[float], list[float]]:
    if first.keys() != second.keys():
        raise AssertionError("Full/TPR parameter sets differ")
    a = []
    b = []
    for name in sorted(first):
        left, right = first[name], second[name]
        for key in ("shape", "indices"):
            if key == "indices":
                if not torch.equal(left[key], right[key]):
                    raise AssertionError(f"Sampled parameter coordinates differ: {name}")
            elif tuple(left[key]) != tuple(right[key]):
                raise AssertionError(f"Parameter shape differs: {name}")
        a.extend(float(x) for x in left["values"].tolist())
        b.extend(float(x) for x in right["values"].tolist())
    return a, b


def compare(root: Path, *, steps: int, full_prefix: str, tpr_prefix: str) -> dict:
    if not 1 <= steps <= 20:
        raise ValueError("Expected 1..20 step comparison")
    records = []
    for step in range(1, steps + 1):
        full_ranks = []
        tpr_ranks = []
        for rank in range(8):
            full_file = root / f"{full_prefix}_step{step}_rank{rank}.pt"
            tpr_file = root / f"{tpr_prefix}_step{step}_rank{rank}.pt"
            if not (full_file.is_file() and tpr_file.is_file()):
                raise FileNotFoundError(f"Missing pair: {full_file} / {tpr_file}")
            full = torch.load(full_file, map_location="cpu", weights_only=False)
            tpr = torch.load(tpr_file, map_location="cpu", weights_only=False)
            if (full["execution_backend"] != "verl_mindspeed_megatron_engine_full"
                    or tpr["execution_backend"] != "verl_mindspeed_megatron_engine_tpr"):
                raise AssertionError("Must compare genuine Full and TPR Engines")
            common = (
                "checkpoint", "source_capture", "source_rows", "source_keys",
                "source_online_global_step", "source_capture_index",
                "data_sha256", "capture_manifest", "tokens", "tp_rank", "cp_rank",
                "optimizer_lr", "optimizer_clip_norm",
            )
            for field in common:
                if full.get(field) != tpr.get(field):
                    raise AssertionError(
                        f"step={step} rank={rank}: Full/TPR {field} differs"
                    )
            first, second = _flatten_samples(
                full["param_initial"], tpr["param_initial"]
            )
            if first != second:
                raise AssertionError(
                    f"step={step} rank={rank}: initial parameter witnesses differ"
                )
            full_ranks.append(full)
            tpr_ranks.append(tpr)

        primary_full = full_ranks[0]
        primary_tpr = tpr_ranks[0]
        log_full = _pairs(primary_full["logprobs"])
        log_tpr = _pairs(primary_tpr["logprobs"])
        if log_full.keys() != log_tpr.keys():
            raise AssertionError(
                f"step={step}: Full/TPR supervise different logical PPO tokens"
            )
        if len(log_full) != primary_full["tokens"]:
            raise AssertionError(f"step={step}: not all PPO tokens have logprobs")
        logical_keys = sorted(log_full)
        log_metrics = _stats(
            [log_full[k] for k in logical_keys], [log_tpr[k] for k in logical_keys]
        )

        # CP ranks contain replicated parameter shards. Measure only one
        # gradient/weight copy per TP partition (CP rank 0) to avoid
        # double-counting the same logical parameter coordinates.
        grad_left, grad_right = [], []
        weight_left, weight_right = [], []
        storage = []
        for full, tpr in zip(full_ranks, tpr_ranks, strict=True):
            if tpr["cp_rank"] != 0:
                continue
            x, y = _flatten_samples(full["gradient_samples"], tpr["gradient_samples"])
            grad_left.extend(x)
            grad_right.extend(y)
            x, y = _flatten_samples(full["param_after"], tpr["param_after"])
            weight_left.extend(x)
            weight_right.extend(y)
            storage.append(tpr["qkv_grad_storage"])
        if len(storage) != 2:
            raise AssertionError("Expected one unique QKV grad-storage witness per TP shard")
        full_loss = sum(float(x) for x in primary_full["loss"])
        tpr_loss = sum(float(x) for x in primary_tpr["loss"])
        fb_full = max(float(x["fwd_bwd_seconds"]) for x in full_ranks)
        fb_tpr = max(float(x["fwd_bwd_seconds"]) for x in tpr_ranks)
        opt_full = max(float(x["optimizer_seconds"]) for x in full_ranks)
        opt_tpr = max(float(x["optimizer_seconds"]) for x in tpr_ranks)
        record = {
            "step": step,
            "capture": primary_full["source_capture"],
            "data_sha256": primary_full["data_sha256"],
            "ppo_tokens": primary_full["tokens"],
            "loss_full": full_loss,
            "loss_tpr": tpr_loss,
            "loss_abs_gap": abs(full_loss - tpr_loss),
            "logprobs": log_metrics,
            "sampled_gradient": _stats(grad_left, grad_right),
            "sampled_parameter_after_step": _stats(weight_left, weight_right),
            "qkv_grad_storage_tp_shards": storage,
            "full_fb_critical_seconds": fb_full,
            "tpr_fb_critical_seconds": fb_tpr,
            "fb_speedup": fb_full / max(fb_tpr, 1e-30),
            "full_fb_optimizer_critical_seconds": fb_full + opt_full,
            "tpr_fb_optimizer_critical_seconds": fb_tpr + opt_tpr,
            "fb_optimizer_speedup": (fb_full + opt_full) / max(fb_tpr + opt_tpr, 1e-30),
            "full_peak_allocated_gib": max(x["peak_allocated_bytes"] for x in full_ranks) / 2**30,
            "tpr_peak_allocated_gib": max(x["peak_allocated_bytes"] for x in tpr_ranks) / 2**30,
            "full_peak_reserved_gib": max(x["peak_reserved_bytes"] for x in full_ranks) / 2**30,
            "tpr_peak_reserved_gib": max(x["peak_reserved_bytes"] for x in tpr_ranks) / 2**30,
        }
        records.append(record)
    return {
        "scope": "genuine_8NPU_TP2_CP4_DP1_MindspeedEngine_optimizer_step",
        "steps": steps,
        "same_captures": True,
        "sampled_gradients_not_full_parameter_l2": True,
        "no_rollout": True,
        "diagnostic_only": True,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--full-prefix", default="verl_full_smoke")
    parser.add_argument("--tpr-prefix", default="verl_tpr_smoke")
    args = parser.parse_args()
    result = compare(
        args.root, steps=args.steps,
        full_prefix=args.full_prefix, tpr_prefix=args.tpr_prefix,
    )
    output = args.root / f"comparison_{args.full_prefix}_vs_{args.tpr_prefix}.json"
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite: {output}")
    output.write_text(json.dumps(result, indent=2))
    for row in result["records"]:
        print(
            f"VERL_ENGINE_TRAIN_COMPARISON step={row['step']} "
            f"loss_gap={row['loss_abs_gap']:.8g} "
            f"logprob_rel_l2={row['logprobs']['relative_l2']:.8g} "
            f"sampled_grad_rel_l2={row['sampled_gradient']['relative_l2']:.8g} "
            f"fb_speedup={row['fb_speedup']:.4f} "
            f"fbo_speedup={row['fb_optimizer_speedup']:.4f}",
            flush=True,
        )
    print(f"VERL_ENGINE_TRAIN_COMPARISON_SAVED {output}", flush=True)


if __name__ == "__main__":
    main()
