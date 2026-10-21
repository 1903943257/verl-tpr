"""CPU-only four-way CP4 N8 diagnostic: Native Full / TPR None / KV / KV+Query.

Requires TPR_MERGE_TOKEN_DIAG=1 snapshots, saved by run_cp4_merge_validation.sh.
Samples are NOT complete-model gradients; Q/K Norm TP contributions are SUMMED
in the TP-aware metric. Clipping is an estimated vanilla-PPO branch decision,
not an exact trace of the PPO kernel's internal control flow.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

BASE = Path(os.getenv(
    "TPR_VALIDATE_ROOT",
    "/workspace/tpr_experiments/performance/cp4_merge_validation_v1",
))
CAP = Path(os.getenv(
    "TPR_VALIDATE_CAPTURE",
    "/workspace/tpr_experiments/performance/cp4_n8_merge_perf_none_v1/data/cp4_n8_p17440_s17440.pt",
))
MODES = ("full", "none", "kv", "kv_query")


def read(mode):
    kind = "full" if mode == "full" else "tpr"
    path = BASE / mode / f"step0_{kind}_fixed_shape.json"
    return json.loads(path.read_text())


def vector_metrics(a, b):
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError("Witness vectors differ in shape or contain nonfinite values")
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    return {
        "relative_l2": float(np.linalg.norm(x-y) / max(nx, 1e-30)),
        "cosine": float(np.dot(x, y) / max(nx*ny, 1e-30)),
        "max_abs": float(np.max(np.abs(x-y))),
    }


def tp_aware_sample(witness):
    """Local-shard samples: sum shared Q/K Norm TP partials; count replicas once."""
    x = np.asarray(witness["tp_shards"], dtype=np.float64)
    assert x.shape[0] == 2
    result = []
    offset = 0
    for name, count in zip(witness["names"], witness["samples_per_name"], strict=True):
        sub = x[:, offset:offset+count]
        offset += count
        if name.endswith((
            "self_attention.q_layernorm.weight", "self_attention.k_layernorm.weight",
        )):
            result.append(sub.sum(axis=0))
        elif name.endswith((
            "embedding.word_embeddings.weight",
            "self_attention.linear_qkv.weight",
            "self_attention.linear_proj.weight",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc2.weight",
        )):
            result.append(sub.ravel())
        else:
            result.append(sub[0])
    assert offset == x.shape[1]
    return np.concatenate(result)


def token_values(snapshot):
    runs = [r for r in snapshot["runs"] if not r["warmup"]]
    assert len(runs) == 1 and runs[0].get("token_logprobs")
    mapping = {}
    for row, offset, value in runs[0]["token_logprobs"]:
        key = (int(row), int(offset))
        if key in mapping:
            raise AssertionError(f"Duplicate response token {key}")
        mapping[key] = float(value)
    return mapping


def main():
    data = {mode: read(mode) for mode in MODES}
    reference = data["full"]
    for mode, snapshot in data.items():
        for field in ("input_sha256", "checkpoint", "topology", "source_capture"):
            if snapshot[field] != reference[field]:
                raise AssertionError(f"{mode}: {field} differs from Native Full")
        witness = snapshot.get("gradient_witness")
        assert witness and witness.get("replica_exact"), f"{mode}: missing or inconsistent gradient witness"
        a = reference["gradient_witness"]
        for field in ("names", "shapes", "samples_per_name"):
            assert witness[field] == a[field], f"{mode}: {field} mismatch"

    tokens = {mode: token_values(snapshot) for mode, snapshot in data.items()}
    keys = sorted(tokens["full"])
    assert keys and all(sorted(tokens[m]) == keys for m in MODES), "PPO token ownership mismatch"

    raw = torch.load(CAP, map_location="cpu", weights_only=False)["tensordict"]
    expected = sum(int(v.bool().sum()) for v in raw["response_mask"].unbind(0))
    assert len(keys) == expected, f"Missing PPO tokens: {len(keys)} != {expected}"
    old_rows = raw["old_log_probs"]
    adv_rows = raw["advantages"]
    clip_mask = {}
    near_clip = {}
    for mode in MODES:
        decisions = []
        near = 0
        for row, off in keys:
            old = float(old_rows[row][off].item())
            advantage = float(adv_rows[row][off].item())
            delta = tokens[mode][row, off] - old
            ratio = math.exp(min(30.0, max(-30.0, delta)))
            decisions.append((advantage > 0 and ratio > 1.2) or
                             (advantage < 0 and ratio < 0.8))
            near += int(min(abs(ratio-0.8), abs(ratio-1.2)) < 0.01)
        clip_mask[mode] = np.asarray(decisions, dtype=np.bool_)
        near_clip[mode] = near

    report = {
        "metadata": {
            "capture": str(CAP), "sha256": reference["input_sha256"],
            "checkpoint": reference["checkpoint"], "topology": reference["topology"],
            "logical_ppo_tokens": len(keys),
            "gradient_note": "TP2 sampled post-finalize BF16 gradients, not full-parameter L2",
            "clip_note": "Estimated vanilla PPO ratio branch using clip_ratio=0.2, not traced kernel mask",
        },
        "modes": {}, "comparisons": {},
    }
    for mode in MODES:
        result = data[mode]
        measured = next(x for x in result["runs"] if not x["warmup"])
        report["modes"][mode] = {
            "loss": float(measured["loss"]),
            "clip_tokens_estimate": int(clip_mask[mode].sum()),
            "near_clip_tokens": near_clip[mode],
            "n_logprobs": len(tokens[mode]),
            "replica_exact": result["gradient_witness"]["replica_exact"],
        }

    comparisons = (
        ("full", "none"), ("full", "kv"), ("full", "kv_query"),
        ("none", "kv"), ("kv", "kv_query"), ("none", "kv_query"),
    )
    for a, b in comparisons:
        va = np.asarray([tokens[a][k] for k in keys], dtype=np.float64)
        vb = np.asarray([tokens[b][k] for k in keys], dtype=np.float64)
        wa, wb = (data[m]["gradient_witness"] for m in (a, b))
        report["comparisons"][f"{a}_vs_{b}"] = {
            "loss_abs_gap": abs(report["modes"][a]["loss"]-report["modes"][b]["loss"]),
            "logprob": vector_metrics(va, vb),
            "logprob_mean_abs": float(np.mean(np.abs(va-vb))),
            "clip_xor_tokens_estimate": int(np.logical_xor(clip_mask[a], clip_mask[b]).sum()),
            "grad_raw": vector_metrics(wa["tp_shards"], wb["tp_shards"]),
            "grad_tp_aware": vector_metrics(tp_aware_sample(wa), tp_aware_sample(wb)),
        }

    for mode, value in report["modes"].items():
        print(f"MODE {mode:9s} loss={value['loss']:.9g} tokens={value['n_logprobs']}"
              f" clip={value['clip_tokens_estimate']}")
    print("PAIRWISE: loss_gap, logprob_mean_abs, logprob_rel_l2, clip_xor, grad_tp_aware_rel_l2/cosine")
    for pair, value in report["comparisons"].items():
        lp = value["logprob"]
        grad = value["grad_tp_aware"]
        print(f"{pair:23s} loss={value['loss_abs_gap']:.6g}"
              f" lp_mean={value['logprob_mean_abs']:.6g} lp_rel={lp['relative_l2']:.6g}"
              f" clip_xor={value['clip_xor_tokens_estimate']}"
              f" grad_rel={grad['relative_l2']:.6g} grad_cos={grad['cosine']:.6f}")
    path = BASE / "four_way_diagnostics.json"
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite {path}")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("SAVED", path)


if __name__ == "__main__":
    main()
