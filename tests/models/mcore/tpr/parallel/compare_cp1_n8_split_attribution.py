"""Attribute N8 gradient deltas to sequence splitting, prefix sharing, or TreeBatch.

This is a diagnostic report, not a pass/fail numerical-equivalence gate. It
compares Full MB1, split-without-sharing, TPR K1, and TPR K2 on the same N8
capture and initial weights. Full-vs-split estimates the pre-existing segment
execution delta; K1-vs-K2 is the TreeBatch-only delta.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

ARMS = ("full_mb1", "split_no_share", "tpr_k1", "tpr_k2")
PAIRS = (
    ("full_mb1", "split_no_share"),
    ("split_no_share", "tpr_k1"),
    ("tpr_k1", "tpr_k2"),
    ("full_mb1", "tpr_k1"),
    ("full_mb1", "tpr_k2"),
)


def _read_result(root: Path, arm: str) -> dict:
    path = root / arm / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {arm} result: {path}")
    item = json.loads(path.read_text())
    if item.get("full_parameter_grad_dir") is None:
        raise AssertionError(f"{arm} did not save full parameter gradients")
    return item


def _read_logprobs(item: dict, arm: str) -> dict[tuple[int, int], float]:
    result = {}
    for row, offset, value in item["logprobs"]:
        key = (int(row), int(offset))
        if key in result:
            raise AssertionError(f"{arm} has duplicate logprob coordinate {key}")
        result[key] = float(value)
    return result


def _manifest(directory: Path) -> dict[str, dict]:
    path = directory / "manifest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing gradient manifest: {path}")
    entries = torch.load(path, map_location="cpu", weights_only=False)
    result = {item["name"]: item for item in entries}
    if len(result) != len(entries):
        raise AssertionError(f"Duplicate parameter names in {path}")
    return result


def _metric_from_sums(diff_sq: float, left_sq: float, right_sq: float, dot: float) -> dict:
    left_norm = math.sqrt(left_sq)
    right_norm = math.sqrt(right_sq)
    return {
        "relative_l2_to_left": math.sqrt(diff_sq) / max(left_norm, 1e-30),
        "absolute_l2": math.sqrt(diff_sq),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "gradient_cosine": dot / max(left_norm * right_norm, 1e-30),
    }


def _compare_gradients(items: dict[str, dict]) -> dict:
    dirs = {arm: Path(items[arm]["full_parameter_grad_dir"]).resolve() for arm in ARMS}
    manifests = {arm: _manifest(directory) for arm, directory in dirs.items()}
    names = set(manifests[ARMS[0]])
    for arm in ARMS[1:]:
        if set(manifests[arm]) != names:
            raise AssertionError(f"Gradient parameter names differ between {ARMS[0]} and {arm}")

    sums = {
        pair: {"diff_sq": 0.0, "left_sq": 0.0, "right_sq": 0.0, "dot": 0.0}
        for pair in PAIRS
    }
    # Error vectors: E_segment = split - full; E_prefix = shared K1 - split;
    # E_treebatch = shared K2 - shared K1.
    error_alignment = {
        "segment_vs_treebatch": {"dot": 0.0, "segment_sq": 0.0, "treebatch_sq": 0.0},
        "prefix_sharing_vs_treebatch": {"dot": 0.0, "prefix_sq": 0.0, "treebatch_sq": 0.0},
    }
    top = {pair: [] for pair in PAIRS}
    for name in sorted(names):
        grads = {}
        for arm in ARMS:
            entry = manifests[arm][name]
            tensor = torch.load(
                dirs[arm] / entry["filename"], map_location="cpu", weights_only=True,
            )
            if tuple(tensor.shape) != tuple(entry["shape"]):
                raise AssertionError(f"{arm} gradient shape mismatch for {name}")
            grads[arm] = tensor.detach().to(torch.float32)
        if len({tuple(x.shape) for x in grads.values()}) != 1:
            raise AssertionError(f"Gradient tensor shapes differ for {name}")

        for left, right in PAIRS:
            a, b = grads[left], grads[right]
            d = b - a
            stat = sums[(left, right)]
            stat["diff_sq"] += float(torch.sum(d * d, dtype=torch.float64).item())
            stat["left_sq"] += float(torch.sum(a * a, dtype=torch.float64).item())
            stat["right_sq"] += float(torch.sum(b * b, dtype=torch.float64).item())
            stat["dot"] += float(torch.sum(a * b, dtype=torch.float64).item())
            diff_norm = float(torch.linalg.vector_norm(d).item())
            left_norm = float(torch.linalg.vector_norm(a).item())
            top[(left, right)].append({
                "name": name,
                "absolute_l2": diff_norm,
                "relative_l2_to_left": diff_norm / max(left_norm, 1e-30),
            })

        e_segment = grads["split_no_share"] - grads["full_mb1"]
        e_prefix = grads["tpr_k1"] - grads["split_no_share"]
        e_tree = grads["tpr_k2"] - grads["tpr_k1"]
        for key, left_error in (
            ("segment_vs_treebatch", e_segment),
            ("prefix_sharing_vs_treebatch", e_prefix),
        ):
            stat = error_alignment[key]
            stat["dot"] += float(torch.sum(left_error * e_tree, dtype=torch.float64).item())
            stat["segment_sq" if key == "segment_vs_treebatch" else "prefix_sq"] += float(
                torch.sum(left_error * left_error, dtype=torch.float64).item()
            )
            stat["treebatch_sq"] += float(torch.sum(e_tree * e_tree, dtype=torch.float64).item())
        del grads, e_segment, e_prefix, e_tree

    pair_metrics = {}
    for left, right in PAIRS:
        stat = sums[(left, right)]
        pair_metrics[f"{left}_vs_{right}"] = _metric_from_sums(
            stat["diff_sq"], stat["left_sq"], stat["right_sq"], stat["dot"],
        )
        pair_metrics[f"{left}_vs_{right}"]["top_parameter_differences"] = sorted(
            top[(left, right)], key=lambda x: x["absolute_l2"], reverse=True,
        )[:12]

    alignment = {}
    for key, stat in error_alignment.items():
        left_key = "segment_sq" if key == "segment_vs_treebatch" else "prefix_sq"
        left_norm = math.sqrt(stat[left_key])
        tree_norm = math.sqrt(stat["treebatch_sq"])
        alignment[key] = {
            "error_vector_cosine": stat["dot"] / max(left_norm * tree_norm, 1e-30),
            "left_error_norm": left_norm,
            "treebatch_error_norm": tree_norm,
            "treebatch_over_left_error_norm": tree_norm / max(left_norm, 1e-30),
        }
    return {"pairwise": pair_metrics, "error_vector_alignment": alignment}


def _compare_logprobs(items: dict[str, dict]) -> dict:
    values = {arm: _read_logprobs(items[arm], arm) for arm in ARMS}
    baseline = values["full_mb1"]
    result = {}
    for arm in ARMS[1:]:
        other = values[arm]
        if baseline.keys() != other.keys():
            raise AssertionError(f"Supervised token coordinates differ: full_mb1 vs {arm}")
        coords = sorted(baseline)
        a = torch.tensor([baseline[k] for k in coords], dtype=torch.float64)
        b = torch.tensor([other[k] for k in coords], dtype=torch.float64)
        result[f"full_mb1_vs_{arm}"] = {
            "relative_l2": float(torch.linalg.vector_norm(a - b) /
                                 torch.linalg.vector_norm(a).clamp_min(1e-30)),
            "max_abs": float((a - b).abs().max().item()),
            "count": len(coords),
        }
    k1, k2 = values["tpr_k1"], values["tpr_k2"]
    if k1.keys() != k2.keys():
        raise AssertionError("K1/K2 supervised token coordinates differ")
    coords = sorted(k1)
    a = torch.tensor([k1[k] for k in coords], dtype=torch.float64)
    b = torch.tensor([k2[k] for k in coords], dtype=torch.float64)
    result["tpr_k1_vs_tpr_k2"] = {
        "relative_l2": float(torch.linalg.vector_norm(a - b) /
                             torch.linalg.vector_norm(a).clamp_min(1e-30)),
        "max_abs": float((a - b).abs().max().item()),
        "count": len(coords),
    }
    return result


def compare(root: Path) -> dict:
    items = {arm: _read_result(root, arm) for arm in ARMS}
    invariants = (
        "input_sha256", "source_capture", "checkpoint", "original_keys",
        "initial_weights_witness_sha256", "ppo_tokens", "logical_trajectories",
        "logical_input_tokens", "input_lengths",
    )
    for key in invariants:
        values = {arm: items[arm].get(key) for arm in ARMS}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) != 1:
            raise AssertionError(f"Attribution arms differ in {key}: {values}")
    if items["full_mb1"]["mode"] != "full_mb1":
        raise AssertionError("full_mb1 arm is not Full MB1")
    if items["split_no_share"]["mode"] != "split_no_share":
        raise AssertionError("split_no_share arm has wrong mode")
    if items["tpr_k1"]["mode"] != "tpr" or items["tpr_k1"]["configured_k"] != 1:
        raise AssertionError("tpr_k1 arm is not TPR K1")
    if items["tpr_k2"]["mode"] != "tpr" or items["tpr_k2"]["configured_k"] != 2:
        raise AssertionError("tpr_k2 arm is not TPR K2")
    split = items["split_no_share"]
    if split["forest_trees"] != 8 or split["forest_segments"] != 16:
        raise AssertionError("split_no_share did not run eight independent Root/Leaf chains")
    if split["unique_tree_tokens"] != split["logical_input_tokens"]:
        raise AssertionError("split_no_share unexpectedly reused physical prefix tokens")
    if items["tpr_k2"]["leaf_batch_stats"]["grouped_leaf_calls"] < 1:
        raise AssertionError("K2 did not execute actual grouped leaf forwards")
    if items["tpr_k1"]["leaf_batch_stats"]["grouped_leaf_calls"] != 0:
        raise AssertionError("K1 unexpectedly grouped leaves")

    gradients = _compare_gradients(items)
    loss = {}
    for left, right in PAIRS:
        loss[f"{left}_vs_{right}"] = abs(float(items[left]["loss"]) - float(items[right]["loss"]))
    treebatch_rel = gradients["pairwise"]["tpr_k1_vs_tpr_k2"]["relative_l2_to_left"]
    segment_rel = gradients["pairwise"]["full_mb1_vs_split_no_share"]["relative_l2_to_left"]
    prefix_rel = gradients["pairwise"]["split_no_share_vs_tpr_k1"]["relative_l2_to_left"]
    report = {
        "root": str(root.resolve()),
        "input_sha256": items["full_mb1"]["input_sha256"],
        "logical_trajectories": items["full_mb1"]["logical_trajectories"],
        "ppo_tokens": items["full_mb1"]["ppo_tokens"],
        "arms": {
            arm: {
                "mode": items[arm]["mode"],
                "configured_k": items[arm]["configured_k"],
                "loss": items[arm]["loss"],
                "forward_calls": items[arm]["physical_model_forward_calls"],
                "grouped_leaf_calls": (
                    (items[arm].get("leaf_batch_stats") or {}).get("grouped_leaf_calls")
                ),
                "full_gradient_dir": items[arm]["full_parameter_grad_dir"],
            } for arm in ARMS
        },
        "loss_abs_gaps": loss,
        "logprob_differences_from_full_mb1_and_k1_k2": _compare_logprobs(items),
        "gradient_differences": gradients,
        "diagnostic_ratios": {
            "treebatch_grad_rel_l2": treebatch_rel,
            "full_vs_split_grad_rel_l2": segment_rel,
            "split_vs_shared_k1_grad_rel_l2": prefix_rel,
            "treebatch_over_split_delta": treebatch_rel / max(segment_rel, 1e-30),
            "treebatch_over_prefix_sharing_delta": treebatch_rel / max(prefix_rel, 1e-30),
        },
        "interpretation_note": (
            "Diagnostic only, no acceptance gate. Compare K1-vs-K2 TreeBatch delta "
            "against Full-vs-split-no-share baseline and inspect error-vector cosine. "
            "Similar magnitudes alone do not prove a common cause."
        ),
    }
    out = root / "split_attribution.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("TPR_CP1_N8_SPLIT_ATTRIBUTION " + json.dumps(report, sort_keys=True), flush=True)
    print(f"Saved attribution report: {out}", flush=True)
    for pair, stat in gradients["pairwise"].items():
        print(
            f"{pair}: grad_rel_l2={stat['relative_l2_to_left']:.8g} "
            f"grad_cos={stat['gradient_cosine']:.8f}",
            flush=True,
        )
    for key, stat in gradients["error_vector_alignment"].items():
        print(
            f"{key}: error_vector_cosine={stat['error_vector_cosine']:.8f} "
            f"treebatch_over_reference_delta={stat['treebatch_over_left_error_norm']:.4f}",
            flush=True,
        )
    print(
        "NO CORRECTNESS GATE: use this report only to decide whether the observed "
        "TreeBatch delta aligns with the pre-existing split delta.",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    compare(args.root)


if __name__ == "__main__":
    main()
