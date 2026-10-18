"""Automatically diagnose 4B CP1 Full vs segmented vs shared TPRSelfAttention.

Never infer a complete 4B model's numerical parity from sampled stage vectors
or one first-layer GEMM. The output always states established evidence and
remaining hypotheses, and preserves an audit JSON report.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .cp1_attention_attribution import _difference, SELECTED_LAYERS, STAGES


def _load(path: Path, mode: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj["mode"] != mode or obj.get("no_fixed_tile") is not True:
        raise AssertionError(f"{path}: missing mode or no-Fixed-Tile provenance")
    return obj


def _intersection(reference, candidate):
    a, b = reference["positions"], candidate["positions"]
    intersection = sorted(set(a) & set(b))
    if not intersection:
        return None
    index_a = torch.tensor([a.index(i) for i in intersection], dtype=torch.long)
    index_b = torch.tensor([b.index(i) for i in intersection], dtype=torch.long)
    t0 = reference["values"].index_select(0, index_a)
    t1 = candidate["values"].index_select(0, index_b)
    if t0.shape != t1.shape:
        return {"unmatched_shapes": [list(t0.shape), list(t1.shape)]}
    return {"absolute_tokens": intersection, **_difference(t0, t1)}


def _stage(full, contender, *, baseline_phase: str, phase: str):
    output = {}
    for row in (0, 1):
        for layer in SELECTED_LAYERS:
            for stage in (*STAGES, "post_rope_k", "raw_v",
                          "qkv_out_grad_out", "proj_out_grad_out", "fc2_out_grad_out"):
                ka = f"r{row}|{baseline_phase}|L{layer}|{stage}"
                kb = f"r{row}|{phase}|L{layer}|{stage}"
                if ka not in full["records"] or kb not in contender["records"]:
                    continue
                result = _intersection(full["records"][ka], contender["records"][kb])
                if result is not None:
                    output[f"r{row}|L{layer}|{stage}"] = result
    return output


def _first_difference(results):
    ordered = (*STAGES, "post_rope_k", "raw_v")
    for layer in SELECTED_LAYERS:
        for stage in ordered:
            match = results.get(f"r0|L{layer}|{stage}")
            if match and match.get("changed", 0) > 0:
                return {"layer": layer, "stage": stage, **match}
    return None


def _metrics_summary(pairs):
    nonzero = [
        (name, item["rel_l2"], item["max_abs"])
        for name, item in pairs.items()
        if item.get("changed", 0) > 0
    ]
    return {
        "nonzero_stage_checks": len(nonzero),
        "checked_stage_pairs": len(pairs),
        "worst_relative_l2": max(nonzero, key=lambda x: x[1]) if nonzero else None,
        "worst_max_abs": max(nonzero, key=lambda x: x[2]) if nonzero else None,
    }


def analyze(root: Path) -> dict:
    snapshots = {
        mode: _load(root / mode / "trace.pt", mode)
        for mode in ("full", "split_no_share", "shared")
    }
    f, s, shared = (
        snapshots["full"], snapshots["split_no_share"], snapshots["shared"]
    )
    if (f["row_lengths"] != s["row_lengths"] or s["row_lengths"] != shared["row_lengths"]
            or s["root_length"] != shared["root_length"]
            or not 0 < s["root_length"] < min(f["row_lengths"])):
        raise AssertionError("Three probes do not describe identical token/Root bounds")
    if f["root_length"] != s["root_length"]:
        raise AssertionError("Full prompt endpoint does not match actual shared Root split")

    root_diff = _stage(f, s, baseline_phase="full", phase="push")
    leaf_diff = _stage(f, s, baseline_phase="full", phase="leaf")
    root_pop = _stage(s, s, baseline_phase="push", phase="pop")
    full_to_split_root_backward = _stage(f, s, baseline_phase="full", phase="pop")
    full_to_split_leaf_backward = _stage(f, s, baseline_phase="full", phase="leaf")
    split_shared_leaf = _stage(s, shared, baseline_phase="leaf", phase="leaf")
    split_shared_root = _stage(s, shared, baseline_phase="push", phase="push")
    square = s["same_qkv_attention"]
    gemm = s["same_input_gemm_replay"]
    linear_replays = gemm.get("individual_original_linear_replays", {})
    relay_split = s.get("kv_relay_roundoff", {})
    relay_shared = shared.get("kv_relay_roundoff", {})

    first = _first_difference(root_diff)
    first_leaf = _first_difference(leaf_diff)
    known_mismatch = (
        gemm.get("input_equal") is True
        and gemm.get("qkv_same_input_m_replay", {}).get("changed", 0) > 0
        and gemm.get("full_qkv_replay_vs_actual", {}).get("exact") is True
        and gemm.get("split_root_qkv_replay_vs_actual", {}).get("exact") is True
    )
    square_mismatch = sum(v["changed"] for v in square.values())
    root_replay_kv_diffs = {
        key: value for key, value in root_pop.items()
        if key.endswith(("post_rope_k", "raw_v")) and value.get("changed", 0)
    }
    prefix_grad_roundoff = [
        {
            "case": group, "key": key, "count": val["count"],
            "dK_rel_l2": val["bf16_vs_fp32_accumulate"]["dK"]["rel_l2"],
            "dV_rel_l2": val["bf16_vs_fp32_accumulate"]["dV"]["rel_l2"],
        }
        for group, records in (("split", relay_split), ("shared", relay_shared))
        for key, val in records.items()
    ]
    prefix_grad_roundoff.sort(
        key=lambda x: max(x["dK_rel_l2"], x["dV_rel_l2"]), reverse=True
    )

    comparison_path = root / "comparison_split_no_share.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    comparison = json.loads(comparison_path.read_text())
    if comparison.get("ppo_tokens", 0) <= 0:
        raise AssertionError("PPO comparator has no true supervised tokens")
    report = {
        "model": "Qwen3-4B-128K BF16",
        "actual_ppo_four_arm_metrics": {
            label: {
                "logprob_rel_l2": data["logprob"]["relative_l2"],
                "loss_abs_gap": data["loss_abs_gap"],
                "sampled_grad_rel_l2": data["canonical_sampled_grad"]["relative_l2"],
                "complete_norm_grad_rel_l2": data["complete_norm_grad"]["relative_l2"],
                "sampled_grad_cosine": data["canonical_sampled_grad"]["cosine"],
                "global_grad_norm_reference": data["global_grad_norm_reference"],
                "global_grad_norm_candidate": data["global_grad_norm_candidate"],
                "global_grad_norm_relative_gap": data["global_grad_norm_relative_gap"],
            }
            for label, data in comparison["comparisons"].items()
        },
        "real_supervised_ppo_tokens": comparison["ppo_tokens"],
        "topology": "TP1_CP1_DP1",
        "source_capture": "controlled_2k_real_actor_ppo_capture",
        "root_length": s["root_length"],
        "row_lengths": s["row_lengths"],
        "selected_layers": list(SELECTED_LAYERS),
        "fixed_tile_enabled": False,
        "replayed_original_megatron_linear": True,
        "first_root_stage_difference": first,
        "first_leaf_stage_difference": first_leaf,
        "same_input_qkv_physical_m_replay": gemm,
        "same_input_original_projection_m_replays": linear_replays,
        "qkv_m_shape_effect_confirmed": known_mismatch,
        "same_qkv_square_vs_rectangular": square,
        "same_qkv_attention_numerically_different": square_mismatch > 0,
        "same_qkv_square_vs_rectangular_changed_elements": square_mismatch,
        "root_push_vs_pop": _metrics_summary(root_pop),
        "root_push_pop_kv_mismatch": root_replay_kv_diffs,
        "split_vs_shared_root": _metrics_summary(split_shared_root),
        "split_vs_shared_leaf": _metrics_summary(split_shared_leaf),
        "full_vs_split_root": _metrics_summary(root_diff),
        "full_vs_split_leaf": _metrics_summary(leaf_diff),
        "full_vs_split_root_backward": _metrics_summary(full_to_split_root_backward),
        "full_vs_split_leaf_backward": _metrics_summary(full_to_split_leaf_backward),
        "bf16_prefix_grad_accumulation_vs_fp32_reference": prefix_grad_roundoff,
        "limits": [
            "Stage comparisons use a bounded fixed subset of absolute token positions; "
            "they are NOT full-layer/full-model relative L2.",
            "QKV replay establishes shape sensitivity of the first-layer projection "
            "only, not sufficiency to explain all 36-layer PPO gradient differences.",
            "The same-QKV Attention probe changes query physical shape while keeping "
            "K/V, causal positions, projection inputs fixed; it is numerical, "
            "not an all-layer backward equivalence proof.",
            "BF16-vs-FP32 dKV sum is an observation on the original production "
            "accumulator; no FP32 gradient is injected into training.",
            "The short capture preserves captured labels but has altered conditioning.",
            "No optimizer step or on-policy rollout; production training acceptance "
            "requires independent multi-step PPO/GRPO validation.",
        ],
    }
    report["conclusions"] = []
    if known_mismatch:
        report["conclusions"].append(
            "CONFIRMED: first-layer QKV on the SAME full/Root input and original "
            "Megatron weights produces different BF16 values solely when physical "
            "GEMM M changes. Replays agree exactly with both recorded branches."
        )
    elif gemm.get("input_equal") is False:
        report["conclusions"].append(
            "NOT YET A GEMM-M ATTRIBUTION: first-layer QKV input already differs. "
            "Investigate the input/RMSNorm branch before blaming GEMM shape."
        )
    elif gemm.get("qkv_same_input_m_replay", {}).get("changed", 0) == 0:
        report["conclusions"].append(
            "NOT CONFIRMED: first-layer QKV same-input physical-M replay was "
            "identical; investigate Attention and deeper-layer GEMMs."
        )
    else:
        report["conclusions"].append(
            "PARTIAL: QKV M-dependent result observed, but actual recorded "
            "Full/Split QKV could not be reproduced exactly. Attribution incomplete."
        )
    for module_name, module_result in linear_replays.items():
        m = module_result["same_full_input_m_replay"]
        compatible = module_result["input_equal"]
        report["conclusions"].append(
            f"FIRST-LAYER {module_name.upper()} BF16 SHAPE REPLAY: "
            f"same-Full-input M-only max_abs={m['max_abs']:.8g}, "
            f"changed={m['changed']}/{m['count']}, "
            f"natural Root input identical={compatible}. "
            + (
                "It is a candidate direct contributor to natural Root drift."
                if compatible and m["changed"] and
                module_result["full_replay_vs_actual"]["exact"] and
                module_result["root_replay_vs_actual"] is not None and
                module_result["root_replay_vs_actual"]["exact"]
                else "Do not identify this shape-only replay with the natural Root "
                     "output without matching inputs AND exact replay."
            )
        )
    if first:
        report["conclusions"].append(
            f"EARLIEST SAMPLED ROOT DIFFERENCE: L{first['layer']} "
            f"{first['stage']} (max_abs={first['max_abs']:.8g}). "
            "Earlier sampled stages equal; unobserved token positions are not ruled out."
        )
    if square_mismatch:
        report["conclusions"].append(
            "CONFIRMED: same-(Q,K,V) CANN Square vs Rectangular Attention differs "
            f"on {square_mismatch} probed output elements in selected layers. "
            "This may be an additional numerical contributor."
        )
    else:
        report["conclusions"].append(
            "SAMPLED CAUSAL ORACLE: no same-QKV Square/Rectangular output differences "
            "on selected layers; this does NOT cover backward dQ/dK/dV."
        )
    if root_replay_kv_diffs:
        report["conclusions"].append(
            "CHECK: Root Push/Pop K/V is not bitwise identical on sampled coordinates; "
            "inspect recomputation/determinism separately."
        )
    else:
        report["conclusions"].append(
            "PASS (SAMPLED): Split Root Push and Pop post-RoPE K/raw V agree on "
            "the recorded absolute positions."
        )
    report["conclusions"].append(
        "NOT PROVEN: matching upstream dK/dV and all parameter gradients. "
        "This attribution is not a numerical training PASS. "
        "Never enable Fixed Tile as a workaround."
    )
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    report = analyze(root)
    output = root / "ATTENTION_ATTRIBUTION_FINAL.json"
    if output.exists():
        raise FileExistsError(f"Final report is immutable; use a fresh root: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = [
        "========== TPRSELFATTENTION CP1 NUMERICAL ATTRIBUTION ==========",
        f"Model: {report['model']} | Root={report['root_length']} | "
        f"rows={report['row_lengths']} | Fixed Tile=OFF",
        f"Confirmed QKV physical M effect: {report['qkv_m_shape_effect_confirmed']}",
        f"Same-QKV Square/Rect output differing values: "
        f"{report['same_qkv_square_vs_rectangular_changed_elements']}",
        "========== AUTOGENERATED FINDINGS ==========",
    ]
    lines.append("========== ACTUAL FOUR-ARM PPO NUMERICAL RESULTS ==========")
    for name in ("full_vs_one_node", "one_node_vs_split_no_share",
                 "split_no_share_vs_shared", "full_vs_shared"):
        observed = report["actual_ppo_four_arm_metrics"][name]
        lines.append(
            f"{name}: logprob_rel={observed['logprob_rel_l2']:.4%}; "
            f"loss_gap={observed['loss_abs_gap']:.8g}; "
            f"sampled_grad_rel={observed['sampled_grad_rel_l2']:.4%}; "
            f"complete_norm_grad_rel={observed['complete_norm_grad_rel_l2']:.4%}; "
            f"gradnorm={observed['global_grad_norm_reference']:.7g}"
            f"->{observed['global_grad_norm_candidate']:.7g} "
            f"(gap={observed['global_grad_norm_relative_gap']:.4%})"
        )
    lines.append("========== AUTOGENERATED ROOT-CAUSE FINDINGS ==========")
    lines.extend("  " + sentence for sentence in report["conclusions"])
    lines += [
        "========== NEXT-GATE ==========",
        "Do not use Fixed Tile; isolate remaining backward gradient contribution, "
        "then validate 5-20 steps authentic PPO FP32-master AdamW.",
        "JSON report: " + str(output),
        "=============================================================",
    ]
    summary = "\n".join(lines) + "\n"
    (root / "ATTENTION_ATTRIBUTION_CONCLUSION.txt").write_text(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
