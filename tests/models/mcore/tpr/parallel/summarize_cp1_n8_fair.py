"""N=8 Full physical MB1/MB2 versus CP1 TPR K1/K2/K4, with stage attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("full_mb1", "full_mb2", "tpr_k1", "tpr_k2", "tpr_k4")
STAGES = ("push", "visit_leaf", "visit_leaf_group", "pop")


def _load(root, arm):
    path = root / arm / "result.json"
    return json.loads(path.read_text()) if path.is_file() else None


def summarize(root: Path, *, memory_margin_gib: float = 0.5) -> str:
    results = {arm: _load(root, arm) for arm in ARMS}
    # Full MB1 is supplementary: existing roots remain fully reportable.
    missing = [arm for arm in ARMS if results[arm] is None and arm != "full_mb1"]
    lines = [
        "=" * 101,
        "TPR CP1 N=8 | FAIR FULL MB=1/2 vs TPR K=1/2/4",
        "Eight DISTINCT actual same-session PPO trajectories per run; one NPU, Qwen3-4B BF16",
        "Controlled short-crop PPO; square-CANN Full B=1/2; model F/B, no optimizer/Rollout/CP",
        f"root={root}",
        "=" * 101,
        f"{'Metric':<37} {'Full MB1':>11} {'Full MB2':>11} {'TPR K1':>11} {'TPR K2':>11} {'TPR K4':>11}",
        "-" * 101,
    ]
    present = [record for record in results.values() if record is not None]
    if present:
        keys = ("input_sha256", "source_capture", "checkpoint",
                "original_keys", "initial_weights_witness_sha256", "ppo_tokens",
                "logical_input_tokens", "input_lengths")
        for key in keys:
            if len({json.dumps(record.get(key), sort_keys=True) for record in present}) != 1:
                raise AssertionError(f"N8 Full/TPR comparison input/checkpoint mismatch: {key}")
        for arm, item in results.items():
            if item is None:
                continue
            exp_mode = arm if arm.startswith("full_mb") else "tpr"
            exp_k = 1 if arm.startswith("full_mb") else int(arm[-1])
            if item["mode"] != exp_mode or item["configured_k"] != exp_k:
                raise AssertionError(f"{arm} mode/K mismatch")
            if item["logical_trajectories"] != 8 or item["ppo_tokens"] <= 0:
                raise AssertionError(f"{arm} does not cover 8 valid PPO trajectories")
            if item.get("stage_profile") or len(item["bench_times_s"]) != item["bench_repeats"]:
                raise AssertionError(f"{arm} was traced/stage-profiled or missed repeats; not fair F/B")
            if arm.startswith("full_mb"):
                physical_b = 1 if arm == "full_mb1" else 2
                if item["physical_model_forward_calls"] != 8 // physical_b:
                    raise AssertionError(f"{arm} forward calls don't match physical B={physical_b}")
                if item.get("physical_full_micro_batch", physical_b) != physical_b:
                    raise AssertionError(f"{arm} physical batch size does not match B={physical_b}")
            if arm.startswith("tpr_k") and (
                arm != "tpr_k1" and not item["leaf_batch_stats"]["grouped_leaf_calls"]
            ):
                raise AssertionError(f"{arm} did not actually group leaves")
        lines.append(
            f"Input SHA={present[0]['input_sha256'][:20]} | PPO tokens={present[0]['ppo_tokens']}"
        )

    def get(record, key):
        return None if record is None else record.get(key)

    def fmt(value, precision=3):
        return "—" if value is None else f"{value:.{precision}f}"

    metrics = (
        ("Logical trajectories", "logical_trajectories", 0),
        ("Logical token count", "logical_input_tokens", 0),
        ("Unique physical tree tokens", "unique_tree_tokens", 0),
        ("Model forward calls", "physical_model_forward_calls", 0),
        ("NPU F/B median (seconds)", "fwd_bwd_seconds", 4),
        ("Peak allocated MAX (GiB)", "peak_allocated_gib", 3),
        ("Peak reserved MAX (GiB)", "peak_reserved_gib", 3),
        ("PPO Loss", "loss", 6),
        ("Global grad norm", "global_grad_norm", 4),
        ("Measured iterations", "bench_repeats", 0),
    )
    for label, key, p in metrics:
        lines.append(
            f"{label:<37} " + " ".join(
                f"{fmt(get(results[arm],key),p):>11}" for arm in ARMS
            )
        )
    lines.extend(["-" * 101])
    comparison = {}
    full = results["full_mb2"]
    if full:
        for arm in ("tpr_k1", "tpr_k2", "tpr_k4"):
            item = results[arm]
            if item is None:
                continue
            ratio = float(full["fwd_bwd_seconds"] / item["fwd_bwd_seconds"])
            alloc = float(full["peak_allocated_gib"] - item["peak_allocated_gib"])
            reserved = float(full["peak_reserved_gib"] - item["peak_reserved_gib"])
            pass_margin = alloc >= memory_margin_gib and reserved >= memory_margin_gib
            comparison[arm] = {
                "speedup_over_full_mb2": ratio,
                "peak_alloc_headroom_gib": alloc,
                "peak_reserved_headroom_gib": reserved,
                "memory_margin_pass": pass_margin,
                "actual_grouped_leaf_calls": item["leaf_batch_stats"]["grouped_leaf_calls"],
                "actual_grouped_leaf_segments": item["leaf_batch_stats"]["grouped_leaf_segments"],
            }
            lines.append(
                f"{arm}: FullMB2/TPR speedup={ratio:.4f}x "
                f"alloc_headroom={alloc:+.3f}GiB reserved_headroom={reserved:+.3f}GiB "
                f"memory_margin={('PASS' if pass_margin else 'FAIL')}"
            )
        if results["tpr_k1"] and results["tpr_k2"]:
            k1, k2 = results["tpr_k1"], results["tpr_k2"]
            lines.append(f"Internal batching K1/K2 speedup={k1['fwd_bwd_seconds']/k2['fwd_bwd_seconds']:.4f}x")
        if results["tpr_k2"] and results["tpr_k4"]:
            k2, k4 = results["tpr_k2"], results["tpr_k4"]
            lines.append(f"Internal batching K2/K4 speedup={k2['fwd_bwd_seconds']/k4['fwd_bwd_seconds']:.4f}x")
        lines.append(
            f"Actual N8 unique-token saving: {100*(1-results['tpr_k1']['unique_tree_tokens']/results['tpr_k1']['logical_input_tokens']):.2f}%"
            if results["tpr_k1"] else "Unique-token saving: TPR K1 unavailable"
        )

    parity_report = {}
    if full:
        from .compare_real_actor_cp1_short_full_tpr import _metrics
        baseline_lp = {
            (int(row), int(offset)): float(value)
            for row, offset, value in full["logprobs"]
        }
        for arm in ("tpr_k1", "tpr_k2", "tpr_k4"):
            item = results[arm]
            if item is None:
                continue
            test_lp = {
                (int(row), int(offset)): float(value)
                for row, offset, value in item["logprobs"]
            }
            if baseline_lp.keys() != test_lp.keys() or len(baseline_lp) != full["ppo_tokens"]:
                raise AssertionError(f"N8 Full vs {arm}: supervised response-token coordinates differ")
            keys = sorted(baseline_lp)
            lp_metric = _metrics([baseline_lp[key] for key in keys], [test_lp[key] for key in keys])
            a, b = full["gradient_witness"], item["gradient_witness"]
            if (a["names"] != b["names"]
                    or a["canonical_shapes"] != b["canonical_shapes"]):
                raise AssertionError(f"N8 Full vs {arm}: canonical gradient sample coordinates mismatch")
            grad_metric = _metrics(a["values"], b["values"])
            parity_report[arm] = {
                "ppo_loss_abs": abs(float(full["loss"])-float(item["loss"])),
                "logprob_rel_l2": lp_metric["relative_l2"],
                "sampled_parameter_grad_rel_l2": grad_metric["relative_l2"],
            }
            lines.append(
                f"PARITY Full MB2 vs {arm}: loss_gap={parity_report[arm]['ppo_loss_abs']:.7f} "
                f"logprob_rel_l2={lp_metric['relative_l2']:.6f} "
                f"sampled_grad_rel_l2={grad_metric['relative_l2']:.6f}"
            )

    # Supplemental Full MB1: existing N8 arms are reused verbatim and only
    # the new B1 measurement changes this part of the report.
    full_mb1_comparison = {}
    mb1 = results["full_mb1"]
    if mb1 is not None:
        full_time = float(mb1["fwd_bwd_seconds"])
        if full is not None:
            # MB2 speedup over MB1 is T(MB1) / T(MB2); the old
            # label accidentally described this inverse ratio as MB1 speedup.
            speed = full_time / float(full["fwd_bwd_seconds"])
            full_mb1_comparison["full_mb2_vs_full_mb1_speedup"] = speed
            lines.append(f"Full MB2 speedup over Full MB1={speed:.4f}x")
        for arm in ("tpr_k1", "tpr_k2", "tpr_k4"):
            item = results[arm]
            if item is not None:
                # This ratio is T(Full MB1) / T(TPR), i.e. TPR
                # speedup over Full MB1, with Push/Pop included.
                speed = full_time / float(item["fwd_bwd_seconds"])
                full_mb1_comparison[f"{arm}_speedup_over_full_mb1"] = speed
                label = arm.upper().replace("_", " ")
                lines.append(f"{label} speedup over Full MB1={speed:.4f}x (Push/Pop included)")
        # A separate numerical check for physical Full batch size itself,
        # and the original no-Leaf-batching TPR baseline.
        from .compare_real_actor_cp1_short_full_tpr import _metrics
        baseline_lp = {
            (int(r), int(offset)): float(v) for r, offset, v in mb1["logprobs"]
        }
        for arm in ("full_mb2", "tpr_k1"):
            item = results[arm]
            if item is None:
                continue
            test_lp = {
                (int(r), int(offset)): float(v) for r, offset, v in item["logprobs"]
            }
            if baseline_lp.keys() != test_lp.keys() or len(test_lp) != mb1["ppo_tokens"]:
                raise AssertionError(f"Full MB1 vs {arm}: PPO token coordinates differ")
            keys = sorted(baseline_lp)
            lp_metric = _metrics([baseline_lp[k] for k in keys], [test_lp[k] for k in keys])
            a, b = mb1["gradient_witness"], item["gradient_witness"]
            if a["names"] != b["names"] or a["canonical_shapes"] != b["canonical_shapes"]:
                raise AssertionError(f"Full MB1 vs {arm}: gradient witness coordinates differ")
            grad_metric = _metrics(a["values"], b["values"])
            parity = {
                "loss_abs_gap": abs(float(mb1["loss"]) - float(item["loss"])),
                "logprob_rel_l2": lp_metric["relative_l2"],
                "sampled_grad_rel_l2": grad_metric["relative_l2"],
            }
            full_mb1_comparison[f"parity_vs_{arm}"] = parity
            lines.append(
                f"PARITY Full MB1 vs {arm}: loss_gap={parity['loss_abs_gap']:.7f} "
                f"logprob_rel_l2={parity['logprob_rel_l2']:.6f} "
                f"sampled_grad_rel_l2={parity['sampled_grad_rel_l2']:.6f}"
            )
    else:
        lines.append("Full MB1 supplemental baseline: NOT YET MEASURED.")

    stage_report = {}
    for k in (2, 4):
        path = root / f"stage_tpr_k{k}" / "result.json"
        if not path.is_file():
            continue
        item = json.loads(path.read_text())
        if not item["stage_profile"] or item["configured_k"] != k:
            raise AssertionError(f"Stage K{k} is not a real measured TPR stage")
        if present and item["input_sha256"] != present[0]["input_sha256"]:
            raise AssertionError(f"Stage K{k} input mismatches performance arms")
        samples = item["stages_by_iteration"]
        if len(samples) != 1:
            raise AssertionError(f"Expected one isolated stage sample K={k}")
        stages = samples[0]
        stage_report[f"k{k}"] = stages
        lines.extend(["", f"TPR K={k} STAGE NPU EVENT TIMER (diagnostic; event overhead not in fair timing)"])
        for stage in STAGES:
            entry = stages.get(stage)
            if entry:
                lines.append(
                    f"  {stage:<19} calls={entry['calls']:<3} "
                    f"NPU_event_sum={entry['npu_event_ms']:.3f}ms "
                    f"enqueue_sum={entry['cpu_enqueue_ms']:.3f}ms "
                    f"groups={[g['batch'] for g in entry['groups']]}"
                )
        lines.append(
            f"  profile_total_npu_event_sum_ms={sum(e['npu_event_ms'] for e in stages.values()):.3f}"
        )
    if missing:
        lines.extend(["", f"MISSING ARMS (check their run.log): {', '.join(missing)}"])
    lines.extend([
        "WARNING: Speedups are diagnostic until BF16 PPO gradients converge.",
        "Model F/B only; CP1, no optimizer, no production native VERL microbatch schedule.",
        f"Summary: {root / 'SUMMARY.txt'}",
        "=" * 101,
    ])
    report = {
        "input_sha256": present[0]["input_sha256"] if present else None,
        "results": {
            arm: {
                key: item.get(key)
                for key in ("fwd_bwd_seconds","peak_allocated_gib","peak_reserved_gib",
                            "logical_trajectories","logical_input_tokens","unique_tree_tokens",
                            "physical_model_forward_calls","ppo_tokens","leaf_batch_stats","loss")
            }
            for arm, item in results.items() if item is not None
        },
        "comparison": comparison, "full_mb1_comparison": full_mb1_comparison,
        "parity": parity_report, "stage": stage_report,
        "missing_arms": missing, "memory_margin_gib": memory_margin_gib,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "SUMMARY.txt").write_text("\n".join(lines)+"\n")
    (root / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--memory-margin-gib",type=float,default=0.5)
    args=p.parse_args()
    if args.memory_margin_gib < 0:
        raise ValueError("memory-margin-gib must be >=0")
    print(summarize(args.root, memory_margin_gib=args.memory_margin_gib),flush=True)


if __name__=="__main__":
    main()
