"""Report CP1/CP4 stage diagnostics alongside original *unprofiled* F/B.

NPU event sums are max across ranks of per-rank event sums; phases can
overlap and may omit native schedule's custom backward. CANN kernel sums
are single-rank totals, not latency: no fabricated stage wall-times.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LABELS = [
    ("Forward", "forward"),
    ("Attention (NPU kernel sum, rank0)", "attention"),
    ("Ring/HCCL (NPU kernel sum, rank0)", "hccl"),
    ("PPO Loss (callback)", "ppo_loss"),
    ("Backward (autograd entry)", "backward"),
    ("Gradient Finalize", "gradient_finalize"),
    ("构树/调度 Host", "setup"),
    ("总 F/B (原始无 Profile)", "unprofiled"),
    ("总 F/B (本次 Profile)", "profiled"),
]


def load_result(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def collect_diagnostic(path: Path, reference: Path, *, topology: str) -> dict:
    current, original = load_result(path), load_result(reference)
    if not current.get("phase_diagnostic"):
        raise ValueError(f"Not a phase-diagnostic result: {path}")
    if current.get("input_sha256", current.get("source_input_sha256")) != original.get(
        "input_sha256", original.get("source_input_sha256")
    ):
        raise AssertionError(f"Diagnostic and reference inputs differ: {path}")
    for field in ("mode", "checkpoint"):
        if current[field] != original[field]:
            raise AssertionError(f"Mismatch on {field} in {path}")
    if current["topology"] != original["topology"]:
        raise AssertionError(f"Topology mismatch in {path}")
    reps = [r for r in current["runs"] if not r["warmup"]]
    if not reps:
        raise AssertionError(f"No measured profile iteration in {path}")
    by_key = {}
    for key in ("forward", "ppo_loss", "backward", "gradient_finalize"):
        vals = []
        for rep in reps:
            ranks = rep.get("phase_diagnostics", [])
            if not ranks:
                continue
            measured = [
                float(rank["stages"][key]["npu_event_sum_ms"])
                for rank in ranks if key in rank["stages"]
            ]
            if measured:
                vals.append(max(measured))
        by_key[key] = round(statistics.median(vals), 3) if vals else None
    for op, field in (("attention", "attention_kernel_sum_ms"),
                      ("hccl", "hccl_kernel_sum_ms")):
        values = [
            float(rank["kernels"][field])
            for rep in reps for rank in rep.get("phase_diagnostics", [])
            if rank["rank"] == 0 and rank["kernels"]
            and rank["kernels"].get(field) is not None
        ]
        by_key[op] = round(statistics.median(values), 3) if values else None
    setup = []
    for rep in reps:
        per_rank = []
        for rank in rep.get("profiles", []):
            profile = rank.get("stage_ms", {})
            per_rank.append(sum(float(profile.get(k, 0)) for k in (
                "tree_build_cpu_ms", "tp_plan_collective_ms",
                "executor_setup_cpu_ms",
            )))
        if per_rank:
            setup.append(max(per_rank))
    # CP1's separate segment profile is recorded by the CP1 experiment.
    if not setup:
        for rep in reps:
            vals = [
                sum(float(rank.get("stage_ms", {}).get(k, 0)) for k in (
                    "tree_build_cpu_ms", "tp_plan_collective_ms",
                    "executor_setup_cpu_ms",
                ))
                for rank in rep.get("segment_profiles", [])
            ]
            if vals:
                setup.append(max(vals))
    by_key["setup"] = (round(statistics.median(setup), 3)
                       if setup and current["mode"] == "tpr" else None)
    by_key["unprofiled"] = round(float(original["median_fwd_bwd_s"]) * 1000, 3)
    by_key["profiled"] = round(float(current["median_fwd_bwd_s"]) * 1000, 3)
    events = {}
    for rep in reps:
        for rank in rep.get("profiles", []):
            if rank["rank"] == 0:
                for event in rank.get("events", []):
                    kind = event["kind"]
                    events[kind] = events.get(kind, 0) + event["npu_event_ms"]
        for rank in rep.get("segment_profiles", []):
            if rank["rank"] == 0:
                for kind, stats in rank.get("segments", {}).items():
                    events[kind] = events.get(kind, 0) + stats["npu_event_sum_ms"]
    return {
        "mode": current["mode"], "topology": topology, "stages_ms": by_key,
        "tpr_segment_kind_rank0_npu_event_sum_ms": events,
        "diagnostic_json": str(path), "baseline_json": str(reference),
        "input_sha256": current.get("input_sha256", current.get("source_input_sha256")),
        "interpretation": (
            "NPU current-stream events; native backward may not use torch.autograd.backward; "
            "Attention/HCCL are single-rank overlapping kernel sums, NOT wall latency. "
            "Reported profile total is overhead-included; unprofiled total is the speedup baseline."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp1", type=Path, required=True)
    parser.add_argument("--cp4", type=Path, required=True)
    parser.add_argument("--cp1-reference", type=Path, required=True)
    parser.add_argument("--cp4-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for label, folder, base, cp in (
        ("CP1 Full", args.cp1, args.cp1_reference, 1),
        ("CP1 TPR", args.cp1, args.cp1_reference, 1),
        ("CP4 Full", args.cp4, args.cp4_reference, 4),
        ("CP4 TPR", args.cp4, args.cp4_reference, 4),
    ):
        mode = "full" if label.endswith("Full") else "tpr"
        filename = (f"{mode}.json" if cp == 1
                    else f"step0_{mode}_fixed_shape.json")
        runs[label] = collect_diagnostic(
            folder / filename, base / filename, topology=f"TP2CP{cp}"
        )
    def fmt(x):
        return "—" if x is None else f"{x:.1f}"
    columns = list(runs)
    print("## CP1 / CP4 Profiling 阶段分解 (ms)")
    print("| 阶段 | " + " | ".join(columns) + " |")
    print("|---|" + "|".join(["---:"] * len(columns)) + "|")
    for title, metric in LABELS:
        values = [fmt(runs[column]["stages_ms"].get(metric)) for column in columns]
        print("| " + title + " | " + " | ".join(values) + " |")
    print("\nNOTE: Trace kernel time is summed over events for rank0; not critical-path time.")
    print("NOTE: Stage timing uses rank-max per-stage sums, not a sum-to-total partition.")
    print("NOTE: Backward can be null if native Megatron invokes the C++ autograd engine directly.")
    print("\nTPR segment kind NPU-event sums (rank0; diagnostic):")
    for label in ("CP1 TPR", "CP4 TPR"):
        print(label, json.dumps(runs[label]["tpr_segment_kind_rank0_npu_event_sum_ms"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps({
        "stages": runs, "units": "milliseconds",
        "not_additive": True, "profile_overhead_included": True,
    }, indent=2))
    print(f"DIAGNOSTIC_SUMMARY_WRITTEN {args.output}")


if __name__ == "__main__":
    main()
