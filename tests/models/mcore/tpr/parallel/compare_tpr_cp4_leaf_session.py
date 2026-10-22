"""Test-only CP4 comparison: native Session F/B vs TPR Leaf F/B.

Uses per-call NPU events from PhaseSampler and per-Segment events from
SegmentEventProfiler. Both modes must have the same audited N=8 capture.
These values are diagnostic current-stream sums, NOT isolated FA/HCCL kernels.
Native per-row F+B excludes its separate PPO loss callback whereas the TPR
leaf Segment event includes its loss. Never label ratios end-to-end speedups.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _measured(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    assert data["phase_diagnostic"], f"missing PhaseSampler: {path}"
    runs = [r for r in data["runs"] if not r["warmup"]]
    if not runs:
        raise AssertionError(f"no measured runs: {path}")
    return data, runs


def _by_rank(run: dict, field: str) -> dict[int, dict]:
    values = run.get(field, [])
    return {int(x["rank"]): x for x in values}


def _samples(phase: dict, key: str, count: int) -> list[float]:
    entries = phase["stages"].get(key, {}).get("per_call_npu_event_ms")
    if entries is None or len(entries) != count:
        raise AssertionError(
            f"{key}: require {count} per-call NPU events, got {None if entries is None else len(entries)}; "
            "set TPR_LEAF_SESSION_DIAG=1 and use the 8-leaf controlled capture"
        )
    return [float(x) for x in entries]


def compare(full_path: Path, tpr_path: Path) -> list[dict]:
    full, full_runs = _measured(full_path)
    tpr, tpr_runs = _measured(tpr_path)
    if full["topology"] != tpr["topology"] or full["topology"]["cp"] != 4:
        raise AssertionError("Full/TPR must share TP2 CP4 topology")
    if full["input_sha256"] != tpr["input_sha256"]:
        raise AssertionError("Full/TPR capture SHA differs")
    if full["checkpoint"] != tpr["checkpoint"]:
        raise AssertionError("Full/TPR model checkpoint differs")
    if len(full_runs) != len(tpr_runs):
        raise AssertionError("Full/TPR repeat counts differ")
    result = []
    for run_idx, (native, tree) in enumerate(zip(full_runs, tpr_runs, strict=True)):
        nph, tph = _by_rank(native, "phase_diagnostics"), _by_rank(tree, "phase_diagnostics")
        tseg = _by_rank(tree, "profiles")
        if not nph or nph.keys() != tph.keys() or nph.keys() != tseg.keys():
            raise AssertionError("Missing phase/segment observations from one or more ranks")
        per_rank = {}
        for rank in nph:
            fwd = _samples(nph[rank], "forward", 8)
            bwd = _samples(nph[rank], "backward", 8)
            tfwd = _samples(tph[rank], "forward", 10)  # Push, 8 Leaf, Pop
            tbwd = _samples(tph[rank], "backward", 9)  # 8 Leaf, Pop
            segments = tseg[rank].get("events", [])
            leaf = [x for x in segments if x.get("kind") == "visit_leaf"]
            if len(leaf) != 8:
                raise AssertionError(f"rank={rank}: expected 8 Leaf events, found {len(leaf)}")
            per_rank[rank] = (fwd, bwd, tfwd, tbwd, leaf)
        for j in range(8):
            # Every number uses the same event ordinal on every CP/TP rank.
            # Attribution to an original capture row is not asserted here.
            def critical(fn):
                return max(float(fn(per_rank[rank])) for rank in per_rank)
            full_fb = critical(lambda x: x[0][j] + x[1][j])
            leaf_fb = critical(lambda x: x[4][j]["npu_event_ms"])
            row = {
                "run": run_idx, "ordinal": j,
                "full_forward_ms": critical(lambda x: x[0][j]),
                "full_backward_ms": critical(lambda x: x[1][j]),
                "full_fwd_bwd_ms": full_fb,
                "leaf_forward_ms": critical(lambda x: x[2][j + 1]),
                "leaf_backward_ms": critical(lambda x: x[3][j]),
                "leaf_segment_ms": leaf_fb,
                "leaf_length": int(per_rank[min(per_rank)][4][j]["length"]),
                "leaf_prefix_length": int(per_rank[min(per_rank)][4][j]["prefix"]),
                "approx_ratio": full_fb / leaf_fb if leaf_fb > 0 else None,
            }
            result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.results_dir
    rows = compare(
        root / "step0_full_fixed_shape.json",
        root / "step0_tpr_fixed_shape.json",
    )
    print("Session / Leaf single-operation comparison (ms, per-event rank-max)")
    print("| run | order | Full F | Full B | Full F+B | Leaf F | Leaf B | Leaf Segment | F+B / Leaf |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['run']} | {r['ordinal']} | {r['full_forward_ms']:.1f} | "
            f"{r['full_backward_ms']:.1f} | {r['full_fwd_bwd_ms']:.1f} | "
            f"{r['leaf_forward_ms']:.1f} | {r['leaf_backward_ms']:.1f} | "
            f"{r['leaf_segment_ms']:.1f} | {r['approx_ratio']:.3f}x |"
        )
    print(
        "MEDIAN full_fwd_bwd_ms=",
        round(statistics.median(x["full_fwd_bwd_ms"] for x in rows), 3),
        "leaf_segment_ms=",
        round(statistics.median(x["leaf_segment_ms"] for x in rows), 3),
    )
    print(
        "APPROX_MEDIAN_SPEEDUP=",
        round(
            statistics.median(x["full_fwd_bwd_ms"] for x in rows)
            / statistics.median(x["leaf_segment_ms"] for x in rows), 4
        ),
    )
    print("NOTE: execution-order comparison, not verified per-source-session identity.")
    print("NOTE: Native F+B omits separate PPO callback; TPR Leaf Segment includes PPO.")
    print("NOTE: Async HCCL streams may overlap; no isolated Ring/HCCL time is implied.")
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps({
            "rows": rows,
            "method": "NPU per-call critical max across ranks; ordinal comparison only",
            "not_comparable_as_end_to_end": True,
        }, indent=2))


if __name__ == "__main__":
    main()
