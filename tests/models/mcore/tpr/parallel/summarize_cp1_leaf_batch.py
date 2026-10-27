"""Print a concise end-of-run CP1 K=1/K=2 summary, even after a partial failure.

This is reporting only; it never modifies comparison/result JSON or reruns the
model. The shell runner invokes it from EXIT and preserves the original rc.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"_read_error": str(exc)}


def _fmt(value, digits: int = 6) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def summarize(root: Path, exit_code: int) -> str:
    k1 = _read_json(root / "k1" / "result.json")
    k2 = _read_json(root / "k2" / "result.json")
    comparison = _read_json(root / "k1_vs_k2.json")
    lines = [
        "",
        "=" * 70,
        f"TPR LEAF FINAL SUMMARY | runner_exit={exit_code} | {root}",
        "=" * 70,
        "Run status is not a numerical-equivalence gate; inspect the parity errors below.",
        "",
        f"{'Metric':<32} {'K=1':>15} {'K=2':>15}",
        "-" * 65,
    ]
    rows = [
        ("Forward calls", ("leaf_batch_stats", "model_forward_calls"), 0),
        ("Grouped leaf calls", ("leaf_batch_stats", "grouped_leaf_calls"), 0),
        ("Grouped leaf segments", ("leaf_batch_stats", "grouped_leaf_segments"), 0),
        ("Forward + backward (s)", ("fwd_bwd_seconds",), 3),
        ("Peak allocated (GiB)", ("peak_allocated_gib",), 3),
        ("Peak reserved (GiB)", ("peak_reserved_gib",), 3),
        ("PPO loss", ("loss",), 8),
        ("Grad norm", ("global_grad_norm",), 6),
    ]
    def get(item, keys):
        if not isinstance(item, dict) or "_read_error" in item:
            return None
        for key in keys:
            if not isinstance(item, dict):
                return None
            item = item.get(key)
        return item

    for title, keys, digits in rows:
        a, b = get(k1, keys), get(k2, keys)
        lines.append(f"{title:<32} {_fmt(a, digits):>15} {_fmt(b, digits):>15}")
    lines.extend(["-" * 65, ""])

    if isinstance(comparison, dict) and "comparison" in comparison:
        trace = comparison.get("layer_trace") or {}
        first = trace.get("first_forward_rel_l2_gt_1e-6") or {}
        qkv = trace.get("k2_same_input_qkv_single_replay") or {}
        rect = trace.get("k2_grouped_vs_single_rectangular") or []
        lines.extend([
            f"Single-run speedup (K1/K2): {_fmt(comparison.get('speedup'), 4)}x (NOT a benchmark)",
            f"Loss absolute gap:             {_fmt(comparison.get('loss_abs_gap'), 9)}",
            f"Logprob relative L2:           {_fmt(get(comparison, ('logprob','relative_l2')), 6)}",
            f"Sampled grad relative L2:      {_fmt(get(comparison, ('sampled_gradient','relative_l2')), 6)}",
            f"Norm grad relative L2:         {_fmt(get(comparison, ('complete_norm_gradient','relative_l2')), 6)}",
            "Full-parameter grad relative L2: " +
                _fmt(get(comparison, ('full_parameter_gradient','relative_l2')), 6),
            "First forward mismatch:        " +
                (f"{first.get('key')} rel_l2={first.get('relative_l2'):.3e}"
                 if first else "No traced value / trace disabled"),
            "Grouped vs single rectangular Attention: " +
                (f"max rel_l2={rect[0]['relative_l2']:.3e}" if rect else "Not measured"),
        ])
        if qkv:
            for row in (0, 1):
                entry = qkv.get(row, qkv.get(str(row), {}))
                full = entry.get("k1_vs_k2_full_first_layer") or {}
                same_input = full.get("qkv_in") or {}
                replay = full.get("k1_vs_single_replay") or {}
                gemm = entry.get("k2_grouped_vs_same_input_single") or {}
                lines.append(
                    f"L1 QKV row={row} input_exact={same_input.get('exact', 'unknown')} "
                    f"B1_replay_equals_K1={replay.get('exact', 'unknown')} "
                    f"group_vs_B1_rel_l2={_fmt(gemm.get('rel_l2'), 8)}"
                )
        lines.append("Parity interpretation: a finished comparison is NOT proof of matching gradients.")
        lines.append("Full report: " + str(root / "k1_vs_k2.json"))
    else:
        lines.append("Comparison missing/incomplete; check the last failed arm's log.")
        for name, item in (("k1", k1), ("k2", k2)):
            if isinstance(item, dict) and "_read_error" in item:
                lines.append(f"{name} result read error: {item['_read_error']}")
            elif item is None:
                lines.append(f"{name} result missing: {root / name / 'result.json'}")
        if isinstance(comparison, dict) and "_read_error" in comparison:
            lines.append("Comparison read error: " + comparison["_read_error"])
        lines.append("Current logs: " + str(root / "k1.log") + " | " + str(root / "k2.log"))
    lines.extend(["", "Summary file: " + str(root / "SUMMARY.txt"), "=" * 70])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    report = summarize(args.root, args.exit_code)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "SUMMARY.txt").write_text(report + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
