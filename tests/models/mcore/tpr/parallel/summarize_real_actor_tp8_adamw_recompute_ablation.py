#!/usr/bin/env python3
"""Summarize TP8 Full/TPR x recompute replay, tolerating partial/OOM runs."""
import argparse
import json
import re
from pathlib import Path

import torch


def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    configs = {}

    for rec in (0, 1):
        for mode in ("full", "tpr"):
            key = f"{mode}_recompute_{rec}"
            steps = []
            payloads = []
            for step in range(1, 6):
                step_dir = root / f"recompute_{rec}" / f"step_{step:02d}"
                ranks = {}
                for rank in range(8):
                    path = step_dir / f"{mode}_rank{rank}.pt"
                    if not path.is_file():
                        continue
                    try:
                        item = load_payload(path)
                    except Exception as exc:
                        print(f"WARNING: cannot load {path}: {exc}")
                        continue
                    if item.get("mode") != mode or int(item.get("step_index", -1)) != step:
                        print(f"WARNING: snapshot identity mismatch: {path}")
                        continue
                    ranks[rank] = item
                    payloads.append(item)

                if sorted(ranks) != list(range(8)):
                    print(f"INCOMPLETE {key} step={step}/5 ranks={sorted(ranks)}")
                    continue

                rows = [ranks[r] for r in range(8)]
                pure_fb = max(
                    max(0.0, float(x["time_forward_backward_finalize_s"])
                        - float(x["time_finalize_s"]))
                    for x in rows
                )
                fb_finalize = max(float(x["time_forward_backward_finalize_s"]) for x in rows)
                grad_norm = max(float(x["time_grad_norm_s"]) for x in rows)
                adamw = max(float(x["time_adamw_s"]) for x in rows)
                training_side = max(
                    float(x["time_forward_backward_finalize_s"])
                    + float(x["time_grad_norm_s"])
                    + float(x["time_adamw_s"])
                    for x in rows
                )
                steps.append({
                    "step": step,
                    "pure_fb_s": pure_fb,
                    "fb_finalize_s": fb_finalize,
                    "grad_norm_s": grad_norm,
                    "adamw_s": adamw,
                    "training_side_s": training_side,
                    "loss": float(rows[0]["loss"]),
                })

            exit_path = root / f"exit_{mode}_recompute_{rec}.txt"
            rc = int(exit_path.read_text().strip()) if exit_path.exists() else None
            log_path = root / "logs" / f"recompute_{rec}_{mode}.log"
            log = log_path.read_text(errors="ignore").lower() if log_path.exists() else ""
            oom = bool(re.search(
                r"out of memory|oom|memory allocation failed|alloc.*failed|"
                r"cannot allocate memory|acl_error_rt_memory_allocation|507018",
                log,
            ))

            if rc == 0 and len(steps) == 5:
                status = "PASS"
            elif oom:
                status = "OOM"
            elif rc not in (None, 0):
                status = "FAILED"
            else:
                status = "INCOMPLETE"

            fields = ("pure_fb_s", "fb_finalize_s", "grad_norm_s", "adamw_s", "training_side_s")
            sums = {name: sum(row[name] for row in steps) for name in fields}
            configs[key] = {
                "mode": mode,
                "recompute": rec,
                "status": status,
                "exit_code": rc,
                "complete_steps": len(steps),
                "steps": steps,
                "sums_s": sums,
                "peak_allocated_gib": (
                    max(float(x["peak_allocated_bytes"]) / 2**30 for x in payloads)
                    if payloads else None
                ),
                "peak_reserved_gib": (
                    max(float(x["peak_reserved_bytes"]) / 2**30 for x in payloads)
                    if payloads else None
                ),
            }

    lines = [
        "# P16S2 N=8 Full/TPR × Recompute 5-step training-side ablation",
        "",
        f"Output root: {root}",
        "Topology: TP8 × CP1 × DP1; Qwen3-4B; same five captured Actor minibatches.",
        "Optimizer: persistent CPU FP32-master AdamW; beta1=0.9, beta2=0.95, eps=1e-8, weight decay=0.",
        "Training-side time = F/B + gradient finalization + global gradient norm + AdamW update/copy-back.",
        "Rollout is excluded. Totals sum only complete 8-rank steps; check Completed before comparing.",
        "",
        "| Mode | Recompute | Status | Completed | Σ pure F/B (s) | Σ F/B+finalize (s) | Σ grad norm (s) | Σ AdamW (s) | Σ training-side (s) | Peak allocated (GiB) | Peak reserved (GiB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rec in (0, 1):
        for mode in ("full", "tpr"):
            x = configs[f"{mode}_recompute_{rec}"]
            s = x["sums_s"]
            lines.append(
                f"| {mode.upper()} | {rec} | {x['status']} | {x['complete_steps']}/5 | "
                f"{fmt(s['pure_fb_s'])} | {fmt(s['fb_finalize_s'])} | "
                f"{fmt(s['grad_norm_s'])} | {fmt(s['adamw_s'])} | "
                f"{fmt(s['training_side_s'])} | {fmt(x['peak_allocated_gib'])} | "
                f"{fmt(x['peak_reserved_gib'])} |"
            )

    lines += ["", "## Full/TPR speedup", ""]
    speedups = {}
    for rec in (0, 1):
        full = configs[f"full_recompute_{rec}"]
        tpr = configs[f"tpr_recompute_{rec}"]
        if full["status"] == "PASS" and tpr["status"] == "PASS":
            fb = full["sums_s"]["pure_fb_s"] / max(tpr["sums_s"]["pure_fb_s"], 1e-12)
            train = full["sums_s"]["training_side_s"] / max(tpr["sums_s"]["training_side_s"], 1e-12)
            speedups[f"recompute_{rec}"] = {"pure_fb": fb, "training_side": train}
            lines.append(
                f"- Recompute={rec}: pure F/B **{fb:.3f}×**; "
                f"training-side **{train:.3f}×** (all five steps completed)."
            )
        else:
            speedups[f"recompute_{rec}"] = None
            lines.append(
                f"- Recompute={rec}: no 5-step Full/TPR speedup because one or both modes did not complete."
            )

    lines += ["", "## Per-step pure F/B comparison", ""]
    lines.append("| Recompute | Step | Full F/B (s) | TPR F/B (s) | Speedup |")
    lines.append("|---:|---:|---:|---:|---:|")
    for rec in (0, 1):
        full_steps = {x["step"]: x for x in configs[f"full_recompute_{rec}"]["steps"]}
        tpr_steps = {x["step"]: x for x in configs[f"tpr_recompute_{rec}"]["steps"]}
        for step in range(1, 6):
            a, b = full_steps.get(step), tpr_steps.get(step)
            if a and b:
                ratio = a["pure_fb_s"] / max(b["pure_fb_s"], 1e-12)
                lines.append(
                    f"| {rec} | {step} | {a['pure_fb_s']:.3f} | "
                    f"{b['pure_fb_s']:.3f} | {ratio:.3f}× |"
                )
            else:
                lines.append(f"| {rec} | {step} | — | — | unavailable |")

    lines += ["", "## Recompute effect", ""]
    for mode in ("full", "tpr"):
        off = configs[f"{mode}_recompute_0"]
        on = configs[f"{mode}_recompute_1"]
        if off["status"] == "PASS" and on["status"] == "PASS":
            fb = off["sums_s"]["pure_fb_s"] / max(on["sums_s"]["pure_fb_s"], 1e-12)
            train = off["sums_s"]["training_side_s"] / max(on["sums_s"]["training_side_s"], 1e-12)
            lines.append(
                f"- {mode.upper()}: recompute OFF/ON pure F/B ratio **{fb:.3f}×**; "
                f"training-side ratio **{train:.3f}×**."
            )
        else:
            lines.append(f"- {mode.upper()}: OFF/ON comparison unavailable; one configuration did not complete five steps.")

    lines += [
        "",
        "## Interpretation",
        "",
        "- OOM is an expected negative-control outcome only for Full + recompute OFF; it is not a speedup result.",
        "- Peak memory is the maximum rank-local NPU allocation/reservation among saved snapshots, including partial runs.",
        "- Pure F/B subtracts the separately timed gradient-finalization interval from the F/B+finalize interval.",
        "- AdamW uses CPU FP32 master weights/moments with zero weight decay; this is a controlled training-side replay, not a new rollout/E2E measurement.",
        "",
    ]

    summary_text = "\n".join(lines)
    (root / "SUMMARY.md").write_text(summary_text + "\n")
    (root / "summary.json").write_text(json.dumps({
        "output_root": str(root),
        "configs": configs,
        "speedups": speedups,
    }, indent=2) + "\n")
    print(summary_text)
    print(f"\nSUMMARY_PATH={root / 'SUMMARY.md'}")
    print(f"JSON_PATH={root / 'summary.json'}")


if __name__ == "__main__":
    main()
