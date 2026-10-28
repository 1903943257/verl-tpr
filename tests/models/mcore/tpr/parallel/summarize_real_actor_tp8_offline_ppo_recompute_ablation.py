#!/usr/bin/env python3
"""Re-compare and summarize existing five-step TP8 offline replay snapshots without rerunning NPU work."""
import argparse
import json
import tempfile
from pathlib import Path

import torch

from compare_real_actor_tp2cp4_offline_ppo import compare


def load_step(root: Path, mode: str, step: int):
    found = {}
    for path in sorted(root.rglob(f"{mode}_rank*.pt")):
        try:
            item = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if item.get("mode") != mode or item.get("step_index") != step:
            continue
        rank = int(item["rank"])
        found.setdefault(rank, (path, item))
    if sorted(found) != list(range(8)):
        raise RuntimeError(
            f"{root}: expected ranks 0..7 for mode={mode} step={step}; "
            f"found={sorted(found)}"
        )
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for rec in (0, 1):
        mode_steps = {"full": [], "tpr": []}
        peaks = {"full": [], "tpr": []}
        for step in range(1, 6):
            loaded = {
                mode: load_step(root / f"recompute_{rec}", mode, step)
                for mode in ("full", "tpr")
            }
            with tempfile.TemporaryDirectory(
                prefix=f"compare_rec{rec}_step{step:02d}_", dir=logs
            ) as tmp:
                tmpdir = Path(tmp)
                for mode in ("full", "tpr"):
                    for rank, (source, _) in loaded[mode].items():
                        (tmpdir / f"{mode}_rank{rank}.pt").symlink_to(source)
                print(
                    f"TPR_RECOMPARE step={step}/5 recompute={rec} "
                    f"snapshot_dir={tmpdir}",
                    flush=True,
                )
                compare(
                    tmpdir,
                    diagnostic_only=True,
                    require_recompute=bool(rec),
                    sequential_replay=True,
                )

            for mode in ("full", "tpr"):
                rows = [loaded[mode][rank][1] for rank in range(8)]
                pure_fb = max(
                    max(
                        0.0,
                        float(x["time_forward_backward_finalize_s"])
                        - float(x["time_finalize_s"]),
                    )
                    for x in rows
                )
                fbfin = max(
                    float(x["time_forward_backward_finalize_s"]) for x in rows
                )
                side = max(
                    float(x["time_forward_backward_finalize_s"])
                    + float(x["time_sgd_s"])
                    for x in rows
                )
                mode_steps[mode].append(
                    {
                        "fwd_bwd": pure_fb,
                        "fwd_bwd_finalize": fbfin,
                        "training_side_diag_sgd": side,
                    }
                )
                peaks[mode].extend(
                    float(x["peak_allocated_bytes"]) / 2**30 for x in rows
                )

        sums = {
            mode: {
                key: sum(step_result[key] for step_result in mode_steps[mode])
                for key in mode_steps[mode][0]
            }
            for mode in ("full", "tpr")
        }
        all_summaries[rec] = sums
        full, tpr = sums["full"], sums["tpr"]
        print(
            f"TPR_5STEP_SUMMARY recompute={rec} "
            f"full={json.dumps(full, sort_keys=True)} "
            f"tpr={json.dumps(tpr, sort_keys=True)} "
            f"pure_fb_speedup={full['fwd_bwd']/max(tpr['fwd_bwd'], 1e-12):.4f}x "
            f"training_side_diag_sgd_speedup="
            f"{full['training_side_diag_sgd']/max(tpr['training_side_diag_sgd'], 1e-12):.4f}x "
            f"peak_alloc_gib_full={max(peaks['full']):.3f} "
            f"peak_alloc_gib_tpr={max(peaks['tpr']):.3f} "
            f"memory_saved_gib={max(peaks['full'])-max(peaks['tpr']):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
