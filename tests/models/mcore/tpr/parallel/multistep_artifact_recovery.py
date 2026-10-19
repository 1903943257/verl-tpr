"""Safe file-only preflight for TP2xCP4 offline PPO optimizer replays.

Snapshot files are written AFTER the FP32 Master AdamW update. Gradient
directories are written BEFORE it, and are not optimizer checkpoints.
Following an interrupted process, the only safe replay is from the original
initial weights. Preserve all incomplete files by atomic rename into a
quarantine subdirectory; never modify a completed opposite mode.

This module has no torch/Ascend dependencies so it can be tested on CPU.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import uuid


MODES = ("full", "tpr")
MILESTONES = (1, 2, 5, 10, 20, 30, 40, 50)


def expected_snapshots(steps: int, *, has_manifest: bool) -> tuple[int, ...]:
    if not 1 <= steps <= 1000:
        raise ValueError("Replay steps must be within 1..1000")
    # Mirror test_real_actor_tp2cp4_multistep_npu.py EXACTLY. The fixed
    # capture test does not save a last-step snapshot if non-milestone.
    return (tuple(range(1, steps + 1)) if has_manifest else
            tuple(step for step in MILESTONES if step <= steps))


def _is_owned_artifact(name: str, mode: str) -> bool:
    if re.fullmatch(rf"{mode}_step[1-9]\d*_rank[0-7]\.pt", name):
        return True
    if re.fullmatch(rf"{mode}_step[1-9]\d*_tp[01]_full_grads", name):
        return True
    if name == f"{mode}_50step.log":
        return True
    return False


def reconcile(root: Path, *, mode: str, steps: int, has_manifest: bool,
              full_grad_steps: set[int]) -> dict:
    if mode not in MODES:
        raise ValueError(f"Unsupported PPO mode: {mode}")
    checkpoints = expected_snapshots(steps, has_manifest=has_manifest)
    if not full_grad_steps.issubset(set(checkpoints)):
        raise ValueError("Full gradient steps must also be saved checkpoint steps")
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(root)
    root.mkdir(parents=True, exist_ok=True)

    all_outputs = [
        p for p in root.iterdir()
        if _is_owned_artifact(p.name, mode)
    ]
    snapshot_names = {
        f"{mode}_step{step}_rank{rank}.pt"
        for step in checkpoints for rank in range(8)
    }
    present_names = {p.name for p in all_outputs if re.fullmatch(
        rf"{mode}_step[1-9]\d*_rank[0-7]\.pt", p.name
    )}
    saved = len(present_names & snapshot_names)
    done = saved == len(snapshot_names) and present_names == snapshot_names
    # If a previous run completed without the requested full-gradient
    # witness, it cannot be enriched without rerunning its optimizer path.
    needed_grad_paths = [
        root / f"{mode}_step{step}_tp{tp}_full_grads"
        for step in sorted(full_grad_steps) for tp in (0, 1)
    ]
    missing_grad = [
        str(p) for p in needed_grad_paths if
        not p.is_dir() or not (p / "manifest.pt").is_file()
    ]
    if done and missing_grad:
        raise RuntimeError(
            f"Completed {mode} PPO snapshots lack requested full-grad manifests: "
            f"{missing_grad}. Rerun in a fresh directory with full-grad enabled."
        )
    if done:
        return {"state": "reuse", "mode": mode,
                "snapshots": saved, "archive": None}
    if not all_outputs:
        return {"state": "run", "mode": mode,
                "snapshots": 0, "archive": None}

    # Quarantine only the selected mode's exact artifacts; no recursive
    # rm, no wildcard delete, no writes to the successful opposite mode.
    # Rename on the same filesystem; keep the original files available
    # for investigation rather than pretending stale gradients are state.
    archive = root / "_interrupted_replays" / (
        f"{mode}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    for source in sorted(all_outputs, key=lambda p: p.name):
        target = archive / source.name
        source.rename(target)
        moved.append(source.name)
    return {
        "state": "replay", "mode": mode, "snapshots": saved,
        "archive": str(archive), "moved": moved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--full-grad-steps", default="")
    args = parser.parse_args()
    full_grads = {int(x) for x in args.full_grad_steps.split(",") if x}
    result = reconcile(args.root, mode=args.mode, steps=args.steps,
                       has_manifest=args.manifest,
                       full_grad_steps=full_grads)
    print(
        f"TPR_FIXED_PPO_RECOVERY mode={args.mode} state={result['state']} "
        f"snapshots={result['snapshots']} archive={result['archive']}",
        flush=True,
    )
    print(result["state"], flush=True)


if __name__ == "__main__":
    main()
