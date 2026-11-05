# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Resume completed routes; an incomplete route restarts from its initial state."""

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .training_multistep_npu import controls_from_environment
from .training_multistep_report import load, matched, read
from .training_multistep_runtime import STEPS, fixture_digest
from .training_precision_contract import MODES
from .training_precision_resume import _check_reference


def prepare_resume(root):
    root = Path(root).resolve()
    fixture = load(root / "actor_fixture.pt")
    if fixture["controls"] != controls_from_environment() or fixture["sha256"] != fixture_digest(fixture):
        raise AssertionError("Resume requires the original ten-step fixture and controls")
    completed, pending, failed = [], [], []
    baseline = read(root / "base_cp1/result.json")
    if fixture["initial_weights_sha256"] != baseline["initial_weights_sha256"]:
        raise AssertionError("Resume fixture initial weights changed")
    if baseline["input_sha256"] != fixture["sha256"]:
        raise AssertionError("Resume fixture differs from the completed training route")
    for mode in MODES:
        folder = root / mode
        if folder.is_symlink() or folder.resolve().parent != root:
            raise ValueError("Unsafe mode archive path")
        if (folder / "result.json").is_file():
            if pending:
                raise AssertionError("Completed modes must form a contiguous prefix")
            result = read(folder / "result.json")
            matched(baseline, result)
            if (
                result["mode"] != mode
                or result["cp"] != (2 if mode.endswith("cp2") else 1)
                or [row["step"] for row in result["steps"]] != list(range(1, STEPS + 1))
            ):
                raise AssertionError("Saved route is not a completed ten-update result")
            for name in ("samples.pt", "probes.pt"):
                if not (folder / name).is_file():
                    raise AssertionError("Completed route is missing small observer artifacts")
            completed.append(mode)
        else:
            pending.append(mode)
            if folder.exists():
                if not folder.is_dir():
                    raise ValueError("Expected a mode directory")
                failed.append(folder)
    if pending or not (root / "comparison.json").is_file():
        for mode in completed:
            if mode.startswith("base"):
                _check_reference(root / mode / "updates")
    for folder in failed:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = root / f"{folder.name}.failed-{stamp}-{uuid4().hex[:8]}"
        if archive.resolve().parent != root or archive.exists():
            raise ValueError("Unsafe archive target")
        folder.rename(archive)
        print(f"TRAIN10_ARCHIVED {folder.name} -> {archive.name}", flush=True)
    print(f"TRAIN10_RESUME completed={completed} pending={pending}", flush=True)
    return completed, pending


if __name__ == "__main__":
    prepare_resume(os.environ["TPR_TRAIN10_OUTPUT"])
