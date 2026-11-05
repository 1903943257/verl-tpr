# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Validate completed controls and archive incomplete mode folders before retry."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from .training_precision_contract import MODES, PREFIX, SUFFIX, input_digest
from .training_precision_report import _matched


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_reference(folder):
    folder = folder.resolve()
    index = _read(folder / "index.json")
    if len(index["layout"]) != len(index["parts"]):
        raise AssertionError("Invalid saved reference layout")
    for parameter, blocks in zip(index["layout"], index["parts"], strict=True):
        cursor = 0
        for block in blocks:
            path = (folder / block["file"]).resolve()
            if (
                block["start"] != cursor
                or block["end"] <= cursor
                or not path.is_relative_to(folder)
                or path.suffix != ".pt"
                or not path.is_file()
                or path.stat().st_size == 0
            ):
                raise AssertionError("Incomplete or unsafe saved Base gradient reference")
            cursor = block["end"]
        if cursor != parameter["elements"]:
            raise AssertionError("Saved Base gradient reference has missing elements")


def prepare_resume(root):
    """Validate everything before mutation; preserve all files in failed modes."""
    root = Path(root).resolve()
    baseline = _read(root / "base_cp1/result.json")
    fixture = torch.load(str(root / "actor_fixture.pt"), map_location="cpu", weights_only=True)
    if baseline["mode"] != "base_cp1" or baseline["prefix_length"] != PREFIX or baseline["suffix_length"] != SUFFIX:
        raise AssertionError("Resume requires the completed P1024/S1024 Base CP1 control")
    if (
        fixture["weights"] != baseline["initial_weights_sha256"]
        or input_digest(fixture["tokens"], fixture["old_log_probs"], fixture["advantages"]) != baseline["input_sha256"]
    ):
        raise AssertionError("Saved actor fixture differs from completed Base CP1")
    controls = dict(
        seed=int(os.getenv("TPR_TRAIN_SEED", "261105")),
        vanilla_mbridge=os.getenv("TPR_TRAIN_VANILLA_MBRIDGE", "1") == "1",
        delayed_dkv_fp32=os.getenv("TPR_EXPERIMENT_DELAYED_DKV_FP32", "0"),
        root_native_ring=os.getenv("TPR_ROOT_NATIVE_RING", "0"),
        leaf_batch_size=os.getenv("TPR_LEAF_BATCH_SIZE", "1"),
        selected_logits_fp32=os.getenv("TPR_SELECTED_LOGITS_FP32", "0"),
    )
    fusion = os.getenv("TPR_TRAIN_GRAD_ACCUM_FUSION", "1") == "1"
    if controls != baseline["controls"] or fusion != baseline["precision"]["gradient_accumulation_fusion"]:
        raise AssertionError("Resume settings differ from Base CP1; keep the original seed/bridge/precision controls")
    completed, pending, failed = [], [], []
    for mode in MODES:
        folder = root / mode
        # A mode directory may contain partial exports. Never move symlinks or
        # a resolved path outside the explicitly selected experiment directory.
        if folder.is_symlink() or folder.resolve().parent != root:
            raise ValueError("Refuse to archive a mode path outside the experiment")
        if (folder / "result.json").is_file():
            if pending:
                raise AssertionError("Completed modes must form a contiguous prefix; inspect the output directory")
            result = _read(folder / "result.json")
            _matched(baseline, result)
            if result["mode"] != mode or result["cp"] != (2 if mode.endswith("cp2") else 1):
                raise AssertionError("Saved result does not match its mode/topology")
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
                _check_reference(root / mode / "grads")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for folder in failed:
        archive = root / f"{folder.name}.failed-{stamp}-{uuid4().hex[:8]}"
        if archive.resolve().parent != root or archive.exists():
            raise ValueError("Invalid failed-mode archive target")
        folder.rename(archive)
        print(f"TRAIN_PRECISION_ARCHIVED {folder.name} -> {archive.name}", flush=True)
    print(f"TRAIN_PRECISION_RESUME completed={completed} pending={pending}", flush=True)
    return completed, pending


if __name__ == "__main__":
    prepare_resume(os.environ["TPR_TRAIN_OUTPUT"])
