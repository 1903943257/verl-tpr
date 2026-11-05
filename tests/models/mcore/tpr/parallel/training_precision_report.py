# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Aggregate diagnostic files after native training, with no NPU math."""

import json
import math
import os
from pathlib import Path

import numpy as np

from .training_precision_contract import MODES, SUFFIX
from .training_precision_io import index_reference, merge_stats, validate_owned_coverage, write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _logprobs(reports):
    values = {}
    for report in reports:
        for row, offset, value in report["logprobs"]:
            if (row, offset) in values or not math.isfinite(value):
                raise AssertionError("Duplicate or nonfinite response logprob")
            values[row, offset] = value
    expected = {(row, offset) for row in range(2) for offset in range(SUFFIX)}
    if set(values) != expected:
        raise AssertionError("Response logprob coverage is incomplete")
    return [[values[row, offset] for offset in range(SUFFIX)] for row in range(2)]


def _token_comparison(reference, candidate):
    a, b = (np.asarray(value["logprobs"], dtype=np.float64) for value in (reference, candidate))
    if a.shape != (2, SUFFIX) or a.shape != b.shape:
        raise AssertionError("Response metric shape mismatch")
    delta = b - a
    return dict(
        loss_abs_gap=abs(reference["loss"] - candidate["loss"]),
        logprob_relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(a)),
        logprob_max_abs=float(np.abs(delta).max()),
        logprob_mean_abs=float(np.abs(delta).mean()),
        cross_entropy_abs_gap=float(abs(a.mean() - b.mean())),
    )


def _matched(reference, candidate):
    for key in (
        "input_sha256",
        "initial_weights_sha256",
        "prefix_length",
        "suffix_length",
        "objective",
        "controls",
        "packages",
    ):
        if reference[key] != candidate[key]:
            raise AssertionError(f"Comparing unmatched experiments: {key}")
    keys = ("main_grad_dtype", "gradient_accumulation_fusion", "grad_reduce_in_fp32", "use_distributed_optimizer")
    if any(reference["precision"][key] != candidate["precision"][key] for key in keys):
        raise AssertionError("The four modes used different native gradient settings")


def complete_mode(root, mode, cp, layout):
    root = Path(root)
    reports = [_read(root / mode / f"rank{rank}.json") for rank in range(cp)]
    for report in reports:
        _matched(reports[0], report)
        if not math.isclose(report["loss"], reports[0]["loss"], rel_tol=1e-6, abs_tol=1e-7):
            raise AssertionError("CP ranks disagree on the reported global loss")
    gradient_reports = [report["gradients"] for report in reports]
    validate_owned_coverage(layout, gradient_reports)
    if mode.startswith("base"):
        index_reference(root / mode / "grads", gradient_reports)
    result = {key: value for key, value in reports[0].items() if key not in ("rank", "gradients", "logprobs", "calls")}
    result["logprobs"] = _logprobs(reports)
    result["next_token_cross_entropy"] = -float(np.mean(result["logprobs"]))
    result["rank_calls"] = [report["calls"] for report in reports]
    result["rank_backward"] = [report["backward"] for report in reports]
    result["peak_allocated_bytes"] = max(report["peak_allocated_bytes"] for report in reports)
    result["comparisons"] = {}
    elements = sum(parameter["elements"] for parameter in layout)
    for name in reports[0]["gradients"]["stats"]:
        reference = _read(root / name / "result.json")
        _matched(reference, result)
        result["comparisons"][name] = {
            **merge_stats([report["gradients"]["stats"][name] for report in reports], elements),
            **_token_comparison(reference, result),
        }
    write_json(root / mode / "result.json", result)
    print(
        f"TRAIN_PRECISION_DONE mode={mode} loss={result['loss']:.9g} "
        f"ce={result['next_token_cross_entropy']:.9g} peak_GiB={result['peak_allocated_bytes'] / 2**30:.2f}",
        flush=True,
    )
    for reference, pair in result["comparisons"].items():
        print(
            f"TRAIN_PRECISION_COMPARE {reference}_vs_{mode}: "
            f"grad_rel_l2={pair['gradient_relative_l2']:.9g} cosine={pair['gradient_cosine']:.12g} "
            f"loss_gap={pair['loss_abs_gap']:.9g} logprob_max_abs={pair['logprob_max_abs']:.9g}",
            flush=True,
        )


def finish(root, *, keep_gradients):
    root = Path(root).resolve()
    results = {mode: _read(root / mode / "result.json") for mode in MODES}
    pairs = {
        f"{reference}_vs_{mode}": comparison
        for mode, result in results.items()
        for reference, comparison in result["comparisons"].items()
    }
    baseline = pairs["base_cp1_vs_base_cp2"]["gradient_relative_l2"]
    ratios = {
        mode: pairs[f"base_cp1_vs_{mode}"]["gradient_relative_l2"] / baseline if baseline else None
        for mode in ("tpr_cp1", "tpr_cp2")
    }
    report = dict(
        scope="Real VERL/Megatron/MindSpeed training F/B/finalize; two shared P1024/S1024 trajectories",
        optimizer_step=False,
        loss="unmodified VERL vanilla token-mean PPO; Base CP1 old logprobs frozen for all groups",
        pairs=pairs,
        error_ratio_to_native_cp2=ratios,
        precision={mode: result["precision"] for mode, result in results.items()},
        temporary_reference_gradients_retained=keep_gradients,
    )
    # Delete only files listed by this successful experiment's two manifests.
    # Never recursively remove an output directory or follow a path outside it.
    if not keep_gradients:
        paths = []
        for mode in ("base_cp1", "base_cp2"):
            folder = (root / mode / "grads").resolve()
            index = _read(folder / "index.json")
            for blocks in index["parts"]:
                for block in blocks:
                    path = (folder / block["file"]).resolve()
                    if not path.is_relative_to(folder) or path.suffix != ".pt" or not path.is_file():
                        raise ValueError("Refuse unsafe gradient cleanup")
                    paths.append(path)
        for path in paths:
            path.unlink()
    write_json(root / "comparison.json", report)
    print(f"TRAIN_PRECISION_ALL_DONE ratios={ratios} report={root / 'comparison.json'}", flush=True)
    return report


if __name__ == "__main__":
    finish(os.environ["TPR_TRAIN_OUTPUT"], keep_gradients=os.getenv("TPR_TRAIN_KEEP_GRADS", "0") == "1")
