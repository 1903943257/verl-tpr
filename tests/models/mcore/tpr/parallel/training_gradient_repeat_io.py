# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Full-element, bounded-memory comparison of two native Base CP1 backwards."""

import math
import os
import shutil
from pathlib import Path

import numpy as np

from .training_multistep_repeat import PRECISION_KEYS, check_source
from .training_multistep_report import load, matched, probe_metrics, read
from .training_precision_io import (
    BLOCK,
    GradientReference,
    add_stats,
    empty_stats,
    merge_stats,
    owned_gradient_ranges,
    parameter_layout,
    validate_owned_coverage,
    write_json,
)


def sign_counts(reference, candidate, threshold):
    """Zero transitions are separate from strict positive/negative sign flips."""
    x, y = (value.numpy() for value in (reference, candidate))
    opposite = ((x > 0) & (y < 0)) | ((x < 0) & (y > 0))
    zero_transition = (x == 0) != (y == 0)
    small = np.abs(x) <= threshold
    return dict(
        elements=x.size,
        changed_elements=int(np.count_nonzero(x != y)),
        both_nonzero_elements=int(np.count_nonzero((x != 0) & (y != 0))),
        opposite_sign_elements=int(np.count_nonzero(opposite)),
        zero_transition_elements=int(np.count_nonzero(zero_transition)),
        small_reference_elements=int(np.count_nonzero(small)),
        small_reference_opposite_sign_elements=int(np.count_nonzero(opposite & small)),
        small_reference_zero_transition_elements=int(np.count_nonzero(zero_transition & small)),
    )


def compare_gradients(model, optimizer, reference_folder, threshold):
    layout = parameter_layout(model)
    reader = GradientReference(reference_folder, layout)
    parameters, owned = [], []
    for index, name, start, end, gradient in owned_gradient_ranges(model, optimizer):
        owned.append(dict(index=index, name=name, start=start, end=end))
        stats, counts = empty_stats(), {}
        for lo in range(start, end, BLOCK):
            hi = min(lo + BLOCK, end)
            candidate = gradient[lo:hi].to(device="cpu", copy=True).contiguous()
            reference = reader.read(index, lo, hi)
            add_stats(stats, reference, candidate)
            for key, value in sign_counts(reference, candidate, threshold).items():
                counts[key] = counts.get(key, 0) + value
        parameters.append(dict(name=name, stats=stats, counts=counts))
    report = dict(layout=layout, owned=owned, parameters=parameters)
    validate_owned_coverage(layout, [report])
    return report


def summarize_gradients(report, threshold):
    elements = sum(item["elements"] for item in report["layout"])
    metrics = merge_stats([item["stats"] for item in report["parameters"]], elements)
    counts = {}
    for item in report["parameters"]:
        for key, value in item["counts"].items():
            counts[key] = counts.get(key, 0) + value
    if counts["elements"] != elements:
        raise AssertionError("Sign comparison does not cover all parameter elements")
    counts.update(
        opposite_sign_fraction_all_elements=counts["opposite_sign_elements"] / elements,
        opposite_sign_fraction_both_nonzero=(
            counts["opposite_sign_elements"] / counts["both_nonzero_elements"]
            if counts["both_nonzero_elements"]
            else None
        ),
        zero_transition_fraction=counts["zero_transition_elements"] / elements,
        small_reference_abs_threshold=threshold,
    )
    return dict(**metrics, signs=counts)


def reference_paths(folder):
    """Validate every cleanup target before deleting any temporary block."""
    folder = Path(folder).resolve()
    paths = []
    for blocks in read(folder / "index.json")["parts"]:
        for block in blocks:
            path = (folder / block["file"]).resolve()
            if not path.is_relative_to(folder) or path.suffix != ".pt" or not path.is_file():
                raise ValueError("Unsafe or missing gradient reference block")
            paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate gradient reference cleanup target")
    return paths


def settings():
    threshold = float(os.getenv("TPR_GRAD_REPEAT_SMALL_THRESHOLD", "1e-6"))
    keep = os.getenv("TPR_GRAD_REPEAT_KEEP_GRADS", "0")
    if not math.isfinite(threshold) or threshold < 0 or keep not in ("0", "1"):
        raise ValueError("Require finite nonnegative small-gradient threshold and KEEP_GRADS=0 or 1")
    return threshold, keep == "1"


def preflight(source, output, packages):
    context = check_source(source, output=output, packages=packages)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Gradient repeat requires a new output directory")
    settings()
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    required = context["baseline"]["model_elements"] * 4 + 2**30
    if shutil.disk_usage(parent).free < required:
        raise OSError(f"Need at least {required / 2**30:.2f} GiB free for one FP32 gradient reference")
    return context


def finish(root, *, keep_gradients=False):
    root = Path(root).resolve()
    first, second = [read(root / run / "result.json") for run in ("first", "second")]
    matched(first, second)
    for key in PRECISION_KEYS:
        if first["precision"][key] != second["precision"][key]:
            raise AssertionError(f"Two Base CP1 runs have different precision/config: {key}")
    for key in ("execution_environment", "small_gradient_threshold", "native_norm_elements"):
        if first[key] != second[key]:
            raise AssertionError(f"Two Base CP1 runs differ: {key}")
    for report in (first, second):
        if report["optimizer_step"] or report["gradient_stage"] != "after_native_finalize_and_prepare_before_clip":
            raise AssertionError("Gradient check must stop before clipping/update")
        if (
            report["native_norm_elements"] != report["model_elements"]
            or report["precision"]["topology_tp_pp_cp_dp"] != [1] * 4
        ):
            raise AssertionError("Require matching all-parameter norm domain and TP=PP=CP=DP=1")
    for run, report in zip(("first", "second"), (first, second), strict=True):
        audit = read(root / run / "runtime_audit_rank0.json")
        if report["run"] != run or audit["status"] != "validated_before_clip_no_update" or audit["optimizer_step"]:
            raise AssertionError("Both independent native backwards must be validated before reporting")
    detail = read(root / "second/parameter_differences.json")
    validate_owned_coverage(detail["layout"], [detail])
    threshold = second["small_gradient_threshold"]
    gradients = summarize_gradients(detail, threshold)
    if gradients["elements"] != first["model_elements"]:
        raise AssertionError("Gradient comparison has incomplete model coverage")
    forwards = [load(root / run / "forwards.pt") for run in ("first", "second")]
    forward_metrics = {
        name: probe_metrics({0: forwards[0][name]}, {0: forwards[1][name]}, 0)
        for name in ("probe0", "training_log_probs")
    }
    paths = reference_paths(root / "first/grads")
    parameter_top = sorted(detail["parameters"], key=lambda item: item["stats"]["difference_sq"], reverse=True)[:10]
    result = dict(
        scope="Two independent Base CP1 first backwards; all FP32 main_grad elements before clipping/update",
        controls=first["controls"],
        input_sha256=first["input_sha256"],
        initial_weights_sha256=first["initial_weights_sha256"],
        gradient_stage=first["gradient_stage"],
        optimizer_step=False,
        gradients=gradients,
        loss_abs_gap=abs(first["loss"] - second["loss"]),
        forwards=forward_metrics,
        native_norms={
            run: report["native_norms"] for run, report in zip(("first", "second"), (first, second), strict=True)
        },
        native_norm_abs_gap=abs(first["native_norms"][0] - second["native_norms"][0]),
        native_minus_cpu_norm={
            "first": first["native_norms"][0] - gradients["reference_gradient_norm"],
            "second": second["native_norms"][0] - gradients["candidate_gradient_norm"],
        },
        top_parameters_by_squared_difference=parameter_top,
        temporary_gradients_retained=True,
    )
    lines = [
        "# Base CP1 first-backward repeat",
        "",
        f"All {gradients['elements']} FP32 main_grad elements; no clipping or optimizer update.",
        f"Gradient rel L2: {gradients['gradient_relative_l2']:.9g}",
        f"Gradient cosine: {gradients['gradient_cosine']:.12g}",
        f"Gradient max abs: {gradients['gradient_max_abs']:.9g}",
        f"Opposite-sign fraction (all elements): {gradients['signs']['opposite_sign_fraction_all_elements']:.9g}",
        f"Zero-transition fraction: {gradients['signs']['zero_transition_fraction']:.9g}",
        f"Loss gap: {result['loss_abs_gap']:.9g}",
        f"Training logprob max gap: {forward_metrics['training_log_probs']['logprob_max_abs']:.9g}",
        f"Initial probe max gap: {forward_metrics['probe0']['logprob_max_abs']:.9g}",
        f"Native norms, two reads each: {result['native_norms']}",
        f"CPU FP64 norms: {gradients['reference_gradient_norm']:.12g}, {gradients['candidate_gradient_norm']:.12g}",
        "",
        "Sign flips exclude zero transitions. Small-gradient threshold is in raw, unclipped gradient units.",
        "These two runs do not establish a stable noise floor or identify the originating operator.",
    ]
    write_json(root / "comparison.json", result)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not keep_gradients:
        for path in paths:
            path.unlink()
        result["temporary_gradients_retained"] = False
        write_json(root / "comparison.json", result)
    print(
        f"GRAD_REPEAT_DONE rel_l2={gradients['gradient_relative_l2']:.9g} "
        f"cosine={gradients['gradient_cosine']:.12g} max_abs={gradients['gradient_max_abs']:.9g} "
        f"sign_flip_fraction={gradients['signs']['opposite_sign_fraction_all_elements']:.9g} "
        f"report={root / 'comparison.json'}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    from .training_path_precision_npu import _versions

    root = Path(os.environ["TPR_GRAD_REPEAT_OUTPUT"])
    if os.getenv("TPR_GRAD_REPEAT_RUN") == "report":
        finish(root, keep_gradients=settings()[1])
    else:
        context = preflight(os.environ["TPR_TRAIN10_SOURCE"], root, _versions())
        print(f"GRAD_REPEAT_PREFLIGHT reference_GiB={context['baseline']['model_elements'] * 4 / 2**30:.2f}")
