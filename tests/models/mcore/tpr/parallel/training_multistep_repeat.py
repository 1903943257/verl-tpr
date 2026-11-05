# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Read-only historical controls and comparisons for two supplemental jobs."""

import math
import os
from pathlib import Path

import torch

from .training_multistep_io import merge_samples, sample_indices
from .training_multistep_report import load, matched, probe_metrics, read, update_metrics
from .training_multistep_runtime import (
    BASE_CP1_REPEAT,
    CHECKPOINTS,
    STEPS,
    TPR_CP2_QKVMERGE,
    fixture_digest,
    mode_topology,
)
from .training_precision_io import add_stats, empty_stats, validate_owned_coverage, write_json
from .training_precision_resume import _check_reference

ENVIRONMENT_CONTROLS = {
    "seed": "TPR_TRAIN_SEED",
    "learning_rate": "TPR_TRAIN10_LR",
    "vanilla_mbridge": "TPR_TRAIN_VANILLA_MBRIDGE",
    "gradient_accumulation_fusion": "TPR_TRAIN_GRAD_ACCUM_FUSION",
    "delayed_dkv_fp32": "TPR_EXPERIMENT_DELAYED_DKV_FP32",
    "root_native_ring": "TPR_ROOT_NATIVE_RING",
    "leaf_batch_size": "TPR_LEAF_BATCH_SIZE",
    "selected_logits_fp32": "TPR_SELECTED_LOGITS_FP32",
}
PRECISION_KEYS = (
    "parameter_dtype",
    "main_grad_dtype",
    "gradient_accumulation_fusion",
    "grad_reduce_in_fp32",
    "use_distributed_optimizer",
    "topology_tp_pp_cp_dp",
)


def restore_controls(controls):
    for key, variable in ENVIRONMENT_CONTROLS.items():
        value = controls[key]
        os.environ[variable] = str(int(value)) if isinstance(value, bool) else str(value)


def check_source(source, *, output=None, packages=None, mode=BASE_CP1_REPEAT):
    """Read-only preflight; missing historical full blocks permit only sampled comparison."""
    source = Path(source).resolve()
    if output is not None:
        output = Path(output).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Repeat output must be separate from the historical experiment")
    baseline = read(source / "base_cp1/result.json")
    fixture = load(source / "actor_fixture.pt")
    if (
        baseline["mode"] != "base_cp1"
        or baseline["cp"] != 1
        or [step["step"] for step in baseline["steps"]] != list(range(1, STEPS + 1))
        or baseline["controls"] != fixture["controls"]
        or baseline["initial_weights_sha256"] != fixture["initial_weights_sha256"]
        or baseline["input_sha256"] != fixture["sha256"]
        or fixture_digest(fixture) != fixture["sha256"]
        or len(fixture["batches"]) != STEPS
    ):
        raise AssertionError("Require the original completed ten-update Base CP1 and unchanged actor fixture")
    if packages is not None and packages != baseline["packages"]:
        raise AssertionError("Repeat must use the original package versions")
    audit = read(source / "base_cp1/runtime_audit_rank0.json")
    if audit["status"] != "ten_updates_validated" or audit["optimizer"]["completed_updates"] != STEPS:
        raise AssertionError("Historical Base CP1 native optimizer updates were not validated")
    samples, probes = load(source / "base_cp1/samples.pt"), load(source / "base_cp1/probes.pt")
    if set(samples) != set(CHECKPOINTS[1:]) or set(probes) != set(CHECKPOINTS):
        raise AssertionError("Historical sample/probe checkpoints are incomplete")
    for step in CHECKPOINTS:
        probe_metrics(probes, probes, step)
    folder = source / "base_cp1/updates"
    index = read(folder / "index.json")
    layout = index["layout"]
    expected_samples = sum(sample_indices(parameter["elements"]).numel() for parameter in layout)
    if sum(parameter["elements"] for parameter in layout) != baseline["model_elements"]:
        raise AssertionError("Historical parameter layout differs from the result")
    if any(sample.ndim != 1 or sample.numel() != expected_samples for sample in samples.values()):
        raise AssertionError("Historical master samples do not match the current coordinates")
    paths = []
    if len(index["parts"]) != len(layout):
        raise AssertionError("Invalid historical update manifest")
    for parameter, blocks in zip(layout, index["parts"], strict=True):
        cursor = 0
        for block in blocks:
            path = (folder / block["file"]).resolve()
            if (
                not path.is_relative_to(folder.resolve())
                or path.suffix != ".pt"
                or block["start"] != cursor
                or block["end"] <= cursor
            ):
                raise ValueError("Invalid historical update block path or coverage")
            paths.append(path)
            cursor = block["end"]
        if cursor != parameter["elements"]:
            raise AssertionError("Historical update manifest has missing elements")
    full_available = bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)
    if full_available:
        _check_reference(folder)
    reference_mode = "tpr_cp2" if mode == TPR_CP2_QKVMERGE else "base_cp1"
    if mode not in (BASE_CP1_REPEAT, TPR_CP2_QKVMERGE):
        raise ValueError("Unknown historical supplement mode")
    if mode == TPR_CP2_QKVMERGE:
        original = read(source / reference_mode / "result.json")
        matched(baseline, original)
        if original["mode"] != reference_mode or original["cp"] != 2:
            raise AssertionError("QKV merge requires the historical TPR CP2 control")
        if [row["step"] for row in original["steps"]] != list(range(1, STEPS + 1)):
            raise AssertionError("Historical TPR CP2 must contain ten completed updates")
        for rank in range(2):
            audit = read(source / reference_mode / f"runtime_audit_rank{rank}.json")
            if audit["status"] != "ten_updates_validated" or audit["optimizer"]["completed_updates"] != STEPS:
                raise AssertionError("Historical TPR CP2 native updates were not validated")
            if any(value != "0" for value in audit.get("ring_merge_settings", {}).values()):
                raise AssertionError("Historical TPR CP2 control must have merge disabled")
        route_samples = load(source / reference_mode / "samples.pt")
        route_probes = load(source / reference_mode / "probes.pt")
        if set(route_samples) != set(CHECKPOINTS[1:]) or set(route_probes) != set(CHECKPOINTS):
            raise AssertionError("Historical TPR CP2 sample/probe checkpoints are incomplete")
        if any(value.ndim != 1 or value.numel() != expected_samples for value in route_samples.values()):
            raise AssertionError("Historical TPR CP2 sample coordinates differ")
        for step in CHECKPOINTS:
            probe_metrics(route_probes, route_probes, step)
        baseline = original
        # Historical TPR update vectors were never saved, only sampled coordinates.
        full_available = False
    return dict(
        source=source,
        baseline=baseline,
        precision=audit["precision"],
        layout=layout,
        full_update_available=full_available,
        mode=mode,
        reference_mode=reference_mode,
    )


def complete_repeat(root, context):
    root = Path(root)
    mode, reference_mode = context["mode"], context["reference_mode"]
    cp, _ = mode_topology(mode)
    folder = root / mode
    source, baseline = context["source"], context["baseline"]
    reports = [read(folder / f"rank{rank}.json") for rank in range(cp)]
    report = reports[0]
    if mode == TPR_CP2_QKVMERGE:
        for rank in range(cp):
            audit = read(folder / f"runtime_audit_rank{rank}.json")
            counts = audit.get("qkv_merge_execution", {}).get("joint_query_kv_calls", {})
            flags = audit.get("ring_merge_settings", {})
            if (
                audit.get("status") != "ten_updates_validated"
                or not counts.get("forward")
                or not counts.get("backward")
                or any(flags.get(f"TPR_RING_COALESCE_PREFIX_{name}") != "1" for name in ("FULL", "QUERY", "PADDED"))
            ):
                raise AssertionError("QKV merge requested without validated joint forward/backward execution")
    for other in reports:
        matched(report, other)
        if other["mode"] != mode or other["cp"] != cp:
            raise AssertionError("Supplement rank topology differs from its mode")
        for old, new in zip(report["steps"], other["steps"], strict=True):
            if any(not math.isclose(old[key], new[key], rel_tol=1e-6, abs_tol=1e-7) for key in ("loss", "grad_norm")):
                raise AssertionError("CP ranks disagree on supplement loss/gradient norm")
    matched(baseline, report)
    if report["layout"] != context["layout"]:
        raise AssertionError("Repeat parameter layout differs from historical Base CP1")
    if [step["step"] for step in report["steps"]] != list(range(1, STEPS + 1)):
        raise AssertionError("Repeat did not complete ten native updates")
    ranked = [load(folder / f"samples_rank{rank}.pt") for rank in range(cp)]
    if any(set(values) != set(CHECKPOINTS[1:]) for values in ranked):
        raise AssertionError("Repeat sample checkpoints are incomplete")
    samples = {step: merge_samples(report["layout"], [values[step] for values in ranked]) for step in CHECKPOINTS[1:]}
    torch.save(samples, str(folder / "samples.pt"))
    original = load(source / reference_mode / "samples.pt")
    base_samples = load(source / "base_cp1/samples.pt")
    reference_probes, probes = load(source / reference_mode / "probes.pt"), load(folder / "probes.pt")
    if set(probes) != set(CHECKPOINTS):
        raise AssertionError("Repeat probe checkpoints are incomplete")
    original_summary = read(source / "comparison.json") if (source / "comparison.json").is_file() else None
    native_trend = {}
    if original_summary is not None:
        if (
            original_summary["controls"] != baseline["controls"]
            or original_summary["model_elements"] != baseline["model_elements"]
        ):
            raise AssertionError("Historical native CP summary does not match Base CP1")
        native_pair = original_summary["pairs"]["base_cp1_vs_base_cp2"]
        native_trend = {row["step"]: row["sampled_update_relative_l2"] for row in native_pair["trend"]}
    trend = []
    for step in CHECKPOINTS[1:]:
        stats = empty_stats()
        add_stats(stats, original[step], samples[step])
        metrics = update_metrics([stats], samples[step].numel())
        native = native_trend.get(step)
        native_difference_norm = native * float(base_samples[step].double().norm()) if native else None
        ratio = math.sqrt(stats["difference_sq"]) / native_difference_norm if native_difference_norm else None
        trend.append(
            dict(
                step=step,
                sampled_update=metrics,
                **probe_metrics(reference_probes, probes, step),
                native_cp_sampled_update_relative_l2=native,
                repeat_to_native_cp_sampled_ratio=ratio,
            )
        )
    steps = []
    for old, new in zip(baseline["steps"], report["steps"], strict=True):
        if old["learning_rates_used"] != new["learning_rates_used"]:
            raise AssertionError("Historical and repeated Base CP1 used different learning rates")
        metrics = {
            key: abs(value - old.get("native_metrics", {})[key])
            for key, value in new.get("native_metrics", {}).items()
            if key in old.get("native_metrics", {})
        }
        steps.append(
            dict(
                step=new["step"],
                reference_loss=old["loss"],
                repeat_loss=new["loss"],
                loss_abs_gap=abs(new["loss"] - old["loss"]),
                grad_norm_abs_gap=abs(new["grad_norm"] - old["grad_norm"]),
                native_metric_abs_gaps=metrics,
            )
        )
    full = None
    if context["full_update_available"]:
        update = report["full_update"]
        validate_owned_coverage(report["layout"], [update])
        full = update_metrics([update["stats"]["base_cp1"]], baseline["model_elements"])
    elif report["full_update"] is not None:
        raise AssertionError("Unexpected full update report without a historical reference")
    result = dict(
        scope="Historical control supplement: identical initial weights, fresh native optimizer, frozen PPO inputs",
        mode=mode,
        reference_mode=reference_mode,
        qkvmerge_enabled=mode == TPR_CP2_QKVMERGE,
        source=str(source),
        controls=report["controls"],
        input_sha256=report["input_sha256"],
        initial_weights_sha256=report["initial_weights_sha256"],
        packages=report["packages"],
        full_update_comparison=full,
        full_update_unavailable_reason=None if full else "Historical vectors were removed or were never saved",
        sampled_elements=samples[1].numel(),
        model_elements=baseline["model_elements"],
        initial_probe=probe_metrics(reference_probes, probes, 0),
        trend=trend,
        step_differences=steps,
        interpretation="Supplement/native CP ratios compare difference norms at identical sampled coordinates",
    )
    lines = [
        f"# {reference_mode} vs {mode}",
        "",
        "Same initial weights, fresh native optimizer, frozen historical inputs and labels.",
        "",
        "| Step | Sampled update rel L2 | Cosine | Repeat/native CP sampled ratio | Probe mean abs | Probe max abs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in trend:
        metrics = row["sampled_update"]
        ratio = row["repeat_to_native_cp_sampled_ratio"]
        lines.append(
            f"| {row['step']} | {metrics['update_relative_l2']:.9g} | {metrics['update_cosine']:.12g} | "
            f"{ratio if ratio is not None else 'unavailable'} | {row['logprob_mean_abs']:.9g} | "
            f"{row['logprob_max_abs']:.9g} |"
        )
        marker = "TRAIN10_QKVMERGE_COMPARE" if mode == TPR_CP2_QKVMERGE else "TRAIN10_REPEAT_COMPARE"
        print(
            f"{marker} reference={reference_mode} step={row['step']} "
            f"sampled_update_rel_l2={metrics['update_relative_l2']:.9g} "
            f"cosine={metrics['update_cosine']:.12g} probe_mean_abs={row['logprob_mean_abs']:.9g} "
            f"repeat_to_native_cp_sampled_ratio={ratio}",
            flush=True,
        )
    lines += [
        "",
        f"Initial probe mean abs: {result['initial_probe']['logprob_mean_abs']:.9g}",
        f"Max per-step loss gap: {max(row['loss_abs_gap'] for row in steps):.9g}",
        f"Max per-step grad norm gap: {max(row['grad_norm_abs_gap'] for row in steps):.9g}",
    ]
    if full is not None:
        lines += [f"Final all-parameter update rel L2: {full['update_relative_l2']:.9g}"]
    else:
        lines += [
            "Full update comparison unavailable: historical vectors were removed or were never saved.",
            "Sampled metrics cover fixed coordinates in every parameter, not every parameter element.",
        ]
    write_json(folder / "result.json", {**report, "repeat_comparison": result})
    write_json(root / "comparison.json", result)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TRAIN10_SUPPLEMENT_DONE mode={mode} summary={root / 'summary.md'}", flush=True)
    return result


if __name__ == "__main__":
    from .training_path_precision_npu import _versions

    context = check_source(
        os.environ["TPR_TRAIN10_REPEAT_SOURCE"],
        output=os.environ["TPR_TRAIN10_OUTPUT"],
        packages=_versions(),
        mode=os.getenv("TPR_TRAIN_MODE", BASE_CP1_REPEAT),
    )
    if Path(os.environ["TPR_TRAIN10_OUTPUT"]).exists():
        raise FileExistsError("Repeat output must be new")
    print(
        f"TRAIN10_REPEAT_PREFLIGHT full_reference_available={context['full_update_available']} "
        f"source={context['source']}"
    )
