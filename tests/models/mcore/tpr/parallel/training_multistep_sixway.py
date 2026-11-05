# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Six routes compared on the historical coordinates, with no old training reruns."""

import os
from pathlib import Path

from .training_multistep_repeat import check_source
from .training_multistep_report import load, matched, probe_metrics, read, update_metrics
from .training_multistep_runtime import BASE_CP1_REPEAT, CHECKPOINTS, TPR_CP2_QKVMERGE
from .training_precision_contract import MODES
from .training_precision_io import add_stats, empty_stats, write_json


def preflight(source, output, *, packages=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Six-route supplemental output must be new")
    context = check_source(source, output=output, packages=packages)
    check_source(source, output=output, packages=packages, mode=TPR_CP2_QKVMERGE)
    for mode in MODES:
        result = read(source / mode / "result.json")
        matched(context["baseline"], result)
        if result["mode"] != mode:
            raise AssertionError("Historical route does not match its output directory")
        samples, probes = load(source / mode / "samples.pt"), load(source / mode / "probes.pt")
        if set(samples) != set(CHECKPOINTS[1:]) or set(probes) != set(CHECKPOINTS):
            raise AssertionError("Historical route is missing small observer checkpoints")
    print(f"TRAIN10_SIX_PREFLIGHT source={source} supplemental_output={output}", flush=True)


def finish(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    folders = {mode: source / mode for mode in MODES}
    folders[BASE_CP1_REPEAT] = output / "base_cp1_repeat_run" / BASE_CP1_REPEAT
    folders[TPR_CP2_QKVMERGE] = output / "qkvmerge_run" / TPR_CP2_QKVMERGE
    results = {mode: read(folder / "result.json") for mode, folder in folders.items()}
    baseline = results["base_cp1"]
    for result in results.values():
        matched(baseline, result)
    samples = {mode: load(folder / "samples.pt") for mode, folder in folders.items()}
    probes = {mode: load(folder / "probes.pt") for mode, folder in folders.items()}
    comparisons = {}
    for mode in folders:
        trend = []
        for step in CHECKPOINTS[1:]:
            stats = empty_stats()
            add_stats(stats, samples["base_cp1"][step], samples[mode][step])
            metrics = update_metrics([stats], samples["base_cp1"][step].numel())
            trend.append(
                dict(step=step, sampled_update=metrics, **probe_metrics(probes["base_cp1"], probes[mode], step))
            )
        comparisons[mode] = trend
    denominator = comparisons["base_cp2"][-1]["sampled_update"]["update_relative_l2"]
    ratios = {
        mode: trend[-1]["sampled_update"]["update_relative_l2"] / denominator if denominator else None
        for mode, trend in comparisons.items()
    }
    repeat = read(output / "base_cp1_repeat_run/comparison.json")
    merge = read(output / "qkvmerge_run/comparison.json")
    if repeat["mode"] != BASE_CP1_REPEAT or merge["mode"] != TPR_CP2_QKVMERGE:
        raise AssertionError("Supplement comparisons do not match the requested modes")
    report = dict(
        scope="Four historical routes plus one Base CP1 repeat and one TPR CP2 QKV merge run",
        controls=baseline["controls"],
        input_sha256=baseline["input_sha256"],
        sampled_elements=baseline["sampled_elements"],
        model_elements=baseline["model_elements"],
        metric_scope="Fixed global coordinates from every named parameter; six-way table is sampled, not all-element",
        comparisons_to_base_cp1=comparisons,
        final_sampled_error_ratio_to_native_cp2=ratios,
        repeatability=repeat["trend"],
        tpr_cp2_merge_off_vs_on=merge["trend"],
        original_all_parameter_report=str(source / "comparison.json"),
        evaluation="Native whole-sequence inference at route CP size; cross-CP outputs include layout effects",
    )
    lines = [
        "# Six-route ten-step comparison",
        "",
        "Fixed-coordinate master-update samples at step 10; the original four-route full metrics remain separate.",
        "",
        "| Route | Sampled update rel L2 vs Base CP1 | Cosine | Ratio to native CP | Probe mean abs |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, trend in comparisons.items():
        last = trend[-1]
        metrics = last["sampled_update"]
        lines.append(
            f"| {mode} | {metrics['update_relative_l2']:.9g} | {metrics['update_cosine']:.12g} | "
            f"{ratios[mode] if ratios[mode] is not None else 'undefined'} | {last['logprob_mean_abs']:.9g} |"
        )
    lines += ["", "| Check | Step | Sampled update rel L2 | Probe mean abs |", "|---|---:|---:|---:|"]
    for name, values in (("Base CP1 repeat", repeat["trend"]), ("TPR CP2 merge OFF/ON", merge["trend"])):
        for row in values:
            lines.append(
                f"| {name} | {row['step']} | {row['sampled_update']['update_relative_l2']:.9g} | "
                f"{row['logprob_mean_abs']:.9g} |"
            )
    lines += [
        "",
        "The two supplemental jobs reuse the original frozen fixture, initial weights and native precision.",
        "Only the QKV merge job enables Prefix FULL/QUERY/PADDED=1/1/1; both Ring F/B paths are audited.",
        "P1024/S1024 is aligned: this does not exercise ragged physical padding.",
        "Same-CP probe comparisons isolate weight drift more directly than cross-CP comparisons.",
    ]
    write_json(output / "comparison.json", report)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TRAIN10_SIX_DONE summary={output / 'summary.md'}", flush=True)
    return report


if __name__ == "__main__":
    if os.getenv("TPR_TRAIN10_SIX_PREFLIGHT", "0") == "1":
        from .training_path_precision_npu import _versions

        preflight(os.environ["TPR_TRAIN10_REPEAT_SOURCE"], os.environ["TPR_TRAIN10_OUTPUT"], packages=_versions())
    else:
        finish(os.environ["TPR_TRAIN10_REPEAT_SOURCE"], os.environ["TPR_TRAIN10_OUTPUT"])
