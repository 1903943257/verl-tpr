# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compact four-route drift reports; all FP64 observer arithmetic stays on CPU."""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .training_multistep_io import merge_samples
from .training_multistep_runtime import CHECKPOINTS, STEPS
from .training_precision_contract import MODES, SUFFIX
from .training_precision_io import (
    add_stats,
    empty_stats,
    index_reference,
    merge_stats,
    validate_owned_coverage,
    write_json,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(str(path), map_location="cpu", weights_only=True)


def matched(reference, candidate):
    for key in ("controls", "input_sha256", "initial_weights_sha256", "packages"):
        if reference[key] != candidate[key]:
            raise AssertionError(f"Unmatched ten-step routes: {key}")


def update_metrics(parts, elements):
    values = merge_stats(parts, elements)
    return {key.replace("gradient", "update"): value for key, value in values.items()}


def probe_metrics(reference, candidate, step):
    a, b = (values[step].double().numpy() for values in (reference, candidate))
    if a.shape != (2, SUFFIX) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Invalid held-out native probe coverage")
    delta = b - a
    change_delta = delta - (candidate[0].double().numpy() - reference[0].double().numpy())
    return dict(
        logprob_mean_abs=float(np.abs(delta).mean()),
        logprob_max_abs=float(np.abs(delta).max()),
        cross_entropy_abs_gap=float(abs(a.mean() - b.mean())),
        logprob_change_mean_abs=float(np.abs(change_delta).mean()),
    )


def complete_mode(root, mode, cp):
    root = Path(root)
    folder = root / mode
    reports = [read(folder / f"rank{rank}.json") for rank in range(cp)]
    for report in reports:
        matched(reports[0], report)
        if report["mode"] != mode or report["cp"] != cp:
            raise AssertionError("Rank report does not match its training route")
        if [step["step"] for step in report["steps"]] != list(range(1, STEPS + 1)):
            raise AssertionError("A route did not complete ten native updates")
    updates = [report["full_update"] for report in reports]
    layout = updates[0]["layout"]
    validate_owned_coverage(layout, updates)
    if mode.startswith("base"):
        index_reference(folder / "updates", updates)
    elements = sum(parameter["elements"] for parameter in layout)
    own = update_metrics([update["own_stats"] for update in updates], elements)
    rank_samples = [load(folder / f"samples_rank{rank}.pt") for rank in range(cp)]
    if any(set(sample) != set(CHECKPOINTS[1:]) for sample in rank_samples):
        raise AssertionError("Missing selected-step master samples")
    samples = {step: merge_samples(layout, [sample[step] for sample in rank_samples]) for step in CHECKPOINTS[1:]}
    torch.save(samples, str(folder / "samples.pt"))
    probes = load(folder / "probes.pt")
    if set(probes) != set(CHECKPOINTS):
        raise AssertionError("Missing held-out probe checkpoint")
    for step in CHECKPOINTS:
        probe_metrics(probes, probes, step)
    steps = []
    for index in range(STEPS):
        row = dict(reports[0]["steps"][index])
        for report in reports[1:]:
            other = report["steps"][index]
            for key in ("loss", "grad_norm"):
                if not math.isclose(row[key], other[key], rel_tol=1e-6, abs_tol=1e-7):
                    raise AssertionError(f"CP ranks disagree on {key}")
            if row["learning_rates_used"] != other["learning_rates_used"]:
                raise AssertionError("CP ranks used different learning rates")
        for key in row:
            if key.endswith("_seconds") or key == "peak_allocated_bytes":
                row[key] = max(report["steps"][index][key] for report in reports)
        steps.append(row)
    result = {
        key: reports[0][key]
        for key in ("mode", "cp", "controls", "input_sha256", "initial_weights_sha256", "packages", "source")
    }
    result.update(
        steps=steps,
        full_update_norm=own["candidate_update_norm"],
        sampled_elements=samples[1].numel(),
        model_elements=elements,
        peak_allocated_bytes=max(step["peak_allocated_bytes"] for step in steps),
        probe_cross_entropies={step: -float(probes[step].double().mean()) for step in CHECKPOINTS},
        timings={key: max(report["timings"][key] for report in reports) for key in reports[0]["timings"]},
        full_update_comparisons={},
    )
    for name in updates[0]["stats"]:
        reference = read(root / name / "result.json")
        matched(reference, result)
        result["full_update_comparisons"][name] = update_metrics(
            [update["stats"][name] for update in updates], elements
        )
    write_json(folder / "result.json", result)
    print(
        f"TRAIN10_DONE mode={mode} updates={STEPS} update_norm={result['full_update_norm']:.9g} "
        f"peak_GiB={result['peak_allocated_bytes'] / 2**30:.2f}",
        flush=True,
    )


def finish(root, *, keep_updates):
    root = Path(root).resolve()
    results = {mode: read(root / mode / "result.json") for mode in MODES}
    for result in results.values():
        matched(results["base_cp1"], result)
    samples = {mode: load(root / mode / "samples.pt") for mode in MODES}
    probes = {mode: load(root / mode / "probes.pt") for mode in MODES}
    pairs = {}
    for mode, result in results.items():
        for reference, full in result["full_update_comparisons"].items():
            trend = []
            for step in CHECKPOINTS[1:]:
                stat = empty_stats()
                add_stats(stat, samples[reference][step], samples[mode][step])
                small = update_metrics([stat], result["sampled_elements"])
                probe = probe_metrics(probes[reference], probes[mode], step)
                trend.append(dict(step=step, sampled_update_relative_l2=small["update_relative_l2"], **probe))
            pair = dict(
                final_full_update=full,
                initial_probe=probe_metrics(probes[reference], probes[mode], 0),
                trend=trend,
            )
            pairs[f"{reference}_vs_{mode}"] = pair
            print(
                f"TRAIN10_COMPARE {reference}_vs_{mode}: update_rel_l2={full['update_relative_l2']:.9g} "
                f"cosine={full['update_cosine']:.12g} "
                f"probe_mean_abs={trend[-1]['logprob_mean_abs']:.9g} "
                f"probe_change_mean_abs={trend[-1]['logprob_change_mean_abs']:.9g}",
                flush=True,
            )
    baseline = pairs["base_cp1_vs_base_cp2"]["final_full_update"]["update_relative_l2"]
    ratios = {
        mode: pairs[f"base_cp1_vs_{mode}"]["final_full_update"]["update_relative_l2"] / baseline if baseline else None
        for mode in ("tpr_cp1", "tpr_cp2")
    }
    modes = {}
    for mode, result in results.items():
        training = sum(
            sum(value for key, value in step.items() if key.endswith("_seconds") and "observer" not in key)
            for step in result["steps"]
        )
        modes[mode] = dict(
            final_loss=result["steps"][-1]["loss"],
            initial_probe_ce=result["probe_cross_entropies"]["0"],
            final_probe_ce=result["probe_cross_entropies"][str(STEPS)],
            full_update_norm=result["full_update_norm"],
            peak_GiB=result["peak_allocated_bytes"] / 2**30,
            training_seconds=training,
            total_seconds=result["timings"]["total_seconds"],
        )
    report = dict(
        scope="Ten native VERL/Megatron/MindSpeed optimizer updates; two shared P1024/S1024 trajectories per minibatch",
        controls=results["base_cp1"]["controls"],
        full_update_step=STEPS,
        sampled_update_steps=list(CHECKPOINTS[1:]),
        sampled_elements=results["base_cp1"]["sampled_elements"],
        model_elements=results["base_cp1"]["model_elements"],
        evaluation="Native whole-sequence inference at route CP size; cross-CP gaps include layout effects",
        interpretation="Fixed-data PPO drift, not on-policy convergence; steps after 1 compare already-diverged states",
        modes=modes,
        pairs=pairs,
        final_update_error_ratio_to_native_cp2=ratios,
        temporary_update_references_retained=True,
    )
    lines = [
        "# 10-step production-path drift",
        "",
        "All-parameter FP32 master updates at step 10; steps 1/5/10 also have fixed-coordinate samples.",
        "",
        "| Comparison | Full update rel L2 | Cosine | Probe mean abs | Probe change mean abs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, pair in pairs.items():
        full, last = pair["final_full_update"], pair["trend"][-1]
        lines.append(
            f"| {name} | {full['update_relative_l2']:.6g} | {full['update_cosine']:.9g} | "
            f"{last['logprob_mean_abs']:.6g} | {last['logprob_change_mean_abs']:.6g} |"
        )
    lines += [
        "",
        f"Error ratios to native CP1/CP2: {ratios}",
        "",
        "Probe uses native whole-sequence inference at the route's CP size. Cross-CP gaps include layout effects.",
        "Probe change subtracts each route's step-0 output; it does not make different CP evaluators identical.",
        "Per-step losses/grad norms/times: <mode>/result.json. Provenance: runtime_audit_rank*.json.",
    ]
    # Validate every manifest path before deleting only this run's temporary Base blocks.
    paths = []
    if not keep_updates:
        for mode in ("base_cp1", "base_cp2"):
            folder = (root / mode / "updates").resolve()
            if not folder.is_relative_to(root):
                raise ValueError("Unsafe update reference folder")
            for blocks in read(folder / "index.json")["parts"]:
                for block in blocks:
                    path = (folder / block["file"]).resolve()
                    if not path.is_relative_to(folder) or path.suffix != ".pt" or not path.is_file():
                        raise ValueError("Unsafe update reference cleanup")
                    paths.append(path)
    # Publish recoverable scalar results before removing temporary blocks.
    write_json(root / "comparison.json", report)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in paths:
        path.unlink()
    report["temporary_update_references_retained"] = keep_updates
    write_json(root / "comparison.json", report)
    print(f"TRAIN10_ALL_DONE ratios={ratios} summary={root / 'summary.md'}", flush=True)
    return report


if __name__ == "__main__":
    finish(os.environ["TPR_TRAIN10_OUTPUT"], keep_updates=os.getenv("TPR_TRAIN10_KEEP_UPDATES", "0") == "1")
