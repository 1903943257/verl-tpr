"""Compare Route B multi-step Full and Split-No-Share trajectories."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or records[0].get("record_type") != "header":
        raise AssertionError(f"Missing Route B header: {path}")
    header = records[0]
    steps = [record for record in records[1:] if record.get("record_type") == "step"]
    return header, steps


def _sampled_metrics(left: dict, right: dict, schema: dict) -> dict:
    diff_sq = left_sq = right_sq = dot = 0.0
    compared = 0
    for name, meta in schema.items():
        a = left[name]
        b = right[name]
        if a is None and b is None:
            continue
        if a is None or b is None:
            raise AssertionError(f"{name}: gradient missing in only one arm")
        if len(a) != len(b) or len(a) != len(meta["indices"]):
            raise AssertionError(f"{name}: sample shape mismatch")
        scale = meta["numel"] / len(a) if meta["numel"] > len(a) else 1.0
        for x, y in zip(a, b, strict=True):
            if not math.isfinite(x) or not math.isfinite(y):
                raise AssertionError(f"{name}: non-finite sampled value")
            d = x - y
            diff_sq += scale * d * d
            left_sq += scale * x * x
            right_sq += scale * y * y
            dot += scale * x * y
        compared += 1
    left_norm = math.sqrt(max(left_sq, 0.0))
    right_norm = math.sqrt(max(right_sq, 0.0))
    return {
        "sampled_tensors": compared,
        "relative_l2": math.sqrt(max(diff_sq, 0.0)) / max(right_norm, 1e-30),
        "cosine": dot / max(left_norm * right_norm, 1e-30),
        "left_norm_estimate": left_norm,
        "right_norm_estimate": right_norm,
    }


def _logprob_metrics(left_rows, right_rows) -> dict:
    left = {(int(row), int(offset)): float(value) for row, offset, value in left_rows}
    right = {(int(row), int(offset)): float(value) for row, offset, value in right_rows}
    if left.keys() != right.keys() or not left:
        raise AssertionError("Full/Split PPO logprob coordinates differ")
    keys = sorted(left)
    a = [left[key] for key in keys]
    b = [right[key] for key in keys]
    if not all(math.isfinite(x) for x in a + b):
        raise AssertionError("Non-finite PPO logprob")
    diff_sq = sum((x - y) ** 2 for x, y in zip(a, b, strict=True))
    left_sq = sum(x * x for x in a)
    right_sq = sum(y * y for y in b)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    diffs = [abs(x - y) for x, y in zip(a, b, strict=True)]
    return {
        "count": len(keys),
        "relative_l2": math.sqrt(diff_sq) / max(math.sqrt(right_sq), 1e-30),
        "cosine": dot / max(math.sqrt(left_sq * right_sq), 1e-30),
        "mean_abs": sum(diffs) / len(diffs),
        "max_abs": max(diffs),
    }


def _load_summary(path: Path):
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text())


def _exact_final_parameter_metrics(full_path: Path, split_path: Path) -> dict:
    """Exact full-vector comparison, streamed one parameter tensor at a time."""
    left = torch.load(
        full_path, map_location="cpu", weights_only=True, mmap=True,
    )
    right = torch.load(
        split_path, map_location="cpu", weights_only=True, mmap=True,
    )
    if left.keys() != right.keys():
        raise AssertionError("Final Full/Split parameter names differ")
    diff_sq = full_sq = split_sq = dot = 0.0
    per_parameter = []
    for name in left:
        a = left[name].float()
        b = right[name].float()
        if a.shape != b.shape:
            raise AssertionError(f"{name}: final parameter shape mismatch")
        difference = a - b
        tensor_diff_sq = torch.sum(difference.square(), dtype=torch.float64).item()
        tensor_full_sq = torch.sum(a.square(), dtype=torch.float64).item()
        tensor_split_sq = torch.sum(b.square(), dtype=torch.float64).item()
        tensor_dot = torch.sum(a * b, dtype=torch.float64).item()
        diff_sq += tensor_diff_sq
        full_sq += tensor_full_sq
        split_sq += tensor_split_sq
        dot += tensor_dot
        per_parameter.append({
            "name": name,
            "relative_l2": math.sqrt(tensor_diff_sq) / max(math.sqrt(tensor_full_sq), 1e-30),
            "difference_l2": math.sqrt(tensor_diff_sq),
            "full_l2": math.sqrt(tensor_full_sq),
            "split_l2": math.sqrt(tensor_split_sq),
        })
        del a, b, difference
    del left, right
    per_parameter.sort(key=lambda row: row["relative_l2"], reverse=True)
    return {
        "parameter_tensors": len(per_parameter),
        "relative_l2": math.sqrt(diff_sq) / max(math.sqrt(full_sq), 1e-30),
        "cosine": dot / max(math.sqrt(full_sq * split_sq), 1e-30),
        "full_norm": math.sqrt(full_sq),
        "split_norm": math.sqrt(split_sq),
        "top_relative_errors": per_parameter[:10],
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: compare_real_actor_cp1_sequence_split_multistep.py "
            "<full.jsonl> <split_no_share.jsonl> <report.json>"
        )
    full_path, split_path, report_path = map(Path, sys.argv[1:])
    full_header, full_steps = _read_jsonl(full_path)
    split_header, split_steps = _read_jsonl(split_path)
    full_summary = _load_summary(full_path)
    split_summary = _load_summary(split_path)

    for field in ("input_sha256", "initial_weights_witness_sha256", "checkpoint",
                  "steps", "ppo_tokens", "topology", "optimizer"):
        if full_header.get(field) != split_header.get(field):
            raise AssertionError(
                f"Full/Split initial contract mismatch for {field}: "
                f"{full_header.get(field)!r} != {split_header.get(field)!r}"
            )
    if full_header["mode"] != "full" or split_header["mode"] != "split_no_share":
        raise AssertionError("Unexpected Route B arm mode")
    if full_header["sample_schema"] != split_header["sample_schema"]:
        raise AssertionError("Full/Split sampled tensor schemas differ")
    steps_n = int(full_header["steps"])
    if len(full_steps) != steps_n or len(split_steps) != steps_n:
        raise AssertionError(
            f"Expected {steps_n} complete steps; got full={len(full_steps)} split={len(split_steps)}"
        )
    if full_summary["steps"] != steps_n or split_summary["steps"] != steps_n:
        raise AssertionError("Summary step count mismatch")

    schema = full_header["sample_schema"]
    initial = _sampled_metrics(
        full_header["initial_parameter_samples"],
        split_header["initial_parameter_samples"], schema,
    )
    if initial["relative_l2"] != 0.0:
        raise AssertionError(
            f"Initial sampled parameters are not identical: {initial['relative_l2']}"
        )

    rows = []
    for a, b in zip(full_steps, split_steps, strict=True):
        if a["step"] != b["step"]:
            raise AssertionError("Full/Split step indices differ")
        step = int(a["step"])
        if a["missing_gradients"] != b["missing_gradients"]:
            raise AssertionError(
                f"step {step}: different missing-gradient parameter sets"
            )
        logprob = _logprob_metrics(a["logprobs"], b["logprobs"])
        gradient = _sampled_metrics(
            a["gradient_samples"], b["gradient_samples"], schema,
        )
        parameter = _sampled_metrics(
            a["parameter_samples_after_update"],
            b["parameter_samples_after_update"], schema,
        )
        row = {
            "step": step,
            "full_loss": a["loss"],
            "split_loss": b["loss"],
            "loss_abs_gap": abs(a["loss"] - b["loss"]),
            "logprob": logprob,
            "gradient_sampled_estimate": gradient,
            "parameter_sampled_estimate": parameter,
            "full_grad_norm_before_clip": a["grad_norm_before_clip"],
            "split_grad_norm_before_clip": b["grad_norm_before_clip"],
            "clip_status_mismatch": a["clipped"] != b["clipped"],
            "missing_gradient_count": len(a["missing_gradients"]),
        }
        rows.append(row)
        print(
            f"ROUTE_B_STEP step={step:02d} "
            f"loss_full={a['loss']:.9g} loss_split={b['loss']:.9g} "
            f"loss_gap={row['loss_abs_gap']:.6g} "
            f"logprob_rel_l2={logprob['relative_l2']:.6g} "
            f"grad_sample_rel_l2={gradient['relative_l2']:.6g} "
            f"grad_sample_cos={gradient['cosine']:.7f} "
            f"param_sample_rel_l2={parameter['relative_l2']:.6g} "
            f"param_sample_cos={parameter['cosine']:.7f} "
            f"clip_mismatch={row['clip_status_mismatch']}",
            flush=True,
        )

    full_final_state = Path(full_summary["final_state_path"])
    split_final_state = Path(split_summary["final_state_path"])
    if not full_final_state.is_file() or not split_final_state.is_file():
        raise FileNotFoundError("Missing exact final model state snapshot")
    exact_final_parameters = _exact_final_parameter_metrics(
        full_final_state, split_final_state,
    )
    final_eval = _logprob_metrics(
        full_summary["post_update_eval"]["logprobs"],
        split_summary["post_update_eval"]["logprobs"],
    )
    final_eval_loss_gap = abs(
        full_summary["post_update_eval"]["loss"]
        - split_summary["post_update_eval"]["loss"]
    )
    moments = {}
    for state_name in ("exp_avg", "exp_avg_sq"):
        moments[state_name] = _sampled_metrics(
            full_summary["optimizer_state_samples"][state_name],
            split_summary["optimizer_state_samples"][state_name],
            schema,
        )

    first_parameter_gap = rows[0]["parameter_sampled_estimate"]["relative_l2"]
    final_parameter_gap = rows[-1]["parameter_sampled_estimate"]["relative_l2"]
    report = {
        "scope": "Qwen3-4B BF16, CP1, Full Sequence vs Split-No-Share, fixed Actor rollout",
        "steps": steps_n,
        "input_sha256": full_header["input_sha256"],
        "initial_weights_witness_sha256": full_header["initial_weights_witness_sha256"],
        "initial_parameter_sample_rel_l2": initial["relative_l2"],
        "data_policy": full_header["data_policy"],
        "sample_scope": full_header["gradient_sample_scope"],
        "stepwise": rows,
        "summary": {
            "max_loss_abs_gap_before_update": max(row["loss_abs_gap"] for row in rows),
            "max_logprob_relative_l2_before_update": max(
                row["logprob"]["relative_l2"] for row in rows
            ),
            "final_pre_update_logprob_relative_l2": rows[-1]["logprob"]["relative_l2"],
            "max_sampled_gradient_relative_l2": max(
                row["gradient_sampled_estimate"]["relative_l2"] for row in rows
            ),
            "min_sampled_gradient_cosine": min(
                row["gradient_sampled_estimate"]["cosine"] for row in rows
            ),
            "first_post_update_sampled_parameter_relative_l2": first_parameter_gap,
            "final_post_update_sampled_parameter_relative_l2": final_parameter_gap,
            "sampled_parameter_gap_growth_factor": (
                final_parameter_gap / max(first_parameter_gap, 1e-30)
            ),
            "exact_final_parameter_comparison": exact_final_parameters,
            "final_post_update_eval_loss_abs_gap": final_eval_loss_gap,
            "final_post_update_eval_logprob": final_eval,
            "clip_status_mismatch_count": sum(row["clip_status_mismatch"] for row in rows),
            "final_adam_moment_samples": moments,
            "interpretation": (
                "Diagnostic Route B result on a fixed rollout. Gradient/parameter "
                "distances are deterministic per-tensor sample estimates, not exact "
                "full-vector metrics; logprob and loss comparisons are over all saved "
                "PPO coordinates. No arbitrary parity gate is applied."
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    s = report["summary"]
    print("\n========== ROUTE B SUMMARY ==========")
    print(f"steps={steps_n} initial_parameters_identical=True")
    print(f"max_loss_abs_gap={s['max_loss_abs_gap_before_update']:.9g}")
    print(f"max_logprob_rel_l2={s['max_logprob_relative_l2_before_update']:.9g}")
    print(f"max_sampled_gradient_rel_l2={s['max_sampled_gradient_relative_l2']:.9g}")
    print(f"min_sampled_gradient_cos={s['min_sampled_gradient_cosine']:.9g}")
    print(f"first_parameter_sample_rel_l2={first_parameter_gap:.9g}")
    print(f"final_parameter_sample_rel_l2={final_parameter_gap:.9g}")
    print(f"parameter_gap_growth_factor={s['sampled_parameter_gap_growth_factor']:.9g}")
    print(
        "exact_final_parameter_rel_l2="
        f"{exact_final_parameters['relative_l2']:.9g} "
        f"cos={exact_final_parameters['cosine']:.9g}"
    )
    print(f"post{steps_n}_eval_loss_gap={final_eval_loss_gap:.9g}")
    print(f"post{steps_n}_eval_logprob_rel_l2={final_eval['relative_l2']:.9g}")
    print(f"clip_status_mismatches={s['clip_status_mismatch_count']}")
    print(f"report={report_path}")
    print("ROUTE_B_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
