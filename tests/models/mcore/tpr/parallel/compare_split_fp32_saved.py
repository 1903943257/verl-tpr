# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Complete the FP32 Full/Connected/Replay attribution from saved CPU files.

This tool NEVER constructs a Qwen model and NEVER invokes a device backward.
It is safe after run_three_way finished the three manifest/trace dumps but
failed during postprocessing (e.g. torch.load(Path, mmap=True) on torch_npu).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .split_fp32_attribution import ARMS, _compare_traces, _loss, tensor_difference


def _read_manifest(root: Path, arm: str) -> list[dict]:
    path = root / arm / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = json.loads(path.read_text())
    if not entries or len({entry["name"] for entry in entries}) != len(entries):
        raise ValueError(f"{arm}: missing or duplicate parameter manifest entries")
    missing = [entry["file"] for entry in entries if not (root / arm / entry["file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{arm}: missing {len(missing)} gradient shards; first={missing[0]}")
    return entries


def _load_trace(root: Path, arm: str) -> list[dict]:
    path = root / arm / "trace.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(str(path), weights_only=True, map_location="cpu")


def _trace_loss(trace: list[dict], *, total_length: int) -> tuple[float, int]:
    """Reconstruct the fixed synthetic loss from recorded FP32 logits.

    Replay emits both the _forward and objective snapshots for a segment;
    deduplicate by position range, selecting only grad-enabled forwards.
    """
    logits = {}
    for record in trace:
        if record["stage"] != "logits" or not record["grad_enabled"]:
            continue
        start, end = int(record["start"]), int(record["end"])
        key = (start, end)
        logits.setdefault(key, record["forward"])
    if not logits:
        raise ValueError("Missing grad-enabled logits in saved trace")
    ranges = sorted(logits)
    cursor = 0
    loss = 0.0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError(f"Uncovered/overlapping logit range: {ranges}")
        # Trace tensors have [S,V] or [S,1,V] sequence-major layout.
        output = logits[(start, end)]
        if output.ndim == 2:
            output = output.unsqueeze(0)
        elif output.ndim == 3 and output.shape[1] == 1:
            output = output.permute(1, 0, 2)
        else:
            raise ValueError(f"Unexpected saved logits shape: {output.shape}")
        loss += float(_loss(output, start, total_length))
        cursor = end
    if cursor != total_length:
        raise ValueError(f"Saved logits cover {cursor} of {total_length} tokens")
    return loss, len(ranges)


def compare_saved(root: Path, *, prefix: int, suffix: int, atol: float = 1e-5, rtol: float = 1e-4,
                  refresh_layer_summaries: bool = False, output_name: str = "report.json") -> dict:
    if prefix < 1 or suffix < 2:
        raise ValueError("Need prefix >=1 and suffix >=2")
    manifests = {arm: _read_manifest(root, arm) for arm in ARMS}
    reference_manifest = manifests["full"]
    for arm in ARMS[1:]:
        if manifests[arm] != reference_manifest:
            raise ValueError(f"{arm}: mismatched parameter manifest/initial model shapes")
    traces = {arm: _load_trace(root, arm) for arm in ARMS}
    length = prefix + suffix
    report = {
        "diagnostic_only": True,
        "data_source": "saved_gradients_and_traces_no_npu_rerun",
        "dtype": "torch.float32",
        "attention": "explicit_math_gqa",
        "loss": "fixed_linear_all_queries_except_last",
        "prefix_length": prefix,
        "suffix_length": suffix,
        "atol": atol,
        "rtol": rtol,
        "model_parameter_elements": sum(math.prod(entry["shape"]) for entry in reference_manifest),
        "arms": {},
        "pairs": {},
        "layer_summary": {},
        "loss_reconstructed_from_saved_logits": True,
    }
    for arm in ARMS:
        loss, ranges = _trace_loss(traces[arm], total_length=length)
        report["arms"][arm] = {
            "loss": loss,
            "trace_records": len(traces[arm]),
            "parameter_count": len(manifests[arm]),
            "logit_ranges": ranges,
        }
    for candidate in ARMS[1:]:
        dst = root / f"full_vs_{candidate}_layers.json"
        if dst.is_file() and not refresh_layer_summaries:
            stages = json.loads(dst.read_text())
            print(f"Saved layer comparison reused: {dst}", flush=True)
        else:
            print(f"Comparing trace stages: full vs {candidate}", flush=True)
            stages = _compare_traces(traces["full"], traces[candidate], atol=atol, rtol=rtol)
            dst.write_text(json.dumps(stages, indent=2))
        report["layer_summary"][candidate] = {
            kind: next((
                {key: event[key] for key in ("event", "stage", "start", "end", "grad_enabled")}
                for event in stages
                if kind in event and not event[kind]["within_tolerance"]
            ), None)
            for kind in ("forward", "gradient")
        }
    del traces
    pairs = (
        ("full", "split_connected"),
        ("full", "split_replay"),
        ("split_connected", "split_replay"),
    )
    for left, right in pairs:
        print(f"Comparing saved FP32 gradients: {left} vs {right} ({len(reference_manifest)} tensors)", flush=True)
        metrics = []
        for index, entry in enumerate(reference_manifest):
            left_path = root / left / entry["file"]
            right_path = root / right / entry["file"]
            a = torch.load(str(left_path), weights_only=True, map_location="cpu", mmap=True)
            b = torch.load(str(right_path), weights_only=True, map_location="cpu", mmap=True)
            if a.dtype != torch.float32 or b.dtype != torch.float32 or tuple(a.shape) != tuple(entry["shape"]) or tuple(b.shape) != tuple(entry["shape"]):
                raise ValueError(f"Incorrect gradient dtype or shape at {entry['name']}")
            metrics.append({"name": entry["name"], **tensor_difference(a, b, atol=atol, rtol=rtol)})
            del a, b
            if (index + 1) % 32 == 0 or index + 1 == len(reference_manifest):
                print(f"  {left} vs {right}: {index+1}/{len(reference_manifest)} tensors", flush=True)
        a_norm_sq = sum(item["reference_norm"] ** 2 for item in metrics)
        b_norm_sq = sum(item["candidate_norm"] ** 2 for item in metrics)
        diff_sq = sum(item["difference_norm"] ** 2 for item in metrics)
        dot = sum((item["cosine"] or 0.0) * item["reference_norm"] * item["candidate_norm"] for item in metrics)
        report["pairs"][f"{left}_vs_{right}"] = {
            "loss_abs_gap": abs(report["arms"][left]["loss"] - report["arms"][right]["loss"]),
            "parameters": metrics,
            "gradient_relative_l2": math.sqrt(diff_sq / a_norm_sq) if a_norm_sq else None,
            "gradient_max_abs": max(item["max_abs"] for item in metrics),
            "gradient_cosine": dot / math.sqrt(a_norm_sq * b_norm_sq) if a_norm_sq and b_norm_sq else None,
            "all_parameters_within_tolerance": all(item["within_tolerance"] for item in metrics),
            "reference_gradient_norm": math.sqrt(a_norm_sq),
            "candidate_gradient_norm": math.sqrt(b_norm_sq),
        }
        pair = report["pairs"][f"{left}_vs_{right}"]
        print(
            f"OFFLINE_SPLIT_FP32 {left}_vs_{right}: "
            f"rel_l2={pair['gradient_relative_l2']:.9g} "
            f"cosine={pair['gradient_cosine']:.12g} "
            f"max_abs={pair['gradient_max_abs']:.9g} "
            f"loss_gap={pair['loss_abs_gap']:.9g}",
            flush=True,
        )
    destination = root / output_name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {destination}")
    destination.write_text(json.dumps(report, indent=2))
    print(f"SPLIT_FP32_OFFLINE_DONE report={destination}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--suffix", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--refresh-layer-summaries", action="store_true")
    parser.add_argument("--output-name", default="report.json")
    args = parser.parse_args()
    compare_saved(args.root, prefix=args.prefix, suffix=args.suffix, atol=args.atol, rtol=args.rtol,
                  refresh_layer_summaries=args.refresh_layer_summaries, output_name=args.output_name)


if __name__ == "__main__":
    main()
