# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Read-only, bounded-memory export of the native optimizer's owned main_grad."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

BLOCK = 4 * 1024 * 1024


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parameter_layout(model):
    return [{"name": name, "shape": list(p.shape), "elements": p.numel()} for name, p in model.named_parameters()]


def weight_digest(model):
    """Hash every loaded parameter, including the tied embedding only once."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
        flat = parameter.detach().view(-1)
        for start in range(0, flat.numel(), BLOCK):
            block = flat[start : start + BLOCK].to(device="cpu", copy=True).contiguous()
            digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _optimizers(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        yield optimizer
    else:
        for child in children:
            yield from _optimizers(child)


def owned_gradient_ranges(model, optimizer):
    """Use DistOpt's actual ownership map; unowned reduce-scatter tails are stale.

    TPR currently requests native force_all_reduce at finalize. Still select
    exactly the same owned slices as Base, so replicated gradients are never
    counted twice. No gradient copying, normalization or communication here.
    """
    owners = {}
    for child in _optimizers(optimizer):
        if not hasattr(child, "model_param_gbuf_map"):
            raise RuntimeError("This experiment requires native Megatron DistributedOptimizer ownership maps")
        for parameter in child.model_param_gbuf_map:
            if parameter in owners:
                raise AssertionError("A parameter belongs to multiple distributed optimizers")
            span = child._get_model_param_range_map(parameter)["param"]
            owners[parameter] = (span.start, span.end)
    for index, (name, parameter) in enumerate(model.named_parameters()):
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None or gradient.dtype != torch.float32 or gradient.numel() != parameter.numel():
            raise AssertionError(f"Expected real, full-layout FP32 main_grad for {name}")
        if parameter in owners:
            start, end = owners.pop(parameter)
            if not 0 <= start < end <= parameter.numel():
                raise AssertionError(f"Invalid native optimizer range for {name}: {start}:{end}")
            yield index, name, start, end, gradient.detach().view(-1)
    if owners:
        raise AssertionError("Optimizer refers to parameters outside the model")


def empty_stats():
    return dict(elements=0, reference_sq=0.0, candidate_sq=0.0, difference_sq=0.0, dot=0.0, max_abs=0.0)


def add_stats(stats, reference, candidate):
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32 or reference.shape != candidate.shape:
        raise TypeError("Compare matching native FP32 gradient slices")
    x, y = (t.detach().cpu().numpy().astype(np.float64) for t in (reference, candidate))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError("Nonfinite native main_grad")
    delta = y - x
    for key, value in (
        ("reference_sq", x * x),
        ("candidate_sq", y * y),
        ("difference_sq", delta * delta),
        ("dot", x * y),
    ):
        stats[key] += float(value.sum())
    stats["max_abs"] = max(stats["max_abs"], float(np.abs(delta).max()))
    stats["elements"] += x.size


def merge_stats(parts, expected_elements):
    result = empty_stats()
    for part in parts:
        for key, value in part.items():
            result[key] = max(result[key], value) if key == "max_abs" else result[key] + value
    if result["elements"] != expected_elements or result["reference_sq"] <= 0 or result["candidate_sq"] <= 0:
        raise AssertionError("Gradient comparison has missing/duplicated elements or zero norms")
    return {
        "elements": result["elements"],
        "gradient_relative_l2": math.sqrt(result["difference_sq"] / result["reference_sq"]),
        "gradient_cosine": result["dot"] / math.sqrt(result["reference_sq"] * result["candidate_sq"]),
        "gradient_max_abs": result["max_abs"],
        "reference_gradient_norm": math.sqrt(result["reference_sq"]),
        "candidate_gradient_norm": math.sqrt(result["candidate_sq"]),
    }


class GradientReference:
    def __init__(self, folder, layout):
        self.folder = Path(folder)
        value = json.loads((self.folder / "index.json").read_text())
        if value["layout"] != layout:
            raise AssertionError("Reference and candidate parameter layouts differ")
        self.parts = value["parts"]

    def read(self, index, start, end):
        pieces, cursor = [], start
        for part in self.parts[index]:
            lo, hi = max(start, part["start"]), min(end, part["end"])
            if lo >= hi:
                continue
            if lo != cursor:
                raise AssertionError("Gap/overlap in saved reference")
            value = torch.load(self.folder / part["file"], map_location="cpu", weights_only=True, mmap=True)
            if value.dtype != torch.float32 or value.numel() != part["end"] - part["start"]:
                raise AssertionError("Corrupt reference gradient block")
            pieces.append(value[lo - part["start"] : hi - part["start"]])
            cursor = hi
        if cursor != end:
            raise AssertionError("Reference does not cover the requested parameter range")
        return torch.cat(pieces) if len(pieces) != 1 else pieces[0]


def observe_gradients(model, optimizer, folder, rank, references, *, save_reference):
    """Copy bounded slices AFTER native finalize; never modify model gradients."""
    folder = Path(folder)
    layout = parameter_layout(model)
    readers = {name: GradientReference(path, layout) for name, path in references.items()}
    stats = {name: empty_stats() for name in references}
    parts, owned = [], []
    if save_reference:
        (folder / f"rank{rank}").mkdir(parents=True, exist_ok=False)
    for index, name, start, end, gradient in owned_gradient_ranges(model, optimizer):
        owned.append(dict(index=index, name=name, start=start, end=end))
        for lo in range(start, end, BLOCK):
            hi = min(lo + BLOCK, end)
            block = gradient[lo:hi].to(device="cpu", copy=True).contiguous()
            if not bool(torch.isfinite(block).all()):
                raise FloatingPointError(f"Nonfinite main_grad for {name}")
            if save_reference:
                relative = f"rank{rank}/p{index:04d}_{lo:012d}_{hi:012d}.pt"
                torch.save(block, folder / relative)
                parts.append(dict(index=index, start=lo, end=hi, file=relative))
            for reference, reader in readers.items():
                add_stats(stats[reference], reader.read(index, lo, hi), block)
    return dict(layout=layout, parts=parts, owned=owned, stats=stats)


def validate_owned_coverage(layout, rank_reports):
    parts = [[] for _ in layout]
    for report in rank_reports:
        if report["layout"] != layout:
            raise AssertionError("Ranks disagree on parameter layout")
        for item in report["owned"]:
            parts[item["index"]].append(item)
    for metadata, spans in zip(layout, parts, strict=True):
        cursor = 0
        for span in sorted(spans, key=lambda item: item["start"]):
            if span["start"] != cursor or span["name"] != metadata["name"]:
                raise AssertionError("Optimizer shard ownership overlaps or has gaps")
            cursor = span["end"]
        if cursor != metadata["elements"]:
            raise AssertionError(f"Missing owned gradient for {metadata['name']}")


def index_reference(folder, rank_reports):
    layout = rank_reports[0]["layout"]
    validate_owned_coverage(layout, rank_reports)
    parts = [[] for _ in layout]
    for report in rank_reports:
        for part in report["parts"]:
            parts[part["index"]].append(part)
    for metadata, blocks in zip(layout, parts, strict=True):
        blocks.sort(key=lambda item: item["start"])
        cursor = 0
        for block in blocks:
            if block["start"] != cursor:
                raise AssertionError("Saved gradient blocks overlap or have gaps")
            cursor = block["end"]
        if cursor != metadata["elements"]:
            raise AssertionError("Incomplete saved reference")
    write_json(Path(folder) / "index.json", dict(layout=layout, parts=parts))
