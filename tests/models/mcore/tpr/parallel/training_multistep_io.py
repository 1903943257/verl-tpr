# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Read actual DistributedOptimizer master shards; never change training state."""

import hashlib
import math

import numpy as np
import torch

from .training_precision_io import (
    BLOCK,
    GradientReference,
    _optimizers,
    add_stats,
    empty_stats,
    parameter_layout,
)

SAMPLES_PER_PARAMETER = 1024


def sample_indices(elements):
    """Same global coordinates for CP1/CP2, including both ends of a parameter."""
    count = min(elements, SAMPLES_PER_PARAMETER)
    if count <= 1:
        return torch.arange(count, dtype=torch.int64)
    return torch.arange(count, dtype=torch.int64) * (elements - 1) // (count - 1)


def owned_master_ranges(model, optimizer):
    owners = {}
    for child in _optimizers(optimizer):
        for parameter in child.model_param_gbuf_map:
            if parameter in owners:
                raise AssertionError("Duplicate native optimizer owner")
            span = child._get_model_param_range_map(parameter)["param"]
            group, order = child.model_param_group_index_map[parameter]
            master = child.optimizer.param_groups[group]["params"][order]
            if (
                master.dtype != torch.float32
                or master.numel() != span.end - span.start
                or not 0 <= span.start < span.end <= parameter.numel()
            ):
                raise AssertionError("Expected native FP32 master with exact owned parameter range")
            owners[parameter] = (span.start, span.end, master.detach().view(-1))
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if parameter in owners:
            yield index, name, *owners.pop(parameter)
    if owners:
        raise AssertionError("Optimizer master refers to a parameter outside the model")


class InitialWeights:
    """One BF16 host copy per rank, reused for hashing and update subtraction.

    Native FP32 masters must initially equal the BF16 model exactly. This
    avoids a second full FP32 host model and observes sub-BF16 optimizer updates.
    """

    def __init__(self, model, optimizer):
        self.layout = parameter_layout(model)
        self.values = {}
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.dtype != torch.bfloat16:
                raise TypeError("Require the audited BF16 model")
            value = parameter.detach().view(-1).to(device="cpu", copy=True).contiguous()
            self.values[name] = value
            digest.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
            for start in range(0, value.numel(), BLOCK):
                digest.update(value[start : start + BLOCK].view(torch.uint8).numpy().tobytes())
        self.digest = digest.hexdigest()
        parameters = dict(model.named_parameters())
        for _index, name, start, end, master in owned_master_ranges(model, optimizer):
            original = parameters[name].detach().view(-1)
            for lo in range(0, end - start, BLOCK):
                hi = min(lo + BLOCK, end - start)
                if not torch.equal(master[lo:hi], original[start + lo : start + hi].float()):
                    raise AssertionError("Initial FP32 master differs from the BF16 checkpoint; cannot subtract it")
        for child in _optimizers(optimizer):
            if any(float(group.get("step", 0)) != 0 for group in child.optimizer.param_groups):
                raise AssertionError("Require fresh native optimizer groups")
            for state in child.optimizer.state.values():
                if float(state.get("step", 0)) != 0:
                    raise AssertionError("Require a fresh native optimizer with zero completed updates")

    def samples(self, model, optimizer):
        result = {}
        for index, name, start, end, master in owned_master_ranges(model, optimizer):
            indices = sample_indices(self.values[name].numel())
            indices = indices[(indices >= start) & (indices < end)]
            if not indices.numel():
                continue
            values = master.index_select(0, (indices - start).to(master.device)).to(device="cpu", copy=True)
            result[index] = dict(indices=indices, delta=values - self.values[name][indices].float())
        return result

    def full_update(self, model, optimizer, folder, rank, references, *, save_reference):
        """One all-parameter scan at step 10; only the two Base routes save blocks."""
        readers = {name: GradientReference(path, self.layout) for name, path in references.items()}
        stats = {name: empty_stats() for name in references}
        own_stats = empty_stats()
        parts, owned = [], []
        if save_reference:
            (folder / f"rank{rank}").mkdir(parents=True, exist_ok=False)
        for index, name, start, end, master in owned_master_ranges(model, optimizer):
            owned.append(dict(index=index, name=name, start=start, end=end))
            for lo in range(start, end, BLOCK):
                hi = min(lo + BLOCK, end)
                current = master[lo - start : hi - start].to(device="cpu", copy=True).contiguous()
                delta = current - self.values[name][lo:hi].float()
                if not readers:
                    square = float(np.square(delta.numpy(), dtype=np.float64).sum())
                    if not math.isfinite(square):
                        raise FloatingPointError("Nonfinite cumulative FP32 master update")
                    own_stats["elements"] += delta.numel()
                    for key in ("reference_sq", "candidate_sq", "dot"):
                        own_stats[key] += square
                if save_reference:
                    relative = f"rank{rank}/p{index:04d}_{lo:012d}_{hi:012d}.pt"
                    torch.save(delta, str(folder / relative))
                    parts.append(dict(index=index, start=lo, end=hi, file=relative))
                for name_ref, reader in readers.items():
                    add_stats(stats[name_ref], reader.read(index, lo, hi), delta)
        if readers:
            # Reuse candidate norms already computed by a reference comparison.
            candidate = next(iter(stats.values()))
            own_stats["elements"] = candidate["elements"]
            for key in ("reference_sq", "candidate_sq", "dot"):
                own_stats[key] = candidate["candidate_sq"]
        return dict(layout=self.layout, parts=parts, owned=owned, stats=stats, own_stats=own_stats)


def merge_samples(layout, rank_samples):
    result = []
    for index, metadata in enumerate(layout):
        pieces = [sample[index] for sample in rank_samples if index in sample]
        if not pieces:
            raise AssertionError("A parameter has no master-update samples")
        indices = torch.cat([part["indices"] for part in pieces])
        values = torch.cat([part["delta"] for part in pieces])
        order = indices.argsort()
        if not torch.equal(indices[order], sample_indices(metadata["elements"])):
            raise AssertionError("Master samples have missing/duplicate global coordinates")
        if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Invalid FP32 master-update samples")
        result.append(values[order])
    return torch.cat(result)


def audit_adam_updates(optimizer, expected_steps):
    """Observe native Adam state after updates, including MindSpeed group counters."""
    count = 0
    for child in _optimizers(optimizer):
        for group in child.optimizer.param_groups:
            for master in group["params"]:
                state = child.optimizer.state.get(master, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    moment = state.get(key)
                    if moment is None or moment.dtype != torch.float32 or moment.numel() != master.numel():
                        raise AssertionError("Missing native FP32 Adam moment state")
                completed = state.get("step", group.get("step"))
                if completed is None or float(completed) != expected_steps:
                    raise AssertionError("Native Adam counter disagrees with the completed update count")
                count += 1
    if not count:
        raise AssertionError("Native Adam has no owned FP32 master states")
    return dict(completed_updates=expected_steps, owned_master_tensors=count, moment_dtype="torch.float32")
