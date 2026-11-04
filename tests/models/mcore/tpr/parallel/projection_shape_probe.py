# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Matched-input first-layer projection: 64 rows vs two 32-row executions."""

import hashlib

import torch
import torch.nn.functional as F

from .fast_fp32_grad_compare import fast_fp32_tensor_difference
from .split_bf16_quick import _bf16_loss
from .split_fp32_attribution import _tensor


def _cpu(tensor):
    return tensor.detach().to(device="cpu", copy=True)


def _hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def capture_projection_sample(model, tokens):
    """One real whole-sequence F/B; save only first projection X, Y, dY, dX, dW."""
    if tokens.ndim != 1 or tokens.numel() != 64 or tokens.dtype != torch.long:
        raise ValueError("Projection capture requires exactly 64 token IDs")
    module = model.decoder.layers[0].self_attention.linear_proj
    if module.weight.dtype != torch.bfloat16 or getattr(module, "bias", None) is not None:
        raise ValueError("Probe requires the bias-free BF16 Qwen projection")
    sample, handles = {}, []

    def input_hook(_module, args):
        if "input" in sample:
            raise AssertionError("First projection executed more than once; disable recomputation")
        hidden = args[0]
        sample["input"] = _cpu(hidden)
        sample["input_stride"] = list(hidden.stride())
        handles.append(hidden.register_hook(lambda grad: sample.update(input_gradient=_cpu(grad))))

    def output_hook(_module, _args, output):
        output = _tensor(output)
        sample["output"] = _cpu(output)
        handles.append(output.register_hook(lambda grad: sample.update(upstream=_cpu(grad))))

    handles.extend((module.register_forward_pre_hook(input_hook), module.register_forward_hook(output_hook)))
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        positions = torch.arange(64, device=tokens.device)[None, :]
        logits = model(input_ids=tokens[None, :], position_ids=positions, attention_mask=None)
        loss = _bf16_loss(logits, 0, 64)
        loss.backward()
        sample["weight_gradient"] = _cpu(module.weight.grad)
        sample["loss"] = float(loss.detach())
        if not {"input", "output", "upstream", "input_gradient", "weight_gradient"} <= sample.keys():
            raise AssertionError("Projection capture missed forward/backward tensors")
        return module, sample
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def _case(module, sample, mode, case):
    reference = mode == "fp32_reference"
    dtype = torch.float32 if reference else torch.bfloat16
    x = sample["input"].to(device=module.weight.device, dtype=dtype).detach().requires_grad_(True)
    dy = sample["upstream"].to(device=x.device, dtype=dtype)
    weight = module.weight.detach().float().requires_grad_(True) if reference else module.weight
    weight.grad = None

    def project(value):
        if mode == "module_bf16":
            return _tensor(module(value))
        if mode == "fp32_bf16_boundary":
            return F.linear(value.float(), weight.float()).to(value.dtype)
        if reference:
            return F.linear(value, weight)
        raise ValueError(f"Unknown projection precision: {mode}")

    if case in ("full", "repeat_full"):
        output = project(x)
        output.backward(dy)
    elif case == "split_connected":
        output = torch.cat((project(x[:32]), project(x[32:])))
        output.backward(dy)
    elif case == "split_sequential":
        pieces = []
        for start in (0, 32):
            piece = project(x[start : start + 32])
            piece.backward(dy[start : start + 32])
            pieces.append(piece.detach())
        output = torch.cat(pieces)
    else:
        raise ValueError(f"Unknown projection execution: {case}")
    if x.grad is None or weight.grad is None:
        raise AssertionError("Projection probe missed an input or weight gradient")
    result = {"output": _cpu(output), "input_gradient": _cpu(x.grad), "weight_gradient": _cpu(weight.grad)}
    weight.grad = None
    return result


def _compare(reference, candidate):
    # Promoting stored BF16 values to FP32 is exact; final metrics use FP64.
    return {name: fast_fp32_tensor_difference(reference[name].float(), candidate[name].float()) for name in reference}


def run_projection_probe(module, sample):
    """All cells use identical captured X/W/dY; only shape/precision/backward changes."""
    if sample["input"].shape[0] != 64 or sample["upstream"].shape[0] != 64:
        raise ValueError("Projection probe is fixed at P32/S32")
    if any(sample[key].dtype != torch.bfloat16 for key in ("input", "upstream")):
        raise TypeError("Captured projection input and upstream gradient must be BF16")
    if module.weight.dtype != torch.bfloat16 or getattr(module, "bias", None) is not None:
        raise ValueError("Expected bias-free BF16 projection")
    version = module.weight._version
    report = {
        "scope": "matched captured first-layer X/W/dY, one 64-row vs two 32-row projection executions",
        "input_shape": list(sample["input"].shape),
        "weight_shape": list(module.weight.shape),
        "input_capture_stride": sample.get("input_stride"),
        "input_sha256": _hash(sample["input"]),
        "weight_sha256": _hash(module.weight),
        "upstream_sha256": _hash(sample["upstream"]),
        "parameter_or_activation_dumps": False,
        "fp32_reference_scope": "BF16 captured X/W/dY promoted exactly; FP32 computation and gradient storage",
        "fp32_bf16_boundary_scope": "FP32 linear compute; BF16 input/output, parameter and gradient storage",
        "modes": {},
    }
    reference = _case(module, sample, "fp32_reference", "full")
    try:
        for mode in ("module_bf16", "fp32_bf16_boundary", "fp32_reference"):
            full = reference if mode == "fp32_reference" else _case(module, sample, mode, "full")
            connected = _case(module, sample, mode, "split_connected")
            sequential = _case(module, sample, mode, "split_sequential")
            repeated = _case(module, sample, mode, "repeat_full")
            report["modes"][mode] = {
                "full_vs_split_connected": _compare(full, connected),
                "full_vs_split_sequential": _compare(full, sequential),
                "connected_vs_sequential": _compare(connected, sequential),
                "full_vs_repeat_full": _compare(full, repeated),
                "vs_fixed_fp32_full": {
                    name: _compare(reference, value)
                    for name, value in (
                        ("full", full),
                        ("split_connected", connected),
                        ("split_sequential", sequential),
                    )
                },
            }
            if mode == "module_bf16":
                report["captured_vs_isolated_full"] = _compare({name: sample[name] for name in full}, full)
            if module.weight._version != version:
                raise AssertionError("Projection weights changed in-place")
            del full, connected, sequential, repeated
        return report
    finally:
        module.weight.grad = None
