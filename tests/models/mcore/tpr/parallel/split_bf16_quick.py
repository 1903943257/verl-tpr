# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Summary-only BF16 Full/connected/Replay comparison, without tensor dumps."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from .fast_fp32_grad_compare import fast_tensor_difference
from .split_attention_precision import attention_implementation
from .split_forward_diagnostics import write_forward_comparison
from .split_fp32_attribution import ARMS, LayerTrace, _loss, _run_arm


def _bf16_loss(logits, start, total_length):
    # Same coefficients/denominator as FP32; the cast's backward returns BF16.
    return _loss(logits.float(), start, total_length)


class _AttentionCounter:
    """Verify the real attention path without retaining activations or hooks."""

    def __init__(self, implementation, tokens, trace=None):
        self.implementation = implementation
        self.tokens = tokens
        self.trace = trace
        self.calls = 0
        self.reset()

    def reset(self):
        self.calls = 0
        self.logprobs = torch.zeros(self.tokens.numel() - 1, dtype=torch.float32)
        self.coverage = torch.zeros(self.tokens.numel() - 1, dtype=torch.int32)

    def record(self, stage, tensor, **kwargs):
        if tensor.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"Unexpected logits dtype: {tensor.dtype}")
        from verl.models.mcore.tpr.context import get_tpr_attention_context

        context = get_tpr_attention_context()
        start = kwargs.get("start", 0 if context is None else context.prefix_length)
        if self.trace is not None:
            self.trace.record(stage, tensor, **{**kwargs, "start": start})
        with torch.no_grad():
            positions = torch.arange(start, start + tensor.shape[1], device=tensor.device)
            valid = positions < self.tokens.numel() - 1
            positions = positions[valid]
            target = self.tokens.index_select(0, positions + 1)
            logprobs = tensor[:, valid].float().log_softmax(dim=-1).gather(-1, target[None, :, None]).reshape(-1)
            indices = positions.cpu()
            self.logprobs.index_copy_(0, indices, logprobs.cpu())
            self.coverage.index_add_(0, indices, torch.ones_like(indices, dtype=torch.int32))

    def observations(self):
        if not bool((self.coverage == 1).all()) or not bool(torch.isfinite(self.logprobs).all()):
            raise AssertionError("Incorrect or nonfinite global next-token logprob coverage")
        return {"next_token_cross_entropy": float(-self.logprobs.double().mean()), "logprobs": self.logprobs.tolist()}

    def attention(self, query, key, value, **kwargs):
        if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
            raise TypeError("BF16 quick diagnostic received non-BF16 Q/K/V")
        output = self.implementation(query, key, value, **kwargs)
        if output.dtype != torch.bfloat16:
            raise TypeError("BF16 attention returned a non-BF16 output")
        self.calls += 1
        return output


def _checked_gradient(name, parameter):
    gradient = parameter.grad
    if gradient is None or gradient.dtype != torch.bfloat16 or gradient.shape != parameter.shape:
        raise RuntimeError(f"Missing or invalid BF16 parameter gradient: {name}")
    return gradient.detach()


def _snapshot_gradients(model):
    # Explicit copy also makes CPU fixture snapshots independent of later zero_grad.
    return {
        name: _checked_gradient(name, parameter).to(device="cpu", copy=True)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _compare_current(model, references):
    """Copy one current gradient to CPU; reduce all pairs in bounded FP64 blocks."""
    if not references:
        return {}
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any(list(snapshot) != names for snapshot in references.values()):
        raise AssertionError("Parameter gradient coverage/order changed between arms")
    totals = {arm: {"aa": 0.0, "bb": 0.0, "dd": 0.0, "dot": 0.0, "maximum": 0.0, "elements": 0} for arm in references}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        actual = _checked_gradient(name, parameter).cpu()
        for arm, snapshot in references.items():
            metrics = fast_tensor_difference(snapshot[name], actual)
            sums = totals[arm]
            a, b = metrics["reference_norm"], metrics["candidate_norm"]
            sums["aa"] += a * a
            sums["bb"] += b * b
            sums["dd"] += metrics["difference_norm"] ** 2
            sums["dot"] += (metrics["cosine"] or 0.0) * a * b
            sums["maximum"] = max(sums["maximum"], metrics["max_abs"])
            sums["elements"] += metrics["elements"]
        del actual
    result = {}
    for arm, sums in totals.items():
        aa, bb, dd = sums["aa"], sums["bb"], sums["dd"]
        if not aa or not bb:
            raise ValueError("Empty or zero-norm gradients cannot establish parity")
        result[arm] = {
            "elements": sums["elements"],
            "parameter_count": len(names),
            "gradient_relative_l2": math.sqrt(dd / aa),
            "gradient_cosine": sums["dot"] / math.sqrt(aa * bb),
            "gradient_max_abs": sums["maximum"],
            "reference_gradient_norm": math.sqrt(aa),
            "candidate_gradient_norm": math.sqrt(bb),
        }
    return result


def token_metric_difference(reference, candidate):
    a, b = reference["logprobs"], candidate["logprobs"]
    if not a or len(a) != len(b):
        raise ValueError("Mismatched next-token logprob coverage")
    delta = [float(y) - float(x) for x, y in zip(a, b, strict=True)]
    norm = sum(float(x) ** 2 for x in a)
    return {
        "logprob_rel_l2": math.sqrt(sum(x * x for x in delta) / norm) if norm else None,
        "logprob_max_abs": max(abs(x) for x in delta),
        "logprob_mean_abs": sum(abs(x) for x in delta) / len(delta),
        "cross_entropy_abs_gap": abs(sum(delta)) / len(delta),
    }


def run_bf16_quick(
    model,
    tokens,
    prefix_length,
    output_dir,
    *,
    production_attention=True,
    attention_mode="cann",
    trace_forward=False,
    forward_baselines=None,
):
    """Keep two CPU BF16 baselines; write just report.json, never gradient shards."""
    if tokens.ndim != 1 or tokens.dtype != torch.long or not 0 < prefix_length < tokens.numel() - 1:
        raise ValueError("Require 1-D token IDs, nonempty prefix, and suffix of at least two tokens")
    parameters = dict(model.named_parameters())
    if not parameters or any(p.dtype != torch.bfloat16 for p in parameters.values()):
        raise TypeError("BF16 quick diagnostic requires BF16 model parameters")
    delayed = os.getenv("TPR_EXPERIMENT_DELAYED_DKV_FP32", "0")
    if delayed not in ("0", "1"):
        raise ValueError("TPR_EXPERIMENT_DELAYED_DKV_FP32 must be 0 or 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    versions = {name: p._version for name, p in parameters.items()}
    report = {
        "diagnostic_only": True,
        "dtype": "torch.bfloat16",
        "gradient_storage": "BF16 parameter.grad; no optimizer/main_grad wrapper",
        "metric_accumulation": "CPU NumPy FP64 blocks",
        "attention": attention_mode if production_attention else f"CPU test stand-in/{attention_mode}",
        "delayed_dkv_fp32": delayed == "1",
        "loss": "fixed_linear_all_queries_except_last_FP32_reduction",
        "prefix_length": prefix_length,
        "suffix_length": tokens.numel() - prefix_length,
        "input_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
        "token_ids": tokens.cpu().tolist(),
        "model_parameter_elements": sum(p.numel() for p in parameters.values()),
        "parameter_dumps": False,
        "layer_traces": "forward_statistics_only" if trace_forward else False,
        "local_fp32_scope": "QK/softmax/PV only; BF16 input/output boundary" if attention_mode == "math_fp32" else None,
        "arms": {},
        "pairs": {},
    }
    references = {}
    trace = None
    full_trace = None
    model.train()
    try:
        with ExitStack() as stack:
            if production_attention:
                import verl.models.mcore.tpr.attention as attention_module
                import verl.models.mcore.tpr.rectangular_attention as rectangular_module

                implementation = rectangular_module.rectangular_causal_attention
            else:
                implementation = model.math_attention
            implementation = attention_implementation(attention_mode, implementation)
            if trace_forward:
                trace = LayerTrace(forward_only=True, attention_impl=implementation, attention_stage="attention_core")
                trace.attach(model)
            counter = _AttentionCounter(trace.attention if trace is not None else implementation, tokens, trace)
            if production_attention:
                stack.enter_context(patch.object(attention_module, "rectangular_causal_attention", counter.attention))
                stack.enter_context(patch.object(rectangular_module, "rectangular_causal_attention", counter.attention))
            else:
                stack.enter_context(patch.object(model, "math_attention", counter.attention))
            for arm in ARMS:
                print(f"SPLIT_BF16_BEGIN {arm}", flush=True)
                model.zero_grad(set_to_none=True)
                counter.reset()
                if trace is not None:
                    trace.arm, trace.records = arm, []
                loss = _run_arm(model, tokens, prefix_length, arm, counter, loss_fn=_bf16_loss)
                expected = len(model.decoder.layers) * {"full": 1, "split_connected": 2, "split_replay": 3}[arm]
                if counter.calls != expected:
                    raise AssertionError(f"BF16 attention coverage: expected {expected}, got {counter.calls}")
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(f"Nonfinite {arm} loss")
                if versions != {name: p._version for name, p in model.named_parameters()}:
                    raise AssertionError("Model weights changed between arms")
                report["arms"][arm] = {"loss": loss_value, "attention_calls": counter.calls, **counter.observations()}
                for reference, metrics in _compare_current(model, references).items():
                    metrics["loss_abs_gap"] = abs(report["arms"][reference]["loss"] - loss_value)
                    metrics.update(token_metric_difference(report["arms"][reference], report["arms"][arm]))
                    report["pairs"][f"{reference}_vs_{arm}"] = metrics
                if trace is not None:
                    if arm == "full":
                        full_trace = trace.records
                        for mode, baseline in (forward_baselines or {}).items():
                            name = f"full_{mode}_vs_{attention_mode}"
                            report.setdefault("cross_attention_forward", {})[name] = write_forward_comparison(
                                baseline,
                                full_trace,
                                output_dir / f"{name}_layers.json",
                            )
                        if forward_baselines is not None:
                            forward_baselines[attention_mode] = full_trace
                    else:
                        report.setdefault("forward_summary", {})[arm] = write_forward_comparison(
                            full_trace,
                            trace.records,
                            output_dir / f"full_vs_{arm}_layers.json",
                        )
                if arm != ARMS[-1]:
                    references[arm] = _snapshot_gradients(model)
                del loss
                print(f"SPLIT_BF16_ARM_DONE {arm} loss={loss_value:.9g}", flush=True)
        (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        if trace is not None:
            trace.close()
            trace.records.clear()
        references.clear()
        model.zero_grad(set_to_none=True)
