# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FP32 split attribution helpers; no fused attention or PPO in this diagnostic.

Full and connected split use ordinary autograd. Replay uses the production
SegmentExecutor. Every query except the last has the same fixed linear loss,
including the prefix's last query at the split boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch import Tensor

ARMS = ("full", "split_connected", "split_replay")


def math_attention(query, key, value, *, softmax_scale=None, dropout_p=0.0):
    """Explicit GQA attention with right-aligned global causal positions."""
    if any(t.dtype != torch.float32 for t in (query, key, value)):
        raise TypeError("FP32 math attention received a non-FP32 tensor")
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("Expected Q/K/V [sequence,batch,heads,head_dim]")
    sq, batch, hq, dim = query.shape
    sk, bk, hk, dk = key.shape
    if not 0 < sq <= sk or batch != bk or dim != dk or hq % hk:
        raise ValueError("Incompatible rectangular GQA shapes")
    if dropout_p != 0:
        raise ValueError("Attribution requires dropout=0")
    q = query.permute(1, 2, 0, 3)
    k = key.permute(1, 2, 0, 3).repeat_interleave(hq // hk, dim=1)
    v = value.permute(1, 2, 0, 3).repeat_interleave(hq // hk, dim=1)
    scale = 1 / math.sqrt(dim) if softmax_scale is None else softmax_scale
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    q_positions = torch.arange(sq, device=query.device) + sk - sq
    k_positions = torch.arange(sk, device=query.device)
    mask = k_positions[None, :] > q_positions[:, None]
    probabilities = scores.masked_fill(mask, -torch.inf).softmax(dim=-1)
    output = torch.matmul(probabilities, v)
    return output.permute(2, 0, 1, 3).reshape(sq, batch, hq * dim)


def tensor_difference(reference, candidate, *, atol=1e-5, rtol=1e-4):
    """CPU FP64 blockwise comparison, shared with the offline recovery path."""
    from .fast_fp32_grad_compare import fast_fp32_tensor_difference

    return fast_fp32_tensor_difference(reference, candidate, atol=atol, rtol=rtol)


def _tensor(output):
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple | list):
        return next((item for item in output if isinstance(item, Tensor)), None)
    return None


class LayerTrace:
    """Save aligned module outputs/gradients; Push and Replay remain separate."""

    def __init__(self):
        self.records = []
        self.handles = []
        self.active_layer = None
        self.arm = None

    def record(self, stage, tensor, *, sequence_dim=0, start=None, record_grad=True):
        from verl.models.mcore.tpr.context import get_tpr_attention_context

        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            return
        if tensor.dtype != torch.float32:
            raise TypeError(f"{stage}: expected FP32, received {tensor.dtype}")
        context = get_tpr_attention_context()
        if start is None:
            start = 0 if context is None else context.prefix_length
        snapshot = tensor.detach().movedim(sequence_dim, 0).cpu().clone()
        item = {
            "stage": stage,
            "arm": self.arm,
            "start": start,
            "end": start + snapshot.shape[0],
            "forward": snapshot,
            "gradient": None,
            "grad_enabled": torch.is_grad_enabled(),
            "expect_gradient": record_grad and tensor.requires_grad and torch.is_grad_enabled(),
        }
        self.records.append(item)
        if record_grad and tensor.requires_grad and torch.is_grad_enabled():

            def observe(gradient):
                if gradient.dtype != torch.float32:
                    raise TypeError(f"{stage}: non-FP32 backward gradient")
                gradient = gradient.detach().movedim(sequence_dim, 0).cpu().clone()
                previous = item["gradient"]
                item["gradient"] = gradient if previous is None else previous + gradient

            self.handles.append(tensor.register_hook(observe))

    def attach(self, model):
        if hasattr(model, "embedding"):
            self.handles.append(
                model.embedding.register_forward_hook(lambda _m, _a, output: self.record("embedding", _tensor(output)))
            )
        for layer in model.decoder.layers:
            number = layer.layer_number
            base = f"layer.{number:02d}"

            def before(_module, args, kwargs, number=number, base=base):
                self.active_layer = number
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                self.record(f"{base}.input", hidden)

            def after(_module, _args, output, base=base):
                self.record(f"{base}.output", _tensor(output))
                self.active_layer = None

            self.handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(after))
            slots = {
                "input_norm": getattr(layer, "input_layernorm", None),
                "qkv": getattr(layer.self_attention, "linear_qkv", None),
                "q_norm": getattr(layer.self_attention, "q_layernorm", None),
                "k_norm": getattr(layer.self_attention, "k_layernorm", None),
                "attention": layer.self_attention,
                "pre_mlp_norm": getattr(layer, "pre_mlp_layernorm", None),
                "mlp": getattr(layer, "mlp", None),
            }
            for name, module in slots.items():
                if module is not None:

                    def capture(_module, _args, output, stage=f"{base}.{name}"):
                        self.record(stage, _tensor(output))

                    self.handles.append(module.register_forward_hook(capture))
        if hasattr(model.decoder, "final_layernorm"):
            self.handles.append(
                model.decoder.final_layernorm.register_forward_hook(
                    lambda _m, _a, output: self.record("final_norm", _tensor(output))
                )
            )

    def attention(self, query, key, value, **kwargs):
        base = f"layer.{self.active_layer:02d}"
        # Full K/V and suffix K/V overlap in global positions. Their hook
        # gradients are partial/combined contributions, not directly comparable.
        self.record(f"{base}.q_rope", query, record_grad=False)
        self.record(f"{base}.k_all", key, start=0, record_grad=False)
        self.record(f"{base}.v_all", value, start=0, record_grad=False)
        output = math_attention(query, key, value, **kwargs)
        self.record(f"{base}.attention_math", output)
        return output

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _loss(logits, start, total_length):
    """Fixed dense upstream gradient, identical at each global query position."""
    if logits.dtype != torch.float32 or logits.shape[0] != 1:
        raise TypeError("Expected single-trajectory FP32 logits")
    positions = torch.arange(start, start + logits.shape[1], device=logits.device)
    token_weights = torch.sin(positions.float() + 1).masked_fill(positions == total_length - 1, 0)
    vocab_weights = torch.cos(torch.arange(logits.shape[-1], device=logits.device).float() * 0.173)
    return (logits * token_weights[None, :, None] * vocab_weights[None, None, :]).sum() / (
        (total_length - 1) * math.sqrt(logits.shape[-1])
    )


def _forward(model, tokens, start, trace):
    positions = torch.arange(start, start + tokens.numel(), device=tokens.device)[None, :]
    output = model(input_ids=tokens[None, :], position_ids=positions, attention_mask=None)
    trace.record("logits", output, sequence_dim=1)
    return output


def _save_gradients(model, folder):
    manifest = []
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise RuntimeError(f"Missing/non-FP32 parameter gradient: {name}")
        filename = f"grad_{index:05d}.pt"
        torch.save(parameter.grad.detach().cpu(), folder / filename)
        manifest.append({"name": name, "file": filename, "shape": list(parameter.shape)})
    if not manifest:
        raise RuntimeError("No parameter gradients exported")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _compare_traces(reference, candidate, *, atol, rtol):
    by_stage = {item["stage"]: item for item in reference}
    if len(by_stage) != len(reference):
        raise AssertionError("Full trace has duplicate stage records")
    if set(by_stage) != {item["stage"] for item in candidate}:
        raise AssertionError("Full and split trace stages differ")
    result = []
    for index, item in enumerate(candidate):
        baseline = by_stage[item["stage"]]
        start, end = item["start"], item["end"]
        record = {
            "event": index,
            "stage": item["stage"],
            "start": start,
            "end": end,
            "grad_enabled": item["grad_enabled"],
        }
        if item["expect_gradient"] and item["gradient"] is None:
            raise AssertionError(f"Missing split gradient hook: {item['stage']}")
        for kind in ("forward", "gradient"):
            if item[kind] is None:
                continue
            if baseline[kind] is None:
                raise AssertionError(f"Missing Full {kind}: {item['stage']}")
            record[kind] = tensor_difference(baseline[kind][start:end], item[kind], atol=atol, rtol=rtol)
        result.append(record)
    order = {name: index for index, name in enumerate(by_stage)}
    return sorted(result, key=lambda item: (order[item["stage"]], item["start"], item["event"]))


def _run_arm(model, tokens, prefix_length, arm, trace, *, loss_fn=_loss):
    """Shared Full/connected/production Replay execution for precision diagnostics."""
    from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context
    from verl.workers.engine.megatron.tpr.executor import SegmentExecutor, build_suffix_rotary_pos_emb
    from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec

    if arm not in ARMS:
        raise ValueError(f"Unknown diagnostic arm: {arm}")
    length = tokens.numel()
    if arm == "full":
        loss = loss_fn(_forward(model, tokens, 0, trace), 0, length)
        loss.backward()
    elif arm == "split_connected":
        rotary = build_suffix_rotary_pos_emb(model.rotary_pos_emb, prefix_length=0, suffix_length=prefix_length)
        prefix = TPRAttentionContext(0, prefix_length, suffix_rotary_pos_emb=rotary)
        with use_tpr_attention_context(prefix):
            prefix_loss = loss_fn(_forward(model, tokens[:prefix_length], 0, trace), 0, length)
        if len(prefix.new_key_values) != len(model.decoder.layers) or any(
            not tensor.requires_grad for pair in prefix.new_key_values.values() for tensor in pair
        ):
            raise AssertionError("Connected split lost a live prefix KV graph")
        rotary = build_suffix_rotary_pos_emb(
            model.rotary_pos_emb, prefix_length=prefix_length, suffix_length=length - prefix_length
        )
        suffix = TPRAttentionContext(
            prefix_length,
            length - prefix_length,
            past_key_values=prefix.new_key_values,
            suffix_rotary_pos_emb=rotary,
        )
        with use_tpr_attention_context(suffix):
            suffix_loss = loss_fn(_forward(model, tokens[prefix_length:], prefix_length, trace), prefix_length, length)
        loss = prefix_loss + suffix_loss
        loss.backward()
        del prefix, suffix, prefix_loss, suffix_loss
    else:
        plan = SegmentPlan(
            (
                SegmentSpec(0, None, tokens[:prefix_length].cpu(), 0, 0),
                SegmentSpec(1, 0, tokens[prefix_length:].cpu(), prefix_length, prefix_length),
            ),
            root_id=0,
        )

        def objective(segment, logits):
            trace.record("logits", logits, sequence_dim=1, start=segment.position_start)
            return loss_fn(logits, segment.position_start, length), {}

        executor = SegmentExecutor(
            model,
            plan,
            expected_layer_numbers=tuple(layer.layer_number for layer in model.decoder.layers),
            segment_loss_fn=objective,
            segment_loss_term_counts={0: prefix_length, 1: length - prefix_length - 1},
        )
        loss = executor.run()
        del executor
    return loss


def run_three_way(model, tokens, prefix_length, output_dir, *, production_attention=False, atol=1e-5, rtol=1e-4):
    """Run on one unchanged model; stream FP32 parameter gradients to disk."""
    length = tokens.numel()
    if tokens.ndim != 1 or tokens.dtype != torch.long or not 0 < prefix_length < length - 1:
        raise ValueError("Require 1-D token IDs, nonempty prefix, and suffix of at least two tokens")
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise TypeError(f"Non-FP32 model tensor: {name} ({tensor.dtype})")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    trace = LayerTrace()
    model.train()
    report = {
        "diagnostic_only": True,
        "dtype": "torch.float32",
        "attention": "explicit_math_gqa",
        "loss": "fixed_linear_all_queries_except_last",
        "prefix_length": prefix_length,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_parameter_elements": sum(p.numel() for p in model.parameters()),
        "suffix_length": length - prefix_length,
        "atol": atol,
        "rtol": rtol,
        "input_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
        "token_ids": tokens.cpu().tolist(),
        "arms": {},
        "pairs": {},
    }
    try:
        trace.attach(model)
        with ExitStack() as stack:
            if production_attention:
                import verl.models.mcore.tpr.attention as attention_module
                import verl.models.mcore.tpr.rectangular_attention as rectangular_module

                stack.enter_context(patch.object(attention_module, "rectangular_causal_attention", trace.attention))
                stack.enter_context(patch.object(rectangular_module, "rectangular_causal_attention", trace.attention))
            else:
                # Tiny CPU fixture uses the same diagnostic hook as real Qwen.
                stack.enter_context(patch.object(model, "math_attention", trace.attention))
            for arm in ARMS:
                trace.arm, trace.records = arm, []
                model.zero_grad(set_to_none=True)
                folder = output_dir / arm
                folder.mkdir()
                loss = _run_arm(model, tokens, prefix_length, arm, trace)
                expected_calls = len(model.decoder.layers) * {"full": 1, "split_connected": 2, "split_replay": 3}[arm]
                actual_calls = sum(item["stage"].endswith(".attention_math") for item in trace.records)
                if actual_calls != expected_calls:
                    raise AssertionError(f"Math attention coverage: expected {expected_calls}, got {actual_calls}")
                manifest = _save_gradients(model, folder)
                torch.save(trace.records, folder / "trace.pt")
                report["arms"][arm] = {
                    "loss": float(loss.detach()),
                    "parameter_count": len(manifest),
                    "trace_records": len(trace.records),
                    "math_attention_calls": actual_calls,
                }
                if versions != {name: p._version for name, p in model.named_parameters()}:
                    raise AssertionError("Model weights changed between arms")
                del loss
        for candidate in ARMS[1:]:
            full_trace = torch.load(output_dir / "full/trace.pt", weights_only=True, map_location="cpu")
            other_trace = torch.load(output_dir / candidate / "trace.pt", weights_only=True, map_location="cpu")
            stages = _compare_traces(full_trace, other_trace, atol=atol, rtol=rtol)
            (output_dir / f"full_vs_{candidate}_layers.json").write_text(json.dumps(stages, indent=2))
            report.setdefault("layer_summary", {})[candidate] = {
                kind: next(
                    (
                        {key: event[key] for key in ("event", "stage", "start", "end", "grad_enabled")}
                        for event in stages
                        if kind in event and not event[kind]["within_tolerance"]
                    ),
                    None,
                )
                for kind in ("forward", "gradient")
            }
            del full_trace, other_trace
        for left, right in (("full", "split_connected"), ("full", "split_replay"), ("split_connected", "split_replay")):
            manifests = [json.loads((output_dir / arm / "manifest.json").read_text()) for arm in (left, right)]
            if manifests[0] != manifests[1]:
                raise AssertionError("Parameter coverage differs between arms")
            metrics = []
            for entry in manifests[0]:
                a, b = [
                    torch.load(str(output_dir / arm / entry["file"]), weights_only=True, map_location="cpu", mmap=True)
                    for arm in (left, right)
                ]
                metrics.append({"name": entry["name"], **tensor_difference(a, b, atol=atol, rtol=rtol)})
                del a, b
            report["pairs"][f"{left}_vs_{right}"] = {
                "loss_abs_gap": abs(report["arms"][left]["loss"] - report["arms"][right]["loss"]),
                "parameters": metrics,
                "gradient_relative_l2": math.sqrt(
                    sum(item["difference_norm"] ** 2 for item in metrics)
                    / max(sum(item["reference_norm"] ** 2 for item in metrics), 1e-300)
                ),
                "gradient_max_abs": max(item["max_abs"] for item in metrics),
                "all_parameters_within_tolerance": all(item["within_tolerance"] for item in metrics),
            }
            reference_norm = math.sqrt(sum(item["reference_norm"] ** 2 for item in metrics))
            candidate_norm = math.sqrt(sum(item["candidate_norm"] ** 2 for item in metrics))
            dot = sum((item["cosine"] or 0.0) * item["reference_norm"] * item["candidate_norm"] for item in metrics)
            report["pairs"][f"{left}_vs_{right}"].update(
                reference_gradient_norm=reference_norm,
                candidate_gradient_norm=candidate_norm,
                gradient_cosine=dot / (reference_norm * candidate_norm) if reference_norm and candidate_norm else None,
            )
        (output_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        trace.close()
        model.zero_grad(set_to_none=True)
