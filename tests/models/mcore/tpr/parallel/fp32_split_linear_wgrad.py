"""Test-only FP32 dW counterfactual for sequence-split Qwen3 training.

Intercepts the four dense Linear families in every Transformer layer. Forward
outputs are unchanged; each segment's dW is computed and accumulated into one
FP32 buffer per parameter, then cast to the parameter dtype once before the
existing Megatron gradient finalizer. dX remains BF16. This is a diagnostic
counterfactual, not a production optimizer/DDP implementation.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


class _FP32DenseWeightGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, inputs, weight, buffer, owner, name):
        if inputs.shape[:-1] != output.shape[:-1]:
            raise AssertionError(
                f"{name}: input/output token dimensions differ: "
                f"{tuple(inputs.shape)} vs {tuple(output.shape)}"
            )
        if inputs.shape[-1] != weight.shape[-1] or output.shape[-1] != weight.shape[0]:
            raise AssertionError(
                f"{name}: GEMM shape mismatch input={tuple(inputs.shape)} "
                f"output={tuple(output.shape)} weight={tuple(weight.shape)}"
            )
        if inputs.dtype != weight.dtype or output.dtype != weight.dtype:
            raise AssertionError(
                f"{name}: expected matching BF16 input/output/weight dtypes, got "
                f"{inputs.dtype}/{output.dtype}/{weight.dtype}"
            )
        ctx.save_for_backward(inputs, weight)
        ctx.buffer = buffer
        ctx.owner = owner
        ctx.name = name
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        x = inputs.detach().reshape(-1, inputs.shape[-1]).contiguous()
        dy = grad_output.detach().reshape(-1, grad_output.shape[-1]).contiguous()
        if x.shape[0] != dy.shape[0]:
            raise AssertionError(
                f"{ctx.name}: flattened input/dY row mismatch {x.shape[0]} vs {dy.shape[0]}"
            )

        # Real FP32 per-segment dW GEMM and FP32 accumulation; no BF16 dW
        # intermediate is materialized. Only the final parameter writeback
        # is BF16. Keep dX in the model's original BF16 dtype.
        with torch.no_grad():
            ctx.buffer.addmm_(dy.float().transpose(0, 1), x.float())
            dx = grad_output.to(dtype=weight.dtype).matmul(weight)
            if dx.dtype != inputs.dtype:
                dx = dx.to(inputs.dtype)

        ctx.owner.backward_counts[ctx.name] += 1
        return None, dx, None, None, None, None


class FP32SplitLinearWGrad:
    """Opt-in test hook for FP32 dW on all Transformer Linear layers."""

    LINEAR_FAMILIES = ("qkv", "proj", "fc1", "fc2")

    def __init__(self, model: Any):
        self.model = model
        self.original_finalizer: Callable[..., Any] | None = getattr(
            model.config, "finalize_model_grads_func", None
        )
        if self.original_finalizer is None:
            raise AssertionError("FP32 dW intervention requires Megatron gradient finalizer")
        self.buffers: dict[torch.nn.Parameter, Tensor | None] = {}
        self.parameter_names: dict[torch.nn.Parameter, str] = {}
        self.forward_counts: dict[str, int] = {}
        self.backward_counts: dict[str, int] = {}
        self.handles: list[Any] = []
        self.installed = False
        self.flushed = False
        self.closed = False
        self._wrapped_finalizer = None

    def install(self) -> "FP32SplitLinearWGrad":
        if self.installed:
            raise RuntimeError("FP32 dW intervention is already installed")
        layers = self.model.decoder.layers
        for layer in layers:
            layer_id = int(layer.self_attention.layer_number)
            modules = {
                "qkv": layer.self_attention.linear_qkv,
                "proj": layer.self_attention.linear_proj,
                "fc1": layer.mlp.linear_fc1,
                "fc2": layer.mlp.linear_fc2,
            }
            for family in self.LINEAR_FAMILIES:
                module = modules[family]
                weight = getattr(module, "weight", None)
                if not isinstance(weight, torch.nn.Parameter) or not weight.requires_grad:
                    raise AssertionError(f"L{layer_id:02d}_{family}: missing trainable weight")
                if weight.dtype != torch.bfloat16:
                    raise AssertionError(
                        f"L{layer_id:02d}_{family}: expected BF16 weight, got {weight.dtype}"
                    )
                if getattr(module, "bias", None) is not None:
                    raise AssertionError(
                        f"L{layer_id:02d}_{family}: bias is unsupported by this test-only hook"
                    )
                if weight in self.buffers:
                    raise AssertionError(
                        f"Unexpected shared dense Linear parameter at L{layer_id:02d}_{family}"
                    )

                name = f"L{layer_id:02d}_{family}"
                self.buffers[weight] = torch.zeros_like(weight, dtype=torch.float32)
                self.parameter_names[weight] = name
                self.forward_counts[name] = 0
                self.backward_counts[name] = 0
                self.handles.append(
                    module.register_forward_hook(self._make_hook(name, weight))
                )

        if len(self.buffers) != len(layers) * len(self.LINEAR_FAMILIES):
            raise AssertionError(
                f"Expected four dense Linear weights per Transformer layer, got {len(self.buffers)}"
            )

        def finalize_with_fp32_dw(*args, **kwargs):
            self.flush()
            return self.original_finalizer(*args, **kwargs)

        self._wrapped_finalizer = finalize_with_fp32_dw
        self.model.config.finalize_model_grads_func = finalize_with_fp32_dw
        self.installed = True
        return self

    def _make_hook(self, name: str, weight: torch.nn.Parameter):
        def route(_module, args, output):
            if not torch.is_grad_enabled():
                return output
            if not args or not isinstance(args[0], Tensor):
                raise AssertionError(f"{name}: expected first Linear argument to be a Tensor")
            inputs = args[0]
            is_tuple = isinstance(output, tuple)
            if is_tuple:
                if len(output) != 2 or output[1] is not None:
                    raise AssertionError(f"{name}: only bias-free Linear outputs are supported")
                raw_output = output[0]
            else:
                raw_output = output
            if not isinstance(raw_output, Tensor):
                raise AssertionError(f"{name}: unexpected Linear output {type(raw_output).__name__}")
            if not raw_output.requires_grad:
                raise AssertionError(f"{name}: Linear output unexpectedly has no gradient")
            buffer = self.buffers[weight]
            if buffer is None:
                raise AssertionError(f"{name}: FP32 gradient buffer was already flushed")
            self.forward_counts[name] += 1
            surrogate = _FP32DenseWeightGrad.apply(
                raw_output.detach(), inputs, weight, buffer, self, name
            )
            return (surrogate, output[1]) if is_tuple else surrogate

        return route

    def flush(self) -> None:
        if self.flushed:
            raise RuntimeError("FP32 dW buffers were flushed more than once")
        if not self.installed:
            raise RuntimeError("FP32 dW intervention was not installed")

        missing = {
            name: (self.forward_counts[name], self.backward_counts[name])
            for name in self.forward_counts
            if self.forward_counts[name] == 0
            or self.backward_counts[name] != self.forward_counts[name]
        }
        if missing:
            raise AssertionError(
                f"FP32 dW forward/backward counts mismatch: {missing}"
            )

        for weight, buffer in self.buffers.items():
            if buffer is None:
                raise AssertionError("FP32 dW buffer unexpectedly missing before flush")
            if weight.grad is not None:
                raise AssertionError(
                    f"{self.parameter_names[weight]} unexpectedly materialized native .grad"
                )
            weight.grad = buffer.to(dtype=weight.dtype)
            self.buffers[weight] = None

        self.flushed = True

    def summary(self) -> dict[str, Any]:
        forward = list(self.forward_counts.values())
        backward = list(self.backward_counts.values())
        return {
            "enabled": True,
            "scope": "all_transformer_layers_qkv_proj_fc1_fc2_weight_gradients",
            "layers": len(self.model.decoder.layers),
            "linear_modules": len(self.forward_counts),
            "fp32_buffers": len(self.forward_counts),
            "forward_calls_total": sum(forward),
            "backward_calls_total": sum(backward),
            "min_forward_calls_per_module": min(forward, default=0),
            "max_forward_calls_per_module": max(forward, default=0),
            "min_backward_calls_per_module": min(backward, default=0),
            "max_backward_calls_per_module": max(backward, default=0),
            "all_modules_backwarded": bool(forward) and all(
                self.forward_counts[name] == self.backward_counts[name] > 0
                for name in self.forward_counts
            ),
            "accumulation_dtype": "torch.float32",
            "writeback_dtype": "torch.bfloat16",
            "dx_dtype": "torch.bfloat16",
            "flushed_before_native_finalizer": self.flushed,
        }

    def close(self) -> None:
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._wrapped_finalizer is not None:
            if self.model.config.finalize_model_grads_func is self._wrapped_finalizer:
                self.model.config.finalize_model_grads_func = self.original_finalizer
        self.buffers.clear()
        self.closed = True
