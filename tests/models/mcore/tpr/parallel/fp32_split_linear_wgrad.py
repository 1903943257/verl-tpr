"""Test-only FP32 dW counterfactual for sequence-split Qwen3 training.

Forward hooks capture each segment's actual Linear input and dY. Native Linear
backward remains intact for dX; its BF16 weight gradient is discarded by a
parameter hook. The captured dY.T @ X is computed and accumulated in FP32, then
cast to the parameter dtype once before the existing gradient finalizer. This
is a diagnostic counterfactual, not a production optimizer/DDP implementation.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class FP32SplitLinearWGrad:
    """Opt-in FP32 dW accumulation for all dense Linear weights in all layers."""

    LINEAR_FAMILIES = ("qkv", "proj", "fc1", "fc2")

    def __init__(self, model: Any):
        self.model = model
        self.original_finalizer = getattr(
            model.config, "finalize_model_grads_func", None
        )
        if self.original_finalizer is None:
            raise AssertionError("FP32 dW intervention requires Megatron gradient finalizer")
        self.buffers: dict[torch.nn.Parameter, Tensor | None] = {}
        self.parameter_names: dict[torch.nn.Parameter, str] = {}
        self.forward_counts: dict[str, int] = {}
        self.backward_counts: dict[str, int] = {}
        self.native_weight_hook_counts: dict[str, int] = {}
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
                self.native_weight_hook_counts[name] = 0
                self.handles.append(module.register_forward_hook(
                    self._make_forward_hook(name, weight)
                ))
                self.handles.append(weight.register_hook(
                    self._make_parameter_hook(name)
                ))

        expected = len(layers) * len(self.LINEAR_FAMILIES)
        if len(self.buffers) != expected:
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

    def _make_forward_hook(self, name: str, weight: torch.nn.Parameter):
        def capture(_module, args, output):
            if not torch.is_grad_enabled():
                return output
            if not args or not isinstance(args[0], Tensor):
                raise AssertionError(f"{name}: expected first Linear argument to be a Tensor")
            inputs = args[0].detach()
            is_tuple = isinstance(output, tuple)
            if is_tuple:
                if len(output) != 2 or output[1] is not None:
                    raise AssertionError(f"{name}: only bias-free Linear outputs are supported")
                raw_output = output[0]
            else:
                raw_output = output
            if not isinstance(raw_output, Tensor) or not raw_output.requires_grad:
                raise AssertionError(f"{name}: unexpected Linear output or missing grad_fn")
            if inputs.shape[:-1] != raw_output.shape[:-1]:
                raise AssertionError(
                    f"{name}: input/output token dimensions differ: "
                    f"{tuple(inputs.shape)} vs {tuple(raw_output.shape)}"
                )
            if inputs.shape[-1] != weight.shape[-1] or raw_output.shape[-1] != weight.shape[0]:
                raise AssertionError(
                    f"{name}: GEMM shape mismatch input={tuple(inputs.shape)} "
                    f"output={tuple(raw_output.shape)} weight={tuple(weight.shape)}"
                )
            if inputs.dtype != weight.dtype or raw_output.dtype != weight.dtype:
                raise AssertionError(
                    f"{name}: expected BF16 input/output/weight, got "
                    f"{inputs.dtype}/{raw_output.dtype}/{weight.dtype}"
                )
            if self.buffers[weight] is None:
                raise AssertionError(f"{name}: FP32 gradient buffer was already flushed")

            self.forward_counts[name] += 1
            buffer = self.buffers[weight]

            def accumulate_fp32_dw(grad_output, *, x=inputs, target=buffer, module_name=name):
                dy = grad_output.detach().reshape(-1, grad_output.shape[-1]).contiguous()
                x2 = x.reshape(-1, x.shape[-1]).contiguous()
                if x2.shape[0] != dy.shape[0]:
                    raise AssertionError(
                        f"{module_name}: flattened input/dY row mismatch "
                        f"{x2.shape[0]} vs {dy.shape[0]}"
                    )
                with torch.no_grad():
                    target.addmm_(dy.float().transpose(0, 1), x2.float())
                self.backward_counts[module_name] += 1
                return grad_output

            raw_output.register_hook(accumulate_fp32_dw)
            return output

        return capture

    def _make_parameter_hook(self, name: str):
        def discard_native_dw(grad):
            # The module output hook already accumulated this invocation's
            # dW from the same real X and dY into FP32. Preserve native dX by
            # leaving the Linear graph intact, but discard its native dW.
            self.native_weight_hook_counts[name] += 1
            return torch.zeros_like(grad)
        return discard_native_dw

    def flush(self) -> None:
        if self.flushed:
            raise RuntimeError("FP32 dW buffers were flushed more than once")
        if not self.installed:
            raise RuntimeError("FP32 dW intervention was not installed")

        missing = {
            name: {
                "forward": self.forward_counts[name],
                "dY_hook": self.backward_counts[name],
                "native_weight_hook": self.native_weight_hook_counts[name],
            }
            for name in self.forward_counts
            if self.forward_counts[name] == 0
            # Some segment outputs (notably final-layer Root activations whose
            # positions are not supervised) are legitimately disconnected from
            # the PPO loss. Only outputs that receive dY contribute to dW.
            # Require at least one contribution per parameter and reject
            # impossible counts, rather than requiring every forward to backprop.
            or self.backward_counts[name] == 0
            or self.backward_counts[name] > self.forward_counts[name]
            or self.native_weight_hook_counts[name] == 0
        }
        if missing:
            raise AssertionError(f"FP32 dW hook coverage mismatch: {missing}")

        for weight, buffer in self.buffers.items():
            if buffer is None:
                raise AssertionError("FP32 dW buffer unexpectedly missing before flush")
            # The parameter hook's native BF16 gradient is zeroed. Replace it
            # with the once-cast FP32 sum before the existing finalizer runs.
            weight.grad = buffer.to(dtype=weight.dtype)
            self.buffers[weight] = None

        self.flushed = True

    def summary(self) -> dict[str, Any]:
        forward = list(self.forward_counts.values())
        backward = list(self.backward_counts.values())
        native = list(self.native_weight_hook_counts.values())
        return {
            "enabled": True,
            "scope": "all_transformer_layers_qkv_proj_fc1_fc2_weight_gradients",
            "layers": len(self.model.decoder.layers),
            "linear_modules": len(self.forward_counts),
            "fp32_buffers": len(self.forward_counts),
            "forward_calls_total": sum(forward),
            "backward_calls_total": sum(backward),
            "native_weight_hook_calls_total": sum(native),
            "min_forward_calls_per_module": min(forward, default=0),
            "max_forward_calls_per_module": max(forward, default=0),
            "min_backward_calls_per_module": min(backward, default=0),
            "max_backward_calls_per_module": max(backward, default=0),
            "all_modules_have_backward_contribution": bool(forward) and all(
                0 < self.backward_counts[name] <= self.forward_counts[name]
                and self.native_weight_hook_counts[name] > 0
                for name in self.forward_counts
            ),
            "modules_with_unused_forward_outputs": sum(
                self.backward_counts[name] < self.forward_counts[name]
                for name in self.forward_counts
            ),
            "accumulation_dtype": "torch.float32",
            "writeback_dtype": "torch.bfloat16",
            "dx_path": "original_native_linear_backward",
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
