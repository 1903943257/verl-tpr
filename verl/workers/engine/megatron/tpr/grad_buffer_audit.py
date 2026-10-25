"""Opt-in, metadata-only QKV gradient-path audit for real TPR Engine updates.

The probe never casts, clones, retains gradient tensors, or changes grad buffers.
PRE_ACCUM records the autograd contribution, POST_ACCUM the parameter hook
view, AFTER_TREE the completed SegmentExecutor tree, and FINALIZE the native
Megatron finalizer boundary. A reported logical size is not extra HBM usage.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch


def _tensor_meta(tensor: Any) -> dict[str, Any] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
        "logical_bytes": tensor.numel() * tensor.element_size(),
        "data_ptr": tensor.data_ptr(),
        "is_view": tensor._base is not None,
    }


class QKVGradBufferAudit:
    """Read-only hooks on selected TPR QKV parameters (no NPU->CPU copies)."""

    def __init__(self, model: Any, wrapped_model: Any, *, layer_numbers: tuple[int, ...] = (1,),
                 max_hook_events: int = 16):
        if max_hook_events < 0:
            raise ValueError("max_hook_events must be nonnegative")
        layers = getattr(getattr(model, "decoder", None), "layers", ())
        self._parameters: dict[int, torch.nn.Parameter] = {}
        for layer in layers:
            attention = getattr(layer, "self_attention", None)
            layer_no = getattr(attention, "layer_number", None)
            linear = getattr(attention, "linear_qkv", None)
            weight = getattr(linear, "weight", None)
            if layer_no in layer_numbers and isinstance(weight, torch.nn.Parameter):
                self._parameters[layer_no] = weight
        missing = set(layer_numbers) - set(self._parameters)
        if missing:
            raise ValueError(f"QKV grad audit missing layers: {sorted(missing)}")
        self._wrapper = wrapped_model
        self._max_hook_events = max_hook_events
        self._hook_events = 0
        self._handles: list[Any] = []
        self._rank = int(os.environ.get("RANK", "0"))

    def _emit(self, stage: str, layer: int, *, incoming: torch.Tensor | None = None,
              tree_index: int | None = None) -> None:
        weight = self._parameters[layer]
        ddp_config = getattr(self._wrapper, "ddp_config", None)
        payload = {
            "rank": self._rank,
            "stage": stage,
            "layer": layer,
            "tree_index": tree_index,
            "wrapper_type": type(self._wrapper).__name__,
            "ddp_config_type": type(ddp_config).__name__ if ddp_config is not None else None,
            "grad_reduce_in_fp32": getattr(ddp_config, "grad_reduce_in_fp32", None),
            "weight": _tensor_meta(weight),
            "incoming_dw": _tensor_meta(incoming),
            "param_grad": _tensor_meta(weight.grad),
            "main_grad": _tensor_meta(getattr(weight, "main_grad", None)),
        }
        print("TPR_QKV_GRAD_AUDIT " + json.dumps(payload, sort_keys=True), flush=True)

    def _incoming(self, layer: int, grad: torch.Tensor) -> None:
        if self._hook_events < self._max_hook_events:
            self._emit("PRE_ACCUM", layer, incoming=grad)
        self._hook_events += 1
        # Returning None leaves autograd's original gradient unchanged.
        return None

    def _post_accum(self, layer: int) -> None:
        if self._hook_events <= self._max_hook_events:
            self._emit("POST_ACCUM", layer)

    def install(self) -> "QKVGradBufferAudit":
        for layer, weight in self._parameters.items():
            self._handles.append(weight.register_hook(
                lambda grad, layer=layer: self._incoming(layer, grad)
            ))
            if hasattr(weight, "register_post_accumulate_grad_hook"):
                self._handles.append(weight.register_post_accumulate_grad_hook(
                    lambda _param, layer=layer: self._post_accum(layer)
                ))
        for layer in self._parameters:
            self._emit("INSTALLED", layer)
        return self

    def snapshot(self, stage: str, *, tree_index: int | None = None) -> None:
        for layer in self._parameters:
            self._emit(stage, layer, tree_index=tree_index)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
