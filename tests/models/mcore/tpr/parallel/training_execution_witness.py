# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Observe the actual engine's core modules and backend calls without replacing them."""

import inspect
import sys
from collections import Counter

import torch


def function_origin(function):
    try:
        filename = inspect.getsourcefile(function)
    except (TypeError, OSError):
        filename = None
    return {
        "module": getattr(function, "__module__", None),
        "name": getattr(function, "__qualname__", type(function).__qualname__),
        "file": filename,
    }


def _native_core(module):
    origin = function_origin(module.forward)
    filename = (origin["file"] or "").replace("\\", "/").lower()
    packages = ("mindspeed.", "megatron.core.", "transformer_engine.")
    return type(module).__module__.startswith(packages) and any(
        part in filename for part in ("/mindspeed/", "/megatron/core/", "/transformer_engine/")
    )


def _fusion_name(name):
    name = name.split("::")[-1].split(".")[0]
    return name.startswith("npu_fusion_attention") and "grad" not in name


class ExecutionWitness:
    """Read-only hooks identify real core instances; profiling identifies backend calls.

    MindSpeed can install ordinary DPA functions or TE modules whose method is
    simply named forward. A function-name-only allowlist misses the latter.
    Binding the actual core instances also prevents an unrelated FA call from
    qualifying a replacement attention adapter as the native baseline.
    """

    def __init__(self, finalize, core_modules=()):
        self.finalize_code = finalize.__code__
        self.counts = Counter()
        self.cores = tuple(core_modules)
        self.core_records = [
            dict(
                class_name=f"{type(module).__module__}.{type(module).__qualname__}",
                forward=function_origin(module.forward),
                native=_native_core(module),
                calls=0,
                input_signature=None,
            )
            for module in self.cores
        ]
        self.entries = Counter()
        self.handles = []
        self.active_native_cores = 0

    def _before_core(self, index, args, kwargs):
        record = self.core_records[index]
        record["calls"] += 1
        if record["input_signature"] is None:
            tensors = list(args[:3]) if args else [kwargs.get(name) for name in ("query", "key", "value")]
            record["input_signature"] = [
                dict(shape=list(value.shape), dtype=str(value.dtype))
                for value in tensors
                if isinstance(value, torch.Tensor)
            ]
        if record["native"]:
            self.counts["native_attention"] += 1
            self.active_native_cores += 1

    def _after_core(self, index):
        if self.core_records[index]["native"]:
            self.active_native_cores -= 1

    def _profile(self, frame, event, argument):
        if event == "call":
            name = frame.f_code.co_name
            filename = frame.f_code.co_filename.replace("\\", "/")
            module = frame.f_globals.get("__name__", "")
            native_package = module.startswith(("mindspeed.", "megatron.core.", "torch_npu."))
            if native_package and any(word in name.lower() for word in ("attention", "forward", "ring", "wgrad")):
                entry = f"{module}:{frame.f_code.co_qualname}@{filename}"
                if entry in self.entries or len(self.entries) < 64:
                    self.entries[entry] += 1
            if frame.f_code is self.finalize_code:
                self.counts["native_finalize"] += 1
            if (
                module.startswith("mindspeed.")
                and name
                in (
                    "ringattn_context_parallel",
                    "ringattn_context_parallel_tnd_general",
                )
                and self.active_native_cores
            ):
                self.counts["native_ring"] += 1
            if (
                any(package in filename for package in ("/megatron/", "/mindspeed/"))
                and name == "forward_backward_no_pipelining"
            ):
                self.counts["native_schedule"] += 1
            if name == "run_tpr_forward_backward_batch" and filename.endswith("/tpr/adapter.py"):
                self.counts["tpr_schedule"] += 1
            if native_package and _fusion_name(name) and self.active_native_cores:
                self.counts["native_fused_attention"] += 1
            # torch_npu APIs may be Torch dispatcher packets, without a
            # Python function named npu_fusion_attention in the call stack.
            if filename.endswith("/torch/_ops.py") and name == "__call__" and self.active_native_cores:
                packet = frame.f_locals.get("self")
                op_name = getattr(packet, "_qualified_op_name", None) or getattr(packet, "_name", "")
                if isinstance(op_name, str) and _fusion_name(op_name):
                    self.counts["native_fused_attention"] += 1
        elif event == "c_call":
            name = getattr(argument, "__name__", "")
            module = getattr(argument, "__module__", "") or ""
            if module.startswith(("torch", "mindspeed")) and _fusion_name(name) and self.active_native_cores:
                self.counts["native_fused_attention"] += 1
        else:
            return
        if name in ("wgrad_gemm_accum_fp32", "npu_matmul_add_fp32"):
            self.counts["wgrad_fp32"] += 1
        if name in ("wgrad_gemm_accum_fp16", "npu_matmul_add_fp16"):
            self.counts["wgrad_fp16"] += 1

    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError("Run the precision gate without another Python profiler")
        try:
            for index, module in enumerate(self.cores):

                def before(_module, args, kwargs, index=index):
                    self._before_core(index, args, kwargs)

                def after(_module, _args, _output, index=index):
                    self._after_core(index)

                self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                self.handles.append(module.register_forward_hook(after, always_call=True))
            sys.setprofile(self._profile)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_exc):
        sys.setprofile(self.previous)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active_native_cores = 0

    def snapshot(self):
        return dict(calls=dict(self.counts), core_attention=self.core_records, observed_entrypoints=dict(self.entries))

    def validate(self, *, tpr, cp, fusion):
        details = f"calls={dict(self.counts)}; inspect runtime_audit_rank*.json for actual module/function provenance"
        if self.counts["native_finalize"] != 1:
            raise AssertionError(f"Expected exactly one native gradient finalize; {details}")
        if self.counts["tpr_schedule" if tpr else "native_schedule"] != 1:
            raise AssertionError(f"Actual training schedule was not observed; {details}")
        if not tpr:
            if not self.core_records or any(
                not record["native"] or not record["calls"] for record in self.core_records
            ):
                raise AssertionError(f"Not every registered native core attention executed; {details}")
            if not self.counts["native_fused_attention"]:
                raise AssertionError(
                    f"No native fused attention call observed inside the registered core modules; {details}"
                )
            if cp == 2 and not self.counts["native_ring"]:
                raise AssertionError(
                    f"Base CP2 did not enter native MindSpeed Ring through its core modules; {details}"
                )
        if fusion and self.counts["wgrad_fp16"]:
            raise AssertionError(f"Observed a low-precision fused wgrad destination; {details}")
