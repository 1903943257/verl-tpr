# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Optional, test-only observation of ONE already-warmed-up TPR replay.

Importing this module, or using a disabled capture, imports neither torch nor
torch_npu and installs no hooks. Enabled captures preserve every call and return
value. Events measure the current compute stream, NOT aggregate HCCL time;
communication and overlap must be read from the unfiltered native trace.

API reference (Level1, Text/raw retention, memory timeline, MSTX):
https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/ascend_pytorch_profiler/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md
"""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import sys
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from unittest.mock import patch


def _kv_storage(pairs):
    """Host-side metadata only; never retain a Tensor/storage object in records."""
    logical, storage = 0, {}
    for pair in pairs:
        for tensor in pair:
            logical += tensor.numel() * tensor.element_size()
            owner = tensor.untyped_storage()
            storage[(str(tensor.device), owner.data_ptr())] = owner.nbytes()
    return {"logical_bytes": logical, "unique_storage_bytes": sum(storage.values()),
            "storage_count": len(storage)}


def _tensor_shapes(args):
    return [dict(shape=list(arg.shape), dtype=str(arg.dtype), device=str(arg.device))
            for arg in args if hasattr(arg, "shape") and hasattr(arg, "dtype")]


class NPUProfileCapture:
    """One per-rank native profile; directory/rank/config/SHA belong to caller.

    ``with capture: capture.profile_call(label, callback)`` calls callback exactly
    once. Warmups must happen outside. Exit synchronizes once by default; passing
    synchronize_on_exit=False requires the caller to synchronize BEFORE exit.
    Captured timings include instrumentation overhead and are not benchmarks.
    """

    def __init__(self, output_dir, *, metadata=None, enabled=False, engine=None,
                 model=None, synchronize_on_exit=True):
        self.output_dir = Path(output_dir)
        self.metadata = dict(metadata or {})
        self.enabled, self.engine, self.model = enabled, engine, model
        self.synchronize_on_exit = synchronize_on_exit
        self.records, self.cache_records, self.notes, self.artifacts = [], [], [], {}
        self._pending, self._ranges, self._active = [], [], False
        self._tree_index = ContextVar("profile_tree_index", default=None)

    def __enter__(self):
        if not self.enabled:
            return self
        self.torch = importlib.import_module("torch")
        self.torch_npu = importlib.import_module("torch_npu")
        self.npu = self.torch_npu.npu
        if getattr(self.npu, "mstx", None) is None:
            self._note("MSTX API unavailable; CPU record_function ranges and native operator trace remain enabled.")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._stack = ExitStack()
        try:
            api = self.torch_npu.profiler
            experimental = dict(profiler_level=api.ProfilerLevel.Level1,
                                data_simplification=False, export_type=[api.ExportType.Text])
            metrics = getattr(getattr(api, "AiCMetrics", None), "PipeUtilization", None)
            if metrics is not None:
                experimental["aic_metrics"] = metrics
            try:
                parameters = inspect.signature(api._ExperimentalConfig).parameters
            except (TypeError, ValueError):
                parameters = {}
            marker_key = next((key for key in ("mstx", "msprof_tx") if key in parameters), None)
            self._mstx_enabled = marker_key is not None
            if marker_key:
                experimental[marker_key] = True
            else:
                self._note("Optional MSTX config flag not advertised; using CPU record_function ranges.")
            config = api._ExperimentalConfig(**experimental)
            handler = api.tensorboard_trace_handler(str(self.output_dir / "native"))

            def on_trace_ready(profiler):
                handler(profiler)  # Text includes trace_view.json + CSV, no HCCL filtering.
                self._export_memory(profiler)

            self._profiler = api.profile(
                activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
                record_shapes=True, profile_memory=True, with_stack=True,
                experimental_config=config, on_trace_ready=on_trace_ready,
            )
            self._profiler.__enter__()
            self._stack.callback(self._close_ranges)
            self._install_observers()
            self._active = True
        except BaseException:
            self._stack.close()
            if hasattr(self, "_profiler"):
                self._profiler.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.enabled:
            return False
        try:
            try:
                self._stack.close()  # Restore functions/hooks even if replay failed.
            finally:
                try:
                    if self.synchronize_on_exit:
                        self.npu.synchronize()  # No device sync anywhere inside ranges.
                finally:
                    self._profiler.__exit__(exc_type, exc, traceback)
            for record, start, end in self._pending:
                record["compute_stream_elapsed_ms"] = float(start.elapsed_time(end))
            self._write_reports(exc_type)
        except Exception as error:
            self._note(f"Profile finalization failed: {type(error).__name__}: {error}")
            self._write_reports(exc_type or type(error))
            if exc_type is None:
                raise
        finally:
            self._active = False
        return False

    def profile_call(self, label, callback):
        if not self.enabled:
            return callback()
        if not self._active:
            raise RuntimeError("profile_call requires an active NPUProfileCapture context")
        with self._range(label):
            return callback()

    def _memory(self):
        return {name: int(getattr(self.npu, name)()) for name in (
            "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")}

    def _note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    @contextmanager
    def _range(self, label, **metadata):
        start, end = self.npu.Event(enable_timing=True), self.npu.Event(enable_timing=True)
        metadata.setdefault("tree_index", self._tree_index.get())
        trace_label = label
        if metadata.get("tree_index") is not None:
            trace_label += f"/tree_{metadata['tree_index']}"
        if metadata.get("segment_id") is not None:
            trace_label += f"/segment_{metadata['segment_id']}"
        if metadata.get("layer") is not None:
            trace_label += f"/layer_{metadata['layer']}"
        record = dict(label=label, trace_label=trace_label, host_start_ns=time.perf_counter_ns(), **metadata)
        record["memory_before"] = self._memory()
        marker = getattr(self.npu, "mstx", None) if self._mstx_enabled else None
        marker_id = None
        if marker is not None:
            try:
                marker_id = marker.range_start(trace_label, self.npu.current_stream())
            except (AttributeError, RuntimeError, TypeError) as error:
                self._note(f"MSTX unavailable: {type(error).__name__}: {error}")
        start.record()
        try:
            with self.torch.profiler.record_function(trace_label):
                yield record
        finally:
            end.record()
            if marker_id is not None:
                marker.range_end(marker_id)
            record["host_enqueue_ms"] = (time.perf_counter_ns() - record["host_start_ns"]) / 1e6
            record["memory_after"] = self._memory()
            self.records.append(record)
            self._pending.append((record, start, end))

    def _patch(self, owner, name, replacement):
        self._stack.enter_context(patch.object(owner, name, replacement))

    def _install_observers(self):
        executor = importlib.import_module("verl.workers.engine.megatron.tpr.executor")
        context = importlib.import_module("verl.models.mcore.tpr.context")
        original_run = getattr(executor.SegmentExecutor, "run", None)
        if original_run is not None:
            @wraps(original_run)
            def run(instance, *args, **kwargs):
                tree_index = kwargs.get("tree_index", 0)
                token = self._tree_index.set(tree_index)
                try:
                    with self._range("tpr.run"):
                        return original_run(instance, *args, **kwargs)
                finally:
                    self._tree_index.reset(token)
            self._patch(executor.SegmentExecutor, "run", run)
        for name in ("push", "pop", "visit_leaf", "_forward", "_backward_segment"):
            original = getattr(executor.SegmentExecutor, name)

            def wrap_method(original=original, name=name):
                @wraps(original)
                def observed(instance, *args, **kwargs):
                    segment = args[0] if args else kwargs.get("segment", kwargs.get("segment_id"))
                    segment_id = getattr(segment, "segment_id", segment)
                    plan = getattr(instance, "plan", None)
                    if plan is not None and callable(getattr(plan, "get", None)):
                        segment = plan.get(segment_id)
                    details = {key: getattr(segment, key, "unknown")
                               for key in ("parent_id", "prefix_length", "length")}
                    with self._range(f"tpr.{name}", segment_id=int(segment_id),
                                     no_grad=kwargs.get("no_grad"), **details):
                        self._stack_cache(instance, name, "before", segment_id)
                        try:
                            return original(instance, *args, **kwargs)
                        finally:
                            self._stack_cache(instance, name, "after", segment_id)
                return observed
            self._patch(executor.SegmentExecutor, name, wrap_method())
        original = context.TPRAttentionContext.set_new_kv

        @wraps(original)
        def set_new_kv(instance, layer_number, key, value):
            before = _kv_storage(((key, value),))
            compact = bool(getattr(instance, "compact_new_kv", False))
            with self._range("tpr.cache_capture", layer=int(layer_number),
                             compact=compact):
                result = original(instance, layer_number, key, value)
            self.cache_records.append(dict(
                label="cache_capture", layer=int(layer_number),
                tree_index=self._tree_index.get(),
                compact=compact, input=before,
                output=_kv_storage((instance.new_key_values[layer_number],)),
                collector=_kv_storage(instance.new_key_values.values()),
            ))
            return result
        self._patch(context.TPRAttentionContext, "set_new_kv", set_new_kv)
        self._patch_autograd_backward()
        for module_name, names in (
            ("verl.models.mcore.tpr.attention", ("apply_rotary_pos_emb",)),
            ("verl.workers.engine.megatron.tpr.ring", ("_block_attention_forward", "_block_attention_backward")),
            ("verl.workers.engine.megatron.tpr.adapter", (
                "digest_forest_plan", "assert_tp_plan_agreement", "run_forest", "aggregate_cp_loss")),
            # Adapter imports the builder locally inside forward_backward_batch.
            ("verl.workers.engine.megatron.tpr.tree", ("build_tree_execution_plans",)),
        ):
            module = importlib.import_module(module_name)
            for name in names:
                self._patch_range(module, name)
        objective = importlib.import_module("verl.workers.engine.megatron.tpr.objective")
        self._patch_range(objective.SegmentPPOObjectiveAdapter, "bind_tree")
        config = getattr(self.model, "config", None)
        if callable(getattr(config, "finalize_model_grads_func", None)):
            self._patch_range(config, "finalize_model_grads_func")
        else:
            self._note("model.config.finalize_model_grads_func unavailable; run_forest range includes finalization.")
        if self.model is not None:
            self._install_module_hooks()

    def _patch_range(self, module, name):
        original = getattr(module, name)

        @wraps(original)
        def observed(*args, **kwargs):
            with self._range(f"tpr.{name}", inputs=_tensor_shapes(args)):
                return original(*args, **kwargs)
        self._patch(module, name, observed)

    def _patch_autograd_backward(self):
        original = self.torch.autograd.backward

        @wraps(original)
        def backward(*args, **kwargs):
            with self._range("tpr.autograd.backward"):
                return original(*args, **kwargs)
        self._patch(self.torch.autograd, "backward", backward)

    def _stack_cache(self, executor, stage, when, segment_id):
        entries = getattr(executor.kv_stack, "_entries", {})
        pairs = [pair for entry in entries.values() for pair in entry.kv.key_values.values()]
        self.cache_records.append(dict(label=f"{stage}.{when}", segment_id=int(segment_id),
                                       tree_index=self._tree_index.get(),
                                       entries=len(entries), **_kv_storage(pairs)))

    def _install_module_hooks(self):
        selected = (".self_attention", ".linear_qkv", ".linear_proj", ".linear_fc1", ".linear_fc2")
        for name, module in self.model.named_modules():
            if name not in ("decoder", "output_layer", "rotary_pos_emb") and not name.endswith(selected):
                continue

            def before(module, args, name=name):
                scope = self._range(f"tpr.module.{name}", inputs=_tensor_shapes(args))
                scope.__enter__()
                self._ranges.append((module, scope))

            def after(module, args, output):
                if self._ranges and self._ranges[-1][0] is module:
                    self._ranges.pop()[1].__exit__(None, None, None)
                return None  # PyTorch hook must not replace original output.

            self._stack.callback(module.register_forward_pre_hook(before).remove)
            # always_call closes ranges on forward exceptions on modern PyTorch.
            parameters = inspect.signature(module.register_forward_hook).parameters
            options = {"always_call": True} if "always_call" in parameters else {}
            self._stack.callback(module.register_forward_hook(after, **options).remove)

    def _close_ranges(self):
        while self._ranges:
            self._ranges.pop()[1].__exit__(None, None, None)

    def _export_memory(self, profiler):
        exporter = getattr(profiler, "export_memory_timeline", None)
        if exporter is None:
            self._note("Native export_memory_timeline is unavailable; inspect native memory CSV and trace.")
            return
        path = self.output_dir / "memory_timeline.json"
        try:
            exporter(str(path), device=f"npu:{self.npu.current_device()}")
            if path.is_file():
                self.artifacts["memory_timeline"] = str(path)
            else:
                self._note("Native memory timeline exporter produced no file; inspect native memory CSV and trace.")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            self._note(f"Native memory timeline unavailable: {type(error).__name__}: {error}")

    def _write_reports(self, exc_type):
        native_files = sorted(str(path.relative_to(self.output_dir))
                              for path in (self.output_dir / "native").rglob("*") if path.is_file())
        traces = [name for name in native_files if name.endswith("trace_view.json")]
        if not traces:
            self._note("No native trace_view.json found: raw data is retained, inspect profiler analysis logs.")
        # Native tensorboard Text export already produces the full Chrome JSON;
        # older Ascend versions do not support also calling export_chrome_trace.
        self.artifacts.update(native_directory=str(self.output_dir / "native"),
                              chrome_traces=traces, native_files=native_files)
        for filename, records in (("stages.csv", self.records), ("cache.csv", self.cache_records)):
            keys = sorted({key for record in records for key in record})
            with (self.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                                  for key, value in record.items()} for record in records)
            self.artifacts[filename] = str(self.output_dir / filename)
        report = dict(
            metadata=self.metadata,
            collection=dict(profiler_level="Level1", profile_memory=True, record_shapes=True,
                            with_stack=True, data_simplification=False, activities=["CPU", "NPU"]),
            exception=None if exc_type is None else exc_type.__name__,
            stages=self.records, cache=self.cache_records,
            engine_cpu_stages=getattr(self.engine, "_tpr_stage_profile", None),
            notes=self.notes, artifacts=self.artifacts,
            interpretation=(
                "Instrumented one-replay profile, not a performance benchmark. "
                "Compute-stream event spans do not sum to HCCL/overlap time; "
                "use native Level1 trace and communication analysis. Native "
                "record_shapes/memory/stack collection can itself affect tensor lifetimes and peak memory."
            ),
        )
        path = self.output_dir / "profile_summary.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        self.artifacts["summary"] = str(path)
