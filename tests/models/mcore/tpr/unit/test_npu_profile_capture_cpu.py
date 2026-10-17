# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Mock-only observer contracts; these tests do NOT validate NPU profiling."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

_PATH = Path(__file__).parents[1] / "parallel" / "_npu_profile_capture.py"
_SPEC = importlib.util.spec_from_file_location("_test_npu_capture", _PATH)
capture_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(capture_module)
NPUProfileCapture = capture_module.NPUProfileCapture


class _Context:
    def __init__(self):
        self.compact_new_kv = True
        self.new_key_values = {}
        self.calls = 0

    def set_new_kv(self, layer, key, value):
        self.calls += 1
        self.new_key_values[layer] = key, value
        return key


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(3, 3)
        self.output_layer = torch.nn.Linear(3, 2)

    def forward(self, tensor):
        return self.output_layer(self.decoder(tensor))


class _Executor:
    def __init__(self, model=None):
        self.kv_stack = SimpleNamespace(_entries={})
        self.context = _Context()
        self.model = model
        self.return_value = object()
        self.calls = []

    def run(self, *, event_profiler=None, tree_index=0):
        return self.push(1)

    def push(self, segment_id):
        self.calls.append("push")
        self._forward(SimpleNamespace(segment_id=segment_id), no_grad=True)
        return self.return_value

    def _forward(self, segment, *, no_grad=False):
        self.calls.append("_forward")
        key, value = torch.ones(2, 1, 1, 2), torch.zeros(2, 1, 1, 2)
        assert self.context.set_new_kv(1, key, value) is key
        assert self.context.new_key_values[1][0] is key
        assert self.context.new_key_values[1][1] is value
        if self.model is not None:
            self.model(torch.ones(1, 3))
        return self.return_value

    def pop(self, segment_id):
        return self._backward_segment(SimpleNamespace(segment_id=segment_id))

    def visit_leaf(self, segment_id):
        return self._backward_segment(SimpleNamespace(segment_id=segment_id))

    def _backward_segment(self, segment):
        self.calls.append("backward")
        loss = torch.ones((), requires_grad=True)
        loss.backward()
        return self.return_value


def _runtime(monkeypatch):
    state = SimpleNamespace(synchronizations=0, events=0, stops=0, config=None, profile=None)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            state.events += 1

        def record(self):
            pass

        def elapsed_time(self, other):
            assert state.synchronizations == 1, "materialized before final synchronization"
            return 1.25

    def synchronize():
        state.synchronizations += 1

    npu = SimpleNamespace(Event=Event, synchronize=synchronize, current_device=lambda: 0,
                          memory_allocated=lambda: 100, memory_reserved=lambda: 200,
                          max_memory_allocated=lambda: 300, max_memory_reserved=lambda: 400)

    class Experimental:
        def __init__(self, *, profiler_level, data_simplification, export_type, msprof_tx,
                     aic_metrics):
            state.config = dict(profiler_level=profiler_level, data_simplification=data_simplification,
                                export_type=export_type, msprof_tx=msprof_tx, aic_metrics=aic_metrics)

    class Profile:
        def __init__(self, **kwargs):
            state.profile = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.stops += 1
            state.profile["on_trace_ready"](self)

        def export_memory_timeline(self, path, *, device):
            assert state.synchronizations == 1
            Path(path).write_text("[]", encoding="utf-8")

    def handler(directory):
        def write(profiler):
            path = Path(directory) / "test_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
            path.mkdir(parents=True)
            (path / "trace_view.json").write_text('{"traceEvents": []}', encoding="utf-8")
            (path / "communication.json").write_text('{"hccl": []}', encoding="utf-8")
            (path / "kernel_details.csv").write_text("name,time\nSend,1\nMatMul,2\n", encoding="utf-8")
        return write

    profiler = SimpleNamespace(
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        ExportType=SimpleNamespace(Text="text"),
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe"),
        _ExperimentalConfig=Experimental, profile=Profile, tensorboard_trace_handler=handler,
    )

    def cpu_stage(value):
        return value

    class Objective:
        def bind_tree(self, tree_plan, *, shard_for_segment=None):
            return tree_plan

    names = {
        "torch_npu": dict(npu=npu, profiler=profiler),
        "verl.workers.engine.megatron.tpr.executor": dict(SegmentExecutor=_Executor),
        "verl.models.mcore.tpr.context": dict(TPRAttentionContext=_Context),
        "verl.models.mcore.tpr.attention": dict(apply_rotary_pos_emb=lambda value: value),
        "verl.workers.engine.megatron.tpr.ring": dict(
            _block_attention_forward=lambda *args, **kwargs: args,
            _block_attention_backward=lambda *args, **kwargs: args),
        "verl.workers.engine.megatron.tpr.adapter": dict(
            digest_forest_plan=cpu_stage, assert_tp_plan_agreement=cpu_stage,
            run_forest=cpu_stage, aggregate_cp_loss=cpu_stage),
        "verl.workers.engine.megatron.tpr.tree": dict(build_tree_execution_plans=cpu_stage),
        "verl.workers.engine.megatron.tpr.objective": dict(SegmentPPOObjectiveAdapter=Objective),
    }
    for name, values in names.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    return state


def test_disabled_capture_imports_nothing_and_calls_original_once(tmp_path):
    original = _Executor.push
    marker, calls = object(), []
    with patch.object(capture_module.importlib, "import_module", side_effect=AssertionError("import")):
        with NPUProfileCapture(tmp_path / "unused") as capture:
            assert capture.profile_call("replay", lambda: calls.append(1) or marker) is marker
    assert calls == [1] and _Executor.push is original
    assert not (tmp_path / "unused").exists()


def test_native_configuration_observer_identity_exports_and_one_sync(tmp_path, monkeypatch):
    state = _runtime(monkeypatch)
    original_push, original_set = _Executor.push, _Context.set_new_kv
    original_backward = torch.autograd.backward
    model, executor = _Model(), _Executor()
    executor.model = model
    directory = tmp_path / "rank_0"
    with NPUProfileCapture(directory, enabled=True, metadata={"rank": 0, "sha": "abc"},
                           model=model) as capture:
        assert capture.profile_call("step6_replay", lambda: executor.push(7)) is executor.return_value
        assert executor.pop(7) is executor.return_value
        value = torch.ones(1)
        attention = sys.modules["verl.models.mcore.tpr.attention"]
        assert attention.apply_rotary_pos_emb(value) is value
        assert state.synchronizations == 0
    assert executor.calls == ["push", "_forward", "backward"]
    assert executor.context.calls == 1
    assert state.synchronizations == 1 and state.events > 0
    assert _Executor.push is original_push and _Context.set_new_kv is original_set
    assert torch.autograd.backward is original_backward
    assert not model.output_layer._forward_hooks and not model.output_layer._forward_pre_hooks
    assert state.config == dict(profiler_level="level1", data_simplification=False,
                                export_type=["text"], msprof_tx=True, aic_metrics="pipe")
    for key in ("record_shapes", "profile_memory", "with_stack"):
        assert state.profile[key] is True
    assert state.profile["activities"] == ["cpu", "npu"]
    summary = json.loads((directory / "profile_summary.json").read_text(encoding="utf-8"))
    assert summary["metadata"]["sha"] == "abc"
    labels = [row["label"] for row in summary["stages"]]
    assert "tpr.module.output_layer" in labels and "tpr.autograd.backward" in labels
    assert "tpr.apply_rotary_pos_emb" in labels
    assert any(path.endswith("communication.json") for path in summary["artifacts"]["native_files"])
    assert summary["artifacts"]["chrome_traces"]
    assert (directory / "stages.csv").is_file() and (directory / "cache.csv").is_file()
    assert (directory / "memory_timeline.json").is_file()
    # All records must serialize without any Tensor/storage fallback encoder.
    json.dumps(capture.records)
    json.dumps(capture.cache_records)


def test_exception_restores_observers_without_repeating_callback(tmp_path, monkeypatch):
    state = _runtime(monkeypatch)
    original = _Executor.push
    calls = []

    def fail():
        calls.append(1)
        raise ValueError("replay failure")

    with pytest.raises(ValueError, match="replay failure"):
        with NPUProfileCapture(tmp_path / "failed", enabled=True) as capture:
            capture.profile_call("replay", fail)
    assert _Executor.push is original and calls == [1] and state.synchronizations == 1
    summary = json.loads((tmp_path / "failed" / "profile_summary.json").read_text())
    assert summary["exception"] == "ValueError"


def test_missing_native_memory_timeline_is_reported_not_fabricated(tmp_path, monkeypatch):
    _runtime(monkeypatch)
    profiler_class = sys.modules["torch_npu"].profiler.profile
    monkeypatch.setattr(profiler_class, "export_memory_timeline", None)
    with NPUProfileCapture(tmp_path / "no_memory_export", enabled=True) as capture:
        capture.profile_call("replay", lambda: None)
    assert any("export_memory_timeline is unavailable" in note for note in capture.notes)
    assert "memory_timeline" not in capture.artifacts


def test_main_context_without_compaction_field_can_be_observed(tmp_path, monkeypatch):
    _runtime(monkeypatch)
    executor = _Executor()
    del executor.context.compact_new_kv
    with NPUProfileCapture(tmp_path / "main_like", enabled=True) as capture:
        assert capture.profile_call("replay", lambda: executor.push(1)) is executor.return_value
    assert executor.context.calls == 1
    assert next(row for row in capture.cache_records if row["label"] == "cache_capture")["compact"] is False


@pytest.mark.parametrize("failure", ["stack", "synchronize"])
def test_native_profiler_stops_even_if_cleanup_or_final_sync_fails(tmp_path, monkeypatch, failure):
    state = _runtime(monkeypatch)
    original = _Executor.push

    def fail():
        if failure == "synchronize":
            state.synchronizations += 1
        raise RuntimeError(failure)

    with pytest.raises(RuntimeError, match=failure):
        with NPUProfileCapture(tmp_path / failure, enabled=True) as capture:
            capture.profile_call("replay", lambda: None)
            if failure == "stack":
                capture._stack.callback(fail)
            else:
                monkeypatch.setattr(sys.modules["torch_npu"].npu, "synchronize", fail)
    assert state.stops == 1 and _Executor.push is original


def test_caller_can_own_the_single_final_synchronization(tmp_path, monkeypatch):
    state = _runtime(monkeypatch)
    with NPUProfileCapture(tmp_path / "caller_sync", enabled=True,
                           synchronize_on_exit=False) as capture:
        capture.profile_call("replay", lambda: None)
        sys.modules["torch_npu"].npu.synchronize()
    assert state.synchronizations == 1


def test_module_exception_preserves_failure_and_removes_forward_hooks(tmp_path, monkeypatch):
    _runtime(monkeypatch)
    model = _Model()
    executor = _Executor(model)

    def fail(*args, **kwargs):
        raise ValueError("head failed")

    monkeypatch.setattr(model.output_layer, "forward", fail)
    with pytest.raises(ValueError, match="head failed"):
        with NPUProfileCapture(tmp_path / "module_failure", enabled=True, model=model) as capture:
            capture.profile_call("replay", lambda: executor.push(1))
    assert not capture._ranges
    for module in (model.decoder, model.output_layer):
        assert not module._forward_pre_hooks and not module._forward_hooks


def test_storage_metadata_deduplicates_backing_storage():
    backing = torch.ones(4, 1, 4, 2)
    key, value = backing[:, :, :1], backing[:, :, 1:2]
    result = capture_module._kv_storage(((key, value),))
    assert result["storage_count"] == 1
    assert result["logical_bytes"] == 2 * key.numel() * key.element_size()
    assert result["unique_storage_bytes"] == backing.untyped_storage().nbytes()


def test_tree_scopes_disambiguate_reused_segment_ids_and_attach_read_only_plan_metadata(tmp_path, monkeypatch):
    _runtime(monkeypatch)
    executors = [_Executor(), _Executor()]
    for index, executor in enumerate(executors):
        segment = SimpleNamespace(segment_id=1, parent_id=None, prefix_length=0, length=10 + index)
        executor.plan = SimpleNamespace(get=lambda segment_id, segment=segment: segment)
    with NPUProfileCapture(tmp_path / "trees", enabled=True) as capture:
        for index, executor in enumerate(executors):
            assert executor.run(tree_index=index) is executor.return_value
    pushes = [row for row in capture.records if row["label"] == "tpr.push"]
    assert [(row["tree_index"], row["segment_id"], row["length"]) for row in pushes] == [(0, 1, 10), (1, 1, 11)]
    assert all(row["parent_id"] is None and row["prefix_length"] == 0 for row in pushes)
    assert pushes[0]["trace_label"] != pushes[1]["trace_label"]
    assert {row["tree_index"] for row in capture.records if row["label"] == "tpr.cache_capture"} == {0, 1}
    assert capture._tree_index.get() is None


def test_cpu_preparation_bind_and_finalize_ranges_call_originals_once_and_restore(tmp_path, monkeypatch):
    _runtime(monkeypatch)
    model = _Model()
    calls, marker = [], object()

    def finalize(value):
        calls.append(value)
        return value

    model.config = SimpleNamespace(finalize_model_grads_func=finalize)
    adapter = sys.modules["verl.workers.engine.megatron.tpr.adapter"]
    tree = sys.modules["verl.workers.engine.megatron.tpr.tree"]
    objective = sys.modules["verl.workers.engine.megatron.tpr.objective"].SegmentPPOObjectiveAdapter
    original_builder, original_bind = tree.build_tree_execution_plans, objective.bind_tree
    with NPUProfileCapture(tmp_path / "cpu_stages", enabled=True, model=model) as capture:
        assert tree.build_tree_execution_plans(marker) is marker
        for name in ("digest_forest_plan", "assert_tp_plan_agreement", "run_forest", "aggregate_cp_loss"):
            assert getattr(adapter, name)(marker) is marker
        assert objective().bind_tree(marker) is marker
        assert model.config.finalize_model_grads_func(marker) is marker
    labels = {row["label"] for row in capture.records}
    assert {f"tpr.{name}" for name in (
        "build_tree_execution_plans", "digest_forest_plan", "assert_tp_plan_agreement",
        "run_forest", "aggregate_cp_loss", "bind_tree", "finalize_model_grads_func")} <= labels
    assert calls == [marker]
    assert tree.build_tree_execution_plans is original_builder and objective.bind_tree is original_bind
    assert model.config.finalize_model_grads_func is finalize


def test_tree_index_scope_and_observers_restore_on_segment_exception(tmp_path, monkeypatch):
    _runtime(monkeypatch)
    original_run, calls = _Executor.run, []

    def fail(*args, **kwargs):
        calls.append(1)
        raise ValueError("segment failed")

    monkeypatch.setattr(_Executor, "_forward", fail)
    with pytest.raises(ValueError, match="segment failed"):
        with NPUProfileCapture(tmp_path / "tree_failure", enabled=True) as capture:
            _Executor().run(tree_index=9)
    assert calls == [1] and capture._tree_index.get() is None
    assert _Executor.run is original_run and _Executor._forward is fail
    row = next(row for row in capture.records if row["label"] == "tpr.push")
    assert row["tree_index"] == 9 and row["length"] == "unknown"


@pytest.mark.parametrize("inspectable", [False, True])
def test_optional_mstx_flag_is_not_required_for_core_collection(tmp_path, monkeypatch, inspectable):
    state = _runtime(monkeypatch)
    api = sys.modules["torch_npu"].profiler

    class CoreConfig:
        def __init__(self, *, profiler_level, data_simplification, export_type, aic_metrics):
            state.config = dict(level=profiler_level, raw=not data_simplification)

    monkeypatch.setattr(api, "_ExperimentalConfig", CoreConfig)
    original_signature = capture_module.inspect.signature
    if not inspectable:
        def signature(value):
            if value is CoreConfig:
                raise ValueError("opaque factory")
            return original_signature(value)
        monkeypatch.setattr(capture_module.inspect, "signature", signature)
    with NPUProfileCapture(tmp_path / f"core_{inspectable}", enabled=True) as capture:
        capture.profile_call("replay", lambda: None)
    assert state.config == {"level": "level1", "raw": True}
    assert state.profile["profile_memory"] is True
    assert any("Optional MSTX" in note for note in capture.notes)
