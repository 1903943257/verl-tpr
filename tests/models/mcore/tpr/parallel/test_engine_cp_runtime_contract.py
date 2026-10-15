"""Engine Ring CP configuration contracts (alternate CP algorithms retired)."""
from types import SimpleNamespace
import pytest
import verl.models.mcore.tpr.megatron_cp as runtime_module


def _engine(cp_size, *, backend=None, transformer_algo=None):
    return SimpleNamespace(
        engine_config=SimpleNamespace(
            context_parallel_size=cp_size, dynamic_context_parallel=False,
            tpr_cp_backend=backend, override_transformer_config={}, mcore_kwargs={},
        ),
        tf_config=SimpleNamespace(context_parallel_algo=transformer_algo),
    )


def _distributed(monkeypatch, *, size=2, rank=0):
    group = object()
    monkeypatch.setattr(runtime_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime_module.parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(runtime_module.parallel_state, "get_context_parallel_group", lambda: group)
    monkeypatch.setattr(runtime_module, "resolve_cp_group", lambda _group: (size, rank))
    return group


def test_cp1_without_distributed_initialization():
    runtime = runtime_module.resolve_engine_cp_runtime(_engine(1), SimpleNamespace())
    assert not runtime.enabled
    assert runtime.backend is None


def test_cp2_uses_initialized_group_and_ring(monkeypatch):
    group = _distributed(monkeypatch, size=2, rank=1)
    calls = []

    class _Ring:
        backend_name = "ring"

    def resolve(name, **kwargs):
        calls.append((name, kwargs))
        return _Ring()

    monkeypatch.setattr(runtime_module, "resolve_tpr_cp_backend", resolve)
    runtime = runtime_module.resolve_engine_cp_runtime(
        _engine(2, transformer_algo="megatron_cp_algo"), SimpleNamespace(),
    )
    assert runtime.backend_name == "ring"
    assert calls == [("megatron_cp_algo", {
        "cp_group": group, "parallel_size": 2, "parallel_rank": 1,
    })]


def test_unspecified_cp_algorithm_defaults_to_ring():
    assert runtime_module._configured_cp_algorithm(_engine(2), SimpleNamespace()) == "ring"


def test_explicit_unsupported_cp_algorithm_is_rejected(monkeypatch):
    _distributed(monkeypatch)
    with pytest.raises(NotImplementedError, match="only supports Ring"):
        runtime_module.resolve_engine_cp_runtime(
            _engine(2, backend="ulysses"), SimpleNamespace(),
        )


def test_cp_size_mismatch_raises(monkeypatch):
    _distributed(monkeypatch, size=4)
    with pytest.raises(RuntimeError, match="topology mismatch"):
        runtime_module.resolve_engine_cp_runtime(
            _engine(2, backend="ring"), SimpleNamespace(),
        )
