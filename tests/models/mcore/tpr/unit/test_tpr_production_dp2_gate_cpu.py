"""Production TPR DP topology contract without importing Megatron/NPU."""

import ast
import os
from pathlib import Path

import pytest


def _gate():
    source = Path(__file__).resolve().parents[5] / "verl/workers/engine/megatron/tpr/adapter.py"
    node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "_validate_tpr_parallel_sizes")
    ns = {"os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    return ns["_validate_tpr_parallel_sizes"]


def _sizes(tp=2, cp=2, dp=2, pp=1, ep=1):
    return {"TP": tp, "CP": cp, "DP": dp, "PP": pp, "EP": ep}


def test_production_dp2_tp2_cp2_requires_no_offline_test_flags(monkeypatch):
    for key in ("TPR_EXPERIMENTAL_OFFLINE_DP2", "TPR_RUN_TP2CP2DP2_ONE_STEP",
                "TPR_RUN_10K_PARALLEL_PARITY", "TPR_10K_PARITY_ONLY"):
        monkeypatch.delenv(key, raising=False)
    assert _gate()(_sizes(), ppo=True) == 2


@pytest.mark.parametrize("sizes", [
    _sizes(cp=4), _sizes(tp=1, cp=4), _sizes(dp=3),
    _sizes(pp=2), _sizes(ep=2), _sizes(tp=4), _sizes(cp=1),
])
def test_unvalidated_dp_topologies_still_fail_closed(sizes, monkeypatch):
    monkeypatch.delenv("TPR_EXPERIMENTAL_OFFLINE_DP2", raising=False)
    with pytest.raises(NotImplementedError):
        _gate()(sizes, ppo=True)


def test_non_ppo_dp2_is_still_gated():
    with pytest.raises(NotImplementedError):
        _gate()(_sizes(), ppo=False)


def test_dp1_supported_topologies_unchanged():
    for cp in (1, 2, 4):
        assert _gate()(_sizes(dp=1, cp=cp), ppo=True) == 2
    assert _gate()(_sizes(tp=1, cp=2, dp=1), ppo=True) == 1


def test_10k_offline_parity_stays_explicit(monkeypatch):
    monkeypatch.setenv("TPR_EXPERIMENTAL_OFFLINE_DP2", "1")
    monkeypatch.setenv("TPR_RUN_10K_PARALLEL_PARITY", "1")
    monkeypatch.setenv("TPR_10K_PARITY_ONLY", "1")
    assert _gate()(_sizes(tp=1, cp=2), ppo=True) == 1
    with pytest.raises(NotImplementedError):
        _gate()(_sizes(tp=1, cp=4), ppo=True)
