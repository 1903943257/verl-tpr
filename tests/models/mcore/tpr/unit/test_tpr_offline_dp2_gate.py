"""Experimental PPO DP2 route is opt-in; other configurations stay DP1."""
from __future__ import annotations

import pytest

from verl.experimental.tpr.legacy.megatron_adapter import _validate_tpr_parallel_sizes


def _sizes(*, tp=2, cp=2, dp=2, pp=1, ep=1):
    return {"TP": tp, "CP": cp, "DP": dp, "PP": pp, "EP": ep}


def test_production_dp2_remains_rejected(monkeypatch):
    monkeypatch.delenv("TPR_EXPERIMENTAL_OFFLINE_DP2", raising=False)
    monkeypatch.delenv("TPR_RUN_TP2CP2DP2_ONE_STEP", raising=False)
    with pytest.raises(NotImplementedError):
        _validate_tpr_parallel_sizes(_sizes(), ppo=True)


def test_only_opted_in_tp2cp2dp2_ppo_is_allowed(monkeypatch):
    monkeypatch.setenv("TPR_EXPERIMENTAL_OFFLINE_DP2", "1")
    monkeypatch.setenv("TPR_RUN_TP2CP2DP2_ONE_STEP", "1")
    assert _validate_tpr_parallel_sizes(_sizes(), ppo=True) == 2
    for sizes, ppo in (
        (_sizes(cp=4), True),
        (_sizes(tp=1), True),
        (_sizes(dp=3), True),
        (_sizes(pp=2), True),
        (_sizes(ep=2), True),
        (_sizes(), False),
    ):
        with pytest.raises(NotImplementedError):
            _validate_tpr_parallel_sizes(sizes, ppo=ppo)


def test_dp1_previous_topologies_unchanged(monkeypatch):
    monkeypatch.delenv("TPR_EXPERIMENTAL_OFFLINE_DP2", raising=False)
    monkeypatch.delenv("TPR_RUN_TP2CP2DP2_ONE_STEP", raising=False)
    for cp in (1, 2, 4):
        assert _validate_tpr_parallel_sizes(_sizes(cp=cp, dp=1), ppo=True) == 2


def test_single_flag_is_not_sufficient(monkeypatch):
    monkeypatch.setenv("TPR_EXPERIMENTAL_OFFLINE_DP2", "1")
    monkeypatch.delenv("TPR_RUN_TP2CP2DP2_ONE_STEP", raising=False)
    with pytest.raises(NotImplementedError):
        _validate_tpr_parallel_sizes(_sizes(), ppo=True)


def test_production_adapter_dp2_requires_explicit_offline_opt_in(monkeypatch):
    """Exercise the actual production gate without importing Megatron on CPU."""
    import ast
    import os
    from pathlib import Path

    adapter = Path(__file__).resolve().parents[5] / "verl/workers/engine/megatron/tpr/adapter.py"
    tree = ast.parse(adapter.read_text(encoding="utf-8"), filename=str(adapter))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_validate_tpr_parallel_sizes")
    namespace = {"os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(adapter), "exec"), namespace)
    validate = namespace["_validate_tpr_parallel_sizes"]

    for key in ("TPR_EXPERIMENTAL_OFFLINE_DP2", "TPR_RUN_TP2CP2DP2_ONE_STEP",
                "TPR_RUN_10K_PARALLEL_PARITY", "TPR_10K_PARITY_ONLY"):
        monkeypatch.delenv(key, raising=False)
    for cp in (1, 2, 4):
        assert validate(_sizes(cp=cp, dp=1), ppo=True) == 2
    with pytest.raises(NotImplementedError):
        validate(_sizes(), ppo=True)

    monkeypatch.setenv("TPR_EXPERIMENTAL_OFFLINE_DP2", "1")
    with pytest.raises(NotImplementedError):
        validate(_sizes(), ppo=True)

    monkeypatch.setenv("TPR_RUN_TP2CP2DP2_ONE_STEP", "1")
    assert validate(_sizes(), ppo=True) == 2
    for sizes, ppo in (
        (_sizes(cp=4), True), (_sizes(tp=1), True),
        (_sizes(dp=3), True), (_sizes(pp=2), True),
        (_sizes(ep=2), True), (_sizes(), False),
    ):
        with pytest.raises(NotImplementedError):
            validate(sizes, ppo=ppo)
