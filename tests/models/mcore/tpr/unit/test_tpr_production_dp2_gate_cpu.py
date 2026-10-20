"""Production TPR DP topology contract without importing Megatron/NPU."""

import ast
import os
from argparse import Namespace
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


def _mindspeed_cli_sanitizer():
    """Import only the pure stdlib function, not NPU-gated test dependencies."""
    source = Path(__file__).resolve().parents[1] / "parallel/test_real_actor_tp2cp4_offline_ppo_npu.py"
    node = next(
        n for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_sanitize_mindspeed_repatch_args"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("short_option_value", [True, False, "test.py", ["test.py"], None])
def test_dp_mindspeed_pytest_short_options_never_become_dataclass_fields(short_option_value):
    # MindSpeed parser_unknown_args uses option[2:], so pytest -q/-s/-x
    # creates an invalid key "". Its value varies with argv ordering.
    args = Namespace(optimizer_selection="fused_adamw", tensor_model_parallel_size=2)
    vars(args)[""] = short_option_value
    _mindspeed_cli_sanitizer()(args)
    assert "" not in vars(args)
    assert vars(args) == {
        "optimizer_selection": "fused_adamw", "tensor_model_parallel_size": 2,
    }


def test_dp_mindspeed_other_invalid_names_still_fail_closed():
    args = Namespace(optimizer_selection="fused_adamw")
    vars(args).update({"": True, "bad-field": True})
    with pytest.raises(RuntimeError, match="Invalid MindSpeed argument names"):
        _mindspeed_cli_sanitizer()(args)


def test_dp_and_cp4_sanitize_both_argument_sources_before_model_config():
    """Prevent reintroducing this bug by sanitizing only one args cache."""
    folder = Path(__file__).resolve().parents[1] / "parallel"
    for name in ("test_real_actor_tp2cp4_offline_ppo_npu.py",
                 "test_tpr_tp2cp2dp2_one_step_npu.py"):
        source = (folder / name).read_text(encoding="utf-8")
        assert "_sanitize_mindspeed_repatch_args(args)" in source
        assert "_sanitize_mindspeed_repatch_args(get_mindspeed_args())" in source
