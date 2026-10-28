"""CPU-only gates for TP2 CP2 DP2 PPO loss reporting.

Different DP replicas own different forests; *local* PPO loss must not
be mistaken for replicated global loss.
"""

import pytest

from ..parallel.tpr_dp2_loss_consistency import analyze_dp2_loss_report

# Regression contracts formerly in test_dp2_hccl_dtype_gate.py
import json

import pytest

from ..parallel.compare_tpr_tp2cp2dp2_one_step import compare

# Regression contracts formerly in test_tpr_offline_grad_sync_precision.py
import pytest
import torch
import torch.distributed as dist

from ..parallel.grad_sync_precision import (
    offline_grad_sync_dtype,
    reduce_average_weight_grad,
)

def _ranks(dp0=0.006869133561849594, dp1=0.1446971446275711,
           tp1_offset=0.0, *, corrupt=None):
    rows = []
    expected = (dp0 + dp1) / 2 + tp1_offset / 2
    for dp in range(2):
        for cp in range(2):
            for tp in range(2):
                rows.append({
                    "rank": len(rows), "tp_rank": tp, "cp_rank": cp,
                    "dp_rank": dp,
                    "local_loss_dp_scaled": (dp0 if dp == 0 else dp1) +
                                           (tp1_offset if tp == 1 else 0),
                    "global_loss": expected,
                })
    if corrupt is not None:
        rows[corrupt["row"]][corrupt["key"]] = corrupt["value"]
    return rows


def test_distinct_dp_local_forest_losses_are_expected_not_a_replica_mismatch():
    result = analyze_dp2_loss_report(_ranks())
    assert result["dp0_local_loss"] == pytest.approx(0.006869133561849594)
    assert result["dp1_local_loss"] == pytest.approx(0.1446971446275711)
    assert result["dp_local_contribution_gap"] == pytest.approx(0.1378280110657215)
    assert result["cp_replica_max_gap"] == 0
    assert result["global_reporting_max_gap"] == 0
    assert result["global_reporting_loss"] == pytest.approx(
        (0.006869133561849594 + 0.1446971446275711)/2
    )


def test_tp_shard_drift_is_reported_without_misclassifying_dp_partition():
    result = analyze_dp2_loss_report(_ranks(tp1_offset=0.00025))
    assert result["tp_shard_loss_gap"] == pytest.approx(0.00025)
    assert result["tp_local_max_gap"] == pytest.approx(0.00025)
    assert result["global_reporting_max_gap"] == 0


def test_cp_replica_mismatch_remains_hard_error():
    with pytest.raises(AssertionError, match="CP replicas"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 2, "key": "local_loss_dp_scaled", "value": 0.5}
        ))


def test_wrong_tp_cp_dp_coordinates_are_rejected():
    with pytest.raises(AssertionError, match="Invalid or repeated"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 1, "key": "dp_rank", "value": 1}
        ))


def test_wrong_global_reporting_collective_is_rejected():
    with pytest.raises(AssertionError, match="World-reduced"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 7, "key": "global_loss", "value": 2.0}
        ))


def test_nan_scalar_is_rejected():
    with pytest.raises(AssertionError, match="Non-finite"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 5, "key": "local_loss_dp_scaled", "value": float("nan")}
        ))

# Tests consolidated from test_dp2_hccl_dtype_gate.py
def test_stale_float64_full_baseline_is_not_compared_to_current_fp32_tpr(tmp_path):
    full = tmp_path / "full.json"
    tpr = tmp_path / "tpr.json"
    full.write_text(json.dumps({
        "mode": "full",
        "hccl_stats_reduce_dtype": "float64",
    }))
    tpr.write_text(json.dumps({
        "mode": "tpr",
        "hccl_stats_reduce_dtype": "float32",
        "topology": {"tp": 2, "cp": 2, "dp": 2, "world": 8},
    }))
    with pytest.raises(AssertionError, match="Stale Full/TPR benchmark"):
        compare(full, tpr)


def test_collective_benchmark_paths_explicitly_use_fp32_for_world_reductions():
    # Guard the root cause found on Ascend 910B (not an NPU runtime test).
    from pathlib import Path
    parallel = Path(__file__).resolve().parents[1] / "parallel"
    for name in (
        "test_tpr_tp2cp2dp2_one_step_npu.py",
        "test_tpr_fixed_shape_40k_benchmark_npu.py",
    ):
        source = (parallel / name).read_text()
        assert '"hccl_stats_reduce_dtype": "float32"' in source
        assert "dtype=torch.float32" in source
        assert "dist.ReduceOp.MAX" in source
    dp2 = (parallel / "test_tpr_tp2cp2dp2_one_step_npu.py").read_text()
    assert "TPR_DP2_WORLD_FP32_REDUCE_PASS" in dp2
    assert "global_loss = torch.tensor(" in dp2
    assert "local_loss, dtype=torch.float32" in dp2


# Tests consolidated from test_tpr_offline_grad_sync_precision.py
def test_offline_grad_precision_flag_defaults_to_native_bf16(monkeypatch):
    monkeypatch.delenv("TPR_OFFLINE_GRAD_SYNC_DTYPE", raising=False)
    assert offline_grad_sync_dtype() == "bf16"
    monkeypatch.setenv("TPR_OFFLINE_GRAD_SYNC_DTYPE", "fp32")
    assert offline_grad_sync_dtype() == "fp32"
    monkeypatch.setenv("TPR_OFFLINE_GRAD_SYNC_DTYPE", "bad")
    with pytest.raises(ValueError, match="TPR_OFFLINE_GRAD_SYNC_DTYPE"):
        offline_grad_sync_dtype()


@pytest.mark.parametrize("reduce_dtype", ["bf16", "fp32"])
def test_reduce_average_does_not_change_original_grad_dtype(
    monkeypatch, reduce_dtype,
):
    original = torch.tensor([0.25, -0.5, 0.75], dtype=torch.bfloat16)
    param = torch.nn.Parameter(torch.ones_like(original))
    param.grad = original.clone()
    calls = []

    def fake_collective(tensor, *, op, group):
        calls.append((tensor.dtype, op, group))
        tensor.mul_(4)

    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(dist, "all_reduce", fake_collective)
    handle = object()
    reduce_average_weight_grad(
        param, handle, name="w", reduce_dtype=reduce_dtype,
    )
    assert len(calls) == 1
    assert calls[0][0] == (
        torch.float32 if reduce_dtype == "fp32" else torch.bfloat16
    )
    assert calls[0][1] == dist.ReduceOp.SUM
    assert calls[0][2] is handle
    assert param.grad.dtype == torch.bfloat16
    torch.testing.assert_close(param.grad, original, rtol=0, atol=0)


def test_reduce_average_rejects_missing_gradient(monkeypatch):
    param = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="Missing gradient"):
        reduce_average_weight_grad(param, object(), name="missing",
                                   reduce_dtype="fp32")
