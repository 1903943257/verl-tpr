"""Loss replica audit must retain diagnostics rather than invent numerical PASS."""
import pytest

from .compare_real_actor_tp2cp4_multistep import _rank_replica_audit


def _snapshots(tp1_loss=0.125, tp1_lp=-3.0):
    result = []
    for cp in range(4):
        for tp in range(2):
            result.append({
                "rank": tp, "tp_rank": tp, "cp_rank": cp,
                "loss": 0.125 if tp == 0 else tp1_loss,
                "global_grad_norm_preclip": 1.0,
                "logprobs": {(cp, 0): -3.0 if tp == 0 else tp1_lp},
            })
    return result


def test_identical_native_replica_values_pass_gate():
    audit = _rank_replica_audit(_snapshots(), "full", 1)
    assert audit["loss_gate_ok"]
    assert audit["norm_gate_ok"]
    assert audit["tp_logprob_gate_ok"]
    assert audit["loss_global_spread"] == 0
    assert audit["tp_logprob_shared_tokens"] == 4


def test_tp_loss_mismatch_is_detected_and_preserved_per_rank():
    audit = _rank_replica_audit(_snapshots(tp1_loss=0.132), "full", 2)
    assert not audit["loss_gate_ok"]
    assert audit["loss_global_spread"] == pytest.approx(0.007)
    assert audit["loss_spread_by_tp"] == {"0": 0.0, "1": 0.0}
    assert audit["loss_spread_by_cp"]["0"] == pytest.approx(0.007)
    assert audit["per_rank"][1]["loss"] == pytest.approx(0.132)


def test_tp_logprob_mismatch_not_silently_ignored():
    audit = _rank_replica_audit(_snapshots(tp1_lp=-3.003), "full", 3)
    assert audit["loss_gate_ok"]
    assert not audit["tp_logprob_gate_ok"]
    assert audit["tp_logprob_max_abs"] == pytest.approx(0.003)
    assert audit["tp_logprob_keys_identical"]
