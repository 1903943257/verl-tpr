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


def test_cp_gradient_witness_detects_replica_mismatch_not_seen_by_global_norm():
    import torch
    snapshots = _snapshots()
    for item in snapshots:
        item["gradient_samples"] = {
            "decoder.layers.0.mlp.linear_fc1.weight": {
                "shape": (4,), "indices": torch.tensor([0, 2]),
                "values": torch.tensor([1.0, 2.0]),
            }
        }
    mismatched = next(x for x in snapshots
                      if x["tp_rank"] == 0 and x["cp_rank"] == 2)
    mismatched["gradient_samples"][
        "decoder.layers.0.mlp.linear_fc1.weight"
    ]["values"] = torch.tensor([1.0, 2.5])
    audit = _rank_replica_audit(snapshots, "full", 1)
    assert audit["global_grad_norm_spread"] == 0
    assert audit["global_grad_norm_rank_spread_is_not_replica_parity"]
    report = audit["sampled_cp_gradient_replica"]
    assert report["status"] == "measured"
    assert report["cp_vs_cp0_by_tp"]["tp0_cp2"]["sampled_max_abs"] == 0.5
    assert report["cp_vs_cp0_by_tp"]["tp0_cp2"]["sampled_changed_elements"] == 1
    assert report["cp_vs_cp0_by_tp"]["tp1_cp2"]["sampled_max_abs"] == 0.0


def test_missing_cp_gradient_witness_is_explicitly_unavailable():
    audit = _rank_replica_audit(_snapshots(), "tpr", 1)
    assert audit["sampled_cp_gradient_replica"]["status"] == "not_captured"
