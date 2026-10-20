"""Loss replica audit must retain diagnostics rather than invent numerical PASS."""
import pytest

from .compare_real_actor_tp2cp4_multistep import _rank_replica_audit

# Regression contracts formerly in test_cp4_three_arm_comparison_cpu.py
from copy import deepcopy

import pytest
import torch

from .compare_cp4_split_no_share_cpu import analyze

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

# Tests consolidated from test_cp4_three_arm_comparison_cpu.py
def _arms():
    states = {}
    topologies = {
        "full": (2, 2, 0.0),
        "split_no_share": (2, 4, 0.01),
        "tpr": (1, 3, 0.01),
    }
    for mode, (trees, segments, delta) in topologies.items():
        recs = []
        for rank in range(8):
            tp, cp = rank % 2, rank // 2
            rec = {
                "mode": mode, "rank": rank, "tp_rank": tp, "cp_rank": cp,
                "source_capture": "saved_original.pt",
                "source_rows": (0, 1),
                "source_keys": ("s0", "s1"),
                "data_sha256": "identical",
                "checkpoint": "Qwen3-4B",
                "objective": "original_captured_vanilla_PPO_token_mean",
                "logical_ppo_tokens": 2,
                "tp": 2, "cp": 4, "dp": 1,
                "optimizer": "diagnostic_bf16_sgd_lr_0.0",
                "forest_trees": trees,
                "forest_segments": segments,
                "ring_attention_impl": "tpr_custom_ring",
                "param_before": {"weight": {
                    "shape": (2,),
                    "indices": torch.tensor([0,1]),
                    "values": torch.tensor([1.0,2.0]),
                }},
                "gradient_samples": {"weight": {
                    "shape": (2,),
                    "indices": torch.tensor([0,1]),
                    "values": torch.tensor([0.4+delta,0.5+delta]),
                }},
                "global_bf16_grad_norm": 0.9+delta,
                "logprobs": {
                    (cp, 0): -5.0+delta
                } if tp == 0 and cp in (0, 3) else {},
                "loss": -1.0+delta,
                "time_forward_backward_finalize_s": 5+delta,
                "peak_allocated_bytes": 2**30,
            }
            # Logical Token keys are (row, response_offset) not (CP, offset).
            if tp == 0 and cp == 0:
                rec["logprobs"] = {(0, 0): -5.0+delta}
            elif tp == 0 and cp == 3:
                rec["logprobs"] = {(1, 0): -5.0+delta}
            recs.append(rec)
        states[mode] = recs
    return states


def test_cp4_three_arm_separates_physical_segmentation_from_reuse():
    x = analyze(_arms(), read_full_grads=False)
    assert x["topologies"]["split_no_share"]["segments"] == 4
    assert x["comparisons"]["full_vs_split_no_share"]["logprob_mean_abs"] == pytest.approx(.01)
    assert x["comparisons"]["split_no_share_vs_shared"]["logprob_mean_abs"] == 0
    assert x["comparisons"]["split_no_share_vs_shared"]["sampled_grad_rel_l2"] == 0


def test_cp4_three_arm_rejects_different_custom_attention_backend():
    a = _arms()
    a["split_no_share"][0]["ring_attention_impl"] = "mindspeed_native_whole_ring"
    # Every rank must match: mode0 mismatch is caught by follow-up rank gate.
    with pytest.raises(AssertionError, match="Not same custom Ring"):
        analyze(a, read_full_grads=False)


def test_cp4_three_arm_rejects_changed_initial_parameters():
    a = _arms()
    a["split_no_share"][0]["param_before"]["weight"]["values"][0] = 999
    with pytest.raises(AssertionError, match="Initial weights changed"):
        analyze(a, read_full_grads=False)


def test_cp4_three_arm_rejects_different_actor_labels():
    a = _arms()
    a["tpr"][1]["data_sha256"] = "different"
    with pytest.raises(AssertionError, match="Input/optimizer mismatch"):
        analyze(a, read_full_grads=False)
