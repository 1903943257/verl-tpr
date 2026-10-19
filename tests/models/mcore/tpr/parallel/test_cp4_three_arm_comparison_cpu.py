"""CPU-only gates for controlled CP4 Whole/Split/Shared PPO attribution."""
from copy import deepcopy

import pytest
import torch

from .compare_cp4_split_no_share_cpu import analyze


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
