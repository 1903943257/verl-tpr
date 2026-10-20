"""CPU checks for immutable real-PPO one-step capture truncation.

The derived 30K experiment must not regenerate labels or change the original
prompt, and all response-aligned fields must be cropped identically.
"""
from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("tensordict")
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from ..parallel.prepare_tpr_30k_one_step import derive_capture

# Regression contracts formerly in test_tpr_30k_three_way_comparison.py
import copy

import pytest

from ..parallel.compare_tpr_30k_three_way import compare_three

def _source_capture(path, *, overlong_prompt=False):
    prompts = [
        torch.tensor(list(range(1, 13 if (overlong_prompt and i == 3) else 4)),
                     dtype=torch.int64)
        for i in range(8)
    ]
    responses = [torch.tensor(list(range(20, 20 + 7 + (i % 3))),
                              dtype=torch.int64) for i in range(8)]
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses)]
    fields = {
        "input_ids": [torch.cat((p, r)) for p, r in zip(prompts, responses)],
        "prompts": prompts,
        "responses": responses,
        "attention_mask": [torch.ones(n, dtype=torch.int64) for n in lengths],
        "response_mask": [torch.ones(len(x), dtype=torch.bool) for x in responses],
        "loss_mask": [torch.ones(len(x), dtype=torch.int64) for x in responses],
        "old_log_probs": [torch.arange(len(x), dtype=torch.float32) / 10
                          for x in responses],
        "advantages": [torch.arange(len(x), dtype=torch.float32)
                       for x in responses],
    }
    nested = {
        key: tu.nested_tensor_from_tensor_list(rows, ragged_idx=1)
        for key, rows in fields.items()
    }
    nested["temperature"] = torch.ones(8, dtype=torch.float32)
    batch = TensorDict(nested, batch_size=[8])
    # Real captured TensorDicts contain batch-level scalar metadata, not
    # only row-aligned tensors. These values must survive shallow cloning
    # without being revalidated as shape-[8] arrays.
    tu.assign_non_tensor(batch, scalar_metadata=17,
                         scalar_tensor_metadata=torch.tensor(1.25))
    keys = [f"uid1_tpr_{i}" for i in range(8)]
    torch.save({
        "capture_stage": "actor_update_mini_batch",
        "keys": keys, "tensordict": batch,
    }, path)


def test_derived_capture_truncates_all_response_labels_without_mutating_source(tmp_path):
    src, target = tmp_path / "source.pt", tmp_path / "truncated.pt"
    _source_capture(src)
    before = torch.load(src, weights_only=False)
    stats = derive_capture(src, target, cap=8)
    raw = torch.load(target, weights_only=False)
    assert raw["keys"] == before["keys"]
    assert raw["offline_sequence_cap"] == 8
    assert stats["max_sequence_length"] == 8
    cropped, original = raw["tensordict"], before["tensordict"]
    assert set(cropped.keys()) == set(original.keys())
    assert tu.get_non_tensor_data(cropped, key="scalar_metadata", default=None) == 17
    scalar_value = tu.get_non_tensor_data(cropped, key="scalar_tensor_metadata", default=None)
    assert torch.equal(torch.as_tensor(scalar_value), torch.tensor(1.25))
    for row in range(8):
        assert torch.equal(cropped["prompts"][row], original["prompts"][row])
        assert torch.equal(cropped["input_ids"][row], original["input_ids"][row][:8])
        assert torch.equal(
            cropped["responses"][row], original["responses"][row][:5]
        )
        for key in ("response_mask", "loss_mask", "old_log_probs", "advantages"):
            assert torch.equal(cropped[key][row], original[key][row][:5])
    assert stats["ppo_tokens"] == 40
    assert len(before["tensordict"]["input_ids"][0]) == 10


def test_derived_capture_rejects_prompt_longer_than_cap(tmp_path):
    src, target = tmp_path / "source.pt", tmp_path / "truncated.pt"
    _source_capture(src, overlong_prompt=True)
    with pytest.raises(ValueError, match="prompt alone"):
        derive_capture(src, target, cap=8)
    assert not target.exists()

# Tests consolidated from test_tpr_30k_three_way_comparison.py
def _artifact(mode, topology, elapsed, peak, loss):
    witness = {
        "names": ["decoder.layers.0.input_layernorm.weight"],
        "shapes": [[2]],
        "samples_per_name": [2],
        "samples_per_parameter": 2,
        "tp_shards": [[1.0, 2.0], [3.0, 4.0]],
        "replica_sample_max_abs": 0.0,
        "replica_exact": True,
        "replica_consistency_enforced": True,
    }
    runs = [
        {"warmup": True, "ppo_tokens": 36933, "peak_allocated_gib": peak,
         "critical_fwd_bwd_s": elapsed+1, "loss": loss},
        {"warmup": False, "ppo_tokens": 36933, "peak_allocated_gib": peak,
         "critical_fwd_bwd_s": elapsed, "loss": loss},
        {"warmup": False, "ppo_tokens": 36933, "peak_allocated_gib": peak,
         "critical_fwd_bwd_s": elapsed, "loss": loss},
    ]
    return {
        "mode": mode,
        "topology": topology,
        "source_step": 6,
        "source_capture": "/workspace/paired/step6_cap30720.pt",
        "input_sha256": "capture_sha",
        "checkpoint": "/workspace/hf_models/Qwen3-4B-128K",
        "warmups": 1, "repeats": 2,
        "capture_logprobs": True,
        "profiler_enabled": False,
        "effective_sequence_cap": 30720,
        "gradient_sync_reduce_dtype": "bf16",
        "hccl_stats_reduce_dtype": "float32",
        "updated_weights": False,
        "median_fwd_bwd_s": elapsed,
        "runs": runs,
        "gradient_witness": witness,
    }


def _three():
    full = _artifact(
        "full", {"tp":2,"cp":4,"dp":1,"world":8},
        16.0, 34.0, 0.15,
    )
    # Old Full/DP2 reports predate explicit deterministic metadata.
    del full["topology"]
    cp4 = _artifact(
        "tpr", {"tp":2,"cp":4,"dp":1,"world":8},
        14.0, 40.0, 0.1501,
    )
    cp4["hccl_deterministic"] = "true"
    dp2 = _artifact(
        "tpr", {"tp":2,"cp":2,"dp":2,"world":8},
        12.0, 41.0, 0.1502,
    )
    dp2["global_ppo_tokens"] = 36933
    dp2["dp_partitions"] = [[0,1,2,3], [4,5,6,7]]
    dp2["dp_tree_token_costs"] = [60000,70000]
    dp2["duplicated_tree_tokens"] = 17000
    return full,cp4,dp2


def test_distinct_tpr_cp4_incremental_dp2_speedup_and_gradient_parity():
    r = compare_three(*_three())
    assert r["full_vs_tpr_cp4_speedup"] == pytest.approx(16/14)
    assert r["tpr_cp4_vs_tpr_cp2dp2_dp_gain"] == pytest.approx(14/12)
    assert r["full_vs_tpr_cp2dp2_speedup"] == pytest.approx(16/12)
    assert r["ppo_loss_abs_gap_full_vs_tpr_cp4"] == pytest.approx(0.0001)
    assert r["global_ppo_tokens"] == 36933
    assert r["sampled_gradient_parity"]["full_vs_tpr_cp4"]["sampled_rel_l2"] == 0
    assert r["sampled_gradient_parity"]["tpr_cp4_vs_tpr_cp2dp2"]["sampled_cosine"] == pytest.approx(1.0)
    assert r["hccl_deterministic_old_baselines_metadata"] == {
        "full":"not_recorded", "tpr_cp2dp2":"not_recorded"
    }


@pytest.mark.parametrize("field,new_value", [
    ("input_sha256", "different_sha"),
    ("source_capture", "/different_capture.pt"),
    ("checkpoint", "/different_checkpoint"),
    ("gradient_sync_reduce_dtype", "fp32"),
    ("repeats", 3),
    ("effective_sequence_cap", 40960),
])
def test_mismatched_old_new_experiments_rejected(field,new_value):
    full,cp4,dp2 = _three()
    cp4[field] = new_value
    with pytest.raises(AssertionError):
        compare_three(full,cp4,dp2)


def test_missing_or_disagreeing_gradient_replicas_rejected():
    full,cp4,dp2 = _three()
    dp2["gradient_witness"]["replica_sample_max_abs"] = 0.00001
    with pytest.raises(AssertionError, match="replicated"):
        compare_three(full,cp4,dp2)


def test_cp4_must_have_deterministic_hccl():
    full,cp4,dp2 = _three()
    cp4["hccl_deterministic"] = "false"
    with pytest.raises(AssertionError, match="DETERMINISTIC"):
        compare_three(full,cp4,dp2)


def test_30k_comparison_preserves_ppo_sample_count():
    full,cp4,dp2 = _three()
    cp4["runs"][0]["ppo_tokens"] = 36934
    with pytest.raises(AssertionError, match="global_ppo_tokens"):
        compare_three(full,cp4,dp2)
