"""CPU tests for the direct N=8 TreeBatch K1/K2 parity comparator."""
import json

import pytest
import torch

from ..parallel.compare_cp1_n8_treebatch import compare


def _write_arm(root, k, grad_values):
    arm = root / f"tpr_k{k}"
    arm.mkdir(parents=True)
    grad_dir = arm / f"tpr_k{k}_step1_tp0_full_grads"
    grad_dir.mkdir()
    torch.save(
        [{"name": "layer.weight", "shape": (3,), "filename": "grad_0000.pt",
          "tp_sharded": False}],
        grad_dir / "manifest.pt",
    )
    torch.save(torch.tensor(grad_values, dtype=torch.bfloat16), grad_dir / "grad_0000.pt")
    item = {
        "mode": "tpr",
        "configured_k": k,
        "input_sha256": "same-input-sha",
        "source_capture": "/fake/capture.pt",
        "checkpoint": "/fake/qwen3-4b",
        "original_keys": ["session-turn-0", "session-turn-1"],
        "initial_weights_witness_sha256": "same-initial-weights",
        "ppo_tokens": 2,
        "logical_trajectories": 8,
        "logical_input_tokens": 160,
        "input_lengths": [20] * 8,
        "forest_trees": 1,
        "forest_segments": 9,
        "unique_tree_tokens": 90,
        "physical_model_forward_calls": 4 if k == 1 else 3,
        "leaf_batch_stats": {
            "grouped_leaf_calls": 0 if k == 1 else 1,
            "grouped_leaf_segments": 0 if k == 1 else 2,
        },
        "fwd_bwd_seconds": 1.0 if k == 1 else 0.8,
        "peak_allocated_gib": 20.0 if k == 1 else 21.0,
        "peak_reserved_gib": 22.0 if k == 1 else 23.0,
        "loss": -0.5 if k == 1 else -0.501,
        "logprobs": [[0, 0, -0.1], [1, 0, -0.2]],
        "gradient_witness": {"values": [1.0, 2.0, 3.0]},
        "full_parameter_grad_dir": str(grad_dir),
    }
    (arm / "result.json").write_text(json.dumps(item))
    return item


def test_treebatch_k1_k2_comparator_passes_with_real_grouping(tmp_path):
    _write_arm(tmp_path, 1, [1.0, 2.0, 3.0])
    _write_arm(tmp_path, 2, [1.01, 2.01, 3.01])
    report = compare(tmp_path)
    assert report["passed"] is True
    assert report["gates"]["actual_grouping"] is True
    assert report["full_parameter_gradient"]["parameter_count"] == 1
    saved = json.loads((tmp_path / "treebatch_k1_vs_k2.json").read_text())
    assert saved["passed"] is True


def test_treebatch_k1_k2_comparator_fails_large_full_gradient_gap(tmp_path):
    _write_arm(tmp_path, 1, [1.0, 2.0, 3.0])
    _write_arm(tmp_path, 2, [10.0, -20.0, 30.0])
    with pytest.raises(AssertionError, match="correctness gates failed"):
        compare(tmp_path)
    report = json.loads((tmp_path / "treebatch_k1_vs_k2.json").read_text())
    assert report["gates"]["full_grad_rel_l2"] is False


def test_treebatch_k2_comparator_rejects_fake_grouping(tmp_path):
    _write_arm(tmp_path, 1, [1.0, 2.0, 3.0])
    item = _write_arm(tmp_path, 2, [1.01, 2.01, 3.01])
    item["leaf_batch_stats"]["grouped_leaf_calls"] = 0
    (tmp_path / "tpr_k2" / "result.json").write_text(json.dumps(item))
    with pytest.raises(AssertionError, match="did not execute a real grouped-leaf"):
        compare(tmp_path)
