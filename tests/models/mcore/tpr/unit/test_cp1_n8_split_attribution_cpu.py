"""CPU tests for the bounded N8 split-attribution report."""
import json

import torch

from ..parallel.compare_cp1_n8_split_attribution import ARMS, compare


def _write_arm(root, arm, grad_values):
    directory = root / arm
    directory.mkdir(parents=True)
    grad_dir = directory / f"{arm}_full_grads"
    grad_dir.mkdir()
    torch.save(
        [{"name": "layer.weight", "shape": (3,), "filename": "grad_0000.pt"}],
        grad_dir / "manifest.pt",
    )
    torch.save(torch.tensor(grad_values, dtype=torch.bfloat16), grad_dir / "grad_0000.pt")
    mode = "tpr" if arm.startswith("tpr_k") else arm
    k = 1 if arm == "tpr_k1" else 2 if arm == "tpr_k2" else 1
    item = {
        "mode": mode,
        "configured_k": k,
        "input_sha256": "same-capture",
        "source_capture": "/capture/n8.pt",
        "checkpoint": "/model/qwen3-4b",
        "original_keys": [f"trajectory-{i}" for i in range(8)],
        "initial_weights_witness_sha256": "same-initial-weights",
        "ppo_tokens": 100,
        "logical_trajectories": 8,
        "logical_input_tokens": 800,
        "input_lengths": [100] * 8,
        "loss": -0.5 if arm == "full_mb1" else -0.501,
        "logprobs": [[row, offset, -0.1 - offset * 1e-5]
                     for row in range(8) for offset in range(12 if row == 0 else 11)],
        "physical_model_forward_calls": 16 if arm == "split_no_share" else 8,
        "leaf_batch_stats": {
            "grouped_leaf_calls": 0 if arm != "tpr_k2" else 4,
        },
        "forest_trees": 8 if arm in ("full_mb1", "split_no_share") else 1,
        "forest_segments": (
            16 if arm == "split_no_share" else
            8 if arm == "full_mb1" else 9
        ),
        "unique_tree_tokens": 800 if arm in ("full_mb1", "split_no_share") else 500,
        "full_parameter_grad_dir": str(grad_dir),
    }
    (directory / "result.json").write_text(json.dumps(item))


def test_n8_split_attribution_reports_segment_prefix_and_treebatch_deltas(tmp_path):
    values = {
        "full_mb1": [1.0, 2.0, 3.0],
        "split_no_share": [1.2, 2.1, 3.1],
        "tpr_k1": [1.21, 2.11, 3.11],
        "tpr_k2": [1.22, 2.12, 3.12],
    }
    for arm in ARMS:
        _write_arm(tmp_path, arm, values[arm])
    report = compare(tmp_path)
    assert report["diagnostic_ratios"]["full_vs_split_grad_rel_l2"] > 0
    assert report["diagnostic_ratios"]["split_vs_shared_k1_grad_rel_l2"] > 0
    assert report["diagnostic_ratios"]["treebatch_grad_rel_l2"] > 0
    alignment = report["gradient_differences"]["error_vector_alignment"]["segment_vs_treebatch"]
    assert alignment["error_vector_cosine"] > 0.99
    saved = json.loads((tmp_path / "split_attribution.json").read_text())
    assert "diagnostic only" in saved["interpretation_note"]


def test_n8_split_attribution_rejects_mismatched_capture(tmp_path):
    for arm in ARMS:
        _write_arm(tmp_path, arm, [1.0, 2.0, 3.0])
    path = tmp_path / "tpr_k2" / "result.json"
    item = json.loads(path.read_text())
    item["input_sha256"] = "different-capture"
    path.write_text(json.dumps(item))
    import pytest
    with pytest.raises(AssertionError, match="differ in input_sha256"):
        compare(tmp_path)
