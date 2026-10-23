"""CPU coverage for fair N8 benchmark's outputs / failed partial runs."""
import json

import pytest
from ..parallel.summarize_cp1_n8_fair import summarize


def _write_case(root, arm):
    folder = root / arm
    folder.mkdir()
    full = arm == "full_mb2"
    k = 1 if full else int(arm[-1])
    result = {
        "mode": "full_mb2" if full else "tpr", "configured_k": k,
        "logical_trajectories": 8,
        "input_sha256": "sha256-valid",
        "source_capture": "/fake/n8.pt",
        "checkpoint": "/fake/ckpt",
        "original_keys": [f"uid_turn_{i}" for i in range(8)],
        "initial_weights_witness_sha256": "identical-weights",
        "ppo_tokens": 3333, "logical_input_tokens": 16000,
        "unique_tree_tokens": 16000 if full else 9000,
        "input_lengths": [2000] * 8,
        "forest_trees": 8 if full else 1, "forest_segments": 8 if full else 9,
        "physical_model_forward_calls": 4 if full else 8 // k + 2,
        "leaf_batch_stats": None if full else {
            "grouped_leaf_calls": 0 if k == 1 else 8 // k,
            "grouped_leaf_segments": 0 if k == 1 else 8,
        },
        "bench_warmups": 1, "bench_repeats": 3,
        "bench_times_s": [2, 2.1, 2.2],
        "fwd_bwd_seconds": 2.1 if full else 1.9,
        "peak_allocated_gib": 36.0 if full else 22.0,
        "peak_reserved_gib": 39.0 if full else 24.0,
        "loss": -0.8, "global_grad_norm": 7.0,
        "stage_profile": False,
    }
    (folder / "result.json").write_text(json.dumps(result))
    return result


def test_n8_report_includes_workload_speed_memory(tmp_path):
    for arm in ("full_mb2", "tpr_k1", "tpr_k2", "tpr_k4"):
        _write_case(tmp_path, arm)
    printed = summarize(tmp_path)
    assert "Eight DISTINCT actual same-session PPO trajectories" in printed
    assert "Model forward calls" in printed
    assert "Actual N8 unique-token saving" in printed
    assert "memory_margin=PASS" in printed
    assert "TPR K=2" not in printed or "tpr_k2:" in printed
    assert (tmp_path / "summary.json").exists()


def test_n8_missing_arms_show_partial_results(tmp_path):
    _write_case(tmp_path, "full_mb2")
    printed = summarize(tmp_path)
    assert "MISSING ARMS" in printed
    assert "tpr_k1" in printed


def test_n8_report_rejects_fake_input_mismatch(tmp_path):
    _write_case(tmp_path, "full_mb2")
    item = _write_case(tmp_path, "tpr_k2")
    item["input_sha256"] = "not-same"
    (tmp_path / "tpr_k2" / "result.json").write_text(json.dumps(item))
    with pytest.raises(AssertionError, match="input_sha256"):
        summarize(tmp_path)
