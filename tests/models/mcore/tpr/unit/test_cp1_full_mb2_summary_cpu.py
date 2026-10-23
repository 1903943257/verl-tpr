"""CPU-only regression for the controlled Full-MB2/TPR-K2 reporter."""
import json

import pytest

from ..parallel.summarize_cp1_full_mb2_leaf_k2 import summarize


def _case(tmp_path, arm, mode, physical_mb=None, leaf_k=1):
    directory = tmp_path / arm
    directory.mkdir()
    item = {
        "mode": mode,
        "logical_trajectories": 2,
        "physical_full_micro_batch": physical_mb,
        "leaf_batch_configured_k": leaf_k,
        "leaf_batch_stats": {
            "model_forward_calls": 3 if leaf_k == 2 else 4,
            "grouped_leaf_calls": 1 if leaf_k == 2 else 0,
            "grouped_leaf_segments": 2 if leaf_k == 2 else 0,
        } if mode == "shared" else None,
        "input_sha256": "test-input",
        "initial_weights_witness_sha256": "test-weights",
        "ppo_tokens": 2,
        "source_capture": "/path/to/sample.pt",
        "checkpoint": "/path/to/checkpoint",
        "original_keys": ["r0", "r1"],
        "bench_repeats": 2,
        "bench_warmups": 1,
        "bench_times_s": [2.2, 2.0] if arm != "tpr_k2" else [1.2, 1.0],
        "fwd_bwd_seconds": 2.1 if arm != "tpr_k2" else 1.1,
        "peak_allocated_gib": 24.0 if arm != "tpr_k2" else 22.0,
        "peak_reserved_gib": 25.0 if arm != "tpr_k2" else 23.0,
        "loss": -0.4,
        "global_grad_norm": 6.0,
        "logprobs": [[0, 0, -0.1], [1, 0, -0.2]],
        "gradient_witness": {
            "names": ["p"],
            "canonical_shapes": [[2]],
            "values": [0.2, 0.4],
        },
    }
    (directory / "result.json").write_text(json.dumps(item))
    return item


def test_summary_reports_equal_rows_speed_and_capacity(tmp_path):
    _case(tmp_path, "full_mb1", "full", physical_mb=1)
    _case(tmp_path, "full_mb2", "full_batch2", physical_mb=2)
    _case(tmp_path, "tpr_k1", "shared")
    _case(tmp_path, "tpr_k2", "shared", leaf_k=2)
    report = summarize(tmp_path)
    assert "2 logical trajectories" in report
    assert "Full physical MB2 / TPR K2 MEDIAN F/B speedup" in report
    assert "Memory margin gate: PASS" in report
    assert "sampled grad relative L2" in report
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "SUMMARY.txt").exists()


def test_summary_handles_oom_partial_results(tmp_path):
    _case(tmp_path, "full_mb1", "full", physical_mb=1)
    report = summarize(tmp_path)
    assert "MISSING full_mb2" in report
    assert "NO Full MB2/TPR K2 result" in report


def test_summary_rejects_mismatched_inputs(tmp_path):
    _case(tmp_path, "full_mb2", "full_batch2", physical_mb=2)
    item = _case(tmp_path, "tpr_k2", "shared", leaf_k=2)
    item["input_sha256"] = "different-capture"
    (tmp_path / "tpr_k2" / "result.json").write_text(json.dumps(item))
    with pytest.raises(AssertionError, match="input_sha256"):
        summarize(tmp_path)
