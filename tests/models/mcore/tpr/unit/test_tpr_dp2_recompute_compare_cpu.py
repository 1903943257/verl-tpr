"""CPU-only parity regression for the 8-NPU DP2 recompute report comparer."""
from __future__ import annotations

import json

import pytest
import torch

from tests.models.mcore.tpr.parallel.compare_tpr_dp2_recompute import compare


def _write_case(root, *, enabled: bool, grad_delta: float = 0.0):
    name = "on" if enabled else "off"
    directory = root / name
    directory.mkdir()
    grad_paths, sample_paths = [], []
    for tp in (0, 1):
        grads = directory / f"grad_tp{tp}"
        grads.mkdir()
        tensor = torch.tensor([1.0 + grad_delta, 2.0, 3.0], dtype=torch.bfloat16)
        torch.save(tensor, grads / "grad_0000.pt")
        torch.save(
            [{"name": f"tp{tp}.weight", "shape": (3,), "tp_sharded": True,
              "filename": "grad_0000.pt"}],
            grads / "manifest.pt",
        )
        grad_paths.append(str(grads))
        samples = directory / f"weights_tp{tp}.pt"
        before = {f"tp{tp}.weight": {
            "shape": (3,), "indices": torch.tensor([0, 1]),
            "values": torch.tensor([1.0, 2.0]),
        }}
        after = {f"tp{tp}.weight": {
            "shape": (3,), "indices": torch.tensor([0, 1]),
            "values": torch.tensor([1.1, 2.1]),
        }}
        torch.save({"initial": before, "updated": after}, samples)
        sample_paths.append(str(samples))
    payload = {
        "mode": "tpr",
        "topology": {"tp": 2, "cp": 2, "dp": 2, "world": 8},
        "recompute": enabled,
        "recompute_granularity": "full" if enabled else None,
        "recompute_method": "uniform" if enabled else None,
        "recompute_num_layers": 1 if enabled else None,
        "recompute_installed_tpr_blocks": 1 if enabled else 0,
        "source_step": 6, "source_capture": "original_capture.pt",
        "input_sha256": "abcd", "checkpoint": "qwen4b",
        "effective_sequence_cap": 40960, "global_ppo_tokens": 61276,
        "dp_partitions": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "dp_tree_token_costs": [100, 100], "duplicated_tree_tokens": 0,
        "placement_policy": "dfs", "warmups": 1, "repeats": 2,
        "capture_logprobs": True, "gradient_sync_reduce_dtype": "bf16",
        "ppo_logprob_chunk_tokens": 512, "hccl_stats_reduce_dtype": "float32",
        "gradient_replica_validation_passed": True,
        "updated_weights": True, "full_gradient_paths": grad_paths,
        "optimizer_step": {
            "optimizer": "cpu_fp32_master_adamw_step1", "lr": 2e-5,
            "clip_norm": 1.0, "global_grad_norm_preclip": 1.0,
            "clip_factor": 1.0, "critical_optimizer_s": 2.0,
            "updated_replica_sample_max_abs": 0.0,
            "updated_weights_samples": sample_paths,
        },
        "median_fwd_bwd_s": 3.0 if enabled else 4.0,
        "runs": [
            {"recompute": enabled, "loss": 0.1,
             "critical_fwd_bwd_s": 3.0, "peak_allocated_gib": 10.0}
            for _ in range(3)
        ],
    }
    path = directory / "result.json"
    path.write_text(json.dumps(payload))
    return path


def test_dp2_compare_complete_gradients_and_adamw(tmp_path):
    off = _write_case(tmp_path, enabled=False)
    on = _write_case(tmp_path, enabled=True)
    report = compare(off, on)
    assert report["correctness_passed"]
    assert report["parameter_elements_checked"] == 6
    assert report["full_parameter_gradient_rel_l2"] == 0
    assert report["optimizer_update_gate_passed"]


def test_dp2_compare_rejects_full_parameter_gradient_drift(tmp_path):
    off = _write_case(tmp_path, enabled=False)
    on = _write_case(tmp_path, enabled=True, grad_delta=1.0)
    with pytest.raises(AssertionError, match="FULL gradient parity gate FAILED"):
        compare(off, on)


def test_dp2_compare_rejects_missing_gradient_replica_gate(tmp_path):
    off = _write_case(tmp_path, enabled=False)
    on = _write_case(tmp_path, enabled=True)
    data = json.loads(on.read_text())
    data["gradient_replica_validation_passed"] = False
    on.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="replica gate"):
        compare(off, on)
