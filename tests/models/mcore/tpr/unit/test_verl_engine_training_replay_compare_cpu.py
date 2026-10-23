"""CPU-only tests for same-capture real Full/TPR optimizer comparisons."""

import pytest
import torch

from ..parallel.compare_verl_engine_training_replay import compare


def _param(values):
    return {
        "decoder.layers.0.self_attention.linear_qkv.weight": {
            "shape": (2,),
            "indices": torch.tensor([0, 1], dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float32),
        }
    }


def _artifact(mode, rank):
    tpr = mode == "tpr"
    return {
        "execution_backend": (
            "verl_mindspeed_megatron_engine_tpr" if tpr
            else "verl_mindspeed_megatron_engine_full"
        ),
        "checkpoint": "/models/Qwen3-4B",
        "source_capture": "/captures/step1.pt",
        "source_rows": tuple(range(8)),
        "source_keys": tuple(f"session_tpr_{i}" for i in range(8)),
        "source_online_global_step": 1,
        "source_capture_index": 0,
        "data_sha256": "same-sha",
        "capture_manifest": "/captures/manifest.json",
        "tokens": 2,
        "tp_rank": rank % 2,
        "cp_rank": rank // 2,
        "optimizer_lr": 2e-5,
        "optimizer_clip_norm": 1.0,
        "loss": [0.5, 0.2] if not tpr else [0.7],
        "logprobs": (
            {(0, 0): -1.0, (1, 0): -2.0 + (0.001 if tpr else 0.0)}
            if rank == 0 else {}
        ),
        "param_initial": _param([1.0, 2.0]),
        "param_after": _param([0.99 if tpr else 1.0, 2.0]),
        "gradient_samples": _param([0.9 if tpr else 1.0, 2.0]),
        "qkv_grad_storage": {"main_grad_dtype": "torch.bfloat16"} if tpr else None,
        "fwd_bwd_seconds": 2.0 if not tpr else 1.0,
        "optimizer_seconds": 0.5,
        "peak_allocated_bytes": 1024 if tpr else 2048,
        "peak_reserved_bytes": 4096,
    }


def _write(root):
    for rank in range(8):
        torch.save(
            _artifact("full", rank),
            root / f"verl_full_smoke_step1_rank{rank}.pt",
        )
        torch.save(
            _artifact("tpr", rank),
            root / f"verl_tpr_smoke_step1_rank{rank}.pt",
        )


def _compare(root):
    return compare(
        root, steps=1, full_prefix="verl_full_smoke",
        tpr_prefix="verl_tpr_smoke",
    )


def test_real_engine_replay_cpu_comparison(tmp_path):
    _write(tmp_path)
    record = _compare(tmp_path)["records"][0]
    assert record["fb_speedup"] == pytest.approx(2.0)
    assert record["loss_abs_gap"] == pytest.approx(0.0)
    assert record["logprobs"]["count"] == 2
    assert record["sampled_gradient"]["count"] == 4
    assert record["sampled_gradient"]["relative_l2"] > 0
    assert record["qkv_grad_storage_tp_shards"] == [
        {"main_grad_dtype": "torch.bfloat16"},
        {"main_grad_dtype": "torch.bfloat16"},
    ]


def test_comparison_rejects_different_capture(tmp_path):
    _write(tmp_path)
    path = tmp_path / "verl_tpr_smoke_step1_rank3.pt"
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    snapshot["data_sha256"] = "mismatch"
    torch.save(snapshot, path)
    with pytest.raises(AssertionError, match="data_sha256"):
        _compare(tmp_path)


def test_comparison_rejects_different_initial_checkpoint(tmp_path):
    _write(tmp_path)
    path = tmp_path / "verl_tpr_smoke_step1_rank0.pt"
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    snapshot["param_initial"] = _param([2.0, 3.0])
    torch.save(snapshot, path)
    with pytest.raises(AssertionError, match="initial parameter"):
        _compare(tmp_path)


def test_comparison_rejects_incomplete_token_logprobs(tmp_path):
    _write(tmp_path)
    path = tmp_path / "verl_tpr_smoke_step1_rank0.pt"
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    snapshot["logprobs"].pop((1, 0))
    torch.save(snapshot, path)
    with pytest.raises(AssertionError, match="different logical PPO tokens"):
        _compare(tmp_path)
