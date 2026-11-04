# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU gates for offline persisted split parity and CP1/CP2 loss positions."""

import json

import torch

from tests.models.mcore.tpr.parallel.compare_split_fp32_saved import compare_saved
from tests.models.mcore.tpr.parallel.test_split_p32s32_cp2_parity_npu import _objective


def test_offline_split_comparator_uses_persisted_grads_without_model(tmp_path):
    tokens = torch.arange(64, dtype=torch.float32)
    logits = (tokens[:, None, None] / 100).expand(64, 1, 12).contiguous()
    manifest = [{"name": "linear.weight", "file": "grad_00000.pt", "shape": [2, 2]}]
    for arm in ("full", "split_connected", "split_replay"):
        folder = tmp_path / arm
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps(manifest))
        torch.save(torch.arange(4, dtype=torch.float32).reshape(2, 2), folder / "grad_00000.pt")
        torch.save([
            {
                "stage": "logits", "arm": arm, "start": 0, "end": 64,
                "forward": logits, "gradient": None, "grad_enabled": True,
                "expect_gradient": False,
            }
        ], folder / "trace.pt")
    report = compare_saved(tmp_path, prefix=32, suffix=32)
    assert (tmp_path / "report.json").is_file()
    assert report["data_source"] == "saved_gradients_and_traces_no_npu_rerun"
    assert report["pairs"]["full_vs_split_connected"]["gradient_relative_l2"] == 0
    assert report["pairs"]["full_vs_split_replay"]["gradient_max_abs"] == 0
    assert report["arms"]["split_replay"]["logit_ranges"] == 1


def test_cp2_loss_preserves_prefix_last_query_and_gradient():
    torch.manual_seed(100)
    logits = torch.randn(1, 64, 17, dtype=torch.float32, requires_grad=True)
    tokens = torch.randint(0, 17, (64,), dtype=torch.long)
    positions = torch.arange(64)
    full, full_logp, full_coverage = _objective(logits, positions, tokens)
    full.backward()
    expected = logits.grad.clone()

    segmented = logits.detach().clone().requires_grad_(True)
    first, first_logp, first_coverage = _objective(segmented[:, :32], positions[:32], tokens)
    second, second_logp, second_coverage = _objective(segmented[:, 32:], positions[32:], tokens)
    (first + second).backward()
    torch.testing.assert_close(first + second, full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(segmented.grad, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(first_logp + second_logp, full_logp, rtol=1e-5, atol=1e-6)
    assert torch.equal(first_coverage + second_coverage, full_coverage)
    assert bool(first_coverage[31])
    assert not bool(full_coverage[63])
    assert float(segmented.grad[0, 31].abs().sum()) > 0
    assert float(segmented.grad[0, 63].abs().max()) == 0
