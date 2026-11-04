# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""BF16 summary metrics, no-dump contract, and real executor backward fault gate."""

import json
import math

import pytest
import torch

from tests.models.mcore.tpr.parallel import split_bf16_quick as quick
from tests.models.mcore.tpr.parallel.fast_fp32_grad_compare import fast_tensor_difference
from tests.models.mcore.tpr.parallel.split_fp32_attribution import math_attention
from tests.models.mcore.tpr.unit.test_split_fp32_attribution_cpu import TinyFP32Model


def _cpu_bf16_attention(q, k, v, **kwargs):
    # CPU stand-in only; the NPU entry delegates to the unmodified CANN adapter.
    return math_attention(q.float(), k.float(), v.float(), **kwargs).bfloat16()


def _model():
    torch.manual_seed(21)
    model = TinyFP32Model().bfloat16()
    model.math_attention = _cpu_bf16_attention
    return model


def test_bf16_summary_matches_independent_flat_fp64_oracle_without_dumps(tmp_path, monkeypatch):
    model = _model()
    tokens = torch.arange(8, dtype=torch.long)
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    snapshots = []
    compare = quick._compare_current

    def observe(model, references):
        snapshots.append(torch.cat([p.grad.detach().flatten().double() for p in model.parameters()]))
        return compare(model, references)

    def reject_dump(*args, **kwargs):
        raise AssertionError("Quick experiment must never serialize tensors")

    monkeypatch.setattr(quick, "_compare_current", observe)
    monkeypatch.setattr(torch, "save", reject_dump)
    report = quick.run_bf16_quick(model, tokens, 3, tmp_path / "quick", production_attention=False)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        a, b = snapshots[left], snapshots[right]
        pair = report["pairs"][f"{quick.ARMS[left]}_vs_{quick.ARMS[right]}"]
        assert math.isclose(pair["gradient_relative_l2"], float((b - a).norm() / a.norm()), rel_tol=1e-12)
        assert math.isclose(pair["gradient_cosine"], float(torch.dot(a, b) / (a.norm() * b.norm())), rel_tol=1e-12)
        assert pair["gradient_max_abs"] == float((b - a).abs().max())
        assert pair["elements"] == a.numel()
        assert "parameters" not in pair
    assert report["pairs"]["full_vs_split_replay"]["gradient_relative_l2"] < 0.03
    assert list((tmp_path / "quick").iterdir()) == [tmp_path / "quick" / "report.json"]
    assert json.loads((tmp_path / "quick/report.json").read_text())["parameter_dumps"] is False
    assert all(torch.equal(weights[name], p) and p.grad is None for name, p in model.named_parameters())
    assert [report["arms"][arm]["attention_calls"] for arm in quick.ARMS] == [2, 4, 6]


def test_bf16_summary_detects_replay_dkv_corruption(tmp_path, monkeypatch):
    from verl.workers.engine.megatron.tpr.state import KVPrefixState

    tokens = torch.arange(8, dtype=torch.long)
    baseline = quick.run_bf16_quick(_model(), tokens, 3, tmp_path / "baseline", production_attention=False)
    original = KVPrefixState.accumulate_anchor_gradients

    def corrupt(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for pair in self.gradients.values():
            for gradient in pair:
                gradient.mul_(2)

    monkeypatch.setattr(KVPrefixState, "accumulate_anchor_gradients", corrupt)
    fault = quick.run_bf16_quick(_model(), tokens, 3, tmp_path / "fault", production_attention=False)
    name = "split_connected_vs_split_replay"
    assert fault["pairs"][name]["gradient_relative_l2"] > baseline["pairs"][name]["gradient_relative_l2"] * 2
    assert fault["pairs"][name]["loss_abs_gap"] == baseline["pairs"][name]["loss_abs_gap"]


def test_bf16_loss_has_identical_global_boundary_gradient():
    logits = torch.zeros(1, 8, 32, dtype=torch.bfloat16, requires_grad=True)
    quick._bf16_loss(logits, 0, 8).backward()
    expected = logits.grad.clone()
    split = logits.detach().clone().requires_grad_(True)
    (quick._bf16_loss(split[:, :3], 0, 8) + quick._bf16_loss(split[:, 3:], 3, 8)).backward()
    assert torch.equal(split.grad, expected)
    assert split.grad[0, 2].count_nonzero() > 0
    assert split.grad[0, -1].count_nonzero() == 0


def test_bf16_comparator_is_blockwise_and_rejects_nonfinite():
    a = torch.linspace(-2, 3, 211).bfloat16()
    b = (a.float() + 0.012).bfloat16()
    metrics = fast_tensor_difference(a, b, block_elements=17)
    assert math.isclose(
        metrics["relative_l2"], float((b.double() - a.double()).norm() / a.double().norm()), rel_tol=1e-12
    )
    b[70] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        fast_tensor_difference(a, b, block_elements=17)


def test_quick_rejects_fp32_model(tmp_path):
    with pytest.raises(TypeError, match="BF16 model"):
        quick.run_bf16_quick(TinyFP32Model(), torch.arange(8), 3, tmp_path / "invalid", production_attention=False)
