# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Matched projection arithmetic and explicit native Ring routing/normalization."""

import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.models.mcore.tpr.parallel.native_ring_reference import native_cp_positions, native_ring_attention
from tests.models.mcore.tpr.parallel.projection_shape_probe import (
    _case,
    capture_projection_sample,
    run_projection_probe,
)
from tests.models.mcore.tpr.parallel.test_split_p32s32_cp2_parity_npu import _objective, _run_native_cp2
from tests.models.mcore.tpr.unit.test_split_bf16_quick_cpu import _model
from verl.workers.engine.megatron.tpr.ring import make_ring_sequence_shard


def test_model_fixture_capture_and_projection_metrics_do_not_dump_tensors(monkeypatch):
    model = _model()
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    module, sample = capture_projection_sample(model, torch.arange(64) % 32)

    def reject_dump(*args, **kwargs):
        raise AssertionError("Projection experiment must not serialize raw tensors")

    monkeypatch.setattr(torch, "save", reject_dump)
    result = run_projection_probe(module, sample)
    json.dumps(result, allow_nan=False)
    assert result["captured_vs_isolated_full"]["output"]["max_abs"] == 0
    assert result["captured_vs_isolated_full"]["weight_gradient"]["relative_l2"] < 1e-5
    for mode in result["modes"].values():
        assert mode["full_vs_repeat_full"]["output"]["max_abs"] == 0
        assert mode["full_vs_split_connected"]["output"]["elements"] == sample["output"].numel()
    assert all(torch.equal(initial[name], p) and p.grad is None for name, p in model.named_parameters())


def test_fp32_projection_gradients_match_independent_fp64_linear_oracle():
    module, sample = capture_projection_sample(_model(), torch.arange(64) % 32)
    actual = _case(module, sample, "fp32_reference", "full")
    x, w, dy = sample["input"].double(), module.weight.detach().double(), sample["upstream"].double()
    expected = {
        "output": x @ w.T,
        "input_gradient": dy @ w,
        "weight_gradient": dy.flatten(0, 1).T @ x.flatten(0, 1),
    }
    for name in expected:
        torch.testing.assert_close(actual[name].double(), expected[name], atol=2e-6, rtol=2e-6)
    doubled = _case(module, {**sample, "upstream": sample["upstream"] * 2}, "fp32_reference", "full")
    assert torch.equal(doubled["output"], actual["output"])
    torch.testing.assert_close(doubled["weight_gradient"], actual["weight_gradient"] * 2)


def test_local_fp32_compute_does_not_hide_bf16_weight_gradient_accumulation_loss():
    module = torch.nn.Linear(1, 1, bias=False).bfloat16()
    x = torch.ones(64, 1, 1, dtype=torch.bfloat16)
    x[31] = 1 / 64
    x[32:63] = -1
    x[63] = 1 / 64
    sample = {"input": x, "upstream": torch.ones_like(x)}
    full = _case(module, sample, "fp32_bf16_boundary", "full")
    sequential = _case(module, sample, "fp32_bf16_boundary", "split_sequential")
    reference = _case(module, sample, "fp32_reference", "split_sequential")
    assert float(full["weight_gradient"]) == 1 / 32
    assert float(sequential["weight_gradient"]) == 0
    assert float(reference["weight_gradient"]) == 1 / 32


@pytest.mark.parametrize("rank", [0, 1])
def test_native_cp_positions_match_global_token_contract(rank):
    positions = native_cp_positions(64, rank, 2, device="cpu")
    expected = make_ring_sequence_shard(64, cp_rank=rank, cp_size=2).global_indices(device="cpu")
    assert torch.equal(positions, expected)


def _fake_group(monkeypatch, rank):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank if group is not None else (4, 7)[rank])
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [4, 7])


@pytest.mark.parametrize("rank", [0, 1])
def test_native_ring_bridge_preserves_gqa_and_delegates_autograd_to_native_api(monkeypatch, rank):
    _fake_group(monkeypatch, rank)
    group = object()
    q = torch.randn(32, 1, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(32, 1, 2, 8, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    calls = []

    def native(q, k, v, heads, cp_para, **kwargs):
        calls.append(cp_para)
        assert q.shape == (32, 1, 32) and k.shape == v.shape == (32, 1, 16)
        assert heads == 4 and q.dtype == k.dtype == v.dtype == torch.bfloat16
        assert cp_para["cp_group"] is group and cp_para["rank"] == rank
        assert cp_para["causal"] and cp_para["cache_policy"] is None
        assert cp_para["cp_global_ranks"] == [4, 7] and not cp_para["megatron_cp_in_bnsd"]
        assert kwargs == {"softmax_scale": 1 / math.sqrt(8), "dropout_p": 0.0}
        return q * 2

    output = native_ring_attention(q, k, v, cp_group=group, softmax_scale=1 / math.sqrt(8), implementation=native)
    output.float().sum().backward()
    assert len(calls) == 1 and torch.equal(q.grad, torch.full_like(q, 2))


def test_native_ring_bridge_never_substitutes_tpr_if_mindspeed_is_missing(monkeypatch):
    _fake_group(monkeypatch, 0)
    monkeypatch.setitem(sys.modules, "mindspeed", None)
    q = torch.zeros(32, 1, 2, 4, dtype=torch.bfloat16)
    with pytest.raises(ImportError):
        native_ring_attention(q, q, q, cp_group=object(), softmax_scale=0.5)


def test_native_full_path_preserves_boundary_and_global_gradient_normalization():
    class Model(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.nn.Parameter(logits.clone())
            self.decoder = SimpleNamespace(
                layers=[
                    SimpleNamespace(self_attention=SimpleNamespace(core_attention=SimpleNamespace(native_ring_calls=0)))
                ]
            )

        def forward(self, input_ids, position_ids, attention_mask):
            assert attention_mask is None
            positions = position_ids[0]
            assert torch.equal(input_ids[0], tokens.index_select(0, positions))
            self.decoder.layers[0].self_attention.core_attention.native_ring_calls += 1
            return self.logits.index_select(1, positions)

    torch.manual_seed(31)
    values = torch.randn(1, 64, 17)
    tokens = torch.arange(64) % 17
    expected_logits = values.clone().requires_grad_(True)
    loss, _, _ = _objective(expected_logits, torch.arange(64), tokens)
    expected = torch.autograd.grad(loss, expected_logits)[0]
    grads = []
    for rank in (0, 1):
        model = Model(values)
        _run_native_cp2(model, SimpleNamespace(cp_rank=rank), tokens)
        grads.append(model.logits.grad)
    torch.testing.assert_close((grads[0] + grads[1]) / 2, expected)
    assert expected[0, 31].count_nonzero() and not expected[0, 63].count_nonzero()
