# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Precision-boundary, first-divergence, and CP2 global gradient metric gates."""

import json
import math

import pytest
import torch

from tests.models.mcore.tpr.parallel import split_bf16_quick as quick
from tests.models.mcore.tpr.parallel.split_attention_precision import (
    ATTENTION_MODES,
    math_bf16_attention,
    math_fp32_attention,
)
from tests.models.mcore.tpr.parallel.test_split_p32s32_cp2_parity_npu import (
    _compare_current_grads,
    _compare_loss_logprobs,
    _objective,
    _remove_reference_gradients,
    _save_gradients,
)
from tests.models.mcore.tpr.unit.test_split_bf16_quick_cpu import _model
from verl.models.mcore.tpr.context import get_tpr_attention_context
from verl.workers.engine.megatron.tpr.ring import make_ring_sequence_shard


@pytest.mark.parametrize("implementation", [math_bf16_attention, math_fp32_attention])
def test_attention_ablation_preserves_bf16_boundary_gqa_and_rectangular_causality(implementation):
    q = torch.zeros(2, 1, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(5, 1, 1, 2, dtype=torch.bfloat16, requires_grad=True)
    v = torch.arange(5).bfloat16()[:, None, None, None].expand(5, 1, 1, 2).clone().requires_grad_(True)
    output = implementation(q, k, v)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output[:, 0].float(), torch.tensor([[1.5] * 4, [2.0] * 4]))
    output.float().sum().backward()
    assert all(t.grad.dtype == torch.bfloat16 and torch.isfinite(t.grad).all() for t in (q, k, v))


@pytest.mark.parametrize("prefix,suffix", [(3, 5), (32, 32)])
def test_three_attention_modes_reuse_weights_and_export_only_forward_statistics(tmp_path, monkeypatch, prefix, suffix):
    model = _model()
    tokens = torch.arange(prefix + suffix) % 32
    versions = {name: p._version for name, p in model.named_parameters()}
    baselines = {}

    def reject_dump(*args, **kwargs):
        raise AssertionError("Precision diagnostic must not dump raw tensors")

    monkeypatch.setattr(torch, "save", reject_dump)
    for mode in ATTENTION_MODES:
        report = quick.run_bf16_quick(
            model,
            tokens,
            prefix,
            tmp_path / mode,
            production_attention=False,
            attention_mode=mode,
            trace_forward=True,
            forward_baselines=baselines,
        )
        assert report["forward_summary"]["split_connected"]["records"] > 20
        assert all(t["gradient"] is None and not t["expect_gradient"] for t in baselines[mode])
        assert all(not t["forward"].requires_grad for t in baselines[mode])
        assert report["arms"]["full"]["next_token_cross_entropy"] > 0
        assert versions == {name: p._version for name, p in model.named_parameters()}
        assert all(file.suffix == ".json" for file in (tmp_path / mode).iterdir())
    # The CPU CANN stand-in is explicitly FP32 math + BF16 boundary.
    same = report["cross_attention_forward"]["full_cann_vs_math_fp32"]
    assert same["first_nonzero"] is None
    different = report["cross_attention_forward"]["full_math_bf16_vs_math_fp32"]
    assert different["first_nonzero"]["stage"].endswith(".attention_core")
    with torch.no_grad():
        logits = model(input_ids=tokens[None], position_ids=torch.arange(tokens.numel())[None], attention_mask=None)
        expected = -logits[:, :-1].float().log_softmax(-1).gather(-1, tokens[None, 1:, None]).double().mean()
    assert math.isclose(report["arms"]["full"]["next_token_cross_entropy"], float(expected), rel_tol=1e-12)


def test_forward_locator_identifies_injected_qkv_fault(tmp_path):
    model = _model()

    def corrupt(_module, _args, output):
        return output + 0.125 if get_tpr_attention_context() is not None else output

    handle = model.decoder.layers[0].self_attention.linear_qkv.register_forward_hook(corrupt)
    try:
        report = quick.run_bf16_quick(
            model,
            torch.arange(8),
            3,
            tmp_path / "fault",
            production_attention=False,
            trace_forward=True,
        )
    finally:
        handle.remove()
    assert report["forward_summary"]["split_connected"]["first_nonzero"]["stage"] == "layer.01.qkv"
    records = json.loads((tmp_path / "fault/full_vs_split_replay_layers.json").read_text())
    assert any(not row["grad_enabled"] for row in records)  # graph-free Push remains separately visible.
    assert not any("gradient" in row for row in records)


def test_cp2_streaming_gradient_metrics_match_flat_fp64_and_clean_only_manifest_files(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1)).bfloat16()
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = (torch.arange(parameter.numel()).reshape(parameter.shape).float() * 0.1 + index + 1).bfloat16()
    reference = torch.cat([p.grad.flatten().double() for p in model.parameters()])
    folder = tmp_path / "grads"
    _save_gradients(model, folder)
    for parameter in model.parameters():
        parameter.grad.mul_(1.03125)
    candidate = torch.cat([p.grad.flatten().double() for p in model.parameters()])
    metric = _compare_current_grads(model, folder)
    assert math.isclose(
        metric["gradient_relative_l2"], float((candidate - reference).norm() / reference.norm()), rel_tol=1e-12
    )
    assert metric["gradient_max_abs"] == float((candidate - reference).abs().max())
    result = tmp_path / "result.json"
    result.write_text("{}")
    _remove_reference_gradients(folder)
    assert not folder.exists() and result.is_file()
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps([{"file": "../result.json"}]))
    with pytest.raises(ValueError, match="cleanup"):
        _remove_reference_gradients(folder)
    assert result.is_file()


def test_cp2_full_and_split_shards_preserve_global_loss_gradient_and_boundary():
    torch.manual_seed(23)
    logits = torch.randn(1, 64, 17, requires_grad=True)
    tokens = torch.randint(0, 17, (64,))
    loss, _, _ = _objective(logits, torch.arange(64), tokens)
    expected = torch.autograd.grad(loss, logits)[0]
    for segments in (((0, 64),), ((0, 32), (32, 32))):
        split = logits.detach().clone().requires_grad_(True)
        rank_losses = []
        for rank in (0, 1):
            pieces = []
            for start, length in segments:
                positions = (
                    make_ring_sequence_shard(length, cp_rank=rank, cp_size=2).global_indices(device="cpu") + start
                )
                local, _, _ = _objective(split.index_select(1, positions), positions, tokens)
                pieces.append(local * 2)  # executor backward scaling
            rank_losses.append(sum(pieces))
        (sum(rank_losses) / 2).backward()  # CP weight gradient averaging
        torch.testing.assert_close(split.grad, expected)
        assert split.grad[0, 31].count_nonzero() and not split.grad[0, 63].count_nonzero()


def test_cp2_token_metrics_and_precision_controls_are_matched():
    baseline = {
        "checkpoint": "/model",
        "input_sha256": "input",
        "initial_weights_witness": "weights",
        "tokens": list(range(64)),
        "profile": "4b",
        "delayed_dkv_fp32": True,
        "logprobs": [-3.0] * 63,
        "loss": 0.0,
    }
    candidate = {**baseline, "logprobs": [-3.01] * 63}
    assert math.isclose(_compare_loss_logprobs(baseline, candidate)["cross_entropy_abs_gap"], 0.01, abs_tol=1e-14)
    candidate["delayed_dkv_fp32"] = False
    with pytest.raises(AssertionError, match="precision controls"):
        _compare_loss_logprobs(baseline, candidate)
