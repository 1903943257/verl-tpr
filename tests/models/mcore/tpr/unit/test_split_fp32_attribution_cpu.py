# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Verify the three-way recorder with real production SegmentExecutor on CPU."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.models.mcore.tpr.parallel.split_fp32_attribution import (
    _loss,
    math_attention,
    run_three_way,
)
from verl.models.mcore.tpr.context import get_tpr_attention_context


class _Rotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.01]))

    def forward(self, length, offset=0, **kwargs):
        del kwargs
        phase = (torch.arange(length).float() + offset)[:, None] * self.inv_freq[None, :]
        return phase.repeat_interleave(2, dim=-1)[:, None, None, :]


def _rope(tensor, phase):
    pairs = tensor.reshape(*tensor.shape[:-1], 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    cosine, sine = phase[..., ::2].cos(), phase[..., ::2].sin()
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1).flatten(-2)


class _Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(32, 16)

    def forward(self, ids):
        return self.table(ids).transpose(0, 1)


class _Attention(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.layer_number = number
        self.linear_qkv = nn.Linear(16, 32, bias=False)
        self.q_layernorm = nn.LayerNorm(4)
        self.k_layernorm = nn.LayerNorm(4)
        self.linear_proj = nn.Linear(16, 16, bias=False)

    def forward(self, hidden, phase):
        q, k, v = self.linear_qkv(hidden).split((16, 8, 8), dim=-1)
        q = _rope(self.q_layernorm(q.reshape(-1, 1, 4, 4)), phase)
        k = _rope(self.k_layernorm(k.reshape(-1, 1, 2, 4)), phase)
        v = v.reshape(-1, 1, 2, 4)
        context = get_tpr_attention_context()
        if context is not None:
            context.set_new_kv(self.layer_number, k, v)
            past = context.get_past_kv(self.layer_number)
            if past is not None:
                k, v = torch.cat((past[0], k)), torch.cat((past[1], v))
        return self.linear_proj(self.attend(q, k, v))


class _Layer(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.layer_number = number
        self.input_layernorm = nn.LayerNorm(16)
        self.self_attention = _Attention(number)
        self.pre_mlp_layernorm = nn.LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 24), nn.SiLU(), nn.Linear(24, 16))

    def forward(self, hidden_states, phase):
        hidden = hidden_states + self.self_attention(self.input_layernorm(hidden_states), phase)
        return hidden + self.mlp(self.pre_mlp_layernorm(hidden))


class TinyFP32Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Embedding()
        self.rotary_pos_emb = _Rotary()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_Layer(1), _Layer(2)])
        self.decoder.final_layernorm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 32, bias=False)
        self.config = SimpleNamespace(tensor_model_parallel_size=1, sequence_parallel=False)
        self.math_attention = math_attention

    def forward(self, *, input_ids, position_ids, attention_mask):
        del attention_mask
        phase = self.rotary_pos_emb(input_ids.shape[1], offset=int(position_ids[0, 0]))
        hidden = self.embedding(input_ids)
        for layer in self.decoder.layers:
            layer.self_attention.attend = self.math_attention
            hidden = layer(hidden, phase)
        return self.head(self.decoder.final_layernorm(hidden)).transpose(0, 1)


@pytest.mark.parametrize("prefix,suffix", [(1, 2), (3, 5), (8, 8)])
def test_three_way_matches_outputs_and_full_parameter_gradients(tmp_path, prefix, suffix):
    torch.manual_seed(2604)
    model = TinyFP32Model()
    tokens = torch.randint(0, 32, (prefix + suffix,))
    report = run_three_way(model, tokens, prefix, tmp_path / "run", atol=2e-6, rtol=2e-4)
    for pair in report["pairs"].values():
        assert pair["all_parameters_within_tolerance"], pair
        assert pair["gradient_relative_l2"] < 1e-5
        assert pair["loss_abs_gap"] < 2e-6
    assert all(summary["forward"] is None for summary in report["layer_summary"].values())
    assert all(summary["gradient"] is None for summary in report["layer_summary"].values())


def test_rectangular_math_mask_uses_global_query_positions():
    q = torch.zeros(2, 1, 2, 2)
    k = torch.zeros(5, 1, 1, 2)
    v = torch.arange(5).float()[:, None, None, None].expand(5, 1, 1, 2)
    output = math_attention(q, k, v)
    assert torch.equal(output[:, 0], torch.tensor([[1.5] * 4, [2.0] * 4]))


def test_loss_includes_prefix_last_query_and_excludes_only_sequence_last():
    logits = torch.zeros(1, 8, 32, requires_grad=True)
    _loss(logits, 0, 8).backward()
    assert logits.grad[:, 2].abs().sum() > 0  # P=3 boundary query owns next token.
    assert logits.grad[:, -1].count_nonzero() == 0
    with pytest.raises(TypeError, match="non-FP32"):
        math_attention(torch.zeros(1, 1, 1, 2).bfloat16(), torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))


def test_probe_exposes_replay_gradient_fault_while_forward_matches(tmp_path, monkeypatch):
    from verl.workers.engine.megatron.tpr.state import KVPrefixState

    original = KVPrefixState.accumulate_anchor_gradients

    def corrupt(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for pair in self.gradients.values():
            for gradient in pair:
                gradient.mul_(1.1)

    monkeypatch.setattr(KVPrefixState, "accumulate_anchor_gradients", corrupt)
    torch.manual_seed(33)
    report = run_three_way(TinyFP32Model(), torch.randint(0, 32, (8,)), 3, tmp_path / "fault")
    assert report["pairs"]["full_vs_split_connected"]["all_parameters_within_tolerance"]
    assert not report["pairs"]["full_vs_split_replay"]["all_parameters_within_tolerance"]
    assert report["layer_summary"]["split_replay"]["forward"] is None
    assert report["layer_summary"]["split_replay"]["gradient"] is not None
