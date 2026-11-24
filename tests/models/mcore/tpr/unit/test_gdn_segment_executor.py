"""CPU integration contract for Hybrid FA KV + GDN state in the production DFS executor."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from verl.experimental.tpr.gdn.context import get_tpr_attention_context
from verl.experimental.tpr.gdn.prefix_state import GDNLayerState
from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec


class _FakeFullAttention(nn.Module):
    layer_number = 1

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        context = get_tpr_attention_context()
        key = hidden.unsqueeze(2)
        value = (hidden * 0.5).unsqueeze(2)
        context.set_new_kv(self.layer_number, key, value)
        past = context.get_past_kv(self.layer_number)
        if past is not None:
            hidden = hidden + 0.01 * (past[0].mean() + past[1].mean())
        return self.scale * hidden


class _FakeGDN(nn.Module):
    tpr_state_kind = "gdn"
    layer_number = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        context = get_tpr_attention_context()
        initial = context.get_initial_gdn_state(self.layer_number)
        if initial is None:
            conv = hidden.new_zeros((1, 1, 2))
            recurrent = hidden.new_zeros((1, 1, 2, 2))
        else:
            conv = initial.conv_state
            recurrent = initial.recurrent_state

        signal = hidden.mean()
        final_conv = conv + (0.1 * signal).expand_as(conv)
        final_recurrent = recurrent + (0.05 * signal).expand_as(recurrent)
        context.set_new_gdn_state(
            self.layer_number,
            GDNLayerState(final_conv, final_recurrent),
        )
        state_signal = 0.01 * (conv.mean() + recurrent.mean())
        return self.scale * hidden + state_signal


class _Layer(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.self_attention = attention


class _ToyHybridModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tensor_model_parallel_size=1, sequence_parallel=False)
        self.embedding = nn.Embedding(32, 4)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(
            [_Layer(_FakeFullAttention()), _Layer(_FakeGDN())]
        )
        self.lm_head = nn.Linear(4, 16, bias=False)

    def rotary_pos_emb(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        return torch.zeros((max_seq_len, 1, 1, 2), device=self.embedding.weight.device)

    def forward(self, *, input_ids, position_ids, attention_mask):
        hidden = self.embedding(input_ids).transpose(0, 1)
        for layer in self.decoder.layers:
            hidden = hidden + layer.self_attention(hidden)
        return self.lm_head(hidden).transpose(0, 1)


class _RelayProbeExecutor(SegmentExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_state_gradients = None

    def pop(self, segment_id):
        state = self.gdn_states[segment_id]
        self.root_state_gradients = {
            layer: GDNLayerState(
                gradient.conv_state.detach().clone(),
                gradient.recurrent_state.detach().clone(),
            )
            for layer, gradient in state.gradients.items()
        }
        return super().pop(segment_id)


def _plan() -> SegmentPlan:
    return SegmentPlan(
        [
            SegmentSpec(0, None, torch.tensor([1, 2, 3, 4]), 0, 0),
            SegmentSpec(1, 0, torch.tensor([5, 6, 7, 8]), 4, 4),
            SegmentSpec(2, 0, torch.tensor([9, 10, 11, 12]), 4, 4),
        ],
        root_id=0,
    )


def test_hybrid_dfs_relays_gdn_state_gradients_and_releases_state() -> None:
    torch.manual_seed(731)
    model = _ToyHybridModel()
    executor = _RelayProbeExecutor(
        model,
        _plan(),
        segment_loss_fn=lambda _segment, logits: (logits.float().square().mean(), {}),
        segment_loss_term_counts={0: 1, 1: 1, 2: 1},
    )

    loss = executor.run()

    assert torch.isfinite(loss)
    assert executor.root_state_gradients is not None
    assert tuple(executor.root_state_gradients) == (2,)
    relayed = executor.root_state_gradients[2]
    assert torch.isfinite(relayed.conv_state).all()
    assert torch.isfinite(relayed.recurrent_state).all()
    assert torch.count_nonzero(relayed.conv_state).item() > 0
    assert torch.count_nonzero(relayed.recurrent_state).item() > 0
    assert executor.kv_stack.segment_ids == ()
    assert not executor.gdn_states
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
