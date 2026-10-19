"""CPU contract for the experimental TPR full/uniform/1 checkpoint adapter.

Uses Megatron-shaped fake TransformerBlocks and a reentrant PyTorch checkpoint
to exercise multi-output KV roots, replay-local ContextVars, and prefix anchors.
The real MindSpeed tensor_parallel checkpoint + Ring tests remain NPU gates.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.transformer_block import TransformerBlock

from verl.workers.engine.megatron.tpr import recompute
from verl.models.mcore.tpr.context import (
    TPRAttentionContext,
    get_tpr_attention_context,
    use_tpr_attention_context,
)


class _FakeAttention(nn.Module):
    pass


class _FakeLayer(nn.Module):
    def __init__(self, layer_number: int):
        super().__init__()
        self.layer_number = layer_number
        self.self_attention = _FakeAttention()
        self.scale = nn.Parameter(torch.tensor(0.7 + layer_number * 0.1, device="cpu"))
        self.observed_contexts = []

    def forward(self, *, hidden_states, **kwargs):
        ctx = get_tpr_attention_context()
        assert ctx is not None, "lost TPR context on checkpoint replay"
        self.observed_contexts.append(ctx)
        pk, pv = ctx.past_key_values[self.layer_number]
        key = (hidden_states * self.scale).unsqueeze(2)
        value = (hidden_states * (self.scale + 0.2)).unsqueeze(2)
        ctx.set_new_kv(self.layer_number, key, value)
        output = hidden_states * (self.scale + 1.0) + 0.1 * (
            pk.squeeze(2) + pv.squeeze(2)
        )
        return output, None


class _FakeBlock(TransformerBlock):
    """Avoid constructing a real parallel model; keep native entry signature."""

    def __init__(self, layer):
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([layer])
        self.config = type("Config", (), {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "distribute_saved_activations": False,
            "fp8": None,
            "fp4": None,
            "cpu_offloading": False,
            # TransformerModule.__call__ checks this even for mocked Blocks.
            "cuda_graph_impl": "none",
        })()

    def _checkpointed_forward(
        self, hidden_states, attention_mask, context, context_mask,
        rotary_pos_emb, attention_bias, packed_seq_params,
        use_inner_quantization_context, padding_mask=None,
    ):
        out, _ = self.layers[0](hidden_states=hidden_states)
        return out

    def forward(self, hidden):
        return self._checkpointed_forward(
            hidden, None, None, None, None, None, None, False,
        )


def _run(*, adapted: bool, monkeypatch):
    monkeypatch.setattr(recompute, "TPRSelfAttention", _FakeAttention)
    monkeypatch.setattr(
        recompute.tensor_parallel,
        "checkpoint",
        lambda fn, _distributed, *a: checkpoint(fn, *a, use_reentrant=True),
    )
    blocks = nn.ModuleList([_FakeBlock(_FakeLayer(i)) for i in (1, 2)])
    if adapted:
        assert recompute.install_tpr_full_recompute(blocks, blocks[0].config) == 2
        recompute.assert_tpr_full_recompute_installed(blocks, blocks[0].config)

    prefix = torch.randn(4, 1, 1, 8, requires_grad=True, device="cpu")
    ctx = TPRAttentionContext(
        prefix_length=4,
        suffix_length=4,
        past_key_values={i: (prefix, prefix) for i in (1, 2)},
        suffix_rotary_pos_emb=torch.zeros(4, 1, 1, 8, device="cpu"),
    )
    x = torch.randn(4, 1, 8, requires_grad=True, device="cpu")
    with use_tpr_attention_context(ctx):
        h = x
        for b in blocks:
            h = b(h)

    assert set(ctx.new_key_values) == {1, 2}
    roots = [h.sum()]
    grads = [None]
    for k, v in ctx.new_key_values.values():
        roots.extend((k, v))
        grads.extend((torch.ones_like(k) * 0.25, torch.ones_like(v) * 0.15))
    torch.autograd.backward(roots, grad_tensors=grads)
    assert prefix.grad is not None
    assert x.grad is not None
    for block in blocks:
        assert block.layers[0].scale.grad is not None
        observed = block.layers[0].observed_contexts
        assert len(observed) == (2 if adapted else 1)
        assert observed[0] is not observed[-1] if adapted else observed[0] is ctx
    return [
        h.detach(), prefix.grad.detach(), x.grad.detach(),
        *(b.layers[0].scale.grad.detach() for b in blocks),
    ]


def test_full_uniform1_kv_side_output_and_context_replay(monkeypatch):
    torch.manual_seed(333)
    baseline = _run(adapted=False, monkeypatch=monkeypatch)
    torch.manual_seed(333)
    checked = _run(adapted=True, monkeypatch=monkeypatch)
    for a, b in zip(baseline, checked):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_push_under_no_grad_exports_graph_free_kv(monkeypatch):
    monkeypatch.setattr(recompute, "TPRSelfAttention", _FakeAttention)
    block = _FakeBlock(_FakeLayer(1))
    recompute.install_tpr_full_recompute(block, block.config)
    prefix = torch.zeros(4, 1, 1, 8, device="cpu")
    ctx = TPRAttentionContext(
        prefix_length=4, suffix_length=4,
        past_key_values={1: (prefix, prefix)},
        suffix_rotary_pos_emb=torch.zeros(4, 1, 1, 8, device="cpu"),
    )
    with torch.no_grad(), use_tpr_attention_context(ctx):
        block(torch.ones(4, 1, 8, device="cpu"))
    assert set(ctx.new_key_values) == {1}
    assert not ctx.new_key_values[1][0].requires_grad


def test_unsupported_recompute_is_rejected():
    class Config:
        recompute_granularity = "full"
        recompute_method = "uniform"
        recompute_num_layers = 2

    with pytest.raises(NotImplementedError, match="full/uniform/1"):
        recompute.validate_tpr_recompute_config(Config())


def test_checkpoint_run_function_does_not_capture_outer_context(monkeypatch):
    """Prevent ctx -> run_function -> outer -> checkpoint KV -> ctx cycles."""
    monkeypatch.setattr(recompute, "TPRSelfAttention", _FakeAttention)
    checkpoint_functions = []

    def record_checkpoint(fn, _distributed, *args):
        checkpoint_functions.append(fn)
        return checkpoint(fn, *args, use_reentrant=True)

    monkeypatch.setattr(recompute.tensor_parallel, "checkpoint", record_checkpoint)
    block = _FakeBlock(_FakeLayer(1))
    assert recompute.install_tpr_full_recompute(block, block.config) == 1

    prefix = torch.randn(4, 1, 1, 8, device="cpu", requires_grad=True)
    context = TPRAttentionContext(
        prefix_length=4,
        suffix_length=4,
        past_key_values={1: (prefix, prefix)},
        suffix_rotary_pos_emb=torch.zeros(4, 1, 1, 8, device="cpu"),
    )
    x = torch.randn(4, 1, 8, device="cpu", requires_grad=True)
    with use_tpr_attention_context(context):
        output = block(x)
    assert checkpoint_functions
    for fn in checkpoint_functions:
        # Megatron's CheckpointFunction holds fn until backward. It must not
        # hold the outer context strongly because it owns returned KV tensors.
        assert "outer" not in fn.__code__.co_freevars
        assert "outer_ref" in fn.__code__.co_freevars
    roots = [output.sum()]
    grad_tensors = [None]
    for k, v in context.new_key_values.values():
        roots.extend((k, v))
        grad_tensors.extend((torch.ones_like(k), torch.ones_like(v)))
    torch.autograd.backward(roots, grad_tensors=grad_tensors)
    assert x.grad is not None and prefix.grad is not None
