"""Opt-in contract for the ACTUAL Megatron tensor_parallel.checkpoint on NPU.

The existing test_tpr_native_recompute_contract_npu.py documents why a naive
config-only integration fails.  This checks the proposed explicit-KV-output
protocol, not a full Qwen3+Ring end-to-end acceptance.
"""

import os

import pytest
import torch
from megatron.core import tensor_parallel

from verl.models.mcore.tpr.context import (
    TPRAttentionContext,
    use_tpr_attention_context,
    get_tpr_attention_context,
)
from ..profiling.test_tpr_qwen3_ring_cp_profile_npu import profile_runtime

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_KV_CHECKPOINT_REPLAY") != "1",
    reason="Set TPR_RUN_KV_CHECKPOINT_REPLAY=1 for the native NPU checkpoint gate",
)


def test_native_checkpoint_explicit_kv_gradients_and_replay_context(profile_runtime):
    device = profile_runtime.device
    hidden = torch.randn(4, 1, 8, device=device, requires_grad=True)
    prefix = torch.randn(4, 1, 1, 8, device=device, requires_grad=True)
    outer = TPRAttentionContext(
        prefix_length=4,
        suffix_length=4,
        past_key_values={1: (prefix, prefix)},
        suffix_rotary_pos_emb=torch.zeros(4, 1, 1, 8, device=device),
    )
    observed = []

    def layer(x):
        local = outer.new_collector()
        with use_tpr_attention_context(local):
            assert get_tpr_attention_context() is local
            observed.append((torch.is_grad_enabled(), local))
            key = (x * 2).unsqueeze(2)
            value = (x * 3).unsqueeze(2)
            local.set_new_kv(1, key, value)
            return x * 4 + prefix.squeeze(2), key, value

    with use_tpr_attention_context(outer):
        output, key, value = tensor_parallel.checkpoint(layer, False, hidden)
    outer.set_new_kv(1, key, value)
    assert key.requires_grad and value.requires_grad
    torch.autograd.backward(
        [output.sum(), key, value],
        grad_tensors=[None, torch.ones_like(key), torch.ones_like(value)],
    )
    assert [enabled for enabled, _ in observed] == [False, True]
    assert observed[0][1] is not observed[1][1]
    assert set(outer.new_key_values) == {1}
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 9.0))
    torch.testing.assert_close(prefix.grad, torch.ones_like(prefix))
