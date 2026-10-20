# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""The optional experiment owns all recurrent state, not the Dense runtime."""

import pytest
import torch

from verl.experimental.tpr.gdn.context import GDNTPRAttentionContext
from verl.experimental.tpr.gdn.prefix_state import GDNLayerState
from verl.models.mcore.tpr.context import (
    TPRAttentionContext,
    get_tpr_attention_context,
    use_tpr_attention_context,
)


def _state(*, requires_grad=False):
    return GDNLayerState(
        torch.randn(1, 12, 4, requires_grad=requires_grad),
        torch.randn(1, 4, 8, 6, requires_grad=requires_grad),
    )


def test_experimental_state_preserves_identity_and_autograd():
    initial = _state(requires_grad=True)
    context = GDNTPRAttentionContext(6, 3, initial_gdn_states={1: initial})
    assert context.get_initial_gdn_state(1) is initial
    assert context.get_initial_gdn_state(1).conv_state.requires_grad
    new = _state(requires_grad=True)
    context.set_new_gdn_state(1, new)
    assert context.new_gdn_states[1] is new
    context.assert_new_gdn_layers([1])
    with pytest.raises(TypeError):
        context.new_gdn_states[2] = new


def test_experimental_state_root_missing_duplicate_and_type_contracts():
    with pytest.raises(ValueError, match="initial_gdn_states"):
        GDNTPRAttentionContext(0, 3, initial_gdn_states={1: _state()})
    assert GDNTPRAttentionContext(0, 3).get_initial_gdn_state(1) is None
    context = GDNTPRAttentionContext(6, 3, initial_gdn_states={1: _state()})
    with pytest.raises(KeyError, match="layer 2"):
        context.get_initial_gdn_state(2)
    context.set_new_gdn_state(1, _state())
    with pytest.raises(RuntimeError, match="already recorded"):
        context.set_new_gdn_state(1, _state())
    with pytest.raises(TypeError, match="GDNLayerState"):
        context.set_new_gdn_state(2, object())
    with pytest.raises(TypeError, match="capture_gdn_final_state"):
        GDNTPRAttentionContext(0, 3, capture_gdn_final_state="yes")


def test_experimental_context_shares_dense_scope_and_restores_on_failure():
    dense = TPRAttentionContext(0, 3)
    experimental = GDNTPRAttentionContext(0, 3)
    with use_tpr_attention_context(dense):
        with pytest.raises(RuntimeError):
            with use_tpr_attention_context(experimental):
                assert get_tpr_attention_context() is experimental
                key, value = (torch.randn(3, 1, 2, 8) for _ in range(2))
                experimental.set_new_kv(2, key, value)
                assert experimental.new_key_values[2] == (key, value)
                raise RuntimeError("failure")
        assert get_tpr_attention_context() is dense
    assert get_tpr_attention_context() is None
