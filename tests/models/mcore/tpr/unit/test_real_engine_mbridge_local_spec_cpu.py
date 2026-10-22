"""CPU-only checks: replay MBridge Local Spec must not alter global MCore state."""
from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from ..parallel._real_engine_mbridge_local_spec import (
    install_replay_local_spec,
    local_qwen3_norm_hf_names,
)


def get_gpt_decoder_layer_specs(config, use_transformer_engine, vp_stage=None):
    return config, use_transformer_engine, vp_stage


def original_provider(self, vp_stage=None):
    return get_gpt_decoder_layer_specs(
        self.config, use_transformer_engine=True, vp_stage=vp_stage
    )


original_provider.__module__ = "mbridge.core.llm_bridge"


class FakeBridge:
    _get_transformer_layer_spec = original_provider

    def __init__(self):
        self.config = SimpleNamespace(normalization="RMSNorm", num_moe_experts=None)
        self.has_vp_stage = False

    def _weight_name_mapping_mcore_to_hf(self, name):
        return [f"original:{name}"]


def test_replay_local_spec_instance_only():
    first, other = FakeBridge(), FakeBridge()
    assert first._get_transformer_layer_spec(2)[1:] == (True, 2)

    install_replay_local_spec(first)

    assert first._get_transformer_layer_spec(3)[1:] == (False, 3)
    assert first.has_vp_stage is True
    assert other._get_transformer_layer_spec(3)[1:] == (True, 3)
    assert first._weight_name_mapping_mcore_to_hf(
        "decoder.layers.11.input_layernorm.weight"
    ) == ["model.layers.11.input_layernorm.weight"]
    assert first._weight_name_mapping_mcore_to_hf(
        "decoder.layers.11.pre_mlp_layernorm.weight"
    ) == ["model.layers.11.post_attention_layernorm.weight"]
    assert first._weight_name_mapping_mcore_to_hf(
        "decoder.layers.11.self_attention.linear_qkv.weight"
    ) == ["original:decoder.layers.11.self_attention.linear_qkv.weight"]

    # Idempotence: TPR wraps the same instance after Local Spec selection.
    install_replay_local_spec(first)
    assert first._get_transformer_layer_spec(1)[1:] == (False, 1)


def test_mapping_only_strict_local_norm_names():
    assert local_qwen3_norm_hf_names(
        "decoder.layers.0.input_layernorm.weight"
    ) == ["model.layers.0.input_layernorm.weight"]
    assert local_qwen3_norm_hf_names(
        "decoder.layers.35.pre_mlp_layernorm.weight"
    ) == ["model.layers.35.post_attention_layernorm.weight"]
    assert local_qwen3_norm_hf_names(
        "decoder.layers.3.self_attention.linear_qkv.layer_norm_weight"
    ) is None
    assert local_qwen3_norm_hf_names("encoder.layers.0.input_layernorm.weight") is None


def test_reject_custom_layer_spec_without_patching():
    obj = FakeBridge()

    def custom_provider(self, vp_stage=None):
        return None

    obj._get_transformer_layer_spec = MethodType(custom_provider, obj)
    with pytest.raises(RuntimeError, match="Unexpected MBridge layer-spec"):
        install_replay_local_spec(obj)
    assert "_real_engine_local_spec_installed" not in obj.__dict__


def test_reject_non_dense():
    obj = FakeBridge()
    obj.config.num_moe_experts = 8
    with pytest.raises(RuntimeError, match="dense Qwen3"):
        install_replay_local_spec(obj)
