"""Instance-local, test-only Ascend MBridge Local Spec for real Engine PPO replay.

Vanilla mbridge's generic LLMBridge hardcodes use_transformer_engine=True.
The installed Megatron Core on Ascend has no TENorm and therefore cannot
instantiate TE specs. This helper targets ONLY the actual vanilla Qwen3
bridge instance in the real Engine replay, before VERL's TPR spec wrapper.

IMPORTANT: Local Spec separates input/post-attention RMSNorm from the
fused TE ColumnParallelLinear. Vanilla MBridge's Qwen2/Qwen3 mapping assumes
the fused names; remap those two norm keys or the loader may silently
skip them, invalidating Full/TPR numerical equivalence.

No edits to installed MBridge/Megatron, no global monkeypatches, and no
changes to production VERL/TPR. Unknown installed MBridge API fails closed.
"""
from __future__ import annotations

import inspect
import re
from types import MethodType


_SEPARATE_NORM_RE = re.compile(
    r"^decoder\.layers\.(\d+)\.(input_layernorm|pre_mlp_layernorm)\.weight$"
)


def local_qwen3_norm_hf_names(mcore_name: str) -> list[str] | None:
    """Translate *separate* Local Spec norm params to original HF Qwen3 keys."""
    match = _SEPARATE_NORM_RE.fullmatch(mcore_name)
    if match is None:
        return None
    layer_id, kind = match.groups()
    hf_component = (
        "input_layernorm" if kind == "input_layernorm"
        else "post_attention_layernorm"
    )
    return [f"model.layers.{layer_id}.{hf_component}.weight"]


def install_replay_local_spec(bridge) -> None:
    """Use Megatron Local Block Spec on exactly one vanilla MBridge instance.

    The original method in mbridge/core/llm_bridge.py calls either
    get_gpt_decoder_layer_specs or get_gpt_decoder_block_spec with
    use_transformer_engine=True, depending on installed MBridge/MCore revision.
    Discover the *actual* function called, preserve vp_stage support, and
    pass only use_transformer_engine=False. Do not replace that module's globals.
    """
    if getattr(bridge, "_real_engine_local_spec_installed", False):
        return
    original = getattr(bridge, "_get_transformer_layer_spec", None)
    if not callable(original):
        raise TypeError("Replay expects vanilla MBridge's _get_transformer_layer_spec")
    underlying = getattr(original, "__func__", original)
    if not getattr(underlying, "__module__", "").startswith("mbridge.core.llm_bridge"):
        raise RuntimeError(
            "Unexpected MBridge layer-spec implementation "
            f"{getattr(underlying, '__module__', 'unknown')}; "
            "do not bypass model-specific special specs"
        )
    candidates = (
        "get_gpt_decoder_layer_specs",
        "get_gpt_decoder_block_spec",
    )
    called = [name for name in candidates if name in underlying.__code__.co_names]
    if len(called) != 1:
        raise RuntimeError(f"Unknown vanilla MBridge spec callable list: {called}")
    name = called[0]
    spec_fn = underlying.__globals__.get(name)
    if not callable(spec_fn):
        raise RuntimeError(f"Installed MBridge has no callable {name}")
    if "use_transformer_engine" not in inspect.signature(spec_fn).parameters:
        raise RuntimeError(f"{name} cannot explicitly select Local Spec")
    if getattr(bridge.config, "normalization", None) != "RMSNorm":
        raise RuntimeError("Replay Local Spec is limited to Qwen3 RMSNorm")
    if getattr(bridge.config, "num_moe_experts", None) not in (None, 0):
        raise RuntimeError("Replay Local Spec is limited to dense Qwen3")

    signature = inspect.signature(spec_fn)
    has_vp_stage = "vp_stage" in signature.parameters

    def local_spec(self, vp_stage=None):
        self.has_vp_stage = has_vp_stage
        kwargs = {"vp_stage": vp_stage} if has_vp_stage else {}
        return spec_fn(self.config, use_transformer_engine=False, **kwargs)

    original_map = getattr(bridge, "_weight_name_mapping_mcore_to_hf", None)
    if not callable(original_map):
        raise TypeError("Replay MBridge lacks original HF weight mapping")

    def local_weight_mapping(self, name):
        replacement = local_qwen3_norm_hf_names(name)
        if replacement is not None:
            return replacement
        return original_map(name)

    bridge._get_transformer_layer_spec = MethodType(local_spec, bridge)
    bridge._weight_name_mapping_mcore_to_hf = MethodType(local_weight_mapping, bridge)
    bridge._real_engine_local_spec_installed = True
    print(
        "REAL_ENGINE_MBRIDGE_LOCAL_SPEC "
        f"bridge={type(bridge).__name__} original_spec={name} "
        "use_transformer_engine=False separate_rmsnorm_mapping=True",
        flush=True,
    )


def assert_replay_local_qwen3_norms(model, bridge, *, num_layers: int = 36) -> None:
    """Reject silent norm drops after the real model has loaded HF weights."""
    from verl.utils.megatron_utils import unwrap_model

    base = unwrap_model(model)
    params = dict(base.named_parameters())
    expected = {}
    for layer in range(num_layers):
        for kind in ("input_layernorm", "pre_mlp_layernorm"):
            name = f"decoder.layers.{layer}.{kind}.weight"
            expected[name] = local_qwen3_norm_hf_names(name)
    missing = sorted(set(expected) - params.keys())
    if missing:
        raise AssertionError(
            f"Local Qwen3 RMSNorm structure mismatch: expected {2*num_layers}, "
            f"missing {len(missing)}: {missing[:5]}"
        )
    if not hasattr(bridge, "safetensor_io"):
        raise AssertionError("Replay MBridge must have loaded HF checkpoint weights")
    io = bridge.safetensor_io
    available = set(io.load_hf_weight_names())
    for name, hf_names in expected.items():
        if bridge._weight_name_mapping_mcore_to_hf(name) != hf_names:
            raise AssertionError(f"Wrong HF norm mapping for {name}")
        hf_key = hf_names[0]
        if hf_key not in available:
            raise AssertionError(f"Missing required Qwen3 norm checkpoint key: {hf_key}")
        # Exact BF16/FP16 comparison after the same dtype cast as MBridge's
        # loader; checking only key names is insufficient because MBridge may
        # silently skip a parameter and leave random init behind.
        actual = params[name].detach().cpu()
        checkpoint = io.load_one_hf_weight(hf_key).detach().cpu().to(actual.dtype)
        if actual.shape != checkpoint.shape or not actual.equal(checkpoint):
            raise AssertionError(
                f"HF Qwen3 norm weight was not loaded exactly: "
                f"local={name} hf={hf_key} "
                f"local_shape={tuple(actual.shape)} hf_shape={tuple(checkpoint.shape)}"
            )
    print(
        f"REAL_ENGINE_QWEN3_LOCAL_NORMS_PASS separate_norms={len(expected)} "
        f"hf_mappings_checked=True actual_hf_values_verified={len(expected)}",
        flush=True,
    )
