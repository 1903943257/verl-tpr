"""Shared NPU helpers for the migrated Qwen3.5 GDN TPR tests."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

DTYPE = torch.bfloat16
HIDDEN_SIZE = 1024
VOCAB_SIZE = 248320


def initialize_npu_runtime(*, world_size: int = 1):
    """Initialize MindSpeed exactly once for the session-scoped test fixture."""
    expected_world_size = int(os.getenv("WORLD_SIZE", str(world_size)))
    if expected_world_size != world_size:
        raise RuntimeError(f"expected WORLD_SIZE={world_size}, got {expected_world_size}")

    import torch_npu  # noqa: F401

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29561")
        dist.init_process_group(
            backend="hccl",
            rank=int(os.getenv("RANK", "0")),
            world_size=world_size,
        )

    saved_argv = sys.argv[:]
    try:
        # MindSpeed parses sys.argv while applying its patches. Keep pytest's
        # flags out of that parser, and invoke repatch only once per session.
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
        from mindspeed.args_utils import get_full_args

        vars(get_full_args()).pop("", None)
        repatch(
            {
                "context_parallel_size": world_size,
                "context_parallel_algo": "megatron_cp_algo",
                "experimental_attention_variant": "gated_delta_net",
                "use_naive_l2norm": True,
                "use_flash_attn": True,
                "deterministic_mode": False,
            }
        )
    finally:
        sys.argv[:] = saved_argv

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=world_size,
            expert_model_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(350800)

    import mindspeed
    import mindspeed_ops
    import megatron.core as megatron_core

    def git_head(module):
        import subprocess
        from pathlib import Path

        module_file = getattr(module, "__file__", None)
        if module_file is None:
            return "unknown"
        directory = Path(module_file).resolve().parent
        try:
            root = subprocess.check_output(
                ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return subprocess.check_output(
                ["git", "-C", root, "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return "not-a-git-checkout"

    runtime = SimpleNamespace(
        rank=dist.get_rank(),
        world_size=world_size,
        device=torch.device("npu", local_rank),
        cp_group=parallel_state.get_context_parallel_group(),
        tp_group=parallel_state.get_tensor_model_parallel_group(),
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
    )
    if runtime.rank == 0:
        print(
            "TPR_GDN_TEST_ENV"
            f"\n  MindSpeed: {mindspeed.__file__} ({git_head(mindspeed)})"
            f"\n  MindSpeed-Ops: {mindspeed_ops.__file__} ({git_head(mindspeed_ops)})"
            f"\n  Megatron-Core: {megatron_core.__file__} ({git_head(megatron_core)})"
            f"\n  PyTorch: {torch.__version__}"
            f"\n  torch_npu: {torch_npu.__version__}"
            f"\n  device: {torch.npu.get_device_name(local_rank)}",
            flush=True,
        )
    return runtime


def destroy_npu_runtime(runtime):
    if not dist.is_initialized():
        return
    dist.barrier(group=runtime.cp_group)
    from megatron.core import parallel_state

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def process_groups(runtime, *, cp_size: int):
    from megatron.core.process_groups_config import ProcessGroupCollection

    groups = ProcessGroupCollection()
    groups.tp = runtime.tp_group
    groups.cp = runtime.tp_group if cp_size == 1 else runtime.cp_group
    groups.pp = runtime.pp_group
    groups.embd = None
    return groups


class AllToAllProbe:
    """Count MindSpeed GDN CP collectives without changing their computation."""

    def __init__(self, gated_delta_net_module):
        self.module = gated_delta_net_module
        self.original_cp2hp = gated_delta_net_module._all_to_all_cp2hp
        self.original_hp2cp = gated_delta_net_module._all_to_all_hp2cp
        self.calls = []

    def __enter__(self):
        def cp2hp(input_, cp_group):
            output = self.original_cp2hp(input_, cp_group)
            self.calls.append("cp2hp")
            return output

        def hp2cp(input_, cp_group):
            output = self.original_hp2cp(input_, cp_group)
            self.calls.append("hp2cp")
            return output

        self.module._all_to_all_cp2hp = cp2hp
        self.module._all_to_all_hp2cp = hp2cp
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module._all_to_all_cp2hp = self.original_cp2hp
        self.module._all_to_all_hp2cp = self.original_hp2cp

    def count(self, direction):
        return self.calls.count(direction)


def qwen35_config(*, cp_size: int):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=24,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=3584,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        attention_output_gate=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        layernorm_zero_centered_gamma=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rotary_percent=0.25,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        use_cpu_initialization=False,
        params_dtype=DTYPE,
        pipeline_dtype=DTYPE,
        autocast_dtype=DTYPE,
        bf16=True,
        deterministic_mode=False,
        transformer_impl="transformer_engine",
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
    )


def make_qwen35_model(runtime, *, cp_size: int, tpr: bool = False, num_layers: int = 24):
    """Build the native or current-branch TPR Qwen3.5 Hybrid model."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    config = qwen35_config(cp_size=cp_size)
    if num_layers not in (4, 24):
        raise ValueError("the migrated test helper supports 4-layer and 24-layer models")
    config.num_layers = num_layers
    spec = get_transformer_block_with_experimental_attention_variant_spec(config)
    if tpr:
        if cp_size not in (1, 2):
            raise NotImplementedError("Hybrid TPR test helper requires CP=1/2")
        os.environ.setdefault("TPR_ENABLE_GDN", "1")
        from verl.workers.engine.megatron.tpr.adapter import replace_self_attention_with_tpr

        spec = replace_self_attention_with_tpr(spec)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=max(4096, int(os.getenv("TPR_GDN_TEST_MAX_SEQ", "4096"))),
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=0.25,
        rotary_base=10_000_000.0,
        pg_collection=process_groups(runtime, cp_size=cp_size),
    ).to(device=runtime.device, dtype=DTYPE)
    if hasattr(model.rotary_pos_emb, "inv_freq"):
        model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(runtime.device)
    for module in model.modules():
        if hasattr(module, "tp_group") and module.tp_group is None:
            module.tp_group = runtime.tp_group
    model.train()
    return model


def assert_hybrid_architecture(model):
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    layers = tuple(model.decoder.layers)
    gdn_layers = tuple(
        layer for layer in layers if isinstance(layer.self_attention, mindspeed_gdn.GatedDeltaNet)
    )
    fa_layers = tuple(layer for layer in layers if layer not in gdn_layers)
    if len(layers) == 24:
        assert len(gdn_layers) == 18
        assert len(fa_layers) == 6
        assert tuple(layer.layer_number for layer in fa_layers) == (4, 8, 12, 16, 20, 24)
    return gdn_layers, fa_layers


@dataclass(frozen=True)
class GradientPairMetrics:
    reference_norm: float
    actual_norm: float
    norm_ratio: float
    absolute_l2: float
    relative_l2: float
    cosine: float


@dataclass(frozen=True)
class GradientMapDiagnostics:
    aggregate: GradientPairMetrics
    worst_name: str
    worst: GradientPairMetrics


def _pair_metrics(reference, actual):
    ref_norm = float(torch.linalg.vector_norm(reference).item())
    act_norm = float(torch.linalg.vector_norm(actual).item())
    diff_norm = float(torch.linalg.vector_norm(actual - reference).item())
    relative = diff_norm / max(ref_norm, 1e-24)
    ratio = act_norm / max(ref_norm, 1e-24)
    cosine = float(torch.dot(reference.reshape(-1), actual.reshape(-1)).item()) / max(
        ref_norm * act_norm, 1e-24
    )
    if ref_norm == 0.0 and act_norm == 0.0:
        cosine = 1.0
    return GradientPairMetrics(ref_norm, act_norm, ratio, diff_norm, relative, cosine)


def gradient_map_diagnostics(reference, actual):
    if reference.keys() != actual.keys():
        raise AssertionError(
            f"gradient parameter sets differ: missing={reference.keys() - actual.keys()}, "
            f"unexpected={actual.keys() - reference.keys()}"
        )
    if not reference:
        raise AssertionError("cannot compare empty gradient maps")
    ref2 = act2 = diff2 = dot = 0.0
    worst_name, worst = None, None
    for name in reference:
        left = reference[name].detach().float()
        right = actual[name].detach().float()
        if left.shape != right.shape:
            raise AssertionError(f"gradient shape differs for {name}: {left.shape} vs {right.shape}")
        ref2 += torch.sum(left.square()).item()
        act2 += torch.sum(right.square()).item()
        diff2 += torch.sum((right - left).square()).item()
        dot += torch.sum(left * right).item()
        local = _pair_metrics(left.reshape(-1), right.reshape(-1))
        if worst is None or local.relative_l2 > worst.relative_l2:
            worst_name, worst = name, local
    aggregate = _pair_metrics(
        torch.tensor([ref2**0.5], dtype=torch.float64),
        torch.tensor([act2**0.5], dtype=torch.float64),
    )
    aggregate = GradientPairMetrics(
        ref2**0.5,
        act2**0.5,
        act2**0.5 / max(ref2**0.5, 1e-24),
        diff2**0.5,
        diff2**0.5 / max(ref2**0.5, 1e-24),
        dot / max((ref2 * act2) ** 0.5, 1e-24) if ref2 and act2 else (1.0 if not ref2 and not act2 else 0.0),
    )
    return GradientMapDiagnostics(aggregate, worst_name, worst)


def assert_gradient_maps_close(reference, actual, *, rtol=8e-2, cosine_min=0.995):
    diagnostic = gradient_map_diagnostics(reference, actual)
    metrics = diagnostic.aggregate
    if metrics.relative_l2 > rtol or metrics.cosine < cosine_min:
        raise AssertionError(
            f"gradient mismatch: relative_l2={metrics.relative_l2:.6e} (limit {rtol:.6e}), "
            f"cosine={metrics.cosine:.9f} (minimum {cosine_min:.9f}), "
            f"worst={diagnostic.worst_name}:{diagnostic.worst.relative_l2:.6e}"
        )
    return metrics.relative_l2, metrics.cosine, (
        diagnostic.worst.relative_l2, diagnostic.worst_name
    )
