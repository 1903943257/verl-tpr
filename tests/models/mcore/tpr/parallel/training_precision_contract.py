# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Inputs and read-only execution witnesses for the production training gate."""

import hashlib
import inspect
import sys
from collections import Counter

import torch

MODES = ("base_cp1", "base_cp2", "tpr_cp1", "tpr_cp2")
PREFIX = SUFFIX = 1024


def synthetic_tokens(vocab_size, seed):
    if vocab_size < 32:
        raise ValueError("Expected a real language-model vocabulary")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (2, PREFIX + SUFFIX), generator=generator)
    tokens[1, :PREFIX] = tokens[0, :PREFIX]
    tokens[1, PREFIX] = (tokens[0, PREFIX] + 1) % vocab_size
    return tokens


def input_digest(tokens, old_log_probs, advantages):
    digest = hashlib.sha256()
    for tensor in (tokens, old_log_probs, advantages):
        digest.update(str((tensor.shape, tensor.dtype)).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def actor_batch(tokens, old_log_probs, advantages):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    if tokens.shape != (2, PREFIX + SUFFIX):
        raise ValueError("Use two trajectories, each with exactly 1K prompt + 1K response")
    if old_log_probs.shape != (2, SUFFIX) or advantages.shape != (2, SUFFIX):
        raise ValueError("Actor labels must cover all response tokens")
    if old_log_probs.dtype != torch.float32 or advantages.dtype != torch.float32:
        raise TypeError("Use the fixed FP32 actor old_log_probs and advantages")
    if not bool(torch.isfinite(old_log_probs).all()) or not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("Nonfinite actor labels")

    def nested(value):
        return torch.nested.as_nested_tensor(list(value.unbind()), layout=torch.jagged)

    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :PREFIX] = False
    batch = TensorDict(
        dict(
            input_ids=nested(tokens),
            prompts=nested(tokens[:, :PREFIX]),
            responses=nested(tokens[:, PREFIX:]),
            attention_mask=nested(torch.ones_like(tokens)),
            position_ids=nested(torch.arange(PREFIX + SUFFIX).repeat(2, 1)),
            response_mask=nested(torch.ones((2, SUFFIX), dtype=torch.bool)),
            loss_mask=nested(mask),
            old_log_probs=nested(old_log_probs),
            advantages=nested(advantages),
        ),
        batch_size=[2],
    )
    tu.assign_non_tensor(
        batch,
        temperature=1.0,
        global_batch_size=2,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        tpr_capture_log_probs=True,
        tpr_trajectory_keys=("precision_0_0", "precision_0_1"),
        pad_mode=DatasetPadMode.NO_PADDING,
    )
    return batch


def validate_forest(batch):
    from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans

    forest = build_tree_execution_plans(("precision_0_0", "precision_0_1"), batch, require_loss_mask_alignment=True)
    if len(forest.trees) != 1 or forest.segment_count != 3 or forest.logical_loss_tokens != 2 * SUFFIX:
        raise AssertionError("Production tree builder did not produce a shared 1K prefix and two 1K suffixes")
    segments = list(forest.trees[0].segment_plan.segments.values())
    if [segment.length for segment in segments] != [PREFIX, SUFFIX, SUFFIX]:
        raise AssertionError("Unexpected physical segment lengths")
    return {"trees": 1, "segments": 3, "lengths": [segment.length for segment in segments]}


def function_origin(function):
    return {
        "module": getattr(function, "__module__", None),
        "name": getattr(function, "__qualname__", type(function).__qualname__),
        "file": inspect.getsourcefile(function),
    }


def audit_engine(engine, *, tpr, fusion):
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, finalize_model_grads

    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.utils.megatron_utils import unwrap_model

    if len(engine.module) != 1 or not isinstance(engine.module[0], DistributedDataParallel):
        raise AssertionError("Require production Megatron DDP, PP=VPP=1")
    wrapped = engine.module[0]
    model = unwrap_model(wrapped)
    config = model.config
    topology = (
        parallel_state.get_tensor_model_parallel_world_size(),
        parallel_state.get_pipeline_model_parallel_world_size(),
        parallel_state.get_context_parallel_world_size(),
        parallel_state.get_data_parallel_world_size(),
    )
    if topology != (1, 1, engine.engine_config.context_parallel_size, 1):
        raise AssertionError(f"Wrong native TP/PP/CP/DP groups: {topology}")
    if config.finalize_model_grads_func is not finalize_model_grads:
        raise AssertionError("Gradient finalize was replaced instead of using native Megatron/MindSpeed")
    if not wrapped.ddp_config.grad_reduce_in_fp32 or not wrapped.ddp_config.use_distributed_optimizer:
        raise AssertionError("Require native FP32 DDP buffers and DistributedOptimizer")
    if config.gradient_accumulation_fusion != fusion:
        raise AssertionError("Requested and actual gradient accumulation fusion disagree")
    if config.calculate_per_token_loss or config.recompute_granularity is not None:
        raise AssertionError("Matched token-mean PPO controls require no per-token schedule or recomputation")
    attention = [layer.self_attention for layer in model.decoder.layers]
    if any(isinstance(module, TPRSelfAttention) != tpr for module in attention):
        raise AssertionError("Base/TPR layer replacement does not match the requested mode")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.dtype != torch.bfloat16
            or getattr(parameter, "main_grad", None) is None
            or parameter.main_grad.dtype != torch.float32
        ):
            raise AssertionError(f"Expected BF16 parameter + real FP32 main_grad: {name}")
    linears = [module for module in model.modules() if hasattr(module, "gradient_accumulation_fusion")]
    if not linears or any(module.gradient_accumulation_fusion != fusion for module in linears):
        raise AssertionError("Native linear modules did not inherit the requested wgrad fusion")
    return model, {
        "engine": f"{type(engine).__module__}.{type(engine).__name__}",
        "topology_tp_pp_cp_dp": list(topology),
        "ddp": f"{type(wrapped).__module__}.{type(wrapped).__name__}",
        "parameter_dtype": "torch.bfloat16",
        "main_grad_dtype": "torch.float32",
        "gradient_accumulation_fusion": fusion,
        "fused_linear_modules": len(linears),
        "grad_reduce_in_fp32": wrapped.ddp_config.grad_reduce_in_fp32,
        "use_distributed_optimizer": wrapped.ddp_config.use_distributed_optimizer,
        "finalize": function_origin(config.finalize_model_grads_func),
        "core_forward": [function_origin(module.core_attention.forward) for module in attention[:1]],
        "transformer_config": {key: str(value) for key, value in vars(config).items() if not callable(value)},
    }


class ExecutionWitness:
    """Observe Python/C call entrypoints; do not replace any training function."""

    def __init__(self, finalize):
        self.finalize_code = finalize.__code__
        self.counts = Counter()

    def _profile(self, frame, event, argument):
        if event == "call":
            name, filename = frame.f_code.co_name, frame.f_code.co_filename.replace("\\", "/")
            if frame.f_code is self.finalize_code:
                self.counts["native_finalize"] += 1
            if "/mindspeed/" in filename and name == "ringattn_context_parallel":
                self.counts["native_ring"] += 1
            if "/mindspeed/" in filename and name in ("dot_product_attention_forward", "flash_attention_forward"):
                self.counts["native_attention"] += 1
            if (
                any(package in filename for package in ("/megatron/", "/mindspeed/"))
                and name == "forward_backward_no_pipelining"
            ):
                self.counts["native_schedule"] += 1
            if name == "run_tpr_forward_backward_batch" and filename.endswith("/tpr/adapter.py"):
                self.counts["tpr_schedule"] += 1
        elif event == "c_call":
            name = getattr(argument, "__name__", "")
        else:
            return
        if name in ("wgrad_gemm_accum_fp32", "npu_matmul_add_fp32"):
            self.counts["wgrad_fp32"] += 1
        if name in ("wgrad_gemm_accum_fp16", "npu_matmul_add_fp16"):
            self.counts["wgrad_fp16"] += 1

    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError("Run the precision gate without another Python profiler")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *_exc):
        sys.setprofile(self.previous)

    def validate(self, *, tpr, cp, fusion):
        if self.counts["native_finalize"] != 1:
            raise AssertionError(f"Expected exactly one native gradient finalize: {dict(self.counts)}")
        if self.counts["tpr_schedule" if tpr else "native_schedule"] != 1:
            raise AssertionError(f"Actual training schedule was not observed: {dict(self.counts)}")
        if not tpr and not self.counts["native_attention"]:
            raise AssertionError("Native MindSpeed attention wrapper was not observed; no adapter fallback is allowed")
        if not tpr and cp == 2 and not self.counts["native_ring"]:
            raise AssertionError("Base CP2 did not enter native MindSpeed Ring through the model")
        if fusion and self.counts["wgrad_fp16"]:
            raise AssertionError(f"Observed a low-precision fused wgrad destination: {dict(self.counts)}")


def audit_wgrad_completion(model, *, fusion):
    # Autograd can run backward on engine threads, outside sys.setprofile's
    # thread. Native per-parameter completion flags are the decisive witness.
    weights = {
        module.weight
        for module in model.modules()
        if hasattr(module, "gradient_accumulation_fusion") and getattr(module, "weight", None) is not None
    }
    if fusion and (not weights or any(not getattr(weight, "grad_added_to_main_grad", False) for weight in weights)):
        raise AssertionError("Native fused linear backward did not mark all its FP32 main_grad contributions")
    return {"linear_weight_count": len(weights), "native_fused_wgrad_completed": fusion}
