# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Four real VERL training routes; no attention/linear/finalize substitutes.

Run with the shell launcher, one topology per torchrun. Deliberately stop
after native gradient finalize, before optimizer clipping/update.
"""

import importlib.metadata
import os
from functools import partial
from pathlib import Path

from .training_precision_io import write_json


def _build_engine(cp, tpr, fusion):
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import ActorConfig, HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine.mindspeed import MindspeedEngineWithLMHead
    from verl.workers.utils.losses import ppo_loss

    model_config = HFModelConfig(
        path=os.environ["TPR_TRAIN_MODEL_PATH"],
        trust_remote_code=False,
        enable_gradient_checkpointing=False,
        use_remove_padding=True,
    )
    overrides = dict(
        context_parallel_algo="megatron_cp_algo",
        use_flash_attn=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=fusion,
        calculate_per_token_loss=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
    )
    engine_config = McoreEngineConfig(
        forward_only=False,
        use_mbridge=True,
        vanilla_mbridge=os.getenv("TPR_TRAIN_VANILLA_MBRIDGE", "1") == "1",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        use_distributed_optimizer=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        infer_micro_batch_size_per_gpu=1,
        use_remove_padding=True,
        use_fused_kernels=False,
        param_offload=False,
        optimizer_offload=False,
        tpr_enabled=tpr,
        tpr_cp_backend="ring" if cp > 1 else "auto",
        seed=int(os.getenv("TPR_TRAIN_SEED", "261105")),
        override_transformer_config=overrides,
    )
    optimizer_config = McoreOptimizerConfig(lr=1e-6, lr_decay_steps=1)
    engine = MindspeedEngineWithLMHead(model_config, engine_config, optimizer_config, CheckpointConfig())
    engine.initialize()
    actor_config = ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode="token-mean",
        entropy_coeff=0.0,
        use_kl_loss=False,
    )
    return engine, partial(ppo_loss, config=actor_config)


def _native_response_logprobs(output, batch):
    from verl.workers.utils.padding import no_padding_2_padding

    return no_padding_2_padding(output["model_output"]["log_probs"], batch).detach().float().cpu()


def _versions():
    result = {}
    for name in ("torch", "torch-npu", "megatron-core", "mindspeed", "mbridge", "megatron-bridge", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main():
    # Import ordering and repatch belong to the actual NPU engine. There are
    # no pytest flags for MindSpeed to mistake for TransformerConfig fields.
    import mindspeed.megatron_adaptor  # noqa: F401
    import torch
    import torch.distributed as dist

    from verl.utils.metric import reduce_metrics

    from .experiment_source import source_metadata
    from .training_precision_contract import (
        MODES,
        PREFIX,
        SUFFIX,
        ExecutionWitness,
        actor_batch,
        audit_engine,
        audit_wgrad_completion,
        input_digest,
        synthetic_tokens,
        validate_forest,
    )
    from .training_precision_io import observe_gradients, parameter_layout, weight_digest
    from .training_precision_report import complete_mode

    mode = os.environ["TPR_TRAIN_MODE"]
    if mode not in MODES:
        raise ValueError(f"Unknown training mode {mode}")
    cp, tpr = (2 if mode.endswith("cp2") else 1), mode.startswith("tpr")
    if int(os.environ["WORLD_SIZE"]) != cp:
        raise ValueError("Require TP=PP=DP=1 and WORLD_SIZE=CP")
    fusion_flag = os.getenv("TPR_TRAIN_GRAD_ACCUM_FUSION", "1")
    if fusion_flag not in ("0", "1"):
        raise ValueError("TPR_TRAIN_GRAD_ACCUM_FUSION must be 0 or 1")
    fusion = fusion_flag == "1"
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    root = Path(os.environ["TPR_TRAIN_OUTPUT"]).resolve()
    folder = root / mode
    try:
        if rank == 0:
            print(f"TRAIN_PRECISION_BUILD mode={mode} using=production_MindspeedEngineWithLMHead", flush=True)
        engine, loss_function = _build_engine(cp, tpr, fusion)
        model, audit = audit_engine(engine, tpr=tpr, fusion=fusion)
        layout = parameter_layout(model)
        weights = weight_digest(model)
        versions = {name: parameter._version for name, parameter in model.named_parameters()}
        fixture_path = root / "actor_fixture.pt"
        if mode == "base_cp1":
            seed = engine.engine_config.seed
            tokens = synthetic_tokens(engine.model_config.hf_config.vocab_size, seed)
            advantages = torch.sin(torch.arange(2 * SUFFIX).reshape(2, SUFFIX).float() * 0.17) + 0.5
            temporary = actor_batch(tokens, torch.zeros_like(advantages), advantages)
            with engine.eval_mode(), torch.no_grad():
                old_log_probs = _native_response_logprobs(
                    engine.forward_backward_batch(temporary, None, forward_only=True), temporary
                )
            fixture = dict(tokens=tokens, old_log_probs=old_log_probs, advantages=advantages, weights=weights)
            torch.save(fixture, fixture_path)
        else:
            fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
        if fixture["weights"] != weights:
            raise AssertionError("Four groups must load exactly the same initial weights")
        batch = actor_batch(fixture["tokens"], fixture["old_log_probs"], fixture["advantages"])
        forest = validate_forest(batch)
        digest = input_digest(fixture["tokens"], fixture["old_log_probs"], fixture["advantages"])
        references = {name: root / name / "grads" for name in ("base_cp1", "base_cp2") if name != mode}
        if mode == "base_cp1":
            references = {}
        if rank == 0:
            print(f"TRAIN_PRECISION_BEGIN mode={mode} P={PREFIX} S={SUFFIX} rows=2 fusion={fusion}", flush=True)
        with engine.train_mode():
            engine.optimizer_zero_grad()
            torch.npu.reset_peak_memory_stats()
            with ExecutionWitness(model.config.finalize_model_grads_func) as witness:
                output = engine.forward_backward_batch(batch, loss_function, forward_only=False)
            torch.npu.synchronize()
            witness.validate(tpr=tpr, cp=cp, fusion=fusion)
            completion = audit_wgrad_completion(model, fusion=fusion)
            peak = torch.npu.max_memory_allocated()
            if rank == 0:
                print(f"TRAIN_PRECISION_FINALIZED mode={mode} main_grad=FP32 exporting_owned_slices", flush=True)
            if versions != {name: parameter._version for name, parameter in model.named_parameters()}:
                raise AssertionError("Diagnostic updated the model weights")
            gradients = observe_gradients(
                model,
                engine.optimizer,
                folder / "grads",
                rank,
                references,
                save_reference=mode.startswith("base"),
            )
            if tpr:
                logprobs = [[row, offset, value] for (row, offset), value in engine._tpr_captured_log_probs.items()]
                counts = output["metrics"]
                if counts["tpr/physical_segments"] != [3.0] or counts["tpr/forest_trees"] != [1.0]:
                    raise AssertionError("The training adapter bypassed the expected shared-prefix forest")
            else:
                values = _native_response_logprobs(output, batch) if rank == 0 else None
                logprobs = (
                    []
                    if values is None
                    else [[row, offset, float(values[row, offset])] for row in range(2) for offset in range(SUFFIX)]
                )
            report = dict(
                mode=mode,
                rank=rank,
                cp=cp,
                prefix_length=PREFIX,
                suffix_length=SUFFIX,
                input_sha256=digest,
                initial_weights_sha256=weights,
                loss=sum(float(value) for value in output["loss"]),
                logprobs=logprobs,
                precision=audit,
                backward=completion,
                calls=dict(witness.counts),
                peak_allocated_bytes=peak,
                forest=forest,
                gradients=gradients,
                source=source_metadata(),
                packages=_versions(),
                objective="unmodified_VERL_vanilla_PPO_token_mean",
                fixture="synthetic_two_trajectories_base_cp1_old_logprobs_fixed_advantages",
                optimizer_step=False,
                gradient_stage="after_native_finalize_before_clipping_or_optimizer_step",
                native_metrics={name: float(value) for name, value in reduce_metrics(dict(output["metrics"])).items()},
                controls={
                    "seed": engine.engine_config.seed,
                    "vanilla_mbridge": engine.engine_config.vanilla_mbridge,
                    "delayed_dkv_fp32": os.environ["TPR_EXPERIMENT_DELAYED_DKV_FP32"],
                    "root_native_ring": os.environ["TPR_ROOT_NATIVE_RING"],
                    "leaf_batch_size": os.environ["TPR_LEAF_BATCH_SIZE"],
                    "selected_logits_fp32": os.environ["TPR_SELECTED_LOGITS_FP32"],
                },
            )
            write_json(folder / f"rank{rank}.json", report)
        dist.barrier()
        if rank == 0:
            complete_mode(root, mode, cp, layout)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
