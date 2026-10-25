"""Genuine VERL/MindSpeed MegatronEngine TPR training replay, no rollout.

Mirror test_verl_engine_full_replay_npu on identical recorded actor PPO steps:
real DDP, real BF16 main_grad policy, original PPO objective, native
gradient finalize and optimizer step. Only the TPR Forest execution differs.
This is NOT the older SimpleNamespace Engine + CPU-AdamW numerical control.
"""
from __future__ import annotations

import json
import os
import time
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO
from .test_verl_engine_full_replay_npu import (
    _construct_real_engine, _load_step, _cpu_metrics, _native_witness,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_VERL_ENGINE_TPR_REPLAY") != "1",
    reason="Explicit 8-NPU real-MindspeedEngine TPR Actor PPO update replay",
)


def _merge_tpr_logprobs(local: dict, *, ppo_tokens: int, tp_rank: int) -> dict:
    """Collect unique CP-owned logprobs once, ignoring identical TP replicas.

    Avoid silently reporting only CP0's subset as the complete Actor update.
    All ranks participate to prevent HCCL object-collective deadlock.
    """
    local_owned = local if tp_rank == 0 else {}
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, local_owned)
    combined = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("TPR logprob records must be dictionaries")
        for key, value in record.items():
            if key in combined:
                raise AssertionError(f"CP ranks both own logical logprob {key}")
            combined[key] = float(value)
    if len(combined) != ppo_tokens:
        raise AssertionError(
            f"TPR produced {len(combined)} logical logprobs, expected {ppo_tokens}"
        )
    return combined


def _qkv_grad_storage(engine) -> dict:
    from verl.utils.megatron_utils import unwrap_model

    wrapped = engine.module[0]
    model = unwrap_model(wrapped)
    weight = model.decoder.layers[0].self_attention.linear_qkv.weight
    main = getattr(weight, "main_grad", None)
    if main is None:
        raise AssertionError(
            "Actual Megatron DDP did not allocate QKV main_grad; "
            "this is not an accepted real-Engine gradient-buffer audit"
        )
    ddp = getattr(wrapped, "ddp_config", None)
    return {
        "wrapped_model": type(wrapped).__name__,
        "param_dtype": str(weight.dtype),
        "param_grad_dtype": str(weight.grad.dtype) if weight.grad is not None else None,
        "main_grad_dtype": str(main.dtype),
        "main_grad_bytes": int(main.numel() * main.element_size()),
        "grad_reduce_in_fp32": getattr(ddp, "grad_reduce_in_fp32", None),
    }


def test_verl_engine_tpr_replay():
    if int(os.getenv("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("Real TPR replay requires TP2 x RingCP4 x DP1 (8 ranks)")
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl")
    try:
        from megatron.core import parallel_state as mpu
        from verl.workers.utils.losses import ppo_loss

        manifest_path = Path(os.environ["TPR_OFFLINE_CAPTURE_MANIFEST"])
        checkpoint = Path(os.getenv(
            "TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"
        ))
        output_dir = Path(os.environ["TPR_OFFLINE_OUTPUT_DIR"])
        run_steps = int(os.getenv("TPR_VERL_TPR_REPLAY_STEPS", "1"))
        prefix = os.getenv("TPR_VERL_TPR_REPLAY_PREFIX", "verl_tpr_smoke")
        if not prefix.startswith("verl_tpr") or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in prefix
        ):
            raise ValueError(f"Unsafe TPR replay prefix: {prefix!r}")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("protocol") != "real_actor_session_first_8merged_v3"
                or not 1 <= len(manifest.get("items", [])) <= 20
                or manifest.get("online_steps") != len(manifest["items"])
                or manifest.get("expected_sessions") != 8):
            raise AssertionError("Requires genuine 1..20-step / 8-Session capture manifest")
        if not 1 <= run_steps <= len(manifest["items"]):
            raise ValueError(
                f"TPR_VERL_TPR_REPLAY_STEPS={run_steps} exceeds "
                f"available original Actor steps={len(manifest['items'])}"
            )
        if os.getenv("TPR_TREE_POLICY", "qwen_turn") != "qwen_turn":
            raise AssertionError("Real TPR replay requires qwen_turn")
        output_dir.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank()
        for step in range(1, run_steps + 1):
            target = output_dir / f"{prefix}_step{step}_rank{rank}.pt"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite TPR Replay: {target}")

        lr = float(os.getenv("TPR_OFFLINE_ADAMW_LR", "2e-5"))
        clip_norm = float(os.getenv("TPR_OFFLINE_ADAMW_CLIP_NORM", "1.0"))
        engine = _construct_real_engine(
            checkpoint, lr=lr, clip_norm=clip_norm, steps=20, tpr_enabled=True,
        )
        if not engine.engine_config.tpr_enabled or engine.engine_config.tpr_cp_backend != "ring":
            raise AssertionError("Real Engine did not select TPR Ring")
        storage = _qkv_grad_storage(engine)
        if rank == 0:
            print("VERL_TPR_ENGINE_STORAGE " + json.dumps(storage, sort_keys=True), flush=True)
        ppo_config = _PPO()
        initial_param_witness = _native_witness(engine, grads=False)

        with engine.train_mode():
            for step, item in enumerate(manifest["items"][:run_steps], start=1):
                batch, original_keys, tokens, digest, rows, capture = _load_step(
                    item, mode="tpr",
                )
                engine.optimizer_zero_grad()
                dist.barrier()
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                allocated_before = int(torch.npu.memory_allocated())
                t0 = time.perf_counter()
                result = engine.forward_backward_batch(
                    batch, partial(ppo_loss, config=ppo_config), forward_only=False
                )
                torch.npu.synchronize()
                fb_seconds = time.perf_counter() - t0
                if result is None or len(result.get("loss", [])) != 1:
                    raise AssertionError("TPR Forest must report one global PPO loss")
                local_logprobs = dict(getattr(engine, "_tpr_captured_log_probs", {}))
                logprobs = _merge_tpr_logprobs(
                    local_logprobs, ppo_tokens=tokens,
                    tp_rank=mpu.get_tensor_model_parallel_rank(),
                )
                grad_witness = _native_witness(engine, grads=True)
                grad_storage = _qkv_grad_storage(engine)
                t1 = time.perf_counter()
                grad_norm = engine.optimizer_step()
                engine.lr_scheduler_step()
                torch.npu.synchronize()
                optimizer_seconds = time.perf_counter() - t1
                param_after = _native_witness(engine, grads=False)
                # Only the model-parallel output source saves token-level
                # logprobs; all ranks retain their own TP parameter witnesses.
                snapshot = {
                    "execution_backend": "verl_mindspeed_megatron_engine_tpr",
                    "mode": "verl_tpr",
                    "run_prefix": prefix,
                    "step": step,
                    "steps": run_steps,
                    "rank": rank,
                    "tp_rank": mpu.get_tensor_model_parallel_rank(),
                    "cp_rank": mpu.get_context_parallel_rank(),
                    "checkpoint": str(checkpoint),
                    "source_capture": str(capture),
                    "source_rows": rows,
                    "source_keys": original_keys,
                    "source_online_global_step": item.get("source_global_step", step),
                    "source_capture_index": item.get("source_capture_index"),
                    "data_sha256": digest,
                    "capture_manifest": str(manifest_path),
                    "tokens": tokens,
                    "native_microbatches": None,
                    "tree_policy": "qwen_turn",
                    "recompute_granularity": None,
                    "loss": [float(x) for x in result["loss"]],
                    "metrics": _cpu_metrics(result.get("metrics", {})),
                    "logprobs": logprobs if engine.is_mp_src_rank_with_outputs() else {},
                    "param_initial": initial_param_witness,
                    "param_after": param_after,
                    "gradient_samples": grad_witness,
                    "qkv_grad_storage": grad_storage,
                    "grad_norm_preclip": float(grad_norm) if grad_norm is not None else None,
                    "fwd_bwd_seconds": fb_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
                    "allocated_before_bytes": allocated_before,
                    "optimizer": "verl_native_megatron_precision_aware_cpu_offload_adam",
                    "optimizer_lr": lr,
                    "optimizer_clip_norm": clip_norm,
                }
                target = output_dir / f"{prefix}_step{step}_rank{rank}.pt"
                if target.exists():
                    raise FileExistsError(target)
                torch.save(snapshot, target)
                if rank == 0:
                    print(
                        f"VERL_ENGINE_TPR step={step} "
                        f"loss_sum={sum(snapshot['loss']):.9g} tokens={tokens} "
                        f"fb_s={fb_seconds:.3f} opt_s={optimizer_seconds:.3f} "
                        f"grad_norm={grad_norm} "
                        f"peak_gib={snapshot['peak_allocated_bytes']/2**30:.3f} "
                        f"logprobs={len(logprobs)}",
                        flush=True,
                    )
                dist.barrier()
    finally:
        if dist.is_initialized():
            from megatron.core import parallel_state as mpu
            if mpu.model_parallel_is_initialized():
                mpu.destroy_model_parallel()
            dist.destroy_process_group()
