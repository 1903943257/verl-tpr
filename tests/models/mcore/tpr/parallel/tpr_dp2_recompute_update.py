"""Post-F/B full-gradient and one-step AdamW gates for 8-NPU TPR DP2 replay.

Deliberately outside the F/B timer. This is an offline, fixed-capture optimizer
verification, not a claim of production VERL Optimizer or online throughput.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _global_grad_norm(model, runtime) -> float:
    """Count unique TP parameter shards once; CP and DP hold finalized copies."""
    local = torch.zeros((), dtype=torch.float32, device=runtime.device)
    if runtime.cp_rank == 0 and runtime.dp_rank == 0:
        for name, param in model.named_parameters():
            if param.grad is None:
                raise AssertionError(f"Missing finalized gradient: {name}")
            if getattr(param, "tensor_model_parallel", False) or runtime.tp_rank == 0:
                grad = param.grad.detach().float()
                local += (grad * grad).sum()
                del grad
    dist.all_reduce(local, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    norm = float(torch.sqrt(local).item())
    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite world-global gradient norm")
    return norm


@torch.no_grad()
def _updated_replica_spread(model, runtime) -> float:
    """All four CP×DP replicas of each TP shard must match exactly."""
    chunks = []
    for name, param in model.named_parameters():
        flat = param.detach().reshape(-1)
        if not flat.numel():
            raise AssertionError(f"Empty parameter {name}")
        indices = torch.tensor(
            sorted({0, flat.numel() // 2, flat.numel() - 1}),
            dtype=torch.long, device=flat.device,
        )
        chunks.append(flat.index_select(0, indices).float())
    samples = torch.cat(chunks)
    group = runtime.dp_cp_group
    replicas = [torch.empty_like(samples) for _ in range(dist.get_world_size(group))]
    dist.all_gather(replicas, samples, group=group)
    stacked = torch.stack(replicas)
    spread = (stacked.max(dim=0).values - stacked.min(dim=0).values).abs().max()
    dist.all_reduce(spread, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
    return float(spread.item())


@torch.no_grad()
def _single_cpu_master_adamw_step(model, runtime, *, norm: float, lr: float,
                                   clip_norm: float) -> float:
    """One FP32-master AdamW step (beta=.9/.95, eps=1e-8, wd=0).

    A single CP0,DP0 rank in each TP-local DP×CP group owns master parameters.
    It broadcasts the updated BF16 parameters to its CP/DP replicas. Other
    ranks do not allocate duplicate FP32 optimizer states.
    """
    owner = runtime.cp_rank == 0 and runtime.dp_rank == 0
    owner_rank = torch.tensor(
        [dist.get_rank() if owner else 2**30],
        dtype=torch.int64, device=runtime.device,
    )
    dist.all_reduce(owner_rank, op=dist.ReduceOp.MIN, group=runtime.dp_cp_group)
    src = int(owner_rank.item())
    if not 0 <= src < dist.get_world_size():
        raise RuntimeError("DP×CP optimizer owner not found")
    factor = 1.0 if norm == 0 else min(1.0, clip_norm / norm)
    for name, param in model.named_parameters():
        if param.grad is None:
            raise AssertionError(f"Missing parameter gradient for {name}")
        if owner:
            # At step 1, corrected AdamW moments equal g and g^2. Keep this
            # explicit to match the established CPU FP32-master replay.
            master = param.detach().to("cpu", dtype=torch.float32).clone()
            grad = param.grad.detach().to("cpu", dtype=torch.float32)
            grad.mul_(factor)
            m = torch.zeros_like(master)
            v = torch.zeros_like(master)
            m.mul_(0.9).add_(grad, alpha=0.1)
            v.mul_(0.95).addcmul_(grad, grad, value=0.05)
            denom = (v / 0.05).sqrt_().add_(1e-8)
            master.addcdiv_(m, denom, value=-lr / 0.1)
            param.copy_(master.to(device=param.device, dtype=param.dtype))
            del master, grad, m, v, denom
        dist.broadcast(param.data, src=src, group=runtime.dp_cp_group)
    return factor


def optional_dp2_verification(model, runtime, output: Path, step: int,
                              *, save_full_grads: bool, optimizer_step: bool) -> dict:
    """One post-replay gate; all eight ranks must call with the same options."""
    output.mkdir(parents=True, exist_ok=True)
    saved_paths = None
    if save_full_grads:
        from .test_real_actor_tp2cp4_offline_ppo_npu import _save_complete_grads
        if runtime.dp_rank == 0 and runtime.cp_rank == 0:
            _save_complete_grads(model, runtime, output, "tpr_dp2", step)
        dist.barrier()
        saved_paths = [
            str(output / f"tpr_dp2_step{step}_tp{tp}_full_grads")
            for tp in (0, 1)
        ]
    if not optimizer_step:
        return {"full_gradient_paths": saved_paths, "optimizer_step": None}
    lr = float(os.getenv("TPR_OFFLINE_ADAMW_LR", "2e-5"))
    clip_norm = float(os.getenv("TPR_OFFLINE_ADAMW_CLIP_NORM", "1.0"))
    if not (0 < lr < 0.1 and 0 < clip_norm < float("inf")):
        raise ValueError("Invalid single-step CPU AdamW learning rate or clip norm")

    from .test_real_actor_tp2cp4_offline_ppo_npu import _witness
    owner = runtime.cp_rank == 0 and runtime.dp_rank == 0
    before = _witness(model, grads=False) if owner else None
    norm = _global_grad_norm(model, runtime)
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    factor = _single_cpu_master_adamw_step(
        model, runtime, norm=norm, lr=lr, clip_norm=clip_norm,
    )
    torch.npu.synchronize()
    seconds = torch.tensor(
        [time.perf_counter() - start], dtype=torch.float32, device=runtime.device,
    )
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
    spread = _updated_replica_spread(model, runtime)
    if spread != 0:
        raise AssertionError(f"Updated DP/CP parameter replicas differ: max_abs={spread}")

    if owner:
        target = output / f"step{step}_updated_tp{runtime.tp_rank}.pt"
        if target.exists():
            raise FileExistsError(target)
        torch.save(
            {"initial": before, "updated": _witness(model, grads=False),
             "global_grad_norm": norm, "clip_factor": factor},
            target,
        )
    dist.barrier()
    return {
        "full_gradient_paths": saved_paths,
        "optimizer_step": {
            "optimizer": "cpu_fp32_master_adamw_step1",
            "lr": lr, "clip_norm": clip_norm,
            "global_grad_norm_preclip": norm,
            "clip_factor": factor,
            "critical_optimizer_s": float(seconds.item()),
            "updated_replica_sample_max_abs": spread,
            "updated_weights_samples": [
                str(output / f"step{step}_updated_tp{tp}.pt") for tp in (0, 1)
            ],
            "notes": (
                "Offline single AdamW update from fixed PPO gradients; "
                "optimizer initialization/state residency and production "
                "VERL engine are NOT part of measured F/B timing"
            ),
        },
    }
