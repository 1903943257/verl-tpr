# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Two-Ascend-NPU TP1/CP1/DP2 zero-local-supervision regression.

Uses production tree-building and run_forest logic with a tiny parameterized
loss. This deliberately avoids a Qwen checkpoint and does NOT claim to test
native Megatron DDP's main-grad buffers or the full PPO adapter.
Run with torchrun --nproc_per_node=2, ASCEND_RT_VISIBLE_DEVICES=2,3.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans


def _production_run_forest():
    """Load the exact source function, excluding unrelated Megatron imports."""
    source = (
        Path(__file__).resolve().parents[5]
        / "verl/workers/engine/megatron/tpr/adapter.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_forest"
    )
    namespace = {
        "Tensor": torch.Tensor, "nullcontext": nullcontext,
        "Callable": Callable, "Any": Any,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["run_forest"]


def main():
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("This regression requires exactly two ranks: TP1 CP1 DP2")
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl")
    try:
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        # Communication check precedes any model or TPR-specific work.
        ranks = torch.tensor([rank], dtype=torch.int32, device=device)
        received = [torch.zeros_like(ranks) for _ in range(2)]
        dist.all_gather(received, ranks)
        if [int(x.item()) for x in received] != [0, 1]:
            raise AssertionError("HCCL world group does not contain exactly DP0/DP1")
        if rank == 0:
            print("TPR_DP2_2NPU_HCCL_PASS", flush=True)

        # DP0 contains a valid trajectory but has zero supervised response tokens.
        # DP1 has exactly one. Each DP independently builds its own Forest.
        local_supervision = rank
        forest = build_tree_execution_plans(
            [f"dp{rank}_s_0"],
            {
                "input_ids": [torch.tensor([11, 12, 13], dtype=torch.long)],
                "prompts": [torch.tensor([11, 12], dtype=torch.long)],
                "responses": [torch.tensor([13], dtype=torch.long)],
                "response_mask": [
                    torch.tensor([local_supervision], dtype=torch.bool)
                ],
            },
            tree_policy="token_lcp",
            tree_builder="lcp_sort",
            min_shared_prefix_tokens=1,
        )
        if len(forest.trees) != 1 or forest.logical_loss_tokens != local_supervision:
            raise AssertionError(
                f"DP{rank} wrong local forest: trees={len(forest.trees)} "
                f"supervised={forest.logical_loss_tokens}"
            )
        if rank == 0:
            print("TPR_DP2_2NPU_FOREST_PASS dp0_tokens=0 dp1_tokens=1", flush=True)

        denominator = torch.tensor(local_supervision, device=device, dtype=torch.int32)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        if int(denominator.item()) != 1:
            raise AssertionError("Global supervised denominator must be exactly 1")

        # Weight=1 and x=2 -> raw gradient=8 on DP1.
        # Like VERL PPO, multiply local objective by dp_size=2 before
        # averaged DP gradient finalization, preserving global token mean.
        model = torch.nn.Linear(1, 1, bias=False, device=device, dtype=torch.float32)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model.zero_grad(set_to_none=True)
        calls = {"forward": 0, "finalize": 0}

        def run_tree(tree):
            calls["forward"] += 1
            if tree.logical_loss_tokens != 1:
                raise AssertionError("Only DP1 may execute a supervised tree")
            value = model(torch.tensor([[2.0]], device=device))
            loss = value.square().sum() * (dist.get_world_size() / int(denominator.item()))
            loss.backward()
            return loss

        def finalize():
            calls["finalize"] += 1
            # Reference to native averaged DP weight-gradient semantics:
            # empty DP0 contributes zero, but both ranks enter the same all_reduce.
            # Production must separately validate Megatron's main_grad buffers.
            for param in model.parameters():
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(dist.get_world_size())

        run_forest = _production_run_forest()
        loss = run_forest(
            forest, run_tree, finalize,
            empty_loss=(torch.zeros((), device=device) if rank == 0 else None),
        )
        if calls != {"forward": rank, "finalize": 1}:
            raise AssertionError(f"DP{rank} unexpected forward/finalize counts: {calls}")
        if float(loss.item()) != (8.0 if rank == 1 else 0.0):
            raise AssertionError(f"DP{rank} unexpected local loss: {float(loss.item())}")

        # Expected global gradient = 8; compare replica values before update.
        if not torch.allclose(model.weight.grad, torch.tensor([[8.0]], device=device)):
            raise AssertionError(
                f"DP{rank} gradient mismatch: {model.weight.grad.item()}"
            )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        if not torch.allclose(model.weight, torch.tensor([[0.2]], device=device)):
            raise AssertionError(f"DP{rank} optimizer mismatch: {model.weight.item()}")

        weight = model.weight.detach().flatten().contiguous()
        all_weights = [torch.empty_like(weight) for _ in range(2)]
        dist.all_gather(all_weights, weight)
        if not torch.equal(all_weights[0], all_weights[1]):
            raise AssertionError("DP replicas diverged after optimizer step")
        if rank == 0:
            print(
                "TPR_DP2_EMPTY_2NPU_PASS topology=TP1xCP1xDP2 "
                "dp0_tokens=0 dp1_tokens=1 gradient=8 weight_after_step=0.2",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
