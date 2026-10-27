# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Four-NPU TP2+SP+RingCP2 model-level Push/Visit/Pop smoke test.

This is a layout/autograd smoke gate, not Full-vs-TPR gradient parity or PPO E2E.

Run with four visible Ascend NPUs:
  TPR_RUN_TP2_SP_CP2=1 \
    torchrun --nproc_per_node=4 --master_port=29861 -m pytest -s -x \
    tests/models/mcore/tpr/parallel/test_tpr_tp2_sp_cp2_npu.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

if os.getenv("TPR_RUN_TP2_SP_CP2") == "1":
    import torch_npu  # noqa: F401

# The TPR engine imports go through verl.workers.engine/__init__.py, which
# eagerly imports MindSpeed. Hide pytest's argv during these imports so its
# import-time feature patching cannot turn pytest arguments into config fields.
_original_argv = sys.argv[:]
try:
    sys.argv[:] = [sys.argv[0]]
    from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
    from .test_real_actor_tp2cp4_offline_ppo_npu import _sanitize_mindspeed_repatch_args
    from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
    from verl.workers.engine.megatron.tpr.ring import make_ring_sequence_shard
    from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec
finally:
    sys.argv[:] = _original_argv

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_TP2_SP_CP2") != "1",
    reason="requires explicit four-NPU TP2+SP+RingCP2 acceptance",
)


@pytest.fixture(scope="module")
def runtime():
    if int(os.getenv("WORLD_SIZE", "1")) != 4:
        pytest.skip("TP2 x CP2 x DP1 requires four ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    from mindspeed.args_utils import get_full_args

    argv = sys.argv[:]
    try:
        # MindSpeed parses sys.argv while patching; keep pytest/torchrun flags
        # out until repatch and argument normalization have both completed.
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch

        repatch({
            "tensor_model_parallel_size": 2,
            "context_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "context_parallel_algo": "megatron_cp_algo",
            "sequence_parallel": True,
            "use_flash_attn": True,
        })
        args = get_full_args()
        _sanitize_mindspeed_repatch_args(args)
        assert args.sequence_parallel and args.context_parallel_algo == "megatron_cp_algo"
    finally:
        sys.argv[:] = argv
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        context_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(261027)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    assert dist.get_world_size(tp_group) == dist.get_world_size(cp_group) == 2
    state = SimpleNamespace(
        tp_group=tp_group, cp_group=cp_group,
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
        rank=parallel_state.get_tensor_model_parallel_rank(),
        cp_rank=parallel_state.get_context_parallel_rank(),
        device=torch.device("npu", local_rank),
    )
    yield state
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_tp2_sp_cp2_push_visit_pop_layout_and_backward(runtime):
    checkpoint = Path(os.getenv(
        "TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B",
    ))
    model, _ = make_real_qwen3_tp_model(
        runtime, model_path=checkpoint, tp_size=2, cp_size=2,
        sequence_parallel=True, profile="1.7b",
    )
    prefix_len, suffix_len = 129, 127  # odd logical lengths stress Ring padding
    plan = SegmentPlan(
        segments=[
            SegmentSpec(0, None, torch.arange(17, 17 + prefix_len), 0, 0),
            SegmentSpec(
                1, 0, torch.arange(80101, 80101 + suffix_len),
                prefix_len, prefix_len,
            ),
        ],
        root_id=0,
    )

    attn = model.decoder.layers[0].self_attention
    seen_input = []
    seen_qkv = []
    h1 = attn.register_forward_pre_hook(
        lambda _module, args: seen_input.append(int(args[0].shape[0]))
    )
    h2 = attn.linear_qkv.register_forward_hook(
        lambda _module, _args, output: seen_qkv.append(
            int((output[0] if isinstance(output, tuple) else output).shape[0])
        )
    )

    def objective(_segment, logits):
        assert logits.ndim == 3
        loss = logits.float().square().mean()
        return loss, {}

    try:
        model.zero_grad(set_to_none=True)
        executor = SegmentExecutor(
            model, plan, cp_group=runtime.cp_group, cp_backend="ring",
            segment_loss_fn=objective, segment_loss_term_counts={1: 1},
        )
        loss = executor.run()
        assert bool(torch.isfinite(loss).item())
        grad = attn.linear_qkv.weight.grad
        assert grad is not None and bool(torch.isfinite(grad).all().item())
        assert bool(grad.float().abs().sum().gt(0).item())
        physical = [
            make_ring_sequence_shard(
                length, cp_rank=runtime.cp_rank, cp_size=2,
            ).local_length for length in (prefix_len, suffix_len, prefix_len)
        ]
        assert seen_input == [length // 2 for length in physical], (
            "Attention input must be TP2 SP-local inside each CP shard",
            seen_input, physical,
        )
        assert seen_qkv == physical, (
            "Megatron QKV must TP-allgather the complete CP-local sequence",
            seen_qkv, physical,
        )
        if dist.get_rank() == 0:
            print(
                f"TPR_TP2_SP_CP2 PASS input={seen_input} qkv={seen_qkv} "
                f"loss={float(loss.item()):.7g}",
                flush=True,
            )
    finally:
        h1.remove()
        h2.remove()
