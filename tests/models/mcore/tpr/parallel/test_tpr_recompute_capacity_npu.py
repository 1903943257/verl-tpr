"""Qwen3-4B-128K TPR + Recompute sequence-length capacity probe.

Change ONLY CP and SEQ_LEN below. Run from any directory:
    python tests/models/mcore/tpr/parallel/test_tpr_recompute_capacity_npu.py

Synthetic shared-prefix 2-branch TPR, TP=DP=1, BF16, full/uniform/1
activation recompute. Measures complete forward, PPO loss, backward, and
gradient finalization, NOT optimizer/online PPO training. This is an
experimental NPU capacity test and is not a correctness gate.
"""

# ================= CHANGE ONLY THESE TWO =================
CP = 1                 # 1 / 2 / 4 (number of NPUs)
SEQ_LEN = 8192         # tokens in EACH logical trajectory; P + S
# =========================================================

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    if os.getenv("TPR_CAP_WORKER") != "1":
        import subprocess

        if CP not in (1, 2, 4):
            raise SystemExit("CP must be 1, 2, or 4")
        if SEQ_LEN <= 1024 or SEQ_LEN % 128 != 0:
            raise SystemExit("SEQ_LEN must be > 1024 and divisible by 128")
        root = Path(__file__).resolve().parents[5]
        env = os.environ.copy()
        env.update({
            "TPR_CAP_WORKER": "1",
            "TPR_CAP_CP": str(CP),
            "TPR_CAP_LENGTH": str(SEQ_LEN),
            "TPR_TREE_POLICY": "token_lcp",
            "TPR_TREE_BUILDER": "lcp_sort",
            "TPR_MIN_SHARED_PREFIX_TOKENS": "1024",
            "TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS": "0",
            "TPR_PPO_LOGPROB_CHUNK_TOKENS": "256",
            "TPR_LEAF_BATCH_SIZE": "1",
            "HCCL_DETERMINISTIC": "true",
            "TPR_OFFLINE_GRAD_SYNC_DTYPE": "bf16",
            "TPR_EXEC_PROFILE": "0",
            "PYTHONUNBUFFERED": "1",
        })
        # Preserve the user's physical-device selection, but use only CP ranks.
        configured = env.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
        devices = configured.split(",") if configured else [str(i) for i in range(CP)]
        if len(devices) < CP or any(not v.strip().isdigit() for v in devices[:CP]):
            raise SystemExit(f"CP={CP} needs {CP} visible NPUs, got {devices}")
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(devices[:CP])
        # Prefer this checkout over a separately pip-installed VERL package.
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(root), env.get("PYTHONPATH", "")))
        )
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--master_addr=127.0.0.1", "--master_port=29683",
            f"--nproc_per_node={CP}",
            str(Path(__file__).resolve()),
        ]
        print(
            f"CAP_START CP={CP} SEQ_LEN={SEQ_LEN} "
            f"NPUS={env['ASCEND_RT_VISIBLE_DEVICES']}", flush=True,
        )
        raise SystemExit(subprocess.call(cmd, env=env, cwd=root))

    # MindSpeed reads sys.argv during imports; keep torchrun/pytest flags out.
    sys.argv[:] = [sys.argv[0]]
    root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(root))
    # Make a checkout mismatch fail clearly before collecting test modules.
    import verl
    import verl.workers.engine.megatron.tpr as tpr_package
    expected = root / "verl" / "workers" / "engine" / "megatron" / "tpr"
    if not Path(tpr_package.__file__).resolve().is_relative_to(expected):
        raise RuntimeError(
            f"Wrong VERL package: {verl.__file__}; "
            f"TPR package: {tpr_package.__file__}; expected under {expected}"
        )
    import pytest
    raise SystemExit(pytest.main(["-q", "-s", "-x", str(Path(__file__).resolve())]))


import json
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron.tpr.adapter import run_tpr_forward_backward_batch
from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans

from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO
from .test_tpr_10k_parallel_parity_npu import _runtime, _engine


def _make_batch(length: int, suffix: int = 1024):
    """Two distinct synthetic responses sharing one token-exact prefix."""
    g = torch.Generator(device="cpu").manual_seed(20261118)
    prefix = torch.randint(100, 8000, (length - suffix,), generator=g)
    suffixes = [
        torch.randint(100, 8000, (suffix,), generator=g)
        for _ in range(2)
    ]

    def ragged(rows):
        return tu.nested_tensor_from_tensor_list(rows, ragged_idx=1)

    batch = TensorDict({
        "input_ids": ragged([torch.cat((prefix, s)) for s in suffixes]),
        "prompts": ragged([prefix.clone(), prefix.clone()]),
        "responses": ragged(suffixes),
        "response_mask": ragged([
            torch.ones(suffix, dtype=torch.bool) for _ in range(2)
        ]),
        "loss_mask": ragged([
            torch.ones(suffix, dtype=torch.bool) for _ in range(2)
        ]),
        "old_log_probs": ragged([
            torch.full((suffix,), -10.0) for _ in range(2)
        ]),
        "advantages": ragged([torch.ones(suffix) for _ in range(2)]),
        "temperature": torch.ones(2),
    }, batch_size=[2])
    keys = ("capacity_0_tpr_0", "capacity_1_tpr_1")
    tu.assign_non_tensor(
        batch,
        tpr_trajectory_keys=keys,
        dp_size=1,
        batch_num_tokens=2 * suffix,
        global_batch_size=2,
        tpr_capture_log_probs=False,
    )
    return batch, keys


def test_tpr_recompute_capacity():
    cp = int(os.environ["TPR_CAP_CP"])
    length = int(os.environ["TPR_CAP_LENGTH"])
    model_path = Path(os.getenv(
        "TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K",
    ))
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen3-4B-128K checkpoint missing: {model_path}")

    runtime = _runtime(tp=1, cp=cp, dp=1)
    try:
        from megatron.core.transformer.transformer_block import TransformerBlock
        from verl.workers.engine.megatron.tpr.recompute import (
            install_tpr_full_recompute,
            assert_tpr_full_recompute_installed,
        )
        from verl.workers.utils.losses import ppo_loss

        batch, keys = _make_batch(length)
        forest = build_tree_execution_plans(
            keys, batch,
            tree_policy="token_lcp",
            tree_builder="lcp_sort",
            min_shared_prefix_tokens=1024,
        )
        logical = sum(row.numel() for row in batch["input_ids"].unbind())
        unique = sum(
            segment.length
            for tree in forest.trees
            for segment in tree.segment_plan.segments.values()
        )
        if len(forest.trees) != 1 or forest.segment_count < 3 or unique >= logical:
            raise AssertionError(
                f"Not a shared TPR tree: trees={len(forest.trees)}, "
                f"segments={forest.segment_count}, "
                f"logical={logical}, unique={unique}"
            )

        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=model_path,
            tp_size=1, cp_size=cp,
            sequence_parallel=False, profile="4b", native_attention=False,
        )
        if length > int(hf.max_position_embeddings):
            raise ValueError(
                f"SEQ_LEN={length} exceeds model max_position_embeddings="
                f"{hf.max_position_embeddings}"
            )
        engine = _engine(model, runtime, 1, cp, 1)
        model.config.recompute_granularity = "full"
        model.config.recompute_method = "uniform"
        model.config.recompute_num_layers = 1

        blocks = [
            m for m in model.modules()
            if isinstance(m, TransformerBlock)
        ]
        installed = install_tpr_full_recompute(model, model.config)
        if not blocks or installed != len(blocks):
            raise AssertionError(
                f"Recompute installed on {installed}/{len(blocks)} blocks"
            )
        assert_tpr_full_recompute_installed(model, model.config)
        if dist.get_rank() == 0:
            print("CAP_PREFLIGHT " + json.dumps({
                "tp": 1, "cp": cp, "dp": 1,
                "seq_len": length, "prefix": length - 1024,
                "suffix": 1024, "branches": 2,
                "logical_tokens": logical, "unique_tokens": unique,
                "recompute": "full/uniform/1", "blocks": installed,
            }), flush=True)

        model.zero_grad(set_to_none=True)
        dist.barrier()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        begin = time.perf_counter()
        result = run_tpr_forward_backward_batch(
            engine, batch,
            partial(ppo_loss, config=_PPO()),
            forward_only=False,
        )
        torch.npu.synchronize()

        seconds = torch.tensor(
            time.perf_counter() - begin, device=runtime.device,
        )
        peaks = torch.tensor([
            torch.npu.max_memory_allocated(),
            torch.npu.max_memory_reserved(),
        ], dtype=torch.int64, device=runtime.device)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)

        missing = sum(
            p.requires_grad and p.grad is None for p in model.parameters()
        )
        missing = torch.tensor(missing, dtype=torch.int64, device=runtime.device)
        dist.all_reduce(missing, op=dist.ReduceOp.MAX)
        loss = float(result["loss"][0])
        if missing.item() or not torch.isfinite(torch.tensor(loss)):
            raise AssertionError(
                f"Missing gradients={missing.item()}, loss={loss}"
            )
        if dist.get_rank() == 0:
            print("CAP_RESULT " + json.dumps({
                "status": "PASS",
                "tp": 1, "cp": cp, "dp": 1,
                "seq_len": length,
                "recompute": "full/uniform/1",
                "fb_seconds_max_rank": round(seconds.item(), 3),
                "peak_allocated_gib_max_rank": round(peaks[0].item() / 2**30, 3),
                "peak_reserved_gib_max_rank": round(peaks[1].item() / 2**30, 3),
                "loss": loss,
                "optimizer_step": False,
            }), flush=True)
    finally:
        from megatron.core import parallel_state
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
