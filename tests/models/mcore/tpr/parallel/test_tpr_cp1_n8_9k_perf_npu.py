"""Qwen3-4B TP2×CP1 N8 P8192/S1024 controlled Full-vs-TPR PPO F/B.

Exactly 2 Ascend NPU ranks. Both modes use the SAME checkpoint, captured PPO
labels, model's TP2 weights, CANN CP1 attention and common minibatch. "full"
executes eight whole trajectories without a Prefix context; "tpr" runs one
qwen_turn tree through the production Executor. This is NOT the separate native
MindSpeed/Megatron F/B schedule, nor an optimizer/online training benchmark.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans
from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .prepare_tpr_cp1_n8_perf_capture import BRANCHES, PREFIX, SUFFIX
from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO, _load_real_pair
from .test_real_actor_cp1_short_full_tpr_npu import _run_controlled_full
from .test_tpr_10k_parallel_parity_npu import _runtime, _engine

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_CP1_N8_PERF") != "1",
    reason="Opt in to TP2×CP1 N8 9K controlled Full/TPR NPU performance",
)


def test_cp1_n8_p8192_s1024_perf():
    mode = os.getenv("TPR_CP1_N8_MODE")
    if mode not in ("full", "tpr"):
        raise ValueError("TPR_CP1_N8_MODE=full or tpr")
    if int(os.getenv("WORLD_SIZE", "1")) != 2:
        raise ValueError("CP1 performance needs exactly 2 ranks: TP2 CP1 DP1")

    warmups = int(os.getenv("TPR_BENCH_WARMUPS", "1"))
    repeats = int(os.getenv("TPR_BENCH_REPEATS", "3"))
    if not 0 <= warmups <= 5 or not 1 <= repeats <= 10:
        raise ValueError("Warmups must be 0..5; repeats must be 1..10")
    if os.getenv("TPR_EXEC_PROFILE", "0") != "0":
        raise ValueError("Disable segment profiler for timed measurements")
    if os.getenv("TPR_CP1_SAVE_FULL_GRADS", "0") != "0":
        raise ValueError("Full gradient snapshots cannot be enabled in performance runs")
    if os.getenv("TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS", "0") != "0":
        raise ValueError("Run performance only with Recompute OFF")
    capture = Path(os.environ["TPR_CP1_N8_CAPTURE"]).resolve()
    output = Path(os.environ["TPR_CP1_N8_OUTPUT_DIR"]).resolve()
    checkpoint = Path(os.getenv(
        "TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K",
    ))
    if not capture.is_file() or not capture.with_suffix(".audit.json").is_file():
        raise FileNotFoundError("Missing derived capture or its provenance audit")
    audit = json.loads(capture.with_suffix(".audit.json").read_text())
    if (audit.get("prefix"), audit.get("suffix"), audit.get("n")) != (
        PREFIX, SUFFIX, BRANCHES
    ):
        raise AssertionError("Data manifest does not describe N8 P8192/S1024")

    runtime = _runtime(2, 1, 1)
    try:
        from verl.workers.utils.losses import ppo_loss
        from verl.workers.engine.megatron.tpr.adapter import run_tpr_forward_backward_batch

        batch, keys, tokens, input_sha = _load_real_pair(
            capture, tuple(range(BRANCHES)), "full" if mode == "full" else "tpr",
        )
        if input_sha != audit["input_sha256"]:
            # Audit hash includes the exact same four fields as _load_real_pair;
            # any mismatch means the files do not belong to this benchmark.
            raise AssertionError("Derived capture input hash mismatches its audit")
        if tokens != audit["ppo_tokens"]:
            raise AssertionError("Captured supervised PPO token count mismatch")
        for row in batch["input_ids"].unbind(0):
            if int(row.numel()) != PREFIX + SUFFIX:
                raise AssertionError("Physical length differs from 9216")

        tu.assign_non_tensor(
            batch, dp_size=1, global_batch_size=BRANCHES,
            batch_num_tokens=tokens, tpr_capture_log_probs=False,
        )
        from verl.workers.engine.megatron.tpr.adapter import _trajectory_keys_from_minibatch
        forest = build_tree_execution_plans(
            _trajectory_keys_from_minibatch(batch),
            batch, tree_policy="qwen_turn", turn_min_segment_tokens=128,
        )
        if mode == "full":
            if len(forest.trees) != BRANCHES:
                raise AssertionError("Full accidentally reused a cross-row Prefix")
            if forest.segment_count != BRANCHES:
                raise AssertionError("Full must execute eight whole sequences")
        else:
            if (len(forest.trees) != 1 or forest.segment_count < BRANCHES+1
                    or forest.trees[0].segment_plan.get(0).length != PREFIX):
                raise AssertionError("TPR did not build one N8 shared 8192-token Root")
        if forest.logical_loss_tokens != tokens:
            raise AssertionError("Forest PPO token count differs from capture")

        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=2, cp_size=1,
            sequence_parallel=False, profile="4b", native_attention=False,
        )
        if hf.num_hidden_layers != 36:
            raise AssertionError("Expected original Qwen3-4B 36 layers")
        engine = _engine(model, runtime, 2, 1, 1)
        model.config.recompute_granularity = None
        model.config.recompute_method = None
        model.config.recompute_num_layers = None

        records = []
        for iteration in range(warmups+repeats):
            model.zero_grad(set_to_none=True)
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            if mode == "full":
                result = _run_controlled_full(
                    model, batch, loss_fn=partial(ppo_loss, config=_PPO()),
                    forest=forest,
                )
                # The CP1, DP1 grad finalize is intentionally empty. Do
                # NOT all-reduce the TP parameter shards.
                model.config.finalize_model_grads_func(
                    [model], None, force_all_reduce=True,
                )
                loss = float(result["loss"])
            else:
                tpr_result = run_tpr_forward_backward_batch(
                    engine, batch, partial(ppo_loss, config=_PPO()),
                    forward_only=False,
                )
                loss = float(tpr_result["loss"][0])
            torch.npu.synchronize()
            elapsed = time.perf_counter()-start
            # Report per-iteration rank-max, never rank0 alone.
            metrics = torch.tensor(
                [elapsed, float(torch.npu.max_memory_allocated()),
                 float(torch.npu.max_memory_reserved())],
                device=runtime.device, dtype=torch.float32,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            rec = {
                "iteration": iteration, "warmup": iteration < warmups,
                "critical_fwd_bwd_finalize_s": float(metrics[0]),
                "peak_allocated_gib": float(metrics[1])/2**30,
                "peak_reserved_gib": float(metrics[2])/2**30,
                "loss": loss,
            }
            if not all(map(lambda key: torch.isfinite(torch.tensor(rec[key])).item(),
                           ("critical_fwd_bwd_finalize_s", "loss"))):
                raise FloatingPointError("Nonfinite forward/backward performance result")
            if dist.get_rank() == 0:
                print(
                    f"TPR_CP1_N8_STEP mode={mode} iteration={iteration} "
                    f"warmup={int(iteration < warmups)} "
                    f"seconds={rec['critical_fwd_bwd_finalize_s']:.6f} "
                    f"peak_gib={rec['peak_allocated_gib']:.3f}", flush=True,
                )
                records.append(rec)
            dist.barrier()

        if dist.get_rank() == 0:
            output.mkdir(parents=True, exist_ok=True)
            file = output/f"{mode}.json"
            if file.exists():
                raise FileExistsError(f"Refusing to overwrite previous NPU result: {file}")
            measured = [row for row in records if not row["warmup"]]
            result = {
                "mode": mode,
                "baseline_kind": ("CONTROLLED_WHOLE_SQUARE_CANN_NO_TPR_CONTEXT"
                                  if mode == "full" else "PRODUCTION_TPR_SEGMENT_EXECUTOR_CP1"),
                "model": "Qwen3-4B-128K", "checkpoint": str(checkpoint),
                "topology": {"tp": 2, "cp": 1, "dp": 1, "world": 2},
                "branches": BRANCHES, "prefix": PREFIX, "suffix": SUFFIX,
                "length": PREFIX+SUFFIX, "source_capture": str(capture),
                "source_input_sha256": input_sha,
                "ppo_tokens": tokens,
                "forest_trees": len(forest.trees),
                "forest_segments": forest.segment_count,
                "condition_changed": audit["changed_conditioning"],
                "synthetic_shared_uid": audit["synthetic_shared_uid"],
                "recompute": False, "optimizer_step": False,
                "warmups": warmups, "repeats": repeats,
                "median_fwd_bwd_s": statistics.median(
                    r["critical_fwd_bwd_finalize_s"] for r in measured
                ),
                "worst_peak_allocated_gib": max(
                    r["peak_allocated_gib"] for r in measured
                ),
                "worst_peak_reserved_gib": max(
                    r["peak_reserved_gib"] for r in measured
                ),
                "runs": records,
            }
            file.write_text(json.dumps(result, indent=2))
            print(
                f"TPR_CP1_N8_SUMMARY mode={mode} "
                f"median_s={result['median_fwd_bwd_s']:.6f} "
                f"peak_gib={result['worst_peak_allocated_gib']:.3f} "
                f"output={file}", flush=True,
            )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
