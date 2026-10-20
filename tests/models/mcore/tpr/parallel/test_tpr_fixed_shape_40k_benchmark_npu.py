"""Isolated warm-shape Qwen3-4B Native Full vs TPR, same 40K capture.

For each original run5 online step: load the exact already-selected 8-Session
40K merged capture; load the SAME HF checkpoint independently per mode; run
1+ warmup F/B passes then repeated F/B with NO AdamW/weight updates.
Never use these as sequential PPO training steps or online reward evidence.
"""
from __future__ import annotations

import gc
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

from verl.workers.engine.megatron.tpr.adapter import run_tpr_forward_backward_batch
from verl.utils import tensordict_utils as tu
from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .native_full_megatron_schedule import run_native_full_step
from .test_real_actor_tp2cp4_multistep_npu import _make_engine
from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO, _runtime, _load_real_pair
from .test_tpr_10k_parallel_parity_npu import _runtime as _parallel_runtime, _engine as _parallel_engine

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_FIXED_SHAPE_40K_BENCH") != "1",
    reason="Explicit opt-in: 8-NPU fixed-capture forward/backward benchmark",
)


def test_fixed_shape_40k_one_mode():
    mode = os.environ["TPR_OFFLINE_MODE"]
    if mode not in ("full", "tpr"):
        raise ValueError("TPR_OFFLINE_MODE must be full or tpr")
    step = int(os.environ["TPR_BENCH_SOURCE_STEP"])
    controlled_path = os.getenv("TPR_BENCH_CONTROLLED_N8_AUDIT")
    cp = int(os.getenv("TPR_BENCH_CP_SIZE", "4"))
    if cp not in (2, 4) or (cp == 2 and not controlled_path):
        raise ValueError("CP2 is only supported for audited controlled N8 shapes")
    world = 2 * cp
    controlled = None
    if controlled_path:
        # Direct, explicitly marked performance-only derived capture. Keep
        # the original 40K/20-step manifest path untouched for production
        # historical regression tests.
        if step != 0:
            raise ValueError("The independent controlled shape benchmark uses step=0")
        controlled = json.loads(Path(controlled_path).read_text())
        p, s = int(controlled["prefix"]), int(controlled["suffix"])
        if (
            (cp, p, s) not in (
                (4, 13824, 13824), (4, 24576, 3072),
                (4, 18432, 18432), (4, 32768, 4096),
                (2, 16384, 2048), (2, 9216, 9216),
                (2, 6016, 6016), (2, 12416, 6016),
            )
            or controlled["n"] != 8 or controlled["length"] != p + s
            or controlled["forest_trees"] != 1
            or controlled["forest_segments"] < 9
            or not p - 127 <= controlled.get("actual_shared_root_tokens", -1) < p + s
            or controlled.get("min_turn_segment_tokens") != 128
            or not controlled["changed_conditioning"]
            or not controlled["synthetic_shared_uid"]
        ):
            raise AssertionError(f"CP{cp} controlled benchmark needs an audited eight-leaf 1:1 or 8:1 tree")
        item = {
            "capture": controlled["capture"],
            "ppo_tokens": controlled["ppo_tokens"],
            "effective_sequence_cap": p + s,
        }
    else:
        manifest_path = Path(os.environ["TPR_OFFLINE_CAPTURE_MANIFEST"])
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("protocol") != "real_actor_session_first_8merged_v3"
                or manifest.get("steps") != 20
                or manifest.get("expected_sessions") != 8
                or manifest.get("max_seq_len") != 40960
                or not 1 <= step <= 20):
            raise ValueError("Requires existing, already truncated, session-first 40K/20-step manifest")
        if (manifest.get("only_step_derived") not in (None, step)
                or manifest.get("experimental_sequence_cap", 40960) !=
                    manifest["items"][step - 1].get("effective_sequence_cap", 40960)):
            raise ValueError("Refuse to replay an unrelated or mismatched 30K derived step")
        item = manifest["items"][step - 1]
        if (item["step"] != step or item["source_global_step"] != step
                or item["rows"] != list(range(8)) or item.get("session_count") != 8
                or item["max_sequence_length"] > 40960):
            raise AssertionError("Source step or original eight-session selection changed")
    capture = Path(item["capture"])
    if not capture.is_file():
        raise FileNotFoundError(capture)
    warmups = int(os.environ.get("TPR_BENCH_WARMUPS", "1"))
    repeats = int(os.environ.get("TPR_BENCH_REPEATS", "3"))
    capture_logprobs = os.environ.get("TPR_BENCH_CAPTURE_LOGPROBS", "1") == "1"
    recompute_setting = os.getenv("TPR_BENCH_RECOMPUTE", "0")
    if recompute_setting not in ("0", "1"):
        raise ValueError("TPR_BENCH_RECOMPUTE must be 0 or 1")
    recompute_enabled = recompute_setting == "1"
    if os.getenv("TPR_BENCH_MEM_AUDIT", "0") not in ("0", "1"):
        raise ValueError("TPR_BENCH_MEM_AUDIT must be 0 or 1")
    mem_audit = os.getenv("TPR_BENCH_MEM_AUDIT", "0") == "1"
    ref_audit = os.getenv("TPR_RECOMPUTE_REF_AUDIT", "0") == "1"
    if ref_audit and not mem_audit:
        raise ValueError("TPR_RECOMPUTE_REF_AUDIT=1 requires TPR_BENCH_MEM_AUDIT=1")
    if not 0 <= warmups <= 10 or not 1 <= repeats <= 20:
        raise ValueError("Require 0..10 warmups and 1..20 measured repeats")
    # _qwen3_tp_checkpoint constructs native MCore recompute config before
    # creating TransformerBlock. This benchmark switch is the sole source
    # of truth: reject inherited native-recompute settings on unrelated jobs.
    if os.getenv("TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS", "0") not in ("0",):
        raise ValueError(
            "Unset TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS; "
            "use TPR_BENCH_RECOMPUTE=0/1 for both benchmark modes"
        )
    os.environ["TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS"] = (
        "1" if recompute_enabled and mode == "full" else "0"
    )
    output_dir = Path(os.environ["TPR_BENCH_OUTPUT_DIR"])
    checkpoint = Path(os.environ.get("TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"))
    runtime = _runtime() if cp == 4 else _parallel_runtime(2, cp, 1)
    # The Full reference needs the same supported FP32 collective gate as
    # the TPR DP2 benchmark; FP64 HCCL reductions cannot be trusted on 910B.
    probe = torch.tensor(
        float(dist.get_rank() + 1), dtype=torch.float32, device=runtime.device,
    )
    dist.all_reduce(probe, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    torch.npu.synchronize()
    expected_sum = float(world * (world + 1) // 2)
    if float(probe.item()) != expected_sum:
        raise AssertionError(
            f"Full WORLD FP32 AllReduce failed on rank {dist.get_rank()}: "
            f"rank_sum={probe.item()} expected={expected_sum}"
        )
    if dist.get_rank() == 0:
        print(f"TPR_FULL_WORLD_FP32_REDUCE_PASS rank_sum={expected_sum:g}", flush=True)
    try:
        from verl.workers.utils.losses import ppo_loss
        # Reuse existing original PPO-capture loader. Its original source_keys
        # are preserved: NO selection, no re-truncation, no reward re-labeling.
        batch, source_keys, tokens, data_sha = _load_real_pair(
            capture, tuple(range(8)), mode,
        )
        if tokens != item["ppo_tokens"]:
            raise AssertionError("Capture/manifest PPO token count mismatch")
        if controlled:
            if data_sha != controlled["input_sha256"]:
                raise AssertionError(f"CP{cp} controlled Full/TPR input SHA differs from audited capture")
            if any(int(row.numel()) != controlled["length"] for row in batch["input_ids"].unbind(0)):
                raise AssertionError(f"CP{cp} controlled replay must have eight exact audited-length rows")
        if max(int(x.numel()) for x in batch["input_ids"].unbind(0)) > int(
                item.get("effective_sequence_cap", 40960)):
            raise AssertionError("Capture exceeds its declared experimental sequence cap")
        if any(int(row.numel()) > 40960 for row in batch["input_ids"].unbind(0)):
            raise AssertionError("Encountered an untruncated row")
        if not source_keys or len(source_keys) != 8:
            raise AssertionError("Expected eight selected Sessions")
        tu.assign_non_tensor(batch, tpr_capture_log_probs=capture_logprobs)
        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=2, cp_size=cp,
            sequence_parallel=False, profile="4b", native_attention=(mode == "full"),
        )
        assert hf.num_hidden_layers == 36
        finalize_time = [0.0]
        engine = (
            (_make_engine(model, runtime, finalize_time) if cp == 4
             else _parallel_engine(model, runtime, 2, cp, 1))
            if mode == "tpr" else None
        )
        installed_blocks = 0
        if mode == "tpr" and recompute_enabled:
            from megatron.core.transformer.transformer_block import TransformerBlock
            from verl.workers.engine.megatron.tpr.recompute import (
                assert_tpr_full_recompute_installed,
                install_tpr_full_recompute,
            )
            model.config.recompute_granularity = "full"
            model.config.recompute_method = "uniform"
            model.config.recompute_num_layers = 1
            blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
            if not blocks or any(m.config is not model.config for m in blocks):
                raise AssertionError("TPR TransformerBlock config does not match model")
            installed_blocks = install_tpr_full_recompute(model, model.config)
            if installed_blocks != len(blocks):
                raise AssertionError("Not all TPR TransformerBlocks received the KV-aware adapter")
            assert_tpr_full_recompute_installed(model, model.config)
        elif not recompute_enabled:
            # Preserve the historical OFF baseline path. _make_engine also
            # clears this field for TPR; native starts disabled in its factory.
            model.config.recompute_granularity = None
            model.config.recompute_method = None
            model.config.recompute_num_layers = None

        expected = ("full", "uniform", 1) if recompute_enabled else (None, None, None)
        actual = (
            model.config.recompute_granularity,
            model.config.recompute_method,
            model.config.recompute_num_layers,
        )
        if actual != expected:
            raise AssertionError(
                f"{mode}: incorrect full/uniform/1 checkpoint config: {actual} != {expected}"
            )
        if dist.get_rank() == 0:
            print(
                f"TPR_FIXED_SHAPE_RECOMPUTE mode={mode} enabled={int(recompute_enabled)} "
                f"granularity={actual[0]} method={actual[1]} num_layers={actual[2]} "
                f"installed_tpr_blocks={installed_blocks} "
                f"native_checkpoint={int(mode == 'full' and recompute_enabled)}",
                flush=True,
            )
        if os.getenv("TPR_EXEC_PROFILE", "0") not in ("0", "1"):
            raise ValueError("TPR_EXEC_PROFILE must be 0 or 1")
        from .grad_sync_precision import offline_grad_sync_dtype
        grad_sync_dtype = offline_grad_sync_dtype()
        if dist.get_rank() == 0:
            print(
                f"TPR_FIXED_SHAPE_GRAD_REDUCE_MODE mode={mode} "
                f"dtype={grad_sync_dtype} cp_group_size={cp}",
                flush=True,
            )
        runs = []
        for iteration in range(warmups + repeats):
            model.zero_grad(set_to_none=True)
            # Explicitly sync before and after the measured F/B as in the
            # original 20-step replay. Timer EXCLUDES gradnorm/optimizer.
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            if mode == "full":
                out_full = run_native_full_step(
                    model, batch, runtime, partial(ppo_loss, config=_PPO()),
                    capture_logprobs=capture_logprobs,
                )
                loss = float(out_full.loss)
            else:
                out = run_tpr_forward_backward_batch(
                    engine, batch, partial(ppo_loss, config=_PPO()),
                    forward_only=False,
                )
                loss = float(out["loss"][0])
            torch.npu.synchronize()
            own_seconds = time.perf_counter() - start
            # FP64 HCCL AllReduce is unsupported on Ascend 910B.
            # Use FP32 for world MAX time and memory statistics.
            critical = torch.tensor(
                [own_seconds], dtype=torch.float32, device=runtime.device,
            )
            dist.all_reduce(
                critical, op=dist.ReduceOp.MAX, group=dist.group.WORLD,
            )
            peak = torch.tensor(
                [float(torch.npu.max_memory_allocated())], dtype=torch.float32,
                device=runtime.device,
            )
            dist.all_reduce(
                peak, op=dist.ReduceOp.MAX, group=dist.group.WORLD,
            )

            # AFTER timing: profiler materialization and optional per-rank
            # snapshots, never included in the F/B critical path.
            if mode == "tpr" and os.getenv("TPR_EXEC_PROFILE", "0") == "1":
                profiler = engine._tpr_segment_event_profiler
                events = profiler.materialize()
                stage = dict(engine._tpr_stage_profile)
            else:
                events, stage = [], {}
            profile = {
                "rank": dist.get_rank(),
                "stage_ms": stage,
                "events": [{
                    "kind": e["kind"], "segment": e["segment_id"],
                    "length": e["length"], "prefix": e["prefix_length"],
                    "npu_event_ms": e["npu_event_ms"],
                    "cpu_enqueue_ms": e["cpu_enqueue_ms"],
                } for e in events],
            }
            profiles = [None] * dist.get_world_size()
            if os.getenv("TPR_EXEC_PROFILE", "0") == "1" and mode == "tpr":
                dist.all_gather_object(profiles, profile)
            else:
                profiles = []
            record = {
                "source_step": step,
                "mode": mode,
                "iteration": iteration,
                "warmup": iteration < warmups,
                "critical_fwd_bwd_s": float(critical.item()),
                "peak_allocated_gib": float(peak.item()) / 2**30,
                "loss": loss,
                "input_sha256": data_sha,
                "ppo_tokens": tokens,
                "recompute": recompute_enabled,
                "recompute_method": model.config.recompute_method,
                "recompute_num_layers": model.config.recompute_num_layers,
                "recompute_installed_tpr_blocks": installed_blocks,
                "updated_weights": False,
                "profiler_enabled": os.getenv("TPR_EXEC_PROFILE", "0") == "1",
                "mem_audit": mem_audit,
                "capture_logprobs": capture_logprobs,
                "profiles": profiles,
            }
            if dist.get_rank() == 0:
                print(
                    f"TPR_FIXED_SHAPE step={step} mode={mode} recompute={int(recompute_enabled)} "
                    f"iter={iteration} warmup={int(iteration < warmups)} "
                    f"critical_fwd_bwd_s={record['critical_fwd_bwd_s']:.6f} "
                    f"peak_gib={record['peak_allocated_gib']:.3f} "
                    f"loss={loss:.9g} sha256={data_sha[:16]}",
                    flush=True,
                )
                runs.append(record)
            # Only opt-in diagnostics change the sequence of operations.
            # gc.collect is intentionally outside the timed forward/backward:
            # if allocated memory drops here, tensors were kept in Python
            # reference cycles rather than being required by the next step.
            if mem_audit:
                torch.npu.synchronize()
                live_before = torch.npu.memory_allocated()
                collected = gc.collect()
                torch.npu.synchronize()
                live_after = torch.npu.memory_allocated()
                audit = torch.tensor(
                    [live_before / 2**30, live_after / 2**30],
                    dtype=torch.float32, device=runtime.device,
                )
                dist.all_reduce(audit, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
                ref_counts = None
                if ref_audit:
                    from verl.workers.engine.megatron.tpr.recompute import (
                        tpr_recompute_live_ref_counts,
                    )
                    refs = tpr_recompute_live_ref_counts()
                    ref_max = torch.tensor(
                        [refs["checkpoint_run_functions"], refs["attention_contexts"]],
                        dtype=torch.float32, device=runtime.device,
                    )
                    dist.all_reduce(ref_max, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
                    ref_counts = {
                        "checkpoint_run_functions": int(ref_max[0].item()),
                        "attention_contexts": int(ref_max[1].item()),
                    }
                if dist.get_rank() == 0:
                    record["live_alloc_before_gc_gib_worst"] = float(audit[0].item())
                    record["live_alloc_after_gc_gib_worst"] = float(audit[1].item())
                    if ref_counts is not None:
                        record["live_recompute_refs"] = ref_counts
                    print(
                        f"TPR_FIXED_SHAPE_MEM_AUDIT mode={mode} "
                        f"recompute={int(recompute_enabled)} iter={iteration} "
                        f"live_before_gc_gib={float(audit[0]):.3f} "
                        f"live_after_gc_gib={float(audit[1]):.3f} "
                        f"live_recompute_refs={ref_counts} "
                        f"local_collected_objects={collected}",
                        flush=True,
                    )
            dist.barrier()
        gradient_witness = None
        if os.getenv("TPR_BENCH_GRAD_WITNESS", "0") == "1":
            from .gradient_witness import collect_tp2_gradient_witness
            gradient_witness = collect_tp2_gradient_witness(
                model, runtime, mode=mode,
                require_exact_replicas=(mode == "tpr"),
            )
        if dist.get_rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            suffix = "_recompute_on" if recompute_enabled else ""
            target = output_dir / f"step{step}_{mode}{suffix}_fixed_shape.json"
            if target.exists():
                raise FileExistsError(f"Refuse to overwrite benchmark: {target}")
            measured = [r["critical_fwd_bwd_s"] for r in runs if not r["warmup"]]
            target.write_text(json.dumps({
                "source_step": step, "mode": mode,
                "controlled_performance_shape": (
                    {"prefix": controlled["prefix"], "suffix": controlled["suffix"],
                     "length": controlled["length"],
                     "actual_shared_root_tokens": controlled["actual_shared_root_tokens"],
                     "unique_tree_tokens": controlled["unique_tree_tokens"],
                     "synthetic_shared_uid": controlled["synthetic_shared_uid"],
                     "changed_conditioning": controlled["changed_conditioning"],
                     "original_capture": controlled["source"]}
                    if controlled else None
                ),
                "recompute": recompute_enabled,
                "recompute_granularity": model.config.recompute_granularity,
                "recompute_method": model.config.recompute_method,
                "recompute_num_layers": model.config.recompute_num_layers,
                "recompute_installed_tpr_blocks": installed_blocks,
                "topology": {"tp": 2, "cp": cp, "dp": 1, "world": world},
                "effective_sequence_cap": int(item.get("effective_sequence_cap", 40960)),
                "input_sha256": data_sha, "source_capture": str(capture),
                "checkpoint": str(checkpoint), "warmups": warmups,
                "repeats": repeats, "updated_weights": False,
                "gradient_sync_reduce_dtype": grad_sync_dtype,
                "hccl_deterministic": os.getenv("HCCL_DETERMINISTIC", "unset"),
                "hccl_stats_reduce_dtype": "float32",
                "capture_logprobs": capture_logprobs,
                "profiler_enabled": os.getenv("TPR_EXEC_PROFILE", "0") == "1",
                "median_fwd_bwd_s": statistics.median(measured),
                "gradient_witness": gradient_witness,
                "runs": runs,
            }, indent=2))
            print(
                f"TPR_FIXED_SHAPE_SUMMARY step={step} mode={mode} "
                f"recompute={int(recompute_enabled)} "
                f"median_fwd_bwd_s={statistics.median(measured):.6f} "
                f"n={len(measured)} output={target}", flush=True,
            )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
