"""N=8 Qwen3-4B true captured PPO, controlled CP1 Full MB1/MB2 vs TPR K1/2/4.

Each arm loads same original eight captured distinct trajectories and model
weights. Full MB1 uses eight physical [1,S] forwards; Full MB2 uses four
physical [2,S] square-CANN forwards; TPR runs a
single eight-trajectory turn-tree forest, including Root Push/Pop. No
production VERL scheduler, optimizer, rollout, CP, TP, or activation recompute.
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
from verl.workers.engine.megatron.tpr.adapter import (
    _trajectory_keys_from_minibatch, run_tpr_forward_backward_batch,
)
from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans
from verl.workers.engine.megatron.tpr.objective import SegmentPPOObjectiveAdapter

from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .test_real_actor_cp1_short_full_tpr_npu import _initial_weight_fingerprint
from .test_real_actor_tp2cp4_offline_ppo_npu import (
    _PPO, _load_real_pair, _global_bf16_gradient_norm,
)
from .test_tpr_10k_parallel_parity_npu import _runtime, _engine
from .tpr_10k_canonical_witness import canonical_grad_witness

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_CP1_N8_FAIR") != "1",
    reason="Explicitly opt into isolated N8 real Qwen3-4B NPU PPO benchmark",
)


def _full_physical_batch(model, batch, forest, loss_fn, *, micro_batch_size: int):
    """Eight original trajectories in physical B=1 or B=2 full-sequence forwards.

    Both modes use identical whole-trajectory square-CANN and PPO denominator.
    Changing micro_batch_size changes only the physical GPT batch size.
    """
    if micro_batch_size not in (1, 2):
        raise ValueError("Only physical Full MB=1/2 controls are supported")
    roots = []
    for tree in forest.trees:
        if len(tree.segment_plan.segments) != 1:
            raise AssertionError("Full requires one independent root per row")
        root = tree.segment_plan.get(tree.segment_plan.root_id)
        roots.append((tree, root))
    if len(roots) != 8:
        raise AssertionError("Physical Full baseline requires exactly 8 real distinct rows")
    lengths = {root.length for _, root in roots}
    if len(lengths) != 1:
        raise AssertionError(f"Physical Full MB1/MB2 refuses unequal lengths: {lengths}")
    length = lengths.pop()
    objective = SegmentPPOObjectiveAdapter(
        batch, loss_fn,
        log_prob_chunk_size=int(os.getenv("TPR_PPO_LOGPROB_CHUNK_TOKENS", "256")) or None,
    )
    losses = []
    physical_calls = 0
    for offset in range(0, len(roots), micro_batch_size):
        chunk = roots[offset:offset + micro_batch_size]
        ids = torch.stack([segment.token_ids for _, segment in chunk]).to(
            device=next(model.parameters()).device
        )
        positions = torch.arange(length, device=ids.device, dtype=torch.long).unsqueeze(0).expand(micro_batch_size, -1)
        logits = model(input_ids=ids, position_ids=positions, attention_mask=None)
        if tuple(logits.shape[:2]) != (micro_batch_size, length):
            raise AssertionError(f"Full MB={micro_batch_size} did not execute physical [{micro_batch_size},S] GPT")
        physical_calls += 1
        chunk_losses = []
        for j, (tree, segment) in enumerate(chunk):
            callback, counts = objective.bind_tree(tree)
            if counts[segment.segment_id] != len(tree.refs_for_segment(segment.segment_id)):
                raise AssertionError("Full PPO row ownership mismatch")
            loss, _ = callback(segment, logits[j:j + 1])
            chunk_losses.append(loss)
        chunk_loss = torch.stack(chunk_losses).sum()
        chunk_loss.backward()
        losses.append(chunk_loss.detach())
        del logits, chunk_loss
    return {
        "loss": float(torch.stack(losses).sum().item()),
        "logprobs": dict(objective.debug_new_log_probs),
        "physical_model_forward_calls": physical_calls,
        "leaf_batch_stats": None,
    }


def _tpr(model, engine, batch, loss_fn):
    output = run_tpr_forward_backward_batch(engine, batch, loss_fn, forward_only=False)
    metrics = output["metrics"]
    stats = {
        key: int(metrics[f"tpr/{key}"][0])
        for key in ("model_forward_calls", "grouped_leaf_calls", "grouped_leaf_segments")
    }
    configured_k = int(os.getenv("TPR_LEAF_BATCH_SIZE", "1"))
    if configured_k == 1 and stats["grouped_leaf_calls"]:
        raise AssertionError("K1 unexpectedly grouped Leaves")
    if configured_k > 1 and not stats["grouped_leaf_calls"]:
        raise AssertionError(
            f"Configured K={configured_k} but NO real-model sibling leaf groups. "
            "Inspect N8 Turn-tree topology / shared-prefix branch positions."
        )
    return {
        "loss": float(output["loss"][0]),
        "logprobs": dict(engine._tpr_captured_log_probs),
        "physical_model_forward_calls": stats["model_forward_calls"],
        "leaf_batch_stats": stats,
    }


def _summarize_stages(records: list[dict]) -> dict:
    by_kind = {}
    for rec in records:
        kind = rec["kind"]
        stat = by_kind.setdefault(
            kind, {"calls": 0, "npu_event_ms": 0., "cpu_enqueue_ms": 0.,
                   "total_segment_tokens": 0, "groups": []}
        )
        stat["calls"] += 1
        stat["npu_event_ms"] += rec["npu_event_ms"]
        stat["cpu_enqueue_ms"] += rec["cpu_enqueue_ms"]
        stat["total_segment_tokens"] += int(rec["length"]) * int(rec.get("group_size", 1))
        stat["groups"].append({
            "batch": int(rec.get("group_size", 1)), "length": int(rec["length"]),
            "npu_event_ms": round(rec["npu_event_ms"], 4),
            "cpu_enqueue_ms": round(rec["cpu_enqueue_ms"], 4),
        })
    return by_kind


def test_real_actor_cp1_n8_full_mb2_vs_tpr():
    mode = os.getenv("TPR_CP1_N8_MODE", "")
    if mode not in ("full_mb1", "full_mb2", "tpr"):
        raise ValueError("N8 benchmark only supports physical full_mb1/full_mb2 or TPR")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise AssertionError("N8 CP1 benchmark requires exactly one NPU")
    if os.getenv("TPR_TREE_POLICY", "qwen_turn") != "qwen_turn":
        raise ValueError("N8 benchmark must use production Qwen turn-tree policy")
    cap = Path(os.environ["TPR_CP1_N8_CAPTURE"]).resolve()
    destination = Path(os.environ["TPR_CP1_N8_OUTPUT"]).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    ckpt = Path(os.getenv(
        "TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"
    )).resolve()
    if not cap.is_file() or not ckpt.is_dir():
        raise FileNotFoundError(f"Missing capture/checkpoint: {cap}, {ckpt}")
    warmups = int(os.getenv("TPR_CP1_N8_WARMUPS", "1"))
    repeats = int(os.getenv("TPR_CP1_N8_REPEATS", "3"))
    stage_profile = os.getenv("TPR_CP1_N8_STAGE_PROFILE", "0") == "1"
    if warmups < 0 or repeats < 1 or stage_profile and (warmups, repeats) != (0, 1):
        raise ValueError("Stage probe requires 0 warmups and 1 iteration; bench requires repeats>=1")
    if stage_profile != (os.getenv("TPR_EXEC_PROFILE", "0") == "1"):
        raise AssertionError("Separate stage profiling from fair F/B timing via TPR_EXEC_PROFILE")
    if stage_profile and mode != "tpr":
        raise ValueError("Stage event profiler is defined on TPR Forest only")

    runtime = _runtime(1, 1, 1)
    try:
        from verl.workers.utils.losses import ppo_loss
        batch, original_keys, ppo_tokens, input_sha = _load_real_pair(
            cap, tuple(range(8)), "tpr" if mode == "tpr" else "full",
        )
        tu.assign_non_tensor(
            batch, dp_size=1, batch_num_tokens=ppo_tokens, global_batch_size=8,
            tpr_capture_log_probs=True,
        )
        keys = _trajectory_keys_from_minibatch(batch)
        forest = build_tree_execution_plans(keys, batch, tree_policy="qwen_turn",
                                          turn_min_segment_tokens=128)
        logical = sum(int(ids.numel()) for ids in batch["input_ids"].unbind())
        unique = sum(seg.length for tree in forest.trees for seg in tree.segment_plan.segments.values())
        input_lengths = [int(ids.numel()) for ids in batch["input_ids"].unbind()]
        if len(set(input_lengths)) != 1 or max(input_lengths) > 4096:
            raise AssertionError("N8 controlled Full MB2 requires equal <=4K genuine response windows")
        root_length = None
        sibling_suffix_lengths = []
        if mode == "tpr":
            if len(forest.trees) != 1 or unique >= logical:
                raise AssertionError(
                    f"TPR N8 must have one actually shared same-session tree; "
                    f"trees={len(forest.trees)}, unique={unique}, logical={logical}"
                )
            root_plan = forest.trees[0].segment_plan
            root_segment = root_plan.get(root_plan.root_id)
            root_length = root_segment.length
            sibling_suffix_lengths = [
                item.length for item in root_plan.children_of(root_plan.root_id)
            ]
            print(
                "TPR_CP1_N8_TREE " + json.dumps({
                    "root_length": root_length,
                    "direct_child_lengths": sibling_suffix_lengths,
                    "logical_rows": 8, "trees": len(forest.trees),
                    "segments": forest.segment_count,
                    "logical_tokens": logical, "unique_tokens": unique,
                    "dedup_fraction": 1 - unique/logical,
                    "segments_info": [
                        {"id": seg.segment_id, "parent": seg.parent_id,
                         "len": seg.length, "prefix": seg.prefix_length}
                        for seg in forest.trees[0].segment_plan.segments.values()
                    ],
                }), flush=True,
            )
        else:
            if len(forest.trees) != 8 or unique != logical:
                raise AssertionError("Full MB1/MB2 N8 must use eight unrelated complete roots")
        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=ckpt, tp_size=1, cp_size=1,
            sequence_parallel=False, profile="4b", native_attention=False,
        )
        if hf.num_hidden_layers != 36 or model.config.recompute_granularity is not None:
            raise AssertionError("Requires genuine Qwen3-4B with recompute off")
        engine = _engine(model, runtime, 1, 1, 1)
        initial_weights = _initial_weight_fingerprint(model)
        loss_fn = partial(ppo_loss, config=_PPO())
        timings, allocs, reserves, stages = [], [], [], []
        result = None
        for i in range(-warmups, repeats):
            model.zero_grad(set_to_none=True)
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            begin = time.perf_counter()
            if mode in ("full_mb1", "full_mb2"):
                result = _full_physical_batch(
                    model, batch, forest, loss_fn,
                    micro_batch_size=1 if mode == "full_mb1" else 2,
                )
            else:
                result = _tpr(model, engine, batch, loss_fn)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - begin
            max_alloc = int(torch.npu.max_memory_allocated())
            max_reserved = int(torch.npu.max_memory_reserved())
            if stage_profile:
                recorder = engine._tpr_segment_event_profiler
                if recorder is None:
                    raise AssertionError("TPR stage event profiler was not installed")
                stage_events = recorder.materialize()
                stages.append(_summarize_stages(stage_events))
                max_alloc = max(max_alloc, recorder.max_allocated_bytes)
                max_reserved = max(max_reserved, recorder.max_reserved_bytes)
            if i >= 0:
                timings.append(elapsed)
                allocs.append(max_alloc / 2**30)
                reserves.append(max_reserved / 2**30)
                print(
                    f"TPR_CP1_N8_ITER mode={mode} K={os.getenv('TPR_LEAF_BATCH_SIZE','1')} "
                    f"repetition={i} fwd_bwd_s={elapsed:.6f} "
                    f"max_alloc_gib={allocs[-1]:.3f} max_reserved_gib={reserves[-1]:.3f}",
                    flush=True,
                )
        assert result is not None
        expected_logprob_keys = {
            (row, int(offset))
            for row in range(8)
            for offset in torch.nonzero(batch["response_mask"][row].bool(), as_tuple=False).flatten().tolist()
        }
        if result["logprobs"].keys() != expected_logprob_keys or len(expected_logprob_keys) != ppo_tokens:
            raise AssertionError("N8 Full/TPR did not cover the same genuine PPO loss tokens")
        grad_norm = _global_bf16_gradient_norm(model, runtime)
        witness = canonical_grad_witness(
            model, tp_size=1, tp_rank=0, cp_rank=0, dp_rank=0, device=runtime.device,
        )
        if witness is None:
            raise AssertionError("N8 missing canonical gradient witness")
        report = {
            "mode": mode, "configured_k": int(os.getenv("TPR_LEAF_BATCH_SIZE", "1")),
            "physical_full_micro_batch": (
                1 if mode == "full_mb1" else 2 if mode == "full_mb2" else None
            ),
            "input_sha256": input_sha, "checkpoint": str(ckpt),
            "source_capture": str(cap), "original_keys": list(original_keys),
            "initial_weights_witness_sha256": initial_weights,
            "logical_trajectories": 8, "input_lengths": input_lengths,
            "ppo_tokens": ppo_tokens,
            "logical_input_tokens": logical, "unique_tree_tokens": unique,
            "token_dedup_fraction": 1 - unique / logical,
            "forest_trees": len(forest.trees), "forest_segments": forest.segment_count,
            "root_prefix_tokens": root_length, "direct_child_segment_lengths": sibling_suffix_lengths,
            "physical_model_forward_calls": result["physical_model_forward_calls"],
            "leaf_batch_stats": result["leaf_batch_stats"],
            "bench_warmups": warmups, "bench_repeats": repeats,
            "bench_times_s": timings, "fwd_bwd_seconds": statistics.median(timings),
            "bench_allocs_gib": allocs, "bench_reserved_gib": reserves,
            "peak_allocated_gib": max(allocs),
            "peak_reserved_gib": max(reserves),
            "stage_profile": stage_profile, "stages_by_iteration": stages,
            "loss": result["loss"], "logprobs": sorted(
                [row, off, float(value)]
                for (row, off), value in result["logprobs"].items()
            ),
            "global_grad_norm": grad_norm, "gradient_witness": witness,
            "optimizer_step": False, "original_online_conditioning": False,
            "reference": (
                f"physical_B{1 if mode == 'full_mb1' else 2}_square_CANN"
                if mode != "tpr" else "TPR_CP1_root_leaf"
            ),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2))
        print(
            "TPR_CP1_N8_RESULT "
            + json.dumps({
                "mode": mode, "K": report["configured_k"],
                "speed_median_s": report["fwd_bwd_seconds"],
                "memory_peak_alloc_gib": report["peak_allocated_gib"],
                "memory_peak_reserved_gib": report["peak_reserved_gib"],
                "logical_rows": report["logical_trajectories"],
                "dedup_pct": round(report["token_dedup_fraction"] * 100, 2),
                "root_prefix_tokens": root_length, "direct_child_lengths": sibling_suffix_lengths,
                "forward_calls": report["physical_model_forward_calls"],
                "grouped": report["leaf_batch_stats"],
                "stage": stages[0] if stages else None,
                "result": str(destination),
            }, sort_keys=True),
            flush=True,
        )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
