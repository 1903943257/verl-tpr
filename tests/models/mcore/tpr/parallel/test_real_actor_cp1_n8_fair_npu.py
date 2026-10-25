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
    _PPO, _load_real_pair, _global_bf16_gradient_norm, _save_complete_grads,
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


def _make_unshared_split_n8(shared_forest, batch):
    """Recreate the exact Root/Leaf boundary per row, without sharing any Root."""
    from verl.workers.engine.megatron.tpr.tree import (
        ForestExecutionPlan, SegmentObjectiveRef, SegmentPlan, SegmentSpec,
        TreeExecutionPlan,
    )

    if len(shared_forest.trees) != 1:
        raise AssertionError("N8 split attribution needs one original shared tree")
    original = shared_forest.trees[0]
    plan = original.segment_plan
    root = plan.get(plan.root_id)
    children = plan.children_of(plan.root_id)
    rows = tuple(sorted(original.member_rows))
    if len(rows) != 8 or len(children) != 8:
        raise AssertionError(
            f"Expected one Root with eight sibling Leaves, rows={rows}, children={len(children)}"
        )
    boundary = root.length
    independent = []
    for row in rows:
        tokens = batch["input_ids"][row].detach().cpu().long()
        if not 0 < boundary < tokens.numel():
            raise AssertionError(f"Invalid root boundary for row {row}: {boundary}")
        if not torch.equal(tokens[:boundary], root.token_ids):
            raise AssertionError(f"Row {row} does not have the exact shared root prefix")
        local_plan = SegmentPlan(
            (
                SegmentSpec(0, None, tokens[:boundary], 0, 0),
                SegmentSpec(1, 0, tokens[boundary:], boundary, boundary),
            ),
            root_id=0,
        )
        refs = []
        for ref in original.objective_refs:
            if ref.sample_row != row:
                continue
            source_segment = plan.get(ref.segment_id)
            global_query = source_segment.position_start + ref.query_offset
            target_global = global_query + 1
            if target_global >= tokens.numel() or int(tokens[target_global]) != ref.target_token_id:
                raise AssertionError(
                    f"Split objective target mismatch row={row}, query={global_query}"
                )
            local_segment_id = 0 if global_query < boundary else 1
            local_query = global_query if local_segment_id == 0 else global_query - boundary
            refs.append(SegmentObjectiveRef(
                local_segment_id, local_query, ref.target_token_id,
                ref.sample_row, ref.response_offset,
            ))
        if not refs:
            raise AssertionError(f"Split tree row {row} has no supervised targets")
        independent.append(TreeExecutionPlan(
            uid=f"{original.uid}_split_no_share_{row}",
            tree_index=row,
            member_rows=(row,),
            segment_plan=local_plan,
            objective_refs=tuple(refs),
        ))
    split = ForestExecutionPlan(tuple(independent), shared_forest.logical_loss_tokens)
    split_tokens = sum(
        segment.length for tree in split.trees
        for segment in tree.segment_plan.segments.values()
    )
    logical_tokens = sum(int(ids.numel()) for ids in batch["input_ids"].unbind())
    if len(split.trees) != 8 or split.segment_count != 16 or split_tokens != logical_tokens:
        raise AssertionError(
            f"Invalid split-no-share topology: trees={len(split.trees)}, "
            f"segments={split.segment_count}, physical={split_tokens}, logical={logical_tokens}"
        )
    return split


def _run_split_no_share(model, engine, batch, loss_fn, forest):
    """Run independent Root/Leaf chains through the same production SegmentExecutor."""
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.workers.engine.megatron.tpr.adapter import run_forest
    from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
    from verl.workers.engine.megatron.tpr.objective import SegmentPPOObjectiveAdapter

    chunk_size = int(os.getenv("TPR_PPO_LOGPROB_CHUNK_TOKENS", "256"))
    if chunk_size < 0:
        raise ValueError("Negative PPO logprob chunk size")
    objective = SegmentPPOObjectiveAdapter(
        batch, loss_fn, log_prob_chunk_size=chunk_size or None,
    )
    layers = tuple(sorted(
        module.layer_number for module in model.modules()
        if isinstance(module, TPRSelfAttention)
    ))
    if len(layers) != 36:
        raise AssertionError("N8 split attribution requires all 36 TPR attention layers")
    model_config = model.config
    if model_config.recompute_granularity is not None:
        raise AssertionError("N8 split attribution requires recompute disabled")
    stats = {"model_forward_calls": 0, "grouped_leaf_calls": 0, "grouped_leaf_segments": 0}

    def run_one_tree(tree):
        callback, counts = objective.bind_tree(tree)
        executor = SegmentExecutor(
            model,
            tree.segment_plan,
            expected_layer_numbers=layers,
            loss_scale_func=getattr(model_config, "grad_scale_func", None),
            segment_loss_fn=callback,
            segment_loss_term_counts=counts,
        )
        value = executor.run()
        for key in stats:
            stats[key] += int(getattr(executor, key))
        return value

    total_loss = run_forest(
        forest,
        run_one_tree,
        lambda: model_config.finalize_model_grads_func(
            engine.module, None, force_all_reduce=True,
        ),
    )
    return {
        "loss": float(total_loss.detach().item()),
        "logprobs": dict(objective.debug_new_log_probs),
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
    if mode not in ("full_mb1", "full_mb2", "split_no_share", "tpr"):
        raise ValueError("N8 benchmark supports full_mb1/full_mb2/split_no_share/tpr")
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
            cap, tuple(range(8)), "tpr" if mode in ("tpr", "split_no_share") else "full",
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
        if mode in ("tpr", "split_no_share"):
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
            if mode == "split_no_share":
                forest = _make_unshared_split_n8(forest, batch)
                unique = sum(
                    seg.length for tree in forest.trees
                    for seg in tree.segment_plan.segments.values()
                )
                print(
                    "TPR_CP1_N8_SPLIT_NO_SHARE "
                    + json.dumps({
                        "trees": len(forest.trees), "segments": forest.segment_count,
                        "logical_tokens": logical, "physical_tokens": unique,
                        "root_length": root_length,
                    }),
                    flush=True,
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
            elif mode == "split_no_share":
                result = _run_split_no_share(model, engine, batch, loss_fn, forest)
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
        configured_k = int(os.getenv("TPR_LEAF_BATCH_SIZE", "1"))
        full_parameter_grad_dir = None
        if os.getenv("TPR_CP1_N8_SAVE_FULL_GRADS", "0") == "1":
            if mode == "tpr" and configured_k not in (1, 2):
                raise ValueError("TreeBatch parity full-gradient dumps support K=1/2 only")
            # Opt-in attribution diagnostic: save exact BF16 parameter gradients
            # for Full, split-no-share, TPR K1 and TPR K2, outside the timed region.
            grad_arm = f"tpr_k{configured_k}" if mode == "tpr" else mode
            full_parameter_grad_dir = _save_complete_grads(
                model, runtime, destination.parent, grad_arm, step=1,
            )
            if full_parameter_grad_dir is None:
                raise AssertionError("Full gradient dump unexpectedly skipped this CP rank")
            print(
                "TPR_CP1_N8_FULL_GRADS "
                + json.dumps({"K": configured_k, "directory": full_parameter_grad_dir}),
                flush=True,
            )
        report = {
            "mode": mode, "configured_k": configured_k,
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
            "full_parameter_grad_dir": full_parameter_grad_dir,
            "optimizer_step": False, "original_online_conditioning": False,
            "reference": (
                f"physical_B{1 if mode == 'full_mb1' else 2}_square_CANN"
                if mode in ("full_mb1", "full_mb2") else
                "TPR_CP1_independent_root_leaf_chains"
                if mode == "split_no_share" else "TPR_CP1_root_leaf"
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
