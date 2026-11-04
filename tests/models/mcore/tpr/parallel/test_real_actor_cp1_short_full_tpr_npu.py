"""One NPU: controlled Full Sequence vs one-node TPR vs shared-prefix TPR.

All modes load the SAME Qwen3-4B BF16 checkpoint and the SAME cropped Actor
capture, original old_log_probs/advantages/temperature, and VERL PPO objective.

Full: call GPTModel on each whole trajectory WITHOUT a TPRAttentionContext.
      TPRSelfAttention delegates to MCore SelfAttention with the existing
      CP1 square-CANN oracle (NOT an independent native VERL Engine claim).
One-node: production SegmentExecutor, one Root leaf per trajectory, no reuse.
Shared: production qwen_turn Root Push -> two Leaves -> Root Pop.

No CP, TP, recompute, Fixed Tile, optimizer, or invented labels. The cropped
source has altered conditioning and is only a numerical parity control.
"""
from __future__ import annotations

import hashlib
import json
import os
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
from .test_real_actor_tp2cp4_offline_ppo_npu import (
    _PPO, _load_real_pair, _global_bf16_gradient_norm,
)
from .test_tpr_10k_parallel_parity_npu import _runtime, _engine
from .tpr_10k_canonical_witness import canonical_grad_witness

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_CP1_SHORT_ACTOR") != "1",
    reason="Set TPR_RUN_CP1_SHORT_ACTOR=1 for opt-in one-NPU Actor PPO parity",
)

MODES = ("full", "full_batch2", "one_node", "split_no_share", "shared")


@torch.no_grad()
def _initial_weight_fingerprint(model) -> str:
    """Small deterministic identity witness; not a full parameter checksum."""
    digest = hashlib.sha256()
    from ._sample_indices import exact_sample_indices

    for name, param in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(param.shape)).encode())
        indices = torch.tensor(
            exact_sample_indices(param.numel(), 64),
            dtype=torch.long, device=param.device,
        )
        values = param.detach().reshape(-1).index_select(0, indices)
        digest.update(values.float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _make_unshared_split(shared_forest, batch):
    """Clone the exact shared Root/Leaf boundary for each logical sibling.

    No KV is shared between rows. Objectives keep identical target tokens,
    global predicting-query positions, logical row IDs and PPO denominator.
    The ONLY physical difference from shared is that the Root is recomputed
    for each row rather than reused across two siblings.
    """
    from verl.workers.engine.megatron.tpr.tree import (
        ForestExecutionPlan, SegmentObjectiveRef, SegmentPlan, SegmentSpec,
        TreeExecutionPlan,
    )

    if len(shared_forest.trees) != 1:
        raise AssertionError("Need exactly one original shared tree")
    original = shared_forest.trees[0]
    plan = original.segment_plan
    root = plan.get(plan.root_id)
    children = plan.children_of(plan.root_id)
    if len(children) != 2 or any(plan.children_of(child.segment_id) for child in children):
        raise AssertionError("Expected one Root and exactly two sibling Leaves")
    boundary = root.length
    rows = tuple(original.member_rows)
    if set(rows) != {0, 1}:
        raise AssertionError(f"Expected two local sibling rows (0,1), got {rows}")
    independent = []
    for row in sorted(rows):
        tokens = batch["input_ids"][row].detach().cpu().long()
        if not 0 < boundary < tokens.numel():
            raise AssertionError("Invalid shared Root boundary for split-no-share")
        if not torch.equal(tokens[:boundary], root.token_ids):
            raise AssertionError("Shared root is not an exact token prefix of both rows")
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
            parent = plan.get(ref.segment_id)
            global_query = parent.position_start + ref.query_offset
            target_global = global_query + 1
            if target_global >= tokens.numel() or int(tokens[target_global]) != ref.target_token_id:
                raise AssertionError(f"Split ref target mismatch row={row}, query={global_query}")
            new_id = 0 if global_query < boundary else 1
            refs.append(SegmentObjectiveRef(
                new_id, global_query if new_id == 0 else global_query - boundary,
                ref.target_token_id, ref.sample_row, ref.response_offset,
            ))
        if not refs:
            raise AssertionError(f"Split tree row {row} has no supervised targets")
        independent.append(TreeExecutionPlan(
            uid=f"{original.uid}_split_no_share_{row}",
            tree_index=row, member_rows=(row,),
            segment_plan=local_plan, objective_refs=tuple(refs),
        ))
    return ForestExecutionPlan(tuple(independent), shared_forest.logical_loss_tokens)


def _short_forest(batch, *, mode: str):
    from verl.workers.engine.megatron.tpr.adapter import _trajectory_keys_from_minibatch

    result = build_tree_execution_plans(
        _trajectory_keys_from_minibatch(batch),
        batch,
        tree_policy="qwen_turn",
        turn_min_segment_tokens=128,
    )
    logical = sum(int(x.numel()) for x in batch["input_ids"].unbind())
    physical = sum(
        segment.length for tree in result.trees
        for segment in tree.segment_plan.segments.values()
    )
    if mode in ("shared", "split_no_share"):
        if len(result.trees) != 1 or result.segment_count != 3:
            raise AssertionError(
                f"Expected one Root + two sibling Leafs, got "
                f"{len(result.trees)} trees / {result.segment_count} segments"
            )
        root = result.trees[0].segment_plan
        if len(root.children_of(root.root_id)) != 2:
            raise AssertionError("Short capture did not produce two Root siblings")
        if not physical < logical:
            raise AssertionError("Shared mode has no real Prefix reuse")
        if mode == "split_no_share":
            result = _make_unshared_split(result, batch)
            physical = sum(
                seg.length for tree in result.trees
                for seg in tree.segment_plan.segments.values()
            )
            if len(result.trees) != 2 or result.segment_count != 4 or physical != logical:
                raise AssertionError("Unshared split must be two independent Root/Leaf chains")
    else:
        if len(result.trees) != 2 or result.segment_count != 2 or physical != logical:
            raise AssertionError("Full/one-node control must have exactly two unshared roots")
    return result, logical, physical


def _run_controlled_full(model, batch, *, loss_fn, forest):
    """A genuine whole-sequence model call without any active TPR context."""
    from verl.models.mcore.tpr.context import get_tpr_attention_context
    from verl.workers.engine.megatron.tpr.objective import SegmentPPOObjectiveAdapter

    if get_tpr_attention_context() is not None:
        raise AssertionError("Full must execute with NO TPRAttentionContext")
    chunk_size = int(os.getenv("TPR_PPO_LOGPROB_CHUNK_TOKENS", "0"))
    if chunk_size < 0:
        raise ValueError("TPR_PPO_LOGPROB_CHUNK_TOKENS must be nonnegative")
    objective = SegmentPPOObjectiveAdapter(
        batch, loss_fn, log_prob_chunk_size=chunk_size or None,
    )
    losses = []
    for tree in forest.trees:
        plan = tree.segment_plan
        segment = plan.get(plan.root_id)
        refs = tree.refs_for_segment(segment.segment_id)
        if len(refs) == 0 or len(plan.segments) != 1:
            raise AssertionError("Full expects a single complete Root with PPO targets")
        callback, counts = objective.bind_tree(tree)
        if counts[segment.segment_id] != len(refs):
            raise AssertionError("Full PPO reference ownership mismatch")
        ids = segment.token_ids.to(next(model.parameters()).device).unsqueeze(0)
        positions = torch.arange(
            segment.length, device=ids.device, dtype=torch.long,
        ).unsqueeze(0)
        # SelfAttention.forward falls back to MCore square-CANN for CP1.
        logits = model(input_ids=ids, position_ids=positions, attention_mask=None)
        if tuple(logits.shape[:2]) != (1, segment.length):
            raise AssertionError("Native whole-sequence logits shape mismatch")
        loss, _ = callback(segment, logits)
        loss.backward()
        losses.append(loss.detach())
        del logits, loss
    return {
        "loss": float(torch.stack(losses).sum().item()),
        "logprobs": dict(objective.debug_new_log_probs),
    }



def _run_controlled_full_batch2(model, batch, *, loss_fn, forest):
    """Physical MB=2 square-CANN whole-sequence GPT baseline, not VERL native scheduler.

    Unlike _run_controlled_full (two individual B=1 forwards), this makes ONE
    full-sequence model invocation with [2, S] tokens. The same per-row
    SegmentPPOObjectiveAdapter / global denominator as K1/K2 is used.
    This tests the *physical batch* of Full fairly without pretending the
    test's lightweight engine is native VERL prepare_micro_batches.
    """
    from verl.models.mcore.tpr.context import get_tpr_attention_context
    from verl.workers.engine.megatron.tpr.objective import SegmentPPOObjectiveAdapter

    if get_tpr_attention_context() is not None:
        raise AssertionError("Full physical MB2 must not use a TPR KV context")
    if len(forest.trees) != 2 or any(len(tree.segment_plan.segments) != 1 for tree in forest.trees):
        raise AssertionError("Full physical MB2 requires two independent complete trajectories")
    roots = [tree.segment_plan.get(tree.segment_plan.root_id) for tree in forest.trees]
    lengths = [segment.length for segment in roots]
    if lengths[0] != lengths[1]:
        raise NotImplementedError(
            "This Full MB=2 control deliberately requires equally long rows; "
            f"got lengths={lengths}, no silent padding / different valid-token masks."
        )
    if len(batch["input_ids"]) != 2:
        raise AssertionError("Exactly two original logical trajectories required")
    ids = torch.stack([segment.token_ids for segment in roots], dim=0).to(
        device=next(model.parameters()).device,
    )
    positions = torch.arange(lengths[0], device=ids.device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(2, -1).contiguous()
    chunk_size = int(os.getenv("TPR_PPO_LOGPROB_CHUNK_TOKENS", "0"))
    if chunk_size < 0:
        raise ValueError("Negative PPO chunk size")
    objective = SegmentPPOObjectiveAdapter(batch, loss_fn, log_prob_chunk_size=chunk_size or None)
    logits = model(input_ids=ids, position_ids=positions, attention_mask=None)
    if tuple(logits.shape[:2]) != (2, lengths[0]):
        raise AssertionError(f"Unexpected physical Full MB2 logits: {tuple(logits.shape)}")
    losses = []
    for row, (tree, segment) in enumerate(zip(forest.trees, roots, strict=True)):
        callback, counts = objective.bind_tree(tree)
        if counts[segment.segment_id] != len(tree.refs_for_segment(segment.segment_id)):
            raise AssertionError(f"Full MB2 row={row} PPO reference count mismatch")
        local_loss, _ = callback(segment, logits[row:row + 1])
        losses.append(local_loss)
    total_loss = torch.stack(losses).sum()
    total_loss.backward()
    return {
        "loss": float(total_loss.detach().item()),
        "logprobs": dict(objective.debug_new_log_probs),
    }


def _run_split_no_share(model, engine, batch, *, loss_fn, forest):
    """Run two isolated Root/Leaf chains through the SAME production Executor."""
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.workers.engine.megatron.tpr.adapter import run_forest
    from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
    from verl.workers.engine.megatron.tpr.objective import SegmentPPOObjectiveAdapter

    chunk_size = int(os.getenv("TPR_PPO_LOGPROB_CHUNK_TOKENS", "0"))
    if chunk_size < 0:
        raise ValueError("Negative PPO chunk size")
    objective = SegmentPPOObjectiveAdapter(
        batch, loss_fn, log_prob_chunk_size=chunk_size or None,
    )
    layers = tuple(sorted(
        module.layer_number for module in model.modules()
        if isinstance(module, TPRSelfAttention)
    ))
    if len(layers) != 36:
        raise AssertionError("Need all 36 TPR-capable Qwen3-4B layers")
    model_config = model.config
    if model_config.recompute_granularity is not None:
        raise AssertionError("Parity must disable recompute")

    def run_one_tree(tree):
        callback, counts = objective.bind_tree(tree)
        return SegmentExecutor(
            model, tree.segment_plan,
            expected_layer_numbers=layers,
            loss_scale_func=getattr(model_config, "grad_scale_func", None),
            segment_loss_fn=callback, segment_loss_term_counts=counts,
        ).run()

    total_loss = run_forest(
        forest, run_one_tree,
        lambda: model_config.finalize_model_grads_func(
            engine.module, None, force_all_reduce=True,
        ),
    )
    return {
        "loss": float(total_loss.item()),
        "logprobs": dict(objective.debug_new_log_probs),
    }

def test_real_actor_cp1_short_full_vs_tpr():
    mode = os.getenv("TPR_CP1_MODE", "")
    if mode not in MODES:
        raise ValueError(f"TPR_CP1_MODE must be one of {MODES}, got {mode!r}")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise AssertionError("Short Qwen3-4B parity must run on exactly one NPU")
    if os.getenv("TPR_TREE_POLICY", "qwen_turn") != "qwen_turn":
        raise ValueError("Only production qwen_turn topology is supported")
    capture = Path(os.environ["TPR_CP1_CAPTURE"]).resolve()
    if not capture.is_file():
        raise FileNotFoundError(capture)
    destination = Path(os.environ["TPR_CP1_OUTPUT"]).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = Path(os.getenv(
        "TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"
    )).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    runtime = _runtime(1, 1, 1)
    try:
        from verl.workers.utils.losses import ppo_loss

        # Original rows are now two local records (0,1) inside the derived
        # capture. Force synthetic nonsharing keys in Full and one-node arms.
        batch_mode = "tpr" if mode in ("shared", "split_no_share") else "full"
        batch, original_keys, ppo_tokens, input_sha = _load_real_pair(
            capture, (0, 1), batch_mode,
        )
        if max(int(x.numel()) for x in batch["input_ids"].unbind()) > 4096:
            raise AssertionError("Short capture exceeds single-card limit")
        tu.assign_non_tensor(
            batch, dp_size=1, batch_num_tokens=ppo_tokens,
            global_batch_size=2, tpr_capture_log_probs=True,
        )
        forest, logical, physical = _short_forest(batch, mode=mode)
        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=1, cp_size=1,
            sequence_parallel=False, profile="4b", native_attention=False,
        )
        if hf.num_hidden_layers != 36:
            raise AssertionError("Only authentic Qwen3-4B is accepted")
        engine = _engine(model, runtime, 1, 1, 1)
        if model.config.recompute_granularity is not None:
            raise AssertionError("Parity must disable activation recompute")
        initial_weights = _initial_weight_fingerprint(model)
        loss_fn = partial(ppo_loss, config=_PPO())
        warmups = int(os.getenv("TPR_CP1_BENCH_WARMUPS", "0"))
        iterations = int(os.getenv("TPR_CP1_BENCH_REPEATS", "1"))
        if warmups < 0 or iterations < 1:
            raise ValueError("TPR_CP1_BENCH_WARMUPS >= 0 and TPR_CP1_BENCH_REPEATS >= 1 required")
        if os.getenv("TPR_CP1_TRACE", "0") == "1" and (warmups or iterations != 1):
            raise ValueError("Layer trace requires one un-warmed execution")
        trace = None
        if os.getenv("TPR_CP1_TRACE", "0") == "1" and mode != "one_node":
            from .cp1_attention_attribution import CP1AttentionTrace
            trace_dir = Path(os.environ["TPR_CP1_TRACE_DIR"])
            trace = CP1AttentionTrace(
                model=model, batch=batch, forest=forest, mode=mode,
                directory=trace_dir,
            ).install()

        def _execute_once():
            local_leaf_stats = None
            if mode == "full":
                result = _run_controlled_full(
                    model, batch, loss_fn=loss_fn, forest=forest,
                )
            elif mode == "full_batch2":
                result = _run_controlled_full_batch2(
                    model, batch, loss_fn=loss_fn, forest=forest,
                )
            elif mode == "split_no_share":
                result = _run_split_no_share(
                    model, engine, batch, loss_fn=loss_fn, forest=forest,
                )
            else:
                from verl.workers.engine.megatron.tpr.adapter import run_tpr_forward_backward_batch
                result = run_tpr_forward_backward_batch(
                    engine, batch, loss_fn, forward_only=False,
                )
                diagnostics = result["metrics"]
                local_leaf_stats = {
                    key: int(diagnostics[f"tpr/{key}"][0])
                    for key in ("model_forward_calls", "grouped_leaf_calls", "grouped_leaf_segments")
                }
                if mode == "shared":
                    configured_k = int(os.getenv("TPR_LEAF_BATCH_SIZE", "1"))
                    print(
                        f"TPR_CP1_LEAF_BATCH_CHECK configured_k={configured_k} "
                        f"model_forward_calls={local_leaf_stats['model_forward_calls']} "
                        f"grouped_leaf_calls={local_leaf_stats['grouped_leaf_calls']} "
                        f"grouped_leaf_segments={local_leaf_stats['grouped_leaf_segments']}",
                        flush=True,
                    )
                    if configured_k > 1 and local_leaf_stats["grouped_leaf_calls"] == 0:
                        raise AssertionError(
                            "TPR_LEAF_BATCH_SIZE>1 but real Qwen3 shared-tree Actor PPO "
                            "performed zero grouped leaf forwards: check equal-length "
                            "sibling eligibility and token budget."
                        )
                result = {
                    "loss": float(result["loss"][0]),
                    "logprobs": dict(engine._tpr_captured_log_probs),
                }
            return result, local_leaf_stats

        timings = []
        memory_allocated = []
        memory_reserved = []
        leaf_batch_stats = None
        try:
            for repeat in range(-warmups, iterations):
                model.zero_grad(set_to_none=True)
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                start = time.perf_counter()
                result, leaf_batch_stats = _execute_once()
                torch.npu.synchronize()
                elapsed = time.perf_counter() - start
                alloc = torch.npu.max_memory_allocated() / 2**30
                reserved = torch.npu.max_memory_reserved() / 2**30
                if repeat < 0:
                    print(f"TPR_CP1_BENCH_WARMUP mode={mode} duration_s={elapsed:.4f}", flush=True)
                else:
                    timings.append(elapsed)
                    memory_allocated.append(alloc)
                    memory_reserved.append(reserved)
                    print(
                        f"TPR_CP1_BENCH_ITER mode={mode} repeat={repeat} "
                        f"seconds={elapsed:.6f} peak_alloc_gib={alloc:.6f} "
                        f"peak_reserved_gib={reserved:.6f}", flush=True,
                    )
            from statistics import median
            seconds = median(timings)
            # Conservative capacity gate: retain MAX over measured iterations.
            measured_alloc = max(memory_allocated)
            measured_reserved = max(memory_reserved)
        finally:
            if trace is not None:
                trace.close()
        if trace is not None:
            if mode == "split_no_share":
                trace.replay_root_qkv_m(
                    Path(os.environ["TPR_CP1_TRACE_DIR"]) / "full/trace.pt"
                )
            trace_path = Path(os.environ["TPR_CP1_TRACE_DIR"]) / mode / "trace.pt"
            trace.save(trace_path)
            print(f"TPR_CP1_STAGE_TRACE mode={mode} saved={trace_path}", flush=True)
        if not torch.isfinite(torch.tensor(result["loss"])):
            raise AssertionError("Nonfinite single-card PPO loss")
        expected_keys = {
            (row, index)
            for row in range(2)
            for index in torch.nonzero(
                batch["response_mask"][row].bool(), as_tuple=False
            ).flatten().tolist()
        }
        if result["logprobs"].keys() != expected_keys or len(expected_keys) != ppo_tokens:
            raise AssertionError("Full/TPR did not produce identical logical PPO supervision")
        grad_norm = _global_bf16_gradient_norm(model, runtime)
        witness = canonical_grad_witness(
            model, tp_size=1, tp_rank=0,
            cp_rank=0, dp_rank=0, device=runtime.device,
        )
        if witness is None:
            raise AssertionError("Missing CP1 canonical gradient witness")
        full_grad_dir = None
        if os.getenv("TPR_CP1_SAVE_FULL_GRADS", "0") == "1":
            from .test_real_actor_tp2cp4_offline_ppo_npu import _save_complete_grads
            # Single-card TP1/CP1 has no replicated parameter shards:
            # persist every finalized BF16 gradient exactly once. This
            # reuses the tested CP4 file format and never alters autograd.
            full_grad_dir = _save_complete_grads(
                model, runtime, destination.parent, mode, step=1,
            )
        stats = {
            "mode": mode,
            "reference": (
                "whole_GPT_no_TPR_context_CP1_square_CANN"
                if mode == "full" else "whole_GPT_2rows_1physical_forward_CP1_square_CANN"
                if mode == "full_batch2" else "TPR_1root_per_row_CP1"
                if mode == "one_node" else "TPR_independent_root_leaf_chains_CP1"
                if mode == "split_no_share" else "TPR_shared_root_two_leaves_CP1"
            ),
            "source_capture": str(capture),
            "original_keys": list(original_keys),
            "input_sha256": input_sha,
            "checkpoint": str(checkpoint),
            "initial_weights_witness_sha256": initial_weights,
            "topology": {"tp": 1, "cp": 1, "dp": 1, "world": 1},
            "tree_policy": "qwen_turn",
            "forest_trees": len(forest.trees),
            "forest_segments": forest.segment_count,
            "logical_input_tokens": logical,
            "unique_tree_tokens": physical,
            "reuse_ratio": logical / physical,
            "ppo_tokens": ppo_tokens,
            "loss": result["loss"],
            "logprobs": sorted(
                [row, offset, float(value)]
                for (row, offset), value in result["logprobs"].items()
            ),
            "gradient_witness": witness,
            "global_grad_norm": grad_norm,
            "full_grad_dir": full_grad_dir,
            "full_gradient_scope": "ALL_PARAMETERS_BF16_TP1_CP1" if full_grad_dir else None,
            "fwd_bwd_seconds": seconds,
            "bench_warmups": warmups,
            "bench_repeats": iterations,
            "bench_times_s": timings,
            "bench_peak_allocated_gib": memory_allocated,
            "bench_peak_reserved_gib": memory_reserved,
            "logical_trajectories": 2,
            "physical_model_forward_calls": (
                1 if mode == "full_batch2" else
                2 if mode == "full" else
                leaf_batch_stats["model_forward_calls"] if leaf_batch_stats else None
            ),
            "physical_full_micro_batch": 2 if mode == "full_batch2" else 1 if mode == "full" else None,
            "benchmark_scope": "controlled_forward_backward_no_optimizer_no_native_VERL_microbatch_scheduler",
            "leaf_batch_stats": leaf_batch_stats,
            "leaf_batch_configured_k": int(os.getenv("TPR_LEAF_BATCH_SIZE", "1")),
            "leaf_batch_max_tokens": int(os.getenv("TPR_LEAF_BATCH_MAX_TOKENS", "8192")),
            "peak_allocated_gib": measured_alloc,
            "peak_reserved_gib": measured_reserved,
            "optimizer_step": False,
            "original_online_conditioning": False,
            "note": "Controlled cropped real-label PPO parity, NOT on-policy replay",
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(stats, indent=2))
        print(
            f"TPR_CP1_SHORT_RESULT mode={mode} input_sha={input_sha} "
            f"trees={stats['forest_trees']} segments={stats['forest_segments']} "
            f"reuse={stats['reuse_ratio']:.4f} ppo_tokens={ppo_tokens} "
            f"loss={stats['loss']:.9g} grad_norm={grad_norm:.9g} "
            f"fwd_bwd_sec={seconds:.3f} "
            f"peak_alloc_gib={stats['peak_allocated_gib']:.3f} "
            f"saved={destination}",
            flush=True,
        )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
