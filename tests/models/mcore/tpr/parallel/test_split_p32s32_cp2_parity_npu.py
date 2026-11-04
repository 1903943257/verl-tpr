# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Controlled BF16 P32S32 parity: CP1 whole, CP2 whole, CP2 TPR replay.

A real Qwen3-4B checkpoint, one identical 64-token synthetic sequence and the
same fixed global-position linear loss, plus next-token logprobs (diagnostic).
CP1 runs ordinary square attention; CP2 runs production TPR Ring CP. This is
separate from the all-FP32 explicit-math test, and it does not claim the CP1
and CP2 attention kernels have identical BF16 rounding.

No optimizer step. CP2 gradients are averaged across CP replicas in FP32
after SegmentExecutor scales local backward loss by CP size.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_P32S32_CP2_PARITY") != "1",
    reason="Opt in to BF16 64-token CP1/CP2 Full/TPR gradient parity",
)


def _tokens() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20261104)
    return torch.randint(256, 151936 - 256, (64,), dtype=torch.long, generator=generator)


def _objective(logits, positions, tokens, *, global_length=64):
    """Unmodified fixed linear loss from FP32 diagnostic, computed on BF16 logits.

    The diagnostic logprobs do NOT contribute to the backward objective.
    Prefix query 31 predicts suffix token 32; query 63 has no loss.
    """
    assert logits.ndim == 3 and logits.shape[0] == 1
    positions = positions.to(device=logits.device, dtype=torch.long)
    assert logits.shape[1] == positions.numel()
    if bool((positions < 0).any()) or bool((positions >= global_length).any()):
        raise ValueError("Physical CP padding is not supported in this exact-length test")
    vocabulary = logits.shape[-1]
    target = tokens.to(device=logits.device).index_select(0, (positions + 1).clamp_max(global_length - 1))
    positions_loss_mask = positions != global_length - 1
    logits_fp32 = logits.float()
    token_weights = torch.sin(positions.float() + 1).masked_fill(~positions_loss_mask, 0)
    vocab_weights = torch.cos(torch.arange(vocabulary, dtype=torch.float32, device=logits.device) * 0.173)
    local_loss = (
        logits_fp32 * token_weights[None, :, None] * vocab_weights[None, None, :]
    ).sum() / ((global_length - 1) * math.sqrt(vocabulary))
    # Report logprobs for 63 prediction queries; the last token has no target.
    next_logprobs = logits_fp32.log_softmax(dim=-1).gather(
        -1, target[None, :, None]
    ).reshape(-1).detach()
    observations = torch.zeros(global_length, dtype=torch.float32, device=logits.device)
    coverage = torch.zeros(global_length, dtype=torch.int32, device=logits.device)
    valid_positions = positions[positions_loss_mask]
    observations.index_copy_(0, valid_positions, next_logprobs[positions_loss_mask])
    coverage.index_fill_(0, valid_positions, 1)
    return local_loss, observations, coverage


def _initial_fingerprint(model):
    from .test_real_actor_cp1_short_full_tpr_npu import _initial_weight_fingerprint
    return _initial_weight_fingerprint(model)


def _save_gradients(model, folder: Path) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if parameter.grad is None or parameter.grad.dtype != torch.bfloat16:
            raise RuntimeError(f"{name}: missing BF16 parameter grad")
        filename = f"grad_{index:05d}.pt"
        torch.save(parameter.grad.detach().cpu().contiguous(), folder / filename)
        entries.append({"name": name, "file": filename, "shape": list(parameter.shape)})
    (folder / "manifest.json").write_text(json.dumps(entries, indent=2))
    return entries


def _compare_current_grads(model, reference: Path) -> dict:
    """Stream the reference CPU shards; current BF16 candidate stays on NPU."""
    manifest = json.loads((reference / "manifest.json").read_text())
    params = dict(model.named_parameters())
    if list(params) != [entry["name"] for entry in manifest]:
        raise AssertionError("CP1/CP2 parameter names/order changed")
    reference_sq = candidate_sq = diff_sq = dot = 0.0
    maximum = 0.0
    count = 0
    for index, entry in enumerate(manifest):
        name = entry["name"]
        gradient = params[name].grad
        if gradient is None:
            raise RuntimeError(f"Missing CP gradient: {name}")
        baseline = torch.load(str(reference / entry["file"]), weights_only=True, map_location="cpu", mmap=True)
        actual = gradient.detach().cpu().contiguous()
        if baseline.dtype != torch.bfloat16 or actual.dtype != torch.bfloat16 or list(actual.shape) != entry["shape"]:
            raise ValueError(f"{name}: mismatched gradient dtype/shape")
        baseline = baseline.reshape(-1)
        actual = actual.reshape(-1)
        for start in range(0, baseline.numel(), 262144):
            a = baseline[start:start+262144].double()
            b = actual[start:start+262144].double()
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise FloatingPointError(f"{name}: nonfinite gradient")
            delta = b - a
            reference_sq += float(a.square().sum())
            candidate_sq += float(b.square().sum())
            diff_sq += float(delta.square().sum())
            dot += float((a*b).sum())
            maximum = max(maximum, float(delta.abs().max()))
            count += a.numel()
        if (index + 1) % 32 == 0:
            print(f"P32S32_CP_GRAD_COMPARE {reference.name} {index + 1}/{len(manifest)}", flush=True)
    if not reference_sq or not candidate_sq:
        raise ValueError("Empty or zero gradient norm")
    return {
        "elements": count,
        "gradient_relative_l2": math.sqrt(diff_sq / reference_sq),
        "gradient_cosine": dot / math.sqrt(reference_sq * candidate_sq),
        "gradient_max_abs": maximum,
        "reference_grad_norm": math.sqrt(reference_sq),
        "candidate_grad_norm": math.sqrt(candidate_sq),
    }


def _compare_loss_logprobs(a: dict, b: dict) -> dict:
    if any(a[k] != b[k] for k in ("checkpoint", "input_sha256", "initial_weights_witness", "tokens")):
        raise AssertionError("Comparing different inputs or weights")
    if len(a["logprobs"]) != 63 or len(b["logprobs"]) != 63:
        raise AssertionError("Expected 63 causal next-token logprobs")
    diffs = [float(y)-float(x) for x,y in zip(a["logprobs"], b["logprobs"], strict=True)]
    denom = sum(float(x)**2 for x in a["logprobs"])
    return {
        "loss_abs_gap": abs(a["loss"] - b["loss"]),
        "logprob_rel_l2": math.sqrt(sum(d*d for d in diffs) / denom) if denom else None,
        "logprob_max_abs": max(abs(d) for d in diffs),
        "logprob_mean_abs": sum(abs(d) for d in diffs)/len(diffs),
    }


def _collect_observations(local_loss, observations, coverage, cp_group, cp):
    if cp > 1:
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM, group=cp_group)
        dist.all_reduce(observations, op=dist.ReduceOp.SUM, group=cp_group)
        dist.all_reduce(coverage, op=dist.ReduceOp.SUM, group=cp_group)
    coverage_cpu = coverage.cpu().tolist()
    if coverage_cpu[:63] != [1] * 63 or coverage_cpu[63] != 0:
        raise AssertionError(f"Wrong CP token coverage: {coverage_cpu}")
    logps = observations.cpu().tolist()[:63]
    if not all(math.isfinite(float(x)) for x in logps):
        raise FloatingPointError("Nonfinite token logprobs")
    return float(local_loss.item()), logps


def _run_full_cp1(model, tokens):
    positions = torch.arange(tokens.numel(), device=tokens.device)
    logits = model(
        input_ids=tokens[None,:], position_ids=positions[None,:], attention_mask=None
    )
    loss, values, coverage = _objective(logits, positions, tokens)
    loss.backward()
    return loss.detach(), values, coverage


def _run_cp2(model, runtime, tokens, mode: str):
    from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
    from verl.workers.engine.megatron.tpr.ring import make_ring_sequence_shard
    from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec

    if mode == "full_cp2":
        segments = (SegmentSpec(0, None, tokens.cpu(), 0, 0),)
        counts = {0: 63}
    elif mode == "tpr_cp2":
        segments = (
            SegmentSpec(0, None, tokens[:32].cpu(), 0, 0),
            SegmentSpec(1, 0, tokens[32:].cpu(), 32, 32),
        )
        counts = {0: 32, 1: 31}
    else:
        raise ValueError(mode)
    observations = torch.zeros(64, dtype=torch.float32, device=runtime.device)
    coverage = torch.zeros(64, dtype=torch.int32, device=runtime.device)

    def segment_loss(segment, local_logits):
        shard = make_ring_sequence_shard(segment.length, cp_rank=runtime.cp_rank, cp_size=2)
        positions = shard.global_indices(device=runtime.device) + segment.position_start
        local_loss, seen_logps, seen_positions = _objective(local_logits, positions, tokens)
        observations.add_(seen_logps)
        coverage.add_(seen_positions)
        return local_loss, {}

    executor = SegmentExecutor(
        model, SegmentPlan(segments, root_id=0),
        expected_layer_numbers=tuple(layer.layer_number for layer in model.decoder.layers),
        cp_group=runtime.cp_group, cp_backend="ring",
        segment_loss_fn=segment_loss, segment_loss_term_counts=counts,
    )
    reported_loss = executor.run().detach()
    # The executor multiplies each local CP loss by CP size before backward,
    # and we subsequently average weight gradients across the CP group.
    return reported_loss, observations, coverage


def _sync_cp_weight_grads(model, runtime):
    from .grad_sync_precision import reduce_average_weight_grad
    for name, parameter in model.named_parameters():
        reduce_average_weight_grad(
            parameter, runtime.cp_group, name=name, reduce_dtype="fp32"
        )


def test_p32s32_cp1_cp2_full_vs_tpr():
    import torch_npu  # noqa: F401
    from megatron.core import parallel_state
    from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
    from .test_tpr_10k_parallel_parity_npu import _runtime

    mode = os.environ.get("TPR_P32S32_MODE")
    if mode not in ("full_cp1", "cp2_both"):
        raise ValueError("TPR_P32S32_MODE must be full_cp1 or cp2_both")
    cp = 1 if mode == "full_cp1" else 2
    root = Path(os.environ["TPR_P32S32_OUTPUT"]).resolve()
    checkpoint = Path(os.environ["TPR_QWEN_4B_PATH"]).resolve()
    if cp == 2 and not (root / "full_cp1" / "result.json").is_file():
        raise FileNotFoundError("Run Full CP1 first")
    runtime = _runtime(1, cp, 1)
    try:
        device_tokens = _tokens().to(runtime.device)
        sha = hashlib.sha256(device_tokens.cpu().numpy().tobytes()).hexdigest()
        model, _ = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=1, cp_size=cp,
            profile="4b", params_dtype=torch.bfloat16,
        )
        model.train()
        parameter_versions = {name: parameter._version for name, parameter in model.named_parameters()}
        witness = _initial_fingerprint(model) if runtime.cp_rank == 0 else None
        if runtime.cp_rank == 0:
            print(f"P32S32_CP_READY cp={cp} token_hash={sha} fingerprint={witness}", flush=True)
        modes = ["full_cp1"] if cp == 1 else ["full_cp2", "tpr_cp2"]
        for arm in modes:
            model.zero_grad(set_to_none=True)
            if runtime.cp_rank == 0:
                print(f"P32S32_CP_BEGIN mode={arm}", flush=True)
            if arm == "full_cp1":
                loss, observations, coverage = _run_full_cp1(model, device_tokens)
            else:
                loss, observations, coverage = _run_cp2(model, runtime, device_tokens, arm)
                _sync_cp_weight_grads(model, runtime)
            # CP2 will also all-reduce per-segment partial losses and token
            # logprobs; the reduction happens only after backward collectives.
            total_loss, logps = _collect_observations(loss, observations, coverage, runtime.cp_group, cp)
            if parameter_versions != {
                name: parameter._version for name, parameter in model.named_parameters()
            }:
                raise AssertionError(f"Model parameters changed in-place during {arm}")
            if runtime.cp_rank != 0:
                # Keep every CP rank at the same phase while rank0 streams
                # complete 4B parameter gradients and compares CPU shards.
                dist.barrier(group=runtime.cp_group)
                continue
            output = root / arm
            output.mkdir(parents=True, exist_ok=False)
            payload = {
                "mode": arm, "checkpoint": str(checkpoint), "input_sha256": sha,
                "tokens": _tokens().tolist(), "initial_weights_witness": witness,
                "dtype": "torch.bfloat16", "loss": total_loss, "logprobs": logps,
                "objective": "fixed_global_position_linear_no_ppo",
                "logprobs_scope": "next_token_diagnostic_not_backward_objective",
                "topology": {"tp": 1, "cp": cp, "dp": 1},
                "gradient_cp_reduce": "fp32_sum_divide_by_cp_then_cast_bf16" if cp > 1 else None,
                "optimizer_step": False,
                "model_attention": "CP1 square CANN" if cp == 1 else "TPR Ring CP2",
            }
            if arm == "full_cp1":
                _save_gradients(model, output / "grads")
            elif arm == "full_cp2":
                baseline = json.loads((root / "full_cp1" / "result.json").read_text())
                payload["full_cp1_comparison"] = {
                    **_compare_loss_logprobs(baseline, payload),
                    **_compare_current_grads(model, root / "full_cp1" / "grads"),
                }
                _save_gradients(model, output / "grads")
            else:
                for other in ("full_cp1", "full_cp2"):
                    baseline = json.loads((root / other / "result.json").read_text())
                    payload[f"{other}_comparison"] = {
                        **_compare_loss_logprobs(baseline, payload),
                        **_compare_current_grads(model, root / other / "grads"),
                    }
            (output / "result.json").write_text(json.dumps(payload, indent=2))
            print(
                f"P32S32_CP_DONE mode={arm} loss={total_loss:.9g} "
                f"comparisons={[key for key in payload if key.endswith('_comparison')]} "
                f"output={output/'result.json'}",
                flush=True,
            )
            if cp > 1:
                dist.barrier(group=runtime.cp_group)
        if cp == 2 and runtime.cp_rank == 0:
            full = json.loads((root / "full_cp2" / "result.json").read_text())
            tpr = json.loads((root / "tpr_cp2" / "result.json").read_text())
            summary = {
                "scope": "BF16 real Qwen3-4B, one P32S32 trajectory; CP1 square vs CP2 Ring",
                "full_cp1_vs_full_cp2": full["full_cp1_comparison"],
                "full_cp1_vs_tpr_cp2": tpr["full_cp1_comparison"],
                "full_cp2_vs_tpr_cp2": tpr["full_cp2_comparison"],
            }
            (root / "comparison.json").write_text(json.dumps(summary, indent=2))
            for key, comparison in summary.items():
                if not isinstance(comparison, dict):
                    continue
                print(
                    f"P32S32_CP_COMPARE {key}: grad_rel_l2={comparison['gradient_relative_l2']:.9g} "
                    f"cosine={comparison['gradient_cosine']:.9g} "
                    f"grad_max_abs={comparison['gradient_max_abs']:.9g} "
                    f"loss_abs_gap={comparison['loss_abs_gap']:.9g} "
                    f"logprob_max_abs={comparison['logprob_max_abs']:.9g}",
                    flush=True,
                )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
