"""Route B: 20 optimizer updates, Full Sequence vs Split-No-Share.

This is a controlled fixed-rollout experiment on the same cropped real Actor PPO
capture used by the CP1 4K parity test. It deliberately excludes prefix sharing,
TP/CP/DP, recompute, and FP32 dW intervention. Each arm runs in its own process
so only one Qwen3-4B model and optimizer reside on the NPU at a time.

The rollout/old_log_probs/advantages are held fixed for 20 optimizer updates.
This is a training-path sensitivity test, not on-policy RL or a production
training benchmark. Gradients and parameters are represented by deterministic
per-tensor samples to avoid writing multi-gigabyte gradients at every step.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from verl.utils import tensordict_utils as tu

from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .test_real_actor_cp1_short_full_tpr_npu import (
    MODES,
    _initial_weight_fingerprint,
    _run_controlled_full,
    _run_split_no_share,
    _short_forest,
)
from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO, _load_real_pair
from .test_tpr_10k_parallel_parity_npu import _runtime, _engine

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_CP1_MULTISTEP") != "1",
    reason="Set TPR_RUN_CP1_MULTISTEP=1 for Route B optimizer-path testing",
)

_STEPS = int(os.getenv("TPR_CP1_MULTISTEP_STEPS", "20"))
_LEARNING_RATE = 1e-5
_BETAS = (0.9, 0.95)
_EPS = 1e-8
_WEIGHT_DECAY = 0.1
_MAX_GRAD_NORM = 1.0
_SAMPLE_COUNT = 128


def _sample_indices(name: str, numel: int) -> list[int]:
    """Stable random witness indices shared by separate Full/Split processes."""
    if numel <= 0:
        raise AssertionError(f"{name}: empty parameter")
    if numel <= 256:
        return list(range(numel))
    seed_bytes = hashlib.sha256(f"{name}:{numel}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, "little") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # Sampling with replacement avoids allocating a randperm of a huge weight.
    return torch.randint(
        low=0, high=numel, size=(min(_SAMPLE_COUNT, numel),),
        generator=generator, dtype=torch.int64,
    ).tolist()


def _make_sample_schema(named_tensors) -> dict:
    schema = {}
    for name, tensor in named_tensors:
        schema[name] = {
            "numel": int(tensor.numel()),
            "indices": _sample_indices(name, int(tensor.numel())),
        }
    return schema


def _sample_values(named_tensors, schema: dict, *, allow_missing: bool = False) -> dict:
    values = {}
    observed = dict(named_tensors)
    if observed.keys() != schema.keys():
        missing = sorted(schema.keys() - observed.keys())
        extra = sorted(observed.keys() - schema.keys())
        raise AssertionError(f"Sample schema mismatch missing={missing[:8]} extra={extra[:8]}")
    for name, item in schema.items():
        tensor = observed[name]
        if tensor is None:
            if not allow_missing:
                raise AssertionError(f"{name}: missing gradient")
            values[name] = None
            continue
        flat = tensor.detach().reshape(-1)
        if flat.numel() != item["numel"]:
            raise AssertionError(f"{name}: tensor size changed during training")
        indices = torch.tensor(item["indices"], dtype=torch.long, device=flat.device)
        values[name] = flat.index_select(0, indices).float().cpu().tolist()
    return values


def _logprob_rows(result: dict) -> list[list[float | int]]:
    rows = []
    for (row, offset), value in sorted(result["logprobs"].items()):
        if isinstance(value, torch.Tensor):
            value = value.item()
        rows.append([int(row), int(offset), float(value)])
    if not rows or not all(torch.isfinite(torch.tensor(row[2])).item() for row in rows):
        raise AssertionError("Missing or non-finite Actor PPO logprobs")
    return rows


def _write_json_line(handle, record: dict) -> None:
    handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def test_real_actor_cp1_sequence_split_multistep():
    mode = os.getenv("TPR_CP1_MULTISTEP_MODE", "")
    if mode not in ("full", "split_no_share"):
        raise ValueError("TPR_CP1_MULTISTEP_MODE must be full or split_no_share")
    if _STEPS < 2:
        raise ValueError("Route B requires at least two optimizer steps")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise AssertionError("Route B must run on exactly one NPU")
    if os.getenv("TPR_TREE_POLICY", "qwen_turn") != "qwen_turn":
        raise ValueError("Route B requires the production qwen_turn segmentation policy")

    capture = Path(os.environ["TPR_CP1_CAPTURE"]).resolve()
    destination = Path(os.environ["TPR_CP1_MULTISTEP_OUTPUT"]).resolve()
    summary_path = destination.with_suffix(".summary.json")
    checkpoint = Path(os.getenv(
        "TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"
    )).resolve()
    if not capture.is_file():
        raise FileNotFoundError(capture)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if destination.exists() or summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Route B artifacts: {destination} / {summary_path}"
        )

    # Each arm is a fresh process with the same seed and exact same checkpoint.
    torch.manual_seed(2026)
    runtime = _runtime(1, 1, 1)
    try:
        from verl.workers.utils.losses import ppo_loss

        batch_mode = "full" if mode == "full" else "tpr"
        batch, original_keys, ppo_tokens, input_sha = _load_real_pair(
            capture, (0, 1), batch_mode,
        )
        if max(int(x.numel()) for x in batch["input_ids"].unbind()) > 4096:
            raise AssertionError("Route B capture exceeds the 4K per-trajectory limit")
        tu.assign_non_tensor(
            batch, dp_size=1, batch_num_tokens=ppo_tokens,
            global_batch_size=2, tpr_capture_log_probs=True,
        )
        forest, logical_tokens, physical_tokens = _short_forest(batch, mode=mode)

        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=1, cp_size=1,
            sequence_parallel=False, profile="4b", native_attention=False,
        )
        if hf.num_hidden_layers != 36:
            raise AssertionError(f"Expected Qwen3-4B/36 layers, got {hf.num_hidden_layers}")
        engine = _engine(model, runtime, 1, 1, 1)
        if model.config.recompute_granularity is not None:
            raise AssertionError("Route B must disable activation recompute")
        model.train()
        loss_fn = partial(ppo_loss, config=_PPO())

        named_parameters = dict(model.named_parameters())
        if not named_parameters:
            raise AssertionError("Model has no parameters")
        # Keep FP32 master weights and Adam moments on host memory so the 4B
        # model's optimizer state does not compete with NPU activations/gradients.
        # This preserves FP32 AdamW arithmetic; only state/gradient transfer is
        # offloaded, and both arms use the exact same optimizer path.
        masters = {
            name: torch.nn.Parameter(
                parameter.detach().to(device="cpu", dtype=torch.float32).clone()
            )
            for name, parameter in named_parameters.items()
        }
        optimizer = torch.optim.AdamW(
            masters.values(), lr=_LEARNING_RATE, betas=_BETAS, eps=_EPS,
            weight_decay=_WEIGHT_DECAY, foreach=False,
        )
        schema = _make_sample_schema(named_parameters.items())
        initial_fingerprint = _initial_weight_fingerprint(model)
        initial_parameter_samples = _sample_values(named_parameters.items(), schema)

        header = {
            "record_type": "header",
            "mode": mode,
            "steps": _STEPS,
            "source_capture": str(capture),
            "input_sha256": input_sha,
            "checkpoint": str(checkpoint),
            "original_keys": list(original_keys),
            "initial_weights_witness_sha256": initial_fingerprint,
            "topology": {"tp": 1, "cp": 1, "dp": 1, "world": 1},
            "tree_policy": "qwen_turn",
            "forest_trees": len(forest.trees),
            "forest_segments": forest.segment_count,
            "logical_input_tokens": logical_tokens,
            "physical_tokens_per_update": physical_tokens,
            "ppo_tokens": ppo_tokens,
            "optimizer": {
                "type": "AdamW_FP32_master",
                "lr": _LEARNING_RATE,
                "betas": list(_BETAS),
                "eps": _EPS,
                "weight_decay": _WEIGHT_DECAY,
                "max_grad_norm": _MAX_GRAD_NORM,
            },
            "sample_schema": schema,
            "initial_parameter_samples": initial_parameter_samples,
            "gradient_sample_scope": "deterministic per-tensor sampled estimate, not complete gradients",
            "data_policy": "same cropped real Actor rollout/old_log_probs/advantages reused for every update",
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            _write_json_line(handle, header)
            for step in range(1, _STEPS + 1):
                model.zero_grad(set_to_none=True)
                if mode == "full":
                    result = _run_controlled_full(
                        model, engine, batch, loss_fn=loss_fn, forest=forest,
                    )
                else:
                    result = _run_split_no_share(
                        model, engine, batch, loss_fn=loss_fn, forest=forest,
                    )
                torch.npu.synchronize()
                loss = float(result["loss"])
                if not torch.isfinite(torch.tensor(loss)).item():
                    raise AssertionError(f"step={step}: non-finite PPO loss")
                logprobs = _logprob_rows(result)
                gradients = _sample_values(
                    ((name, parameter.grad) for name, parameter in named_parameters.items()),
                    schema, allow_missing=True,
                )
                missing_gradients = [
                    name for name, parameter in named_parameters.items()
                    if parameter.grad is None
                ]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), _MAX_GRAD_NORM, foreach=False,
                ))
                if not torch.isfinite(torch.tensor(grad_norm)).item():
                    raise AssertionError(f"step={step}: non-finite gradient norm")
                clipped = grad_norm > _MAX_GRAD_NORM

                for name, parameter in named_parameters.items():
                    if parameter.grad is None:
                        masters[name].grad = None
                    else:
                        masters[name].grad = parameter.grad.detach().to(
                            device="cpu", dtype=torch.float32,
                        )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, parameter in named_parameters.items():
                        parameter.copy_(masters[name].to(
                            device=parameter.device, dtype=parameter.dtype,
                        ))
                parameter_samples = _sample_values(named_parameters.items(), schema)

                record = {
                    "record_type": "step",
                    "step": step,
                    "loss": loss,
                    "logprobs": logprobs,
                    "grad_norm_before_clip": grad_norm,
                    "clipped": clipped,
                    "missing_gradients": missing_gradients,
                    "gradient_samples": gradients,
                    "parameter_samples_after_update": parameter_samples,
                }
                _write_json_line(handle, record)
                print(
                    f"TPR_CP1_MULTISTEP_STEP mode={mode} step={step:02d}/{_STEPS} "
                    f"loss={loss:.9g} grad_norm={grad_norm:.7g} clipped={clipped} "
                    f"missing_grads={len(missing_gradients)}",
                    flush=True,
                )

        # Evaluate the parameters AFTER the final optimizer update on the same
        # fixed rollout, without taking a 21st optimizer step.
        model.zero_grad(set_to_none=True)
        if mode == "full":
            final_eval = _run_controlled_full(
                model, engine, batch, loss_fn=loss_fn, forest=forest,
            )
        else:
            final_eval = _run_split_no_share(
                model, engine, batch, loss_fn=loss_fn, forest=forest,
            )
        torch.npu.synchronize()
        model.zero_grad(set_to_none=True)
        final_eval_record = {
            "loss": float(final_eval["loss"]),
            "logprobs": _logprob_rows(final_eval),
        }
        optimizer_state_samples = {}
        for state_name in ("exp_avg", "exp_avg_sq"):
            optimizer_state_samples[state_name] = _sample_values(
                (
                    (
                        name,
                        optimizer.state[masters[name]].get(state_name),
                    )
                    for name in masters
                ),
                schema,
                allow_missing=True,
            )
        summary = {
            "mode": mode,
            "steps": _STEPS,
            "source_capture": str(capture),
            "input_sha256": input_sha,
            "checkpoint": str(checkpoint),
            "initial_weights_witness_sha256": initial_fingerprint,
            "ppo_tokens": ppo_tokens,
            "optimizer": header["optimizer"],
            "post_update_eval_after_step_n": _STEPS,
            "post_update_eval": final_eval_record,
            "optimizer_state_samples": optimizer_state_samples,
            "sample_scope": header["gradient_sample_scope"],
        }
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(
            f"TPR_CP1_MULTISTEP_RESULT mode={mode} steps={_STEPS} "
            f"input_sha={input_sha} final_loss={final_eval_record['loss']:.9g} "
            f"trajectory={destination} summary={summary_path}",
            flush=True,
        )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
