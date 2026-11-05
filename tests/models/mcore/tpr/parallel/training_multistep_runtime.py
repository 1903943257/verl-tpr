# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Fixed PPO inputs, native optimizer updates and synchronized stage timers."""

import hashlib
import json
import math
import time
from contextlib import nullcontext

import torch

from .training_path_precision_npu import _native_response_logprobs
from .training_precision_contract import SUFFIX, actor_batch, input_digest, synthetic_tokens, validate_forest

STEPS = 10
CHECKPOINTS = (0, 1, 5, 10)
BASE_CP1_REPEAT = "base_cp1_repeat"
TPR_CP2_QKVMERGE = "tpr_cp2_qkvmerge"


def mode_topology(mode):
    from .training_precision_contract import MODES

    if mode not in (*MODES, BASE_CP1_REPEAT, TPR_CP2_QKVMERGE):
        raise ValueError(f"Unknown mode: {mode}")
    return (2 if mode in ("base_cp2", "tpr_cp2", TPR_CP2_QKVMERGE) else 1), mode.startswith("tpr")


def timed_call(synchronize, function, *args, **kwargs):
    synchronize()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize()
    return result, time.perf_counter() - start


def input_item(vocab_size, seed, ordinal):
    tokens = synthetic_tokens(vocab_size, seed + ordinal * 7919)
    advantages = torch.sin(torch.arange(2 * SUFFIX).reshape(2, SUFFIX).float() * 0.17 + ordinal * 0.31) + 0.5
    return dict(tokens=tokens, advantages=advantages)


def item_batch(item):
    return actor_batch(item["tokens"], item["old_log_probs"], item["advantages"])


def fixture_digest(fixture):
    digest = hashlib.sha256(json.dumps(fixture["controls"], sort_keys=True).encode())
    digest.update(fixture["initial_weights_sha256"].encode())
    for item in [*fixture["batches"], fixture["probe"]]:
        digest.update(input_digest(item["tokens"], item["old_log_probs"], item["advantages"]).encode())
    return digest.hexdigest()


def prepare_fixture(engine, controls, initial_digest, path, rank, broadcast, barrier):
    """All old logprobs come from the unupdated Base CP1 model, including held-out probe."""
    if controls["mode"] == "base_cp1":
        common = {key: value for key, value in controls.items() if key != "mode"}
        items = [input_item(engine.model_config.hf_config.vocab_size, common["seed"], i) for i in range(STEPS + 1)]
        for ordinal, item in enumerate(items):
            temporary = actor_batch(item["tokens"], torch.zeros_like(item["advantages"]), item["advantages"])
            with engine.eval_mode(), torch.no_grad():
                output = engine.forward_backward_batch(temporary, None, forward_only=True)
                item["old_log_probs"] = _native_response_logprobs(output, temporary)
            if rank == 0:
                print(f"TRAIN10_FIXTURE item={ordinal + 1}/{len(items)}", flush=True)
        fixture = dict(controls=common, initial_weights_sha256=initial_digest, batches=items[:STEPS], probe=items[-1])
        fixture["sha256"] = fixture_digest(fixture)
        torch.save(fixture, str(path))
    barrier()
    fixture = torch.load(str(path), map_location="cpu", weights_only=True)
    expected = {key: value for key, value in controls.items() if key != "mode"}
    if fixture["controls"] != expected or fixture["initial_weights_sha256"] != initial_digest:
        raise AssertionError("Ten-step modes have mismatched controls or initial weights")
    if fixture_digest(fixture) != fixture["sha256"] or len(fixture["batches"]) != STEPS:
        raise AssertionError("Corrupt ten-step actor fixture")
    for item in [*fixture["batches"], fixture["probe"]]:
        validate_forest(item_batch(item))
    if controls["mode"] == BASE_CP1_REPEAT:
        # Match the original Base CP1's eleven native preparation forwards.
        # Keep the historical PPO labels frozen even if these calls differ.
        for ordinal, item in enumerate([*fixture["batches"], fixture["probe"]], 1):
            temporary = actor_batch(item["tokens"], torch.zeros_like(item["advantages"]), item["advantages"])
            with engine.eval_mode(), torch.no_grad():
                output = engine.forward_backward_batch(temporary, None, forward_only=True)
                _native_response_logprobs(output, temporary)
            if rank == 0:
                print(f"TRAIN10_FIXTURE_REPLAY item={ordinal}/{STEPS + 1} old_labels=reused", flush=True)
    digests = broadcast(initial_digest if rank == 0 else None)
    if initial_digest != digests:
        raise AssertionError("CP replicas loaded different initial weights")
    return fixture


def native_update(engine, batch, loss_function, synchronize, *, witness=None, after_backward=None):
    """Same zero/F-B/step/scheduler calls as production, with read-only observers."""
    learning_rates = sorted({float(group["lr"]) for group in engine.optimizer.param_groups})
    if not learning_rates or any(not math.isfinite(lr) or lr <= 0 for lr in learning_rates):
        raise AssertionError("Require positive native learning rates on all ten updates")
    with engine.train_mode():
        _, zero_seconds = timed_call(synchronize, engine.optimizer_zero_grad)
        with witness if witness is not None else nullcontext():
            output, fb_seconds = timed_call(
                synchronize, engine.forward_backward_batch, batch, loss_function, forward_only=False
            )
        if after_backward is not None:
            after_backward()
        norm, optimizer_seconds = timed_call(synchronize, engine.optimizer_step)
        _, scheduler_seconds = timed_call(synchronize, engine.lr_scheduler_step)
    norm = float(norm)
    if not math.isfinite(norm) or norm < 0:
        raise FloatingPointError("Invalid gradient norm from the native optimizer")
    return output, dict(
        grad_norm=norm,
        learning_rates_used=learning_rates,
        clip_coefficient_from_native_norm=min(1.0, engine.optimizer_config.clip_grad / (norm + 1e-6)),
        zero_grad_seconds=zero_seconds,
        forward_backward_seconds=fb_seconds,
        optimizer_seconds=optimizer_seconds,
        scheduler_seconds=scheduler_seconds,
    )


def native_probe(engine, item, rank):
    """Whole-sequence inference for every route, retaining its native CP topology."""
    from verl.models.mcore.tpr.context import get_tpr_attention_context
    from verl.utils.megatron_utils import unwrap_model

    if get_tpr_attention_context() is not None:
        raise AssertionError("TPR attention context leaked out of the training schedule")
    batch = item_batch(item)
    model = unwrap_model(engine.module[0])
    cores = [layer.self_attention.core_attention for layer in model.decoder.layers]
    called, handles = set(), []

    def before(module, _args):
        if get_tpr_attention_context() is not None:
            raise AssertionError("Probe entered segmented TPR attention")
        called.add(id(module))

    try:
        for core in cores:
            handles.append(core.register_forward_pre_hook(before))
        with engine.eval_mode(), torch.no_grad():
            output = engine.forward_backward_batch(batch, None, forward_only=True)
            values = _native_response_logprobs(output, batch).tolist() if rank == 0 else None
        if not cores or called != {id(core) for core in cores}:
            raise AssertionError("Probe did not execute every registered native core attention")
        return values
    finally:
        for handle in handles:
            handle.remove()
