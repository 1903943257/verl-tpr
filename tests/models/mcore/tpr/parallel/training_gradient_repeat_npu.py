# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""One fresh native Base CP1 backward; stop before clipping or Adam."""

import math
import os
from pathlib import Path


def native_norms(optimizer, synchronize):
    """Use the optimizer's own preparation/norm methods, without calling step."""
    from .training_execution_witness import function_origin
    from .training_multistep_runtime import timed_call
    from .training_precision_io import _optimizers

    for child in _optimizers(optimizer):
        if getattr(child, "grad_scaler", None) is not None:
            raise AssertionError("This BF16 check requires no gradient scaler")
        if any(float(group.get("step", 0)) != 0 for group in child.optimizer.param_groups):
            raise AssertionError("Optimizer already updated its parameters")
        if any(float(state.get("step", 0)) != 0 for state in child.optimizer.state.values()):
            raise AssertionError("Optimizer already updated its state")
    if float(optimizer.get_loss_scale()) != 1.0:
        raise AssertionError("Require native BF16 loss scale 1")
    found_inf, prepare_seconds = timed_call(synchronize, optimizer.prepare_grads)
    if found_inf:
        raise FloatingPointError("Native optimizer preparation found nonfinite gradients")
    elements = sum(gradient.numel() for child in _optimizers(optimizer) for gradient in child.get_grads_for_grad_norm())
    norms = [float(timed_call(synchronize, optimizer.get_grad_norm)[0]) for _ in range(2)]
    if any(not math.isfinite(norm) or norm <= 0 for norm in norms):
        raise FloatingPointError("Invalid native gradient norm")
    return dict(
        native_norms=norms,
        native_norm_elements=elements,
        prepare_seconds=prepare_seconds,
        native_optimizer_functions={
            name: function_origin(getattr(optimizer, name)) for name in ("prepare_grads", "get_grad_norm")
        },
    )


def main():
    import mindspeed.megatron_adaptor  # noqa: F401
    import torch
    import torch.distributed as dist

    from verl.utils.metric import reduce_metrics
    from verl.workers.config import McoreOptimizerConfig

    from .experiment_source import source_metadata
    from .training_gradient_repeat_io import compare_gradients, settings
    from .training_multistep_io import InitialWeights
    from .training_multistep_npu import controls_from_environment
    from .training_multistep_repeat import PRECISION_KEYS, check_source, restore_controls
    from .training_multistep_runtime import (
        BASE_CP1_REPEAT,
        STEPS,
        item_batch,
        native_probe,
        prepare_fixture,
        timed_call,
    )
    from .training_path_precision_npu import _build_engine, _native_response_logprobs, _versions
    from .training_precision_contract import ExecutionWitness, audit_engine, audit_wgrad_completion
    from .training_precision_io import index_reference, observe_gradients, write_json

    if int(os.environ["WORLD_SIZE"]) != 1:
        raise ValueError("Require one process, TP=PP=CP=DP=1")
    run = os.environ["TPR_GRAD_REPEAT_RUN"]
    if run not in ("first", "second"):
        raise ValueError("Require first or second gradient repeat")
    root = Path(os.environ["TPR_GRAD_REPEAT_OUTPUT"]).resolve()
    folder = root / run
    context = check_source(os.environ["TPR_TRAIN10_SOURCE"], output=root, packages=_versions())
    restore_controls(context["baseline"]["controls"])
    controls = controls_from_environment()
    if controls != context["baseline"]["controls"]:
        raise AssertionError("Historical controls were not restored exactly")
    threshold, _ = settings()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl")
    audit_path = folder / "runtime_audit_rank0.json"
    audit = dict(run=run, status="building", optimizer_step=False)
    try:
        optimizer_config = McoreOptimizerConfig(
            lr=controls["learning_rate"],
            min_lr=controls["learning_rate"],
            lr_decay_style="constant",
            lr_decay_steps=STEPS,
            total_training_steps=STEPS,
            lr_warmup_steps=0,
            weight_decay=controls["weight_decay"],
            clip_grad=controls["clip_grad"],
        )
        fusion = controls["gradient_accumulation_fusion"]
        print(f"GRAD_REPEAT_BUILD run={run}", flush=True)
        (engine, loss_function), build_seconds = timed_call(
            torch.npu.synchronize, _build_engine, 1, False, fusion, optimizer_config=optimizer_config
        )
        model, precision = audit_engine(engine, tpr=False, fusion=fusion)
        if any(precision[key] != context["precision"][key] for key in PRECISION_KEYS):
            raise AssertionError("Gradient repeat differs from historical Base CP1 precision")
        initial = InitialWeights(model, engine.optimizer)
        digest = initial.digest
        del initial
        fixture, fixture_seconds = timed_call(
            torch.npu.synchronize,
            prepare_fixture,
            engine,
            {**controls, "mode": BASE_CP1_REPEAT},
            digest,
            context["source"] / "actor_fixture.pt",
            0,
            lambda value: value,
            dist.barrier,
        )
        probe, _ = timed_call(torch.npu.synchronize, native_probe, engine, fixture["probe"], 0)
        versions = {name: parameter._version for name, parameter in model.named_parameters()}
        batch = item_batch(fixture["batches"][0])
        witness = ExecutionWitness(
            model.config.finalize_model_grads_func,
            [layer.self_attention.core_attention for layer in model.decoder.layers],
        )
        audit.update(status="initialized", precision=precision, packages=_versions(), controls=controls)
        write_json(audit_path, audit)
        with engine.train_mode():
            engine.optimizer_zero_grad()
            torch.npu.reset_peak_memory_stats()
            print(f"GRAD_REPEAT_BACKWARD run={run}", flush=True)
            with witness:
                output, fb_seconds = timed_call(
                    torch.npu.synchronize, engine.forward_backward_batch, batch, loss_function, forward_only=False
                )
            audit.update(status="backward_observed", **witness.snapshot())
            write_json(audit_path, audit)
            witness.validate(tpr=False, cp=1, fusion=fusion)
            completion = audit_wgrad_completion(model, fusion=fusion)
            norms = native_norms(engine.optimizer, torch.npu.synchronize)
            if norms["native_norm_elements"] != context["baseline"]["model_elements"]:
                raise AssertionError("Native norm and exported gradients cover different parameter elements")
            training_log_probs = _native_response_logprobs(output, batch)
            if run == "first":
                gradients, observer_seconds = timed_call(
                    torch.npu.synchronize,
                    observe_gradients,
                    model,
                    engine.optimizer,
                    folder / "grads",
                    0,
                    {},
                    save_reference=True,
                )
                index_reference(folder / "grads", [gradients])
            else:
                gradients, observer_seconds = timed_call(
                    torch.npu.synchronize, compare_gradients, model, engine.optimizer, root / "first/grads", threshold
                )
                write_json(folder / "parameter_differences.json", gradients)
            if versions != {name: parameter._version for name, parameter in model.named_parameters()}:
                raise AssertionError("Model parameters changed during the backward-only check")
            torch.save(
                dict(probe0=torch.tensor(probe, dtype=torch.float32), training_log_probs=training_log_probs),
                str(folder / "forwards.pt"),
            )
            report = dict(
                run=run,
                controls=controls,
                input_sha256=fixture["sha256"],
                initial_weights_sha256=digest,
                model_elements=context["baseline"]["model_elements"],
                precision=precision,
                packages=_versions(),
                source=source_metadata(),
                small_gradient_threshold=threshold,
                loss=sum(float(value) for value in output["loss"]),
                native_metrics={key: float(value) for key, value in reduce_metrics(dict(output["metrics"])).items()},
                execution_environment={
                    key: os.getenv(key)
                    for key in ("ASCEND_RT_VISIBLE_DEVICES", "HCCL_DETERMINISTIC", "TPR_TRAIN_GRAD_ACCUM_FUSION")
                },
                optimizer_step=False,
                gradient_stage="after_native_finalize_and_prepare_before_clip",
                peak_allocated_bytes=torch.npu.max_memory_allocated(),
                timings=dict(
                    build=build_seconds, fixture=fixture_seconds, backward=fb_seconds, observer=observer_seconds
                ),
                **norms,
            )
            write_json(folder / "result.json", report)
            audit.update(status="validated_before_clip_no_update", backward=completion, **norms)
            write_json(audit_path, audit)
        print(
            f"GRAD_REPEAT_ARM_DONE run={run} loss={report['loss']:.9g} native_norms={norms['native_norms']}", flush=True
        )
    except Exception as error:
        audit.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(audit_path, audit)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
