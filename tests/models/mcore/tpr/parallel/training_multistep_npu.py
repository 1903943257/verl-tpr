# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Four production training routes, ten real native Adam updates per route."""

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

from .training_path_precision_npu import _build_engine, _versions
from .training_precision_io import write_json


def controls_from_environment():
    from .training_multistep_runtime import CHECKPOINTS, STEPS

    learning_rate = float(os.getenv("TPR_TRAIN10_LR", "1e-6"))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("TPR_TRAIN10_LR must be finite and positive")
    return dict(
        steps=STEPS,
        checkpoints=list(CHECKPOINTS),
        seed=int(os.getenv("TPR_TRAIN_SEED", "261105")),
        learning_rate=learning_rate,
        lr_decay_style="constant",
        weight_decay=0.01,
        clip_grad=1.0,
        vanilla_mbridge=os.getenv("TPR_TRAIN_VANILLA_MBRIDGE", "1") == "1",
        gradient_accumulation_fusion=os.getenv("TPR_TRAIN_GRAD_ACCUM_FUSION", "1") == "1",
        delayed_dkv_fp32=os.getenv("TPR_EXPERIMENT_DELAYED_DKV_FP32", "0"),
        root_native_ring=os.getenv("TPR_ROOT_NATIVE_RING", "0"),
        leaf_batch_size=os.getenv("TPR_LEAF_BATCH_SIZE", "1"),
        selected_logits_fp32=os.getenv("TPR_SELECTED_LOGITS_FP32", "0"),
    )


def main():
    import mindspeed.megatron_adaptor  # noqa: F401
    import torch
    import torch.distributed as dist

    from verl.utils.metric import reduce_metrics
    from verl.workers.config import McoreOptimizerConfig

    from .experiment_source import source_metadata
    from .training_multistep_io import InitialWeights, audit_adam_updates
    from .training_multistep_report import complete_mode
    from .training_multistep_runtime import (
        BASE_CP1_REPEAT,
        CHECKPOINTS,
        STEPS,
        TPR_CP2_QKVMERGE,
        item_batch,
        mode_topology,
        native_probe,
        native_update,
        prepare_fixture,
        timed_call,
    )
    from .training_precision_contract import ExecutionWitness, audit_engine, audit_wgrad_completion

    mode = os.environ["TPR_TRAIN_MODE"]
    repeating = mode in (BASE_CP1_REPEAT, TPR_CP2_QKVMERGE)
    qkvmerge = mode == TPR_CP2_QKVMERGE
    cp, tpr = mode_topology(mode)
    if int(os.environ["WORLD_SIZE"]) != cp:
        raise ValueError("Require TP=PP=DP=1, WORLD_SIZE=CP")
    repeat_context = None
    if repeating:
        from .training_multistep_repeat import PRECISION_KEYS, check_source, complete_repeat, restore_controls

        repeat_context = check_source(
            os.environ["TPR_TRAIN10_REPEAT_SOURCE"],
            output=os.environ["TPR_TRAIN10_OUTPUT"],
            packages=_versions(),
            mode=mode,
        )
        restore_controls(repeat_context["baseline"]["controls"])
    controls = controls_from_environment()
    if repeating and controls != repeat_context["baseline"]["controls"]:
        raise AssertionError("Cannot restore the historical training controls exactly")
    merge_settings = {
        name: "1" if qkvmerge else "0"
        for name in (
            "TPR_RING_COALESCE_PREFIX_FULL",
            "TPR_RING_COALESCE_PREFIX_QUERY",
            "TPR_RING_COALESCE_PREFIX_PADDED",
        )
    }
    for name, value in merge_settings.items():
        os.environ[name] = value
    fusion = controls["gradient_accumulation_fusion"]
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    root = Path(os.environ["TPR_TRAIN10_OUTPUT"]).resolve()
    folder = root / mode
    audit_path = folder / f"runtime_audit_rank{rank}.json"
    runtime_audit = dict(mode=mode, rank=rank, status="building")
    current_step = 0
    started = time.perf_counter()

    def broadcast(value):
        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    try:
        if rank == 0:
            print(f"TRAIN10_BUILD mode={mode}", flush=True)
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
        (engine, loss_function), build_seconds = timed_call(
            torch.npu.synchronize, _build_engine, cp, tpr, fusion, optimizer_config=optimizer_config
        )
        model, precision = audit_engine(engine, tpr=tpr, fusion=fusion)
        if repeating and any(precision[key] != repeat_context["precision"][key] for key in PRECISION_KEYS):
            raise AssertionError("Supplement precision/topology differs from its historical control")
        initial, initial_seconds = timed_call(torch.npu.synchronize, InitialWeights, model, engine.optimizer)
        fixture, fixture_seconds = timed_call(
            torch.npu.synchronize,
            prepare_fixture,
            engine,
            {**controls, "mode": mode},
            initial.digest,
            (repeat_context["source"] if repeating else root) / "actor_fixture.pt",
            rank,
            broadcast,
            dist.barrier,
        )
        runtime_audit.update(status="initialized", precision=precision, packages=_versions(), controls=controls)
        runtime_audit["ring_merge_settings"] = merge_settings
        write_json(audit_path, runtime_audit)
        torch.npu.reset_peak_memory_stats()
        probes, samples, steps, evaluations = {}, {}, [], []

        def evaluate(step):
            values, seconds = timed_call(torch.npu.synchronize, native_probe, engine, fixture["probe"], rank)
            if values is not None:
                probes[step] = torch.tensor(values, dtype=torch.float32)
            evaluations.append(dict(step=step, seconds=seconds))
            if rank == 0:
                print(f"TRAIN10_PROBE mode={mode} step={step} seconds={seconds:.3f}", flush=True)

        evaluate(0)
        for current_step, item in enumerate(fixture["batches"], 1):
            if rank == 0:
                print(f"TRAIN10_BEGIN mode={mode} step={current_step}/{STEPS}", flush=True)
            witness = (
                ExecutionWitness(
                    model.config.finalize_model_grads_func,
                    [layer.self_attention.core_attention for layer in model.decoder.layers],
                )
                if current_step == 1
                else None
            )
            merge_witness = None
            if qkvmerge and current_step == 1:
                from .training_qkvmerge_witness import QKVMergeWitness

                merge_witness = QKVMergeWitness()

            def after_backward(witness=witness, merge_witness=merge_witness):
                if witness is not None:
                    runtime_audit.update(status="backward_observed", **witness.snapshot())
                    if merge_witness is not None:
                        runtime_audit["qkv_merge_execution"] = merge_witness.snapshot()
                    write_json(audit_path, runtime_audit)
                    witness.validate(tpr=tpr, cp=cp, fusion=fusion)
                    if merge_witness is not None:
                        merge_witness.validate()
                audit_wgrad_completion(model, fusion=fusion)

            with merge_witness if merge_witness is not None else nullcontext():
                output, metrics = native_update(
                    engine,
                    item_batch(item),
                    loss_function,
                    torch.npu.synchronize,
                    witness=witness,
                    after_backward=after_backward,
                )
            if tpr and (
                output["metrics"]["tpr/physical_segments"] != [3.0] or output["metrics"]["tpr/forest_trees"] != [1.0]
            ):
                raise AssertionError("TPR bypassed the shared 1K prefix / two 1K suffixes")
            metrics.update(
                step=current_step,
                loss=sum(float(value) for value in output["loss"]),
                native_metrics={key: float(value) for key, value in reduce_metrics(dict(output["metrics"])).items()},
                peak_allocated_bytes=torch.npu.max_memory_allocated(),
                profiled=current_step == 1,
            )
            del output
            if current_step in CHECKPOINTS:
                sample, seconds = timed_call(torch.npu.synchronize, initial.samples, model, engine.optimizer)
                samples[current_step] = sample
                metrics["sample_observer_seconds"] = seconds
                evaluate(current_step)
            steps.append(metrics)
            runtime_audit.update(status="updated", completed_updates=current_step)
            if witness is not None:
                runtime_audit.update(witness.snapshot())
            write_json(audit_path, runtime_audit)
            if rank == 0:
                print(
                    f"TRAIN10_STEP mode={mode} step={current_step} loss={metrics['loss']:.9g} "
                    f"grad_norm={metrics['grad_norm']:.9g} lr={metrics['learning_rates_used']} "
                    f"fb_s={metrics['forward_backward_seconds']:.3f} opt_s={metrics['optimizer_seconds']:.3f} "
                    f"peak_GiB={metrics['peak_allocated_bytes'] / 2**30:.2f}",
                    flush=True,
                )
        runtime_audit.update(status="ten_updates_validated", optimizer=audit_adam_updates(engine.optimizer, STEPS))
        write_json(audit_path, runtime_audit)
        references = {name: root / name / "updates" for name in ("base_cp1", "base_cp2") if name != mode}
        if mode == "base_cp1":
            references = {}
        if repeating:
            references = {"base_cp1": repeat_context["source"] / "base_cp1/updates"}
        if repeating and not repeat_context["full_update_available"]:
            update, update_seconds = None, 0.0
            print(
                f"TRAIN10_SUPPLEMENT_FINAL_SCAN mode={mode} skipped=historical_full_reference_unavailable", flush=True
            )
        else:
            if rank == 0:
                print(f"TRAIN10_FINAL_SCAN mode={mode} all_parameters=true", flush=True)
            update, update_seconds = timed_call(
                torch.npu.synchronize,
                initial.full_update,
                model,
                engine.optimizer,
                folder / "updates",
                rank,
                references,
                save_reference=mode in ("base_cp1", "base_cp2"),
            )
        torch.save(samples, str(folder / f"samples_rank{rank}.pt"))
        if rank == 0:
            torch.save(probes, str(folder / "probes.pt"))
        report = dict(
            mode=mode,
            rank=rank,
            cp=cp,
            controls=controls,
            ring_merge_settings=merge_settings,
            input_sha256=fixture["sha256"],
            initial_weights_sha256=initial.digest,
            packages=_versions(),
            source=source_metadata(),
            steps=steps,
            evaluations=evaluations,
            full_update=update,
            layout=initial.layout,
            timings=dict(
                build_seconds=build_seconds,
                initial_snapshot_seconds=initial_seconds,
                fixture_seconds=fixture_seconds,
                final_update_observer_seconds=update_seconds,
                total_seconds=time.perf_counter() - started,
            ),
        )
        write_json(folder / f"rank{rank}.json", report)
        dist.barrier()
        if rank == 0:
            if repeating:
                complete_repeat(root, repeat_context)
            else:
                complete_mode(root, mode, cp)
        dist.barrier()
    except Exception as error:
        runtime_audit.update(status="failed", step=current_step, error=f"{type(error).__name__}: {error}")
        write_json(audit_path, runtime_audit)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
