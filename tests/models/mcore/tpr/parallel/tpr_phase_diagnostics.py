"""Test-only, opt-in stage sampling for CP1/CP4 Full versus TPR.

NPU events are enqueued on the current stream (no internal synchronize);
these timings are diagnostic, NOT production F/B speedups. Hardware kernel
durations/communication overlap are read separately from optional CANN trace.
No production Executor/Attention/PPO implementation is modified.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch


class PhaseSampler:
    def __init__(self, model, *, mode: str):
        self.model, self.mode = model, mode
        self._stack = ExitStack()
        self._pending = []

    @contextmanager
    def phase(self, name: str):
        # Event timings use the host's current NPU stream; they do not measure
        # simultaneous non-default HCCL streams or every host dispatch cost.
        start, end = (torch.npu.Event(enable_timing=True),
                      torch.npu.Event(enable_timing=True))
        with torch.autograd.profiler.record_function("TPR_PHASE/" + name):
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending.append((name, start, end))

    def __enter__(self):
        self._stack.__enter__()
        try:
            original_forward = self.model.forward
            @wraps(original_forward)
            def forward(*args, **kwargs):
                with self.phase("forward"):
                    return original_forward(*args, **kwargs)
            self._stack.enter_context(patch.object(self.model, "forward", forward))

            native_cp = int(getattr(self.model.config, "context_parallel_size", 1))
            # Native Megatron does not necessarily call Python's
            # torch.autograd.backward; mark the actual schedule backward_step.
            # The CP1 controlled Full and TPR Executor use autograd.backward.
            if self.mode == "full" and native_cp > 1:
                import megatron.core.pipeline_parallel.schedules as schedule
                if not hasattr(schedule, "backward_step"):
                    raise RuntimeError("Native schedule missing backward_step profiling hook")
                original_backward = schedule.backward_step
                @wraps(original_backward)
                def backward(*args, **kwargs):
                    with self.phase("backward"):
                        return original_backward(*args, **kwargs)
                self._stack.enter_context(patch.object(schedule, "backward_step", backward))
            else:
                original_backward = torch.autograd.backward
                @wraps(original_backward)
                def backward(*args, **kwargs):
                    with self.phase("backward"):
                        return original_backward(*args, **kwargs)
                self._stack.enter_context(patch.object(torch.autograd, "backward", backward))

            from verl.workers.engine.megatron.tpr.objective import SegmentPPOObjectiveAdapter
            original_ppo = SegmentPPOObjectiveAdapter.compute_loss
            @wraps(original_ppo)
            def ppo(adapter, *args, **kwargs):
                with self.phase("ppo_loss"):
                    return original_ppo(adapter, *args, **kwargs)
            self._stack.enter_context(patch.object(SegmentPPOObjectiveAdapter, "compute_loss", ppo))

            from . import native_full_megatron_schedule as native
            original_native_ppo = native._compact_native_ppo
            @wraps(original_native_ppo)
            def compact_ppo(*args, **kwargs):
                with self.phase("ppo_loss"):
                    return original_native_ppo(*args, **kwargs)
            self._stack.enter_context(patch.object(native, "_compact_native_ppo", compact_ppo))
            # Native Full computes TP-sharded log-probs outside the compact
            # PPO loss; capture this as the same non-overlapping loss category.
            from verl.utils.megatron import tensor_parallel as tp_logprobs
            original_logprobs = tp_logprobs.vocab_parallel_log_probs_from_logits
            @wraps(original_logprobs)
            def logprobs(*args, **kwargs):
                with self.phase("ppo_loss"):
                    return original_logprobs(*args, **kwargs)
            self._stack.enter_context(patch.object(
                tp_logprobs, "vocab_parallel_log_probs_from_logits", logprobs,
            ))

            finalize = getattr(self.model.config, "finalize_model_grads_func", None)
            if callable(finalize):
                @wraps(finalize)
                def wrapped_finalize(*args, **kwargs):
                    with self.phase("gradient_finalize"):
                        return finalize(*args, **kwargs)
                self._stack.enter_context(
                    patch.object(self.model.config, "finalize_model_grads_func", wrapped_finalize)
                )
            return self
        except BaseException:
            self._stack.__exit__(*__import__("sys").exc_info())
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def materialize(self):
        """Call only AFTER caller synchronizes the NPU at F/B completion."""
        by_name = defaultdict(list)
        for name, start, end in self._pending:
            by_name[name].append(float(start.elapsed_time(end)))
        # Opt-in individual entries: CP4 native Full invokes forward/backward
        # per Session; TPR invokes forward for Push/Leaf/Pop and backward for
        # Leaf/Pop. Keep the existing aggregate schema unchanged otherwise.
        per_call = os.getenv("TPR_LEAF_SESSION_DIAG", "0") == "1"
        return {
            key: {
                "npu_event_sum_ms": sum(vals),
                "events": len(vals),
                **({"per_call_npu_event_ms": vals} if per_call else {}),
            }
            for key, vals in by_name.items()
        }


class OperatorTrace:
    """Opt-in per-rank profiler for one measured iteration; no production changes.

    Default: legacy Kineto Chrome trace (preserved for existing tests).
    TPR_OPERATOR_TRACE_FORMAT=ascend_level1: native Ascend Level1 profiler,
    with device kernels and per-rank communication reports. This deliberately
    avoids interpreting CPU HCCL launch events as device communication time.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self.profiler = None
        self.format = os.getenv("TPR_OPERATOR_TRACE_FORMAT", "kineto")

    def __enter__(self):
        if self.path is None:
            return self
        import torch_npu
        activities = [
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ]
        if self.format == "ascend_level1":
            self.path.mkdir(parents=True, exist_ok=True)
            self.profiler = torch_npu.profiler.profile(
                activities=activities,
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1,
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(self.path),
                ),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profiler.__enter__()
        elif self.format == "kineto":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.profiler = torch_npu.profiler._KinetoProfile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profiler.start()
        else:
            raise ValueError(
                f"TPR_OPERATOR_TRACE_FORMAT must be kineto or ascend_level1, "
                f"got {self.format!r}"
            )
        return self

    def __exit__(self, *exc):
        if self.profiler is None:
            return False
        if self.format == "ascend_level1":
            try:
                if exc[0] is None:
                    # A single active step mirrors the previously validated
                    # two-rank Ascend Level1 smoke test.
                    self.profiler.step()
            finally:
                self.profiler.__exit__(*exc)
            if exc[0] is None:
                # Communication reports are not guaranteed for every profiler
                # export. In particular, a device-side kernel failure can
                # interrupt export. Never replace the original NPU failure
                # with a spurious "communication.json missing" assertion.
                reports = list(self.path.rglob("ASCEND_PROFILER_OUTPUT/communication.json"))
                kernels = list(self.path.rglob("ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
                traces = list(self.path.rglob("ASCEND_PROFILER_OUTPUT/trace_view.json"))
                if not reports:
                    print(
                        f"TPR_ASCEND_LEVEL1_WARNING path={self.path} "
                        "communication.json missing; verify NPU device logs "
                        "and profile completeness before interpreting timings",
                        flush=True,
                    )
                if not kernels and not traces:
                    print(
                        f"TPR_ASCEND_LEVEL1_WARNING path={self.path} "
                        "no device-kernel timeline exported; "
                        "the profile is incomplete",
                        flush=True,
                    )
                print(
                    f"TPR_ASCEND_LEVEL1_OUTPUT path={self.path} "
                    f"communication_reports={len(reports)} "
                    f"kernel_details={len(kernels)} trace_view={len(traces)}",
                    flush=True,
                )
            return False
        self.profiler.stop()
        if exc[0] is None:
            self.profiler.export_chrome_trace(str(self.path))
            if not self.path.exists():
                raise RuntimeError(f"Profiler did not export NPU trace: {self.path}")
        return False


def trace_path(output_dir: Path, mode: str, iteration: int, rank: int) -> Path | None:
    if os.getenv("TPR_OPERATOR_TRACE", "0") != "1":
        return None
    ranks = tuple(int(x) for x in os.getenv("TPR_TRACE_RANKS", "0").split(","))
    if rank not in ranks:
        return None
    if os.getenv("TPR_OPERATOR_TRACE_FORMAT", "kineto") == "ascend_level1":
        return output_dir / "ascend_profiler" / f"{mode}_iter{iteration}_rank{rank}"
    return output_dir / "trace" / f"{mode}_iter{iteration}_rank{rank}.json"


def trace_kernel_summary(path: Path | None):
    """Conservative keyword inventory, NOT an additive wall-clock breakdown.

    Event formats differ across CANN versions: on-device tasks may omit
    `cat` but carry CANN Task Type / physical stream metadata in `args`.
    Never classify CPU API latency as device HCCL/kernel time.
    """
    if path is None or not path.exists():
        return {"attention_kernel_sum_ms": None, "hccl_kernel_sum_ms": None,
                "reason": "trace_missing"}
    if path.is_dir():
        # Level1 materializes its own reports. A kernel-sum inventory of a
        # Python/CPU trace is not a substitute for the device communication CSV.
        files = list(path.rglob("ASCEND_PROFILER_OUTPUT/communication.json"))
        return {
            "attention_kernel_sum_ms": None,
            "hccl_kernel_sum_ms": None,
            "reason": "ascend_level1_raw_reports",
            "communication_reports": [str(file) for file in files],
        }
    import re
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeError, OSError) as exc:
        # An incomplete CANN export must not invalidate a completed F/B.
        # Never report kernel timings from a partial trace as complete.
        print(f"TPR_TRACE_PARSE_FAILED path={path} error={type(exc).__name__}: {exc}", flush=True)
        return {
            "attention_kernel_sum_ms": None, "hccl_kernel_sum_ms": None,
            "attention_events": 0, "hccl_events": 0,
            "reason": "trace_invalid_json",
            "trace_error": f"{type(exc).__name__}: {exc}"[:240],
        }
    events = raw.get("traceEvents") if isinstance(raw, dict) else raw
    if not isinstance(events, list):
        return {
            "attention_kernel_sum_ms": None, "hccl_kernel_sum_ms": None,
            "attention_events": 0, "hccl_events": 0,
            "reason": "trace_events_missing",
        }
    totals = defaultdict(float)
    counts = defaultdict(int)
    for item in events:
        if not isinstance(item, dict) or item.get("ph") != "X":
            continue
        category = str(item.get("cat", "")).lower()
        name = str(item.get("name", ""))
        args = item.get("args")
        args = args if isinstance(args, dict) else {}
        # CANN Chrome Trace device events can omit `cat`. Identify them
        # by device task metadata instead of treating them as host callbacks.
        has_device_task = (
            "Physic Stream Id" in args and "Task Id" in args
        ) or (
            str(args.get("Task Type", "")).upper().startswith("KERNEL_")
        )
        has_device_category = any(
            key in category
            for key in ("npu", "kernel", "ascend", "hardware", "cann", "communication")
        )
        # Explicit host CPU records are never hardware operators without
        # supporting CANN device task metadata.
        if not has_device_task and not (
            has_device_category and "cpu" not in category
        ):
            continue
        duration = item.get("dur")
        if not isinstance(duration, (int, float)) or duration <= 0:
            continue
        label = "hccl" if re.search(r"hccl|allreduce|allgather|reducescatter|sendrecv|hcom", name, re.I) else (
            "attention" if re.search(r"flashattention|fusionattention|npu_fusion_attention|\battention\b", name, re.I) else None
        )
        if label:
            totals[label] += duration / 1000
            counts[label] += 1
    return {
        "attention_kernel_sum_ms": round(totals["attention"], 3) if counts["attention"] else None,
        "hccl_kernel_sum_ms": round(totals["hccl"], 3) if counts["hccl"] else None,
        "attention_events": counts["attention"],
        "hccl_events": counts["hccl"],
        "semantics": "SUM of matching NPU trace events on one rank; overlaps possible; not wall time",
    }


def segment_event_summary(events: list[dict]) -> dict:
    """Sum executor segment event latencies by operation, per single rank.

    push = graph-free Root forward; visit_leaf includes forward + PPO +
    backward; pop includes Root recompute + gradient relay + backward.
    These NPU stream timings can overlap separate comm streams and cannot
    be equated with end-to-end elapsed or an isolated Attention/BWD total.
    """
    totals = defaultdict(float)
    counts = defaultdict(int)
    for event in events:
        kind = event["kind"]
        totals[kind] += float(event["npu_event_ms"])
        counts[kind] += 1
    return {
        key: {"npu_event_sum_ms": round(totals[key], 4), "segments": counts[key]}
        for key in ("push", "visit_leaf", "pop")
    }
