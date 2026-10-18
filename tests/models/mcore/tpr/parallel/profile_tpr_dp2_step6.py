# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent, fixed-weight DP2 step-6 replay; observations, not acceptance.

Launch this physical script with torchrun, even when --repo points at an older
checkout. Only the adjacent observer is new; model/runtime/PPO come from --repo.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Full host-verified 40-hex SHA")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", choices=("main", "pr7"), required=True)
    parser.add_argument("--profile", action="store_true", help="Trace only; not comparable timing")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args(argv)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.profile else 10)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a complete 40-hex host SHA")
    if not 0 <= args.warmups <= 10 or not 1 <= args.repeats <= 20:
        parser.error("warmups must be 0..10 and repeats 1..20")
    args.repo = args.repo.resolve(strict=True)
    args.reference_json = args.reference_json.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    args.revision = args.revision.lower()
    return args


def load_reference(path):
    reference = json.loads(path.read_text(encoding="utf-8"))
    expected = {"source_step": 6, "mode": "tpr", "effective_sequence_cap": 30720,
                "topology": {"tp": 2, "cp": 2, "dp": 2, "world": 8},
                "ppo_logprob_chunk_tokens": 512, "gradient_sync_reduce_dtype": "bf16",
                "updated_weights": False, "recompute": False}
    for key, value in expected.items():
        if key not in reference or reference[key] != value:
            raise ValueError(f"Reference invariant {key}: expected {value!r}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", reference.get("input_sha256", "")):
        raise ValueError("Reference requires input_sha256")
    parts = reference.get("dp_partitions", [])
    if (len(parts) != 2 or not all(parts)
            or sorted(row for part in parts for row in part) != list(range(8))):
        raise ValueError("Reference partitions must cover original 8 rows exactly once")
    if not isinstance(reference.get("global_ppo_tokens"), int) or reference["global_ppo_tokens"] <= 0:
        raise ValueError("Reference requires positive global_ppo_tokens")
    for key in ("source_capture", "checkpoint"):
        if not isinstance(reference.get(key), str) or not Path(reference[key]).exists():
            raise FileNotFoundError(f"Reference {key}: {reference.get(key)!r}")
    return reference


def configure_environment(reference, environ):
    visible = environ.get("ASCEND_RT_VISIBLE_DEVICES", "8,9,10,11,12,13,14,15")
    if (not re.fullmatch(r"[0-9]+(?:,[0-9]+){7}", visible)
            or len(set(map(int, visible.split(",")))) != 8):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain exactly 8 unique nonnegative integer IDs")
    controls = {
        "TPR_EXPERIMENTAL_OFFLINE_DP2": "1", "TPR_RUN_TP2CP2DP2_ONE_STEP": "1",
        "TPR_OFFLINE_MODE": "tpr",
        "TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS": "0", "TPR_TREE_POLICY": "qwen_turn",
        "TPR_TURN_MIN_SEGMENT_TOKENS": "128", "TPR_RING_COALESCE_PREFIX_FULL": "0",
        "TPR_RING_COALESCE_PREFIX_QUERY": "0", "TPR_PPO_LOGPROB_CHUNK_TOKENS": "512",
        "TPR_OFFLINE_GRAD_SYNC_DTYPE": "bf16", "TPR_EXEC_PROFILE": "0",
        "TPR_BENCH_CAPTURE_LOGPROBS": "0", "HCCL_DETERMINISTIC": "true",
        "TPR_QWEN_4B_PATH": reference["checkpoint"],
        "ASCEND_RT_VISIBLE_DEVICES": visible,
    }
    forbidden = ("TPR_MEM_TRACE", "TPR_BENCH_MEM_TRACE", "TPR_FIRST_LEAF_DIAG",
                 "TPR_FIRST_LEAF_DIAG_SYNC", "TPR_BENCH_GRAD_WITNESS")
    for key, value in environ.items():
        diagnostic = key in forbidden or (key.startswith("TPR_") and "TILE" in key)
        if diagnostic and value.lower() not in ("0", "false", "off", ""):
            raise ValueError(f"Conflicting diagnostic/FixedTile environment: {key}={value}")
    for key, value in controls.items():
        if key in environ and environ[key] != value:
            raise ValueError(f"Conflicting environment: {key}={environ[key]!r}; require {value!r}")
    environ.update(controls)
    return controls


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ppo_configuration(config):
    values = {}
    for name in dir(config):
        if not name.startswith("_"):
            value = getattr(config, name)
            if not callable(value):
                values[name] = copy.deepcopy(value)
    return values


def group_membership(runtime, distributed):
    return {name: list(distributed.get_process_group_ranks(getattr(runtime, attribute)))
            for name, attribute in (("tp", "tp_group"), ("cp", "cp_group"),
                                    ("dp", "dp_group"), ("dp_cp", "dp_cp_group"), ("pp", "pp_group"))}


def redacted_environment(environ):
    sensitive = re.compile(
        r"SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|(?:^|_)(?:TOKEN|KEY)(?:_|$)", re.I,
    )
    return {key: "<redacted>" if sensitive.search(key) else value
            for key, value in environ.items() if key.startswith(("TPR_", "HCCL_", "ASCEND_"))}


def revision_metadata(repo, revision):
    def git(*arguments):
        return subprocess.run(["git", "-C", str(repo), *arguments], text=True,
                              capture_output=True, check=True, timeout=15).stdout.strip()
    try:
        actual = git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError) as exc:
        return {"revision": revision, "verification": "host_declared_not_verified",
                "git_note": str(exc)}
    if actual != revision or Path(git("rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError(f"Target git HEAD/root differs from declared revision: {actual}")
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", "verl")
    if dirty:
        raise ValueError(f"Refuse dirty production sources: {dirty}")
    return {"revision": revision, "verification": "git_head_and_clean_production_verified"}


def verify_module_paths(repo, modules):
    result = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(repo):
            raise RuntimeError(f"Imported code outside --repo: {module.__name__}: {path}")
        result[module.__name__] = {"path": str(path), "sha256": sha256(path)}
    return result


def load_observer():
    path = Path(__file__).with_name("_npu_profile_capture.py")
    spec = importlib.util.spec_from_file_location("_independent_tpr_profile_capture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NPUProfileCapture


def percentile90(values):
    values = sorted(values)
    index = (len(values) - 1) * 0.9
    return values[math.floor(index)] * (1 - index % 1) + values[math.ceil(index)] * (index % 1)


def package_versions():
    versions = {}
    for name in ("torch", "torch-npu", "megatron-core", "mindspeed", "transformers", "tensordict", "verl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "distribution_metadata_unavailable"
    return versions


def runtime_without_cli(factory):
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        return factory()
    finally:
        sys.argv[:] = original_argv


def load_target_modules(repo):
    """Import every runtime dependency with MindSpeed CLI isolation already active.

    The caller must use runtime_without_cli around this function. In particular
    MindSpeed can parse and cache sys.argv during module import, *before* _runtime
    is called. Passing this driver's --profile flag to that parser changes the
    patch registration path and can duplicate the NVTX range_push patch.
    """
    replay = importlib.import_module("tests.models.mcore.tpr.parallel.test_tpr_tp2cp2dp2_one_step_npu")
    checkpoint = importlib.import_module("tests.models.mcore.tpr.parallel._qwen3_tp_checkpoint")
    actor = importlib.import_module("tests.models.mcore.tpr.parallel.test_real_actor_tp2cp4_offline_ppo_npu")
    adapter = importlib.import_module("verl.workers.engine.megatron.tpr.adapter")
    verify_module_paths(repo, [replay, checkpoint, actor, adapter])
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    parallel_state = importlib.import_module("megatron.core.parallel_state")
    ppo_loss = importlib.import_module("verl.workers.utils.losses").ppo_loss
    return replay, checkpoint, actor, adapter, torch, dist, parallel_state, ppo_loss


def main(argv=None):
    args = parse_args(argv)  # --help/invalid CLI must not import NPU or target packages.
    reference = load_reference(args.reference_json)
    controls = configure_environment(reference, os.environ)
    provenance = revision_metadata(args.repo, args.revision)
    rank = int(os.environ.get("RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) != 8 or not 0 <= rank < 8:
        raise ValueError("Launch with torchrun --nproc_per_node=8")
    rank_dir = args.output_root / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)
    # MindSpeed may initialize/capture its args on import, not just at _runtime.
    # Keep all target imports outside the driver's --profile/--repo CLI.
    (replay, checkpoint, actor, adapter, torch, dist, parallel_state, ppo_loss) = (
        runtime_without_cli(lambda: load_target_modules(args.repo))
    )

    runtime = None
    try:
        runtime = runtime_without_cli(replay._runtime)
        batch, plan, tokens, digest = replay._placement(
            Path(reference["source_capture"]), runtime, reference["global_ppo_tokens"],
        )
        if digest != reference["input_sha256"] or [list(x) for x in plan.partitions] != reference["dp_partitions"]:
            raise AssertionError("Reference input SHA or DP partitions changed")
        if max(int(x.numel()) for x in batch["input_ids"].unbind(0)) > reference["effective_sequence_cap"]:
            raise AssertionError("Input exceeds accepted 30720 sequence cap")
        model, hf = checkpoint.make_real_qwen3_tp_model(
            runtime, model_path=Path(reference["checkpoint"]), tp_size=2, cp_size=2,
            sequence_parallel=False, profile="4b", native_attention=False,
        )
        if hf.num_hidden_layers != 36:
            raise AssertionError("Requires accepted 36-layer Qwen3-4B model")
        engine = replay._engine_with_dp_finalize(model, runtime)
        actor_config = actor._PPO()
        production = [module for name, module in sys.modules.items()
                      if name.startswith("verl.") and getattr(module, "__file__", None)]
        module_paths = verify_module_paths(args.repo, [replay, checkpoint, actor, *production])
        config = {"protocol": "tpr_dp2_step6_observations_v1", "label": args.label,
                  "repo": str(args.repo), **provenance, "rank": rank,
                  "reference_json": str(args.reference_json), "reference_sha256": sha256(args.reference_json),
                  "reference": {key: reference[key] for key in (
                      "source_step", "mode", "effective_sequence_cap", "topology", "source_capture",
                      "checkpoint", "input_sha256", "global_ppo_tokens", "dp_partitions",
                      "ppo_logprob_chunk_tokens", "gradient_sync_reduce_dtype", "recompute", "updated_weights")},
                  "controls": controls, "seed": 261013,
                  "environment": redacted_environment(os.environ),
                  "warmups": args.warmups, "repeats": args.repeats, "profiler_enabled": args.profile,
                  "topology": reference["topology"], "dp_partitions": [list(x) for x in plan.partitions],
                  "input_sha256": digest, "global_ppo_tokens": tokens,
                  "source_capture_sha256": sha256(Path(reference["source_capture"])),
                  "ppo_config": ppo_configuration(actor_config),
                  "process_group_ranks": group_membership(runtime, dist),
                  "local_rank": int(os.environ["LOCAL_RANK"]), "device": str(runtime.device),
                  "coordinates": {key: getattr(runtime, key) for key in ("tp_rank", "cp_rank", "dp_rank")},
                  "versions": package_versions(), "torch_version": torch.__version__,
                  "device_name": torch.npu.get_device_name(runtime.device), "hf_config": hf.to_dict(),
                  "model_config": {key: repr(value) for key, value in vars(model.config).items()},
                  "source_files": module_paths, "updated_weights": False,
                  "tooling_files": {str(path): sha256(path) for path in (
                      Path(__file__).resolve(), Path(__file__).with_name("_npu_profile_capture.py").resolve())},
                  "diagnostic_difference": {"reference_capture_logprobs": reference.get("capture_logprobs"),
                                            "capture_logprobs": False},
                  "scope": "F/B+finalize observations only; no correctness, AdamW or E2E certification"}
        (rank_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        loss_function = partial(ppo_loss, config=actor_config)
        def callback():
            return adapter.run_tpr_forward_backward_batch(engine, batch, loss_function, forward_only=False)
        for _ in range(args.warmups):
            model.zero_grad(set_to_none=True)
            dist.barrier()
            callback()
            torch.npu.synchronize()
        capture_class = load_observer() if args.profile else None
        runs = []
        for iteration in range(args.repeats):
            model.zero_grad(set_to_none=True)
            dist.barrier()
            torch.npu.synchronize()
            baseline = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            if args.profile:
                with capture_class(rank_dir / f"trace{iteration}", metadata=config, enabled=True,
                                   engine=engine, model=model) as capture:
                    result = capture.profile_call("step6_replay", callback)
                own_seconds = None
            else:
                start = time.perf_counter()
                result = callback()
                torch.npu.synchronize()
                own_seconds = time.perf_counter() - start
            run = {"iteration": iteration, "local_seconds": own_seconds,
                   "baseline_allocated_bytes": baseline,
                   "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                   "peak_reserved_bytes": torch.npu.max_memory_reserved(),
                   "loss": float(result["loss"][0])}
            if args.profile:
                run.update(artifacts=capture.artifacts, notes=capture.notes)
            else:
                critical = torch.tensor(own_seconds, device=runtime.device, dtype=torch.float32)
                dist.all_reduce(critical, op=dist.ReduceOp.MAX)
                run["max_rank_seconds"] = float(critical.item())
            runs.append(run)
            del result
        times = [run["max_rank_seconds"] for run in runs if "max_rank_seconds" in run]
        summary = {"config": "config.json", "runs": runs, "profiler_enabled": args.profile,
                   "median_seconds": statistics.median(times) if times else None,
                   "p90_seconds": percentile90(times) if times else None,
                   "std_seconds": statistics.pstdev(times) if times else None,
                   "timing_reduction_dtype": "float32", "performance_claim": None}
        (rank_dir / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(f"TPR_STEP6_OBSERVATIONS rank={rank} output={rank_dir}", flush=True)
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
