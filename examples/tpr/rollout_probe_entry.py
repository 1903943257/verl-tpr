#!/usr/bin/env python3
"""Reuse UniAgent's own parallel_infer_verl CLI with strict session hard cap.

All inference/Ray/TransferQueue logic stays in upstream UniAgent. This entry
only opts in to the vendored Gateway hard-stop and bounds vLLM scheduler size.
The shell launcher applies/verifies the UniAgent patch *before* this process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

root = Path(os.getenv("UNIAGENT_ROOT", "/workspace/uni-agent"))
infer_dir = root / "examples" / "inference"
if not (infer_dir / "parallel_infer_verl.py").is_file():
    raise SystemExit(f"UniAgent inference entry not found: {infer_dir}")
sys.path.insert(0, str(infer_dir))

from uni_agent.framework.framework import GatewayAgentFramework

if not hasattr(GatewayAgentFramework, "_await_runner_with_hard_cap"):
    raise SystemExit(
        "Strict session cap is not installed. Run "
        "python patches/apply_uniagent_session_hardcap.py /workspace/uni-agent "
        "before starting the probe."
    )

import parallel_infer_verl as upstream

_original_init_config = upstream.init_config


def strict_init_config(args, *, served_model_name):
    cfg = _original_init_config(args, served_model_name=served_model_name)
    af_path = "actor_rollout_ref.rollout.custom.agent_framework"
    OmegaConf.update(
        cfg, f"{af_path}.hard_stop_on_trajectory_capacity",
        True, force_add=True,
    )
    rollout_path = "actor_rollout_ref.rollout"
    OmegaConf.update(
        cfg, f"{rollout_path}.max_num_seqs",
        args.concurrency, force_add=True,
    )
    OmegaConf.update(
        cfg, f"{rollout_path}.max_num_batched_tokens",
        args.prompt_length + args.response_length, force_add=True,
    )
    OmegaConf.update(
        cfg, f"{rollout_path}.enable_chunked_prefill",
        True, force_add=True,
    )
    print(
        "ROLLOUT_PROBE_CONFIG "
        f"n={args.n} concurrency={args.concurrency} tp={args.tensor_parallel_size} "
        f"capacity={args.prompt_length + args.response_length} "
        f"gpu_memory_utilization={args.gpu_memory_utilization} "
        "strict_session_cap=1 vllm_stats=1",
        flush=True,
    )
    return cfg


upstream.init_config = strict_init_config

if __name__ == "__main__":
    upstream.main()
