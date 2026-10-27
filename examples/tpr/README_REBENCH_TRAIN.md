# UniAgent SWE-reBench → VERL/Megatron one-step train (Ascend ARM64)

Run the verified Qwen3-4B training topology with **one public entrypoint**:

- **Full:** `MODE=full bash examples/tpr/run_train_rebench.sh`
- **TPR:** `MODE=tpr bash examples/tpr/run_train_rebench.sh`
- **Preflight:** `MODE=tpr CHECK_ONLY=1 bash examples/tpr/run_train_rebench.sh`

The script delegates all Hydra training configuration to
`tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh`.
It is intentionally small and does **not** duplicate the trainer implementation.

## Pinned experimental settings

Qwen3-4B BF16; 8 healthy Ascend 910B4 NPUs with 32 GiB each;
training TP2×CP4×DP1; vLLM rollout TP2;
one task `ASPP__pelita-875` with `rollout.n=2`;
one PPO update; prompt budget 16,384 / response budget 8,192;
activation full recomputation; CPU param/optimizer offloading;
`max_concurrent_sessions=2`, `max_num_seqs=2`;
online Claude Code + local ARM64 Docker sandbox + offline Reward.

The default TRAIN_FILE is
`/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64_train.parquet`.
It must be a **prebuilt one-row** normalized task dataset whose
`extra_info.tools_kwargs.task.sandbox.image` is
`uniagent-swe-pelita-875:latest`, and whose reward metadata does not
run `pip install -e .`. The wrapper validates these properties.
Users can set `TRAIN_FILE=/my/data.parquet`, `MODEL_PATH=/my/model`,
`UNIAGENT_ROOT=/my/uni-agent`, `TASK_CONFIG=/my/task.yaml`, and
`NPU_IDS=0,1,2,3,4,5,6,7` as needed.

Requirements:
- Runtime is the matching TPR VERL checkout and pinned UniAgent source.
- `mbridge==0.15.1`, matching Megatron-Core/MindSpeed and Ascend stack.
- An existing NPU-enabled container with the host Docker socket available.
- Docker daemon has `uniagent-swe-pelita-875:latest` (`linux/arm64`),
  with Claude Code and the reward test dependencies preinstalled.
- At least eight healthy, available NPUs; no other local Ray job (canonical
  launcher issues `ray stop --force`).

## Run and inspect

```bash
cd /workspace/verl-tpr-e2e
MODE=tpr CHECK_ONLY=1 bash examples/tpr/run_train_rebench.sh
MODE=tpr bash examples/tpr/run_train_rebench.sh
```

Each invocation creates a unique directory
`/workspace/tpr_experiments/e2e/rebench-4b-16k8k-<mode>-.../`
with `launcher.log` and `exit_code.txt`. Confirm
`training/global_step:1`, `actor/grad_norm`, `actor/loss`,
`timing_s/gen`, `timing_s/update_actor` and
`actor/perf/max_memory_allocated_gb`.

### Limitations

- This is **online Full/TPR E2E smoke**, not identical-trajectory
  speed benchmarking. Each invocation samples different Claude Code
  trajectories and may have dramatically different lengths and rewards.
- The ARM64 machine's current UniAgent checkout does not accept the old
  strict Session Hard Cap patcher. `SESSION_HARD_CAP=False` is pinned;
  Gateway trajectory **materialization** at 24,576 tokens remains active.
  It is not a session-level time or token limit.
- One long-running or 24K-cap trajectory can dominate
  `timing_s/gen`, even when the other trajectory finishes fast.
- `rollout.n=2` sessions use asynchronous gather under a concurrency
  semaphore. Their actual vLLM replica placement and effective parallelism must be
  verified from Ray/vLLM logs. `NUM_AGENT_WORKERS=1` does not imply serial task execution.
- For a rigorous TPR speedup comparison, collect one trajectory batch,
  replay **that same batch** through Full and TPR, and separately report
  inference/reward/Actor F/B/weight sync times.
