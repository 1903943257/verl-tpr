# SWE-reBench PPO E2E (Ascend / UniAgent)

This branch is based on **latest `main`**, not on the old SWE-reBench
screening branch. It carries the existing short-fix screener and adds an
online Full/TPR launcher and real-prefix audit. **No NPU E2E result has been
validated by this commit.** This is a runnable experiment definition.

## 1. Pick real instances and prepare the right data format

The earlier local Parquet may have legacy
`extra_info.tools_kwargs.{env,reward}` entries. The current UniAgent
framework requires `extra_info.tools_kwargs.task`. Convert **before**
starting the trainer:

```bash
python -m pytest -q tests/models/mcore/tpr/parallel/test_filter_swe_rebench_shortfix_cpu.py

python tests/models/mcore/tpr/parallel/filter_swe_rebench_shortfix.py \
  /workspace/data/swe_agent/swe_rebench_filtered_local.parquet \
  --out /workspace/tpr_experiments/shortfix \
  --limit 50
```

The E2E launcher reads the **normalized** file by default:

```
/workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet
```

The screen uses gold-patch size and test count as proxies for easy tasks.
It **does not** claim an actual token P:S ratio. Inspect each candidate's
`extra_info.tools_kwargs.task.sandbox.image`: this is the *per-instance*
SWE-reBench image, not one global container image. For example, an installed
image `swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest` only covers
its corresponding instance; do not assume every candidate image is installed.
The image must already contain the correct `/testbed` checkout.

For a controlled run, `REBENCH_INSTANCE_IDS` selects exact instance IDs
from the normalized `TRAIN_FILE` and writes a separate, immutable Parquet to
the run directory. For example, use
`REBENCH_INSTANCE_IDS=aspp__pelita-875` with **DP1**, or provide two
comma-separated instance IDs with the default DP2 configuration. The ID
must match `task.metadata.instance_id` exactly. Do not repeat identical
dataset rows to inflate the number of independent tasks.

## 2. Check the new environment before launching Ray

Use the actual UniAgent/MBridge image, task config, model path, and supported
sandbox backend. The default task config is:

```
/workspace/uni-agent-cc/examples/claude_code_swe_task/task_config_claude_code_swe_rebench_docker.yaml
```

Preflight (does not stop Ray, start a trainer, or allocate an NPU):

```bash
cd /workspace/verl-tpr-main  # or your checkout of feat/tpr-rebench-e2e-20261026
PRECHECK_ONLY=True SESSION_HARD_CAP=False \
  bash tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh
```

Preflight checks data schema, instance images, model config and dependencies.
For actual runs, `SESSION_HARD_CAP=True` is the default and requires
`patches/apply_uniagent_session_hardcap.py` from the prior UniAgent Bridge
setup; set `BRIDGE_ROOT` or `UNIAGENT_SESSION_PATCHER` to its **real** location.
Do not disable the cap just to bypass a missing patch during a 48K session.

**Runtime checkout warning:** this launcher keeps the original UniAgent-CC
setup, so `python -m verl.trainer.main_ppo` imports code from
`/workspace/uni-agent-cc/verl`, not automatically from the checkout where
you read the new script. Confirm that the runtime UniAgent VERL has the
matching TPR production implementation before interpreting `TPR_ENABLED=True`
as a valid E2E test. The launcher prints both checkout paths. Do not replace
the UniAgent fork blindly with vanilla VERL if that breaks the V1/TQ trainer.

## 3. Online Full vs TPR

The default parallelism is **TP=2, CP=2, DP=2 on eight NPUs**. Thus a normal
online training batch needs at least **two independent prompts**, with
`TRAIN_PROMPT_BSZ=GEN_BATCH_SIZE=PPO_MINI_BATCH_SIZE=2`.
Keep N=8 rollouts **per prompt**. For exactly one prompt group, use four NPUs
with TP2/CP2/DP1, or explicitly validate a TP2/CP4/DP1 run on eight NPUs.

Smoke (one full PPO step; the same command accepts `TPR_ENABLED=True`):

```bash
SCRIPT=tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh
N=8 N_GPUS_PER_NODE=8 TRAIN_CP=2 TOTAL_TRAINING_STEPS=1 \
  TPR_ENABLED=False CAPTURE_ACTOR=False \
  RUN_NAME=rebench-full-smoke bash "$SCRIPT" 2>&1 | tee /workspace/rebench-full-smoke.log

N=8 N_GPUS_PER_NODE=8 TRAIN_CP=2 TOTAL_TRAINING_STEPS=1 \
  TPR_ENABLED=True CAPTURE_ACTOR=False TPR_TREE_POLICY=qwen_turn \
  RUN_NAME=rebench-tpr-smoke bash "$SCRIPT" 2>&1 | tee /workspace/rebench-tpr-smoke.log
```

Once both smoke paths pass, test `TOTAL_TRAINING_STEPS=10` for Full and
TPR using the same `TRAIN_FILE`, `TASK_CONFIG`, `N`, token caps, model,
recompute, and parallel topology. Use distinct `RUN_NAME` values; do not
reuse actor captures or checkpoints.

The script enables genuine online Actor update by setting
`skip.rollout_tq.enable=False`; it does not present a cached TQ replay or
inference-only timing as PPO E2E. The `ACTOR_RECOMPUTE` knob defaults to
`True` for both arms. `CAPTURE_ACTOR=False` avoids snapshot I/O during
timed E2E comparisons.

## 4. Find the useful *measured* P:S task

For discovery, repeat short candidate runs with `CAPTURE_ACTOR=True`.
This requires the previously patched actor-update capture hook in
`/workspace/uni-agent-cc/verl/verl/workers/engine_workers.py` and writes
captures under:

```
/workspace/tpr_experiments/e2e/<RUN_NAME>/actor_captures/
```

After a discovery run:

```bash
python tests/models/mcore/tpr/integration/report_rebench_prefix.py \
  --captures /workspace/tpr_experiments/e2e/<RUN_NAME>/actor_captures \
  --expected-n 8 \
  --out /workspace/tpr_experiments/e2e/<RUN_NAME>/prefix_stats.csv
```

The report merges all actor mini-batches within the same PPO step and UID,
deduplicates by `session_id`, and outputs the exact common-token prefix
and the per-trajectory remainder, **without zero-padding**. Its
`root_only_dedup_fraction` is an LCP-based proxy and does **not** measure
the actual Qwen-turn tree's unique tokens.

Choose cases based on actual:
- Complete N=8 groups and consistent instance/image mapping
- Suffix distributions (not just mean) and nontrivial PPO supervision
- Reward diversity and valid task outcomes, audited separately
- Measured P:S with a spread of long-prefix/short-suffix and low-reuse
  controls, without editing or truncating successful trajectories

Do not confuse short gold patches with short model completions. No automatic
reward-variance result is available from this prefix analyzer. If Full and
TPR generate independently, online E2E wall time includes rollout and tool
environment variance: compare repeated runs **and** separately compare
training time on *identical captured trajectories* when attributing a TPR
speedup. Stage 1 screening results and online timing should be reported
as different measurements.
