# SWE-reBench PPO E2E (Ascend / UniAgent)

> **Current Docker checkout:** See [the `/workspace/uni-agent` + `/workspace/verl-tpr-e2e` bootstrap guide](../../../../../docs/tpr/GETTING_STARTED_UNIAGENT_NEW.md). Git and Python run inside the container; the host mount is `/ssd_3/jyb/project/uniagent_linear -> /workspace`. The E2E runner explicitly uses the TPR fork as its VERL runtime. `bridge` is no longer needed; the optional Session Hard Cap patcher is in this branch. The launcher is still single-node (`NNODES=1`), not a 3-inference/1-training multi-node setup.


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

## 1a. Controlled prompt-hint experiment for pelita-875

Do not edit or regenerate the source dataset. Create a separate, one-row
variant from the existing normalized Parquet; only the selected row's final
user message is extended, while its task ID, reward metadata, and image stay
unchanged:

```bash
cd /workspace/verl-tpr-e2e
python -m pytest -q tests/models/mcore/tpr/parallel/test_add_swe_rebench_task_hint.py

python tests/models/mcore/tpr/parallel/add_swe_rebench_task_hint.py \
  --input /workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet \
  --output /workspace/tpr_experiments/shortfix/pelita875_hint_h2.parquet \
  --instance-id ASPP__pelita-875 \
  --level h2 \
  --only-target
```

The helper supports three levels: `h1` (file/function localization),
`h2` (explicit diagnosis and regression-test guidance), and `h3`
(answer-assisted/oracle-style code hint). Start with H2. Label H3 results as
assisted; do not report them as unassisted SWE-agent success.

For a one-task, one-rollout inference/Reward screen, use the generated file
as the data path with the existing Uni-Agent inference entrypoint and
`--n 1 --limit 1`. Keep the original model, task config, token cap and
reward evaluator unchanged. Save logs/results to a new run directory so
the baseline and hinted outputs cannot overwrite each other.

Before interpreting token counts, record both model-generated tokens and
the full interaction/response tokens. Compare reward correctness, generation
time, reward time and full session time. Only run N=8 after the hint produces
a correct, complete trajectory and the task's runtime is acceptable. If
testing the training launcher instead, set `TRAIN_FILE` to this one-row
variant and use a topology with DP=1; do not duplicate the row to fill DP2.

## 2. Check the new environment before launching Ray

Use the actual UniAgent/MBridge image, task config, model path, and supported
sandbox backend. The default task config is:

```
/workspace/verl-tpr-e2e/examples/tpr/task_config_swe_rebench_docker.yaml
```

Preflight (does not stop Ray, start a trainer, or allocate an NPU):

```bash
cd /workspace/verl-tpr-e2e
PRECHECK_ONLY=True SESSION_HARD_CAP=False \
  bash tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh
```

Preflight checks data schema, instance images, model config and dependencies.
For actual runs, `SESSION_HARD_CAP=True` is the default and invokes the vendored
`/workspace/verl-tpr-e2e/patches/apply_uniagent_session_hardcap.py`.
Run its `--check /workspace/uni-agent` first; on `PRECHECK_ONLY=True`, the launcher
only checks the patch and will not modify UniAgent.
Do not disable the cap just to bypass a missing patch during a 48K session.

**Runtime import gate:** `ROOT=/workspace/uni-agent` supplies the Agent code;
`TPR_VERL_ROOT=/workspace/verl-tpr-e2e` supplies the **actual** VERL trainer.
The launcher checks the `verl` and `uni_agent` import source paths before Ray.
Do not interpret `TPR_ENABLED=True` as proof of a successful optimizer step;
check the training logs and parity evidence.

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
`/workspace/verl-tpr-e2e/verl/workers/engine_workers.py` and writes
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
