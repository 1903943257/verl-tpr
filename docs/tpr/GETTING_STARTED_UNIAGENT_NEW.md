# VERL-TPR × UniAgent：新服务器 Docker 开发与 E2E 指引

> **2026-10-26 新环境修订**：本指南以正在运行的 `uniagent:qwen36` 容器为起点。镜像根目录已有 `/Megatron-LM`、`/Megatron-Bridge`、`/MindSpeed`、`/verl`、`/uni-agent`、`/vllm-ascend`。不要无目的地重复克隆这些依赖。
>
> 当前文档是**部署流程和待验收清单**，不是新机 E2E PASS 报告。此阶段仅处理单节点路径；现有 E2E 脚本中 `NNODES=1`，四机 16 卡（三机推理、一机训练）另行开发和验收。
>
> 本文件保留旧文件名 `GETTING_STARTED_UNIAGENT_NEW.md` 以兼容仓库已有引用，但**实际镜像是 `uniagent:qwen36`**。

## 1. 文件系统：宿主机映射与镜像自带路径

```text
宿主机 /ssd_3/jyb/project/uniagent_linear/  <=> 容器 /workspace/

容器 /
├── Megatron-LM/        # 镜像预装，仅检查兼容性
├── Megatron-Bridge/
├── MindSpeed/
├── verl/               # 镜像预装；不要默认为 E2E 训练运行时
├── uni-agent/          # 镜像预装；不等于 /workspace/uni-agent
├── vllm-ascend/
└── workspace/
    ├── hf_models/      # 已存在；检查实际权重文件
    ├── uni-agent/      # 当前要使用/更新的 UniAgent checkout
    ├── verl-tpr-e2e/   # TPR + E2E 脚本 + 补丁
    └── tpr_experiments/
```

**从现在起在 Docker 内执行 Git、Python 和实验命令**，因为你已在 Docker 内配置 Git 网络。镜像根目录的组件只作为可用依赖，不自动覆盖 `/workspace` 仓库。

若需核查 bind mount（**宿主机**）：

```bash
docker inspect uniagent_16npu \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

如需重新创建容器，宿主机运行 `scripts/tpr/run_uniagent_new_container.sh`，默认镜像 `uniagent:qwen36`，默认将 `/ssd_3/jyb/project/uniagent_linear` 挂载到 `/workspace`。**已有容器不需要重建**；脚本不自动覆盖同名容器。

## 2. Docker 内确认仓库及版本（不覆盖已有目录）

```bash
cd /workspace
ls -ld hf_models uni-agent verl-tpr-e2e

# 检查现有 UniAgent 是否为 Git 仓库（不是就先辨别目录来源）
git -C /workspace/uni-agent rev-parse --show-toplevel
git -C /workspace/uni-agent remote -v
git -C /workspace/uni-agent status --short --branch
git -C /workspace/uni-agent log -1 --oneline

# 当前 TPR E2E 工作分支
git -C /workspace/verl-tpr-e2e status --short --branch
git -C /workspace/verl-tpr-e2e log -1 --oneline
```

只有当 `/workspace/uni-agent` 不存在时，才新建：

```bash
git clone https://github.com/verl-project/uni-agent.git /workspace/uni-agent
```

若 `/workspace/uni-agent` 已是干净的 Git checkout，更新之前先看它与远端差异：

```bash
git -C /workspace/uni-agent fetch origin
git -C /workspace/uni-agent status --short --branch
git -C /workspace/uni-agent log --oneline --max-count=5
```

确认当前分支、上游与本地修改后再用 `git pull --ff-only`。**最新 UniAgent 与当前 TPR fork 的 V1/TQ 兼容性仍需实测**；不要为了追最新而丢掉已经验证的旧版本，也不要 `git reset --hard`。

E2E 文档和修复已提交在 `feat/tpr-rebench-e2e-20261026`，已有目录更新即可：

```bash
git -C /workspace/verl-tpr-e2e fetch origin \
  feat/tpr-rebench-e2e-20261026
git -C /workspace/verl-tpr-e2e status --short --branch
# 当前分支确为 feat/tpr-rebench-e2e-20261026 且无本地修改时：
git -C /workspace/verl-tpr-e2e pull --ff-only
```

## 3. Bridge 不需要：Session Hard Cap 脚本已移入 TPR 仓库

TPR 本身不依赖 Bridge 源码。此前唯一实际需要的 Bridge 文件是 `apply_uniagent_session_hardcap.py`；现在已收纳到：

```text
/workspace/verl-tpr-e2e/patches/apply_uniagent_session_hardcap.py
```

**先只检查兼容性，不修改 UniAgent：**

```bash
python /workspace/verl-tpr-e2e/patches/apply_uniagent_session_hardcap.py \
  --check /workspace/uni-agent
```

如果检查通过，正式 E2E 的启动脚本会按显式 `SESSION_HARD_CAP=True` 应用该补丁。该脚本按历史 UniAgent Gateway/Framework 代码锚点工作，因此**最新版 UniAgent 不一定适用**；`--check` 失败时应先定位代码差异，不能强行忽略。旧的 `bridge/patches/apply_uniagent_cc_tpr.py` 引用已废弃的 TPR 模块路径，**不能再用于当前 TPR main**。

## 4. 为什么 UniAgent 和 TPR 是两个 checkout？

当前正式路径选择为：

```text
UniAgent / Agent 框架    /workspace/uni-agent
VERL / Megatron TPR     /workspace/verl-tpr-e2e
```

`/workspace/uni-agent` 内可能有原生 VERL 子模块 `verl/`，镜像也有 `/verl`。这两份都**不应默默成为 TPR Actor 的实际运行时**。

E2E launcher 已改成 `ROOT=/workspace/uni-agent`，并在 `/workspace/verl-tpr-e2e` 目录中运行 `python -m verl.trainer.main_ppo`，使 VERL 从 TPR fork 导入。脚本会检查 `verl` 与 `uni_agent` 两个包的实际 import 路径，否则直接失败。

在 Docker 内可先自行检查：

```bash
cd /workspace/verl-tpr-e2e
export ROOT=/workspace/uni-agent
export TPR_VERL_ROOT=/workspace/verl-tpr-e2e
export PYTHONPATH="/workspace/verl-tpr-e2e:/workspace/uni-agent:${PYTHONPATH:-}"

python - <<'PY'
import importlib.util
for name in ("verl", "uni_agent", "torch_npu", "vllm", "ray", "mbridge"):
    try:
        spec = importlib.util.find_spec(name)
        print(name, "=>", getattr(spec, "origin", None))
    except Exception as exc:
        print(name, "ERROR", repr(exc))
PY
```

如果需要安装可编辑源码，先查看镜像现有 Python 环境，不要直接重装 torch/torch-npu/CANN：

```bash
python -m pip list | grep -Ei 'torch|torch-npu|ray|vllm|megatron|mbridge|swebench'
python -m pip show uni-agent verl || true
```

确认依赖齐全后，才考虑 `python -m pip install --no-deps -e /workspace/uni-agent` 和 `-e /workspace/verl-tpr-e2e`。`--no-deps` **不会补齐缺失依赖**，只是避免意外升级底层 NPU 软件栈；真正缺少的包要根据固定版本清单安装。

## 5. Ascend 与模型、数据、Sandbox 预检

```bash
# Docker 内
npu-smi info

python - <<'PY'
import torch
print("torch:", torch.__version__)
try:
    import torch_npu
    print("torch_npu:", torch_npu.__version__)
    print("NPU available:", torch.npu.is_available(), "count:", torch.npu.device_count())
except Exception as exc:
    print("NPU ERROR:", repr(exc))
PY

docker info >/dev/null && echo 'Docker sandbox daemon OK'

test -f /workspace/hf_models/Qwen3-4B-128K/config.json &&
  echo 'Model config OK'
test -f /workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet &&
  echo 'Normalized data OK'
```

当前模型目录名仍按旧实验 `Qwen3-4B-128K`，但目录存在不代表权重已迁移。128K 版本需要匹配的 RoPE/YaRN 模型配置；如果只有官方较短上下文配置，不能只通过增大 `max_model_len` 就当作等价模型。

本分支的本地 Docker SWE-reBench 配置：

```text
/workspace/verl-tpr-e2e/examples/tpr/task_config_swe_rebench_docker.yaml
```

每条 SWE-reBench 数据还引用独立的 Docker sandbox 镜像；例如 `ASPP__pelita-875` 对应 `swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest`。主镜像 `uniagent:qwen36` 并不能代替该实例镜像：

```bash
docker image inspect swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
```

如果镜像缺失，必须通过可信 registry 获取准确实例镜像（保留 image digest），不能随便用别的 Python 镜像代替。

## 6. 单任务推理与 Reward（先于 PPO）

确认模型、数据、Docker daemon、实例镜像和 UniAgent 框架依赖齐全后，在 Docker 内：

```bash
cd /workspace/verl-tpr-e2e
export PYTHONPATH="/workspace/verl-tpr-e2e:/workspace/uni-agent:${PYTHONPATH:-}"
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export DATA_PATH=/workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet
export TASK_CONFIG=/workspace/verl-tpr-e2e/examples/tpr/task_config_swe_rebench_docker.yaml

mkdir -p /workspace/tpr_experiments/shortfix/rollout_smoke1

python /workspace/uni-agent/examples/inference/parallel_infer_verl.py \\
  --data-path "$DATA_PATH" \\
  --model-path "$MODEL_PATH" \\
  --task-config "$TASK_CONFIG" \\
  --n 1 --limit 1 \\
  --response-length 40960 \\
  --log-dir /workspace/tpr_experiments/shortfix/rollout_smoke1 \\
  --result-path /workspace/tpr_experiments/shortfix/rollout_smoke1/result.json
```

这是推理与 Reward 的 Smoke，不是 PPO。数据第一条未必是 pelita；请根据 `result.json` 和日志的 `instance_id` 核对实际测试任务。旧镜像的自定义 CLI 及依赖若与当前 public UniAgent 有差异，需要以脚本实际 `--help` 和预检结果为准。

## 7. 单机 Full / TPR E2E

当且仅当上面预检成功，且 Session Hard Cap 补丁适配当前代码时再运行。以下示例假设**当前节点可见四张 Ascend NPU**：

```bash
cd /workspace/verl-tpr-e2e
export ROOT=/workspace/uni-agent
export TPR_VERL_ROOT=/workspace/verl-tpr-e2e
export TASK_CONFIG=/workspace/verl-tpr-e2e/examples/tpr/task_config_swe_rebench_docker.yaml
export TRAIN_FILE=/workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export REBENCH_INSTANCE_IDS=ASPP__pelita-875
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export N_GPUS_PER_NODE=4 TRAIN_TP=2 TRAIN_CP=2
export TRAIN_PROMPT_BSZ=1 GEN_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1
export N=8 TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1
export CAPTURE_ACTOR=False

SCRIPT=tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh

# 只做环境预检，不启动 Ray/NPU；硬上限补丁仅 --check
SESSION_HARD_CAP=True PRECHECK_ONLY=True TPR_ENABLED=False \\
  RUN_NAME=pelita-precheck bash "$SCRIPT"

# 正式 E2E 首先 Full；不会凭空假设新环境已通过
SESSION_HARD_CAP=True TPR_ENABLED=False \\
  RUN_NAME=pelita-full-step1 bash "$SCRIPT"

# Full PASS 之后，再跑 TPR；使用新的 RUN_NAME
SESSION_HARD_CAP=True TPR_ENABLED=True \\
  RUN_NAME=pelita-tpr-step1 bash "$SCRIPT"
```

注意 `RUN_NAME` 和对应实验产物目录不可重复使用；当前 launcher 中 `REBENCH_INSTANCE_IDS` 会在新实验目录中生成筛选后数据。尤其要审核每个任务的 reward、完整 N=8 分组、真实 Actor optimizer step 和实际 TPR dispatch。独立采样的 Full/TPR E2E 时间差**不能直接等同于相同轨迹上的 TPR 算法加速比**。

现有脚本会在正式启动前执行本地 `ray stop --force`，**不要在共享的多机 Ray 集群上直接运行**。四机三推理一训练需要单独适配资源放置和权重同步，不能仅修改 `NNODES`。

## 8. 版本与验收记录

```text
Host path=/ssd_3/jyb/project/uniagent_linear
Docker workspace=/workspace
Image=uniagent:qwen36 (record docker image digest)
UniAgent root=/workspace/uni-agent; Git SHA=...
TPR VERL=/workspace/verl-tpr-e2e; Git SHA=...
Megatron/MindSpeed/vLLM-Ascend paths & versions=...
torch/torch_npu/CANN/driver=...
Session hardcap --check=PASS/FAIL
Model config & checkpoint SHA=...
Dataset parquet SHA / instance_id=...
Sandbox image digest=...
NPU smoke=PASS/FAIL
Single-task inference & reward=PASS/FAIL
Full PPO 1-step optimizer=PASS/FAIL
TPR PPO 1-step optimizer=PASS/FAIL
Controlled identical-capture F/B performance=...
Four-node placement=NOT VALIDATED
```

后续实验详见 [REBENCH_E2E.md](../../tests/models/mcore/tpr/integration/REBENCH_E2E.md)，TPR 模块设计见 [TPR 开发文档](TPR开发文档.md)。
