# 从零搭建 VERL-TPR + UniAgent SWE-reBench E2E（uniagent:new）

> 状态：**迁移操作指南，不是新服务器验收通过证明**。2026-10-26；仅写入 `feat/tpr-rebench-e2e-20261026`。不修改 TPR 算法和测试整理分支。
>
> 范围：旧机 Ascend 单机实验 → 新机基于 `uniagent:new` 的 Docker；先跑单机 Full/TPR，再规划四机 16 卡。现有 `run_uniagent_rebench_ppo_e2e.sh` **固定 `NNODES=1`**，尚不能声称直接支持三机推理、一机训练。

## 0. 最先确认：新服务器到底是 Ascend 还是 NVIDIA

`docker images` 只能证明镜像存在，不能证明机器是 NPU，也不能证明 `uniagent:new` 装有 torch-npu/CANN。该镜像曾在 GPU 环境中使用；**先检查硬件，不要在 NVIDIA 节点上直接运行 trainer.device=npu**。

宿主机（不要进入 Docker）：

```bash
hostname
docker image inspect uniagent:new --format '{{.Id}} {{.Architecture}} {{.Os}}'
ls -l /dev/davinci* /dev/davinci_manager 2>/dev/null || true
command -v npu-smi && npu-smi info || true
command -v nvidia-smi && nvidia-smi -L || true
ls -ld /usr/local/Ascend/driver /usr/local/dcmi 2>/dev/null || true
```

以下流程仅针对 **Ascend NPU + 驱动已安装在宿主机**。若只有 `nvidia-smi`，停止 NPU E2E 操作：需要另做 CUDA/PyTorch/TPR 适配，不能把镜像名字当作硬件兼容证明。

## 1. 宿主机取得代码（所有 Git 在宿主机）

新服务器初始目录：`/home/j00972288/project`。沿用旧容器工作目录映射：`HOST_WORKSPACE → /workspace`。先用一个稳定根目录收纳所有工程及数据：

```bash
export WS=/home/j00972288/project/uniagent_linear
mkdir -p "$WS" "$WS/hf_models" "$WS/data/swe_agent" \
  "$WS/tpr_experiments/shortfix" "$WS/tpr_experiments/e2e"

# TPR 研发/E2E 工作树；此仓库里的脚本不是运行时 VERL 的自动替身。
git clone --branch feat/tpr-rebench-e2e-20261026 \
  https://github.com/1903943257/verl-tpr.git "$WS/verl-tpr-e2e"

# UniAgent 主项目和其 VERL 子模块。旧服务器修改过的 UniAgent 代码
# 未必等于公开 main；后续会做版本和运行时门禁。
git clone --recurse-submodules https://github.com/RichardFido/uni-agent.git \
  "$WS/uni-agent-cc"

# 旧 Session hard-cap 补丁仅在此历史仓库找到；不要运行其中旧版 TPR hook 安装器。
git clone https://github.com/1903943257/bridge.git "$WS/bridge"

git -C "$WS/verl-tpr-e2e" rev-parse HEAD
git -C "$WS/uni-agent-cc" rev-parse HEAD
git -C "$WS/uni-agent-cc/verl" rev-parse HEAD
git -C "$WS/bridge" rev-parse HEAD
```

### 运行时 VERL 必须具备 TPR

原 E2E launcher 的 `ROOT=/workspace/uni-agent-cc`，`PYTHONPATH` 指向 `ROOT/verl`，所以 `verl-tpr-e2e` 只是保存实验入口，**不等于训练真的用到了 TPR**。

从零部署的一种明确做法是将 UniAgent 子模块的运行时 VERL 切换到**同一 TPR E2E 提交**，不做不透明的 cp/rsync 覆盖；这样新机最少需要两个代码仓：

```bash
# 宿主机。先确认 uni-agent-cc/verl 是干净的子模块 checkout。
git -C "$WS/uni-agent-cc/verl" status --short
git -C "$WS/uni-agent-cc/verl" remote add tpr \
  https://github.com/1903943257/verl-tpr.git
git -C "$WS/uni-agent-cc/verl" fetch tpr \
  feat/tpr-rebench-e2e-20261026
git -C "$WS/uni-agent-cc/verl" checkout --detach FETCH_HEAD
git -C "$WS/uni-agent-cc/verl" rev-parse HEAD
git -C "$WS/verl-tpr-e2e" rev-parse HEAD
```

如果 `tpr` remote 已存在，不重复 `remote add`；先查看 `git remote -v`。这一步**替换了 UniAgent 原先锁定的 VERL 版本**，兼容性仍需通过后面的模块导入和 E2E smoke 实测，不能保证公开 UniAgent main 与此 fork 在所有 V1/TQ API 上兼容。若存在不兼容，恢复旧机的实际 UniAgent commit/差异后再验收，禁止默默切回不带 TPR 的 upstream 代码。

```bash
# 对照旧机若仍可访问：记录原始 checkout 与未提交改动，严禁丢掉。
# 在旧机宿主机分别执行：
# git -C <OLD_WS>/uni-agent-cc status --short
# git -C <OLD_WS>/uni-agent-cc/verl status --short
# git -C <OLD_WS>/uni-agent-cc/verl rev-parse HEAD
```

## 2. 宿主机创建 Docker（复用 2026-10-13 的约定）

之前使用的容器名 `uniagent_16npu`，`--network host`、`--privileged`、`--shm-size=128g`、内存锁定、Ascend driver/dcmi/npu-smi、Docker socket 和 CLI，以及整个工作区挂载到 `/workspace`。原先的镜像入口 `/workspace/entrypoint.sh` 曾被 bind mount 遮住而启动失败；**显式覆盖为 `--entrypoint /bin/bash`** 是之前采用的修复。

仓库已经附有对应启动脚本：

```bash
# 宿主机（镜像必须已经 docker image inspect 成功）
cd "$WS/verl-tpr-e2e"
IMAGE=uniagent:new \
HOST_WORKSPACE="$WS" \
CONTAINER_NAME=uniagent_16npu \
bash scripts/tpr/run_uniagent_new_container.sh

docker ps --filter name=uniagent_16npu
docker inspect uniagent_16npu --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
docker exec -it uniagent_16npu bash
```

该脚本故意沿用 `--privileged` 和宿主 Docker socket，这是**高权限开发环境**，只适合可信测试主机。无需额外 `docker rm -f`：已有同名容器时脚本会拒绝覆盖。脚本在非 Ascend 主机上直接停止。

新机如果每节点 4 张 NPU，`docker` 实际可见设备也应该是每节点 4 张，**不要套用旧单机 `ASCEND_RT_VISIBLE_DEVICES=8,...,15`**。

## 3. Docker 内检查镜像与 Python 环境（缺一项不要跑）

```bash
# Docker 内：从这里开始，除 Git 外的操作均在容器内
cd /workspace
npu-smi info
python - <<'PY'
import sys, importlib.util, torch
print("python", sys.version)
print("torch", torch.__version__)
for name in ("torch_npu", "vllm", "vllm_ascend", "ray", "megatron.core",
             "mbridge", "uni_agent", "verl", "pandas", "pyarrow"):
    try: print(name, importlib.util.find_spec(name))
    except Exception as exc: print(name, "ERROR", repr(exc))
try:
    import torch_npu
    print("NPU available:", torch.npu.is_available(), "count:", torch.npu.device_count())
    assert torch.npu.is_available()
except Exception as exc:
    raise SystemExit("STOP: uniagent:new is not a proven Ascend runtime: " + repr(exc))
PY
docker info >/dev/null && echo 'Nested Docker socket/CLI OK'
```

如果 `torch_npu` 缺失，不要随手 `pip install torch-npu`：须先确认 **宿主机驱动/CANN、torch、torch-npu、vLLM-Ascend 的兼容组合**。旧机版本曾使用 torch-npu 2.9.0/CANN 9.0，但不能断言此镜像相同。建议先恢复旧机可用镜像或锁定依赖快照，避免一条无版本依赖安装破坏镜像。

镜像内原有 `/workspace/.venvs/uniagent-cc` 如果是随镜像打进去的，也可能被 `/workspace` bind mount 遮住，所以必须重新检查：

```bash
test -f /workspace/.venvs/uniagent-cc/bin/activate && \
  source /workspace/.venvs/uniagent-cc/bin/activate || true
python -m pip list | grep -Ei 'torch|torch-npu|ray|vllm|megatron|mbridge|swebench'
```

若镜像全局环境已满足上述依赖，可采用隔离的 system-site-packages venv，避免强行重装底层 NPU 运行时：

```bash
python -m venv --system-site-packages /workspace/.venvs/uniagent-cc
source /workspace/.venvs/uniagent-cc/bin/activate
python -m pip install --no-deps -e /workspace/uni-agent-cc
python -m pip install --no-deps -e /workspace/uni-agent-cc/verl
```

此安装只链接本地包，**不会自动补齐缺失依赖**。运行前必须确认缺失包；若缺失，用与旧机匹配的受控 lock/requirements 单独安装，不能把此步骤描述为“装完即可 E2E”。

## 4. 验证实际被 import 的 TPR 路径

```bash
cd /workspace/uni-agent-cc
source /workspace/.venvs/uniagent-cc/bin/activate
export PYTHONPATH=/workspace/vllm-ascend:/workspace/uni-agent-cc/verl:/workspace/uni-agent-cc:${PYTHONPATH:-}

python - <<'PY'
import inspect
from verl.workers.config.engine import McoreEngineConfig
from verl.workers.engine.megatron.transformer_impl import MegatronEngine
from verl.workers.engine.megatron.tpr import adapter
print("config:", inspect.getfile(McoreEngineConfig))
print("engine:", inspect.getfile(MegatronEngine))
print("TPR adapter:", inspect.getfile(adapter))
assert "tpr_enabled" in inspect.signature(McoreEngineConfig).parameters
assert "/workspace/uni-agent-cc/verl/" in inspect.getfile(adapter)
print("TPR_RUNTIME_IMPORT_PASS")
PY
```

如果失败，先解决代码版本、`PYTHONPATH`、editable install 或依赖问题。不能用 `TPR_ENABLED=True` 的打印值作为 TPR 生效的证据。旧 `bridge/patches/apply_uniagent_cc_tpr.py` 引用已经废弃的 `verl.models.mcore.tpr.megatron_adapter` 路径，**不要应用到当前 main**；当前 fork 的 TPR 接入已经在 config/transformer_impl 内。

## 5. Session cap、Task Config、模型和数据（原镜像不包含这些保证）

### 5.1 UniAgent Session 硬上限

最新 E2E launcher 默认 `SESSION_HARD_CAP=True`，它需要额外的 **UniAgent Gateway/Framework** 补丁，TPR fork 当前没有内置该补丁。旧 `bridge` 仓库中有 `patches/apply_uniagent_session_hardcap.py`：

```bash
python /workspace/bridge/patches/apply_uniagent_session_hardcap.py \
  --check /workspace/uni-agent-cc
# 只有 --check 兼容当前 UniAgent 修订后再真正应用：
python /workspace/bridge/patches/apply_uniagent_session_hardcap.py \
  /workspace/uni-agent-cc
```

该补丁是版本敏感的，`--check` 不通过就停止并核对 UniAgent 版本，**不要忽略失败、不要强行打补丁**。唯一可在补丁未完成时运行的 launcher 配置是 `SESSION_HARD_CAP=False PRECHECK_ONLY=True`（只做静态预检查），不能把 48K 长 session 的真实定时 E2E 建立在绕过 hard-cap 的基础上。

### 5.2 Docker Sandbox Task Config

本分支新增 `examples/tpr/task_config_swe_rebench_docker.yaml`。它使用 UniAgent 公共 `swe_rebench`/Claude Code task schema，`sandbox.provider=docker`、`pull_policy=never`，每行数据提供自己的 sandbox image。不要误用上游 `provider=vefaas` 的配置到本地 Docker。

```bash
export TASK_CONFIG=/workspace/verl-tpr-e2e/examples/tpr/task_config_swe_rebench_docker.yaml
test -f "$TASK_CONFIG"
```

此配置是新环境的**待验收启动配置**，不是此前旧机已经成功验证的那份定制 Task Config；需要一次真实 task + reward smoke 确认。

### 5.3 模型权重

旧实验模型目录为 `/workspace/hf_models/Qwen3-4B-128K`。它是带 YaRN/128K 设置的版本，不能以直接下载原始官方 Qwen3-4B 并改 `max_model_len` 代替。

优先从旧机复制已经验证的模型：

```bash
# 新机宿主机，替换 <OLD_SERVER> 和旧机帐号/IP
rsync -avP <OLD_SERVER>:/mnt/pipeline-data/j00972288/project/uniagent_linear/hf_models/Qwen3-4B-128K/ \
  "$WS/hf_models/Qwen3-4B-128K/"
```

Docker 内检查：

```bash
python - <<'PY'
import json
p="/workspace/hf_models/Qwen3-4B-128K/config.json"
c=json.load(open(p))
print("model_type:",c.get("model_type"),"rope_scaling:",c.get("rope_scaling"))
PY
```

### 5.4 数据集及 SWE sandbox 镜像

最短路径是迁移旧机已归一化的数据文件，再用现有 E2E 脚本按实例 ID 选择 `ASPP__pelita-875`，不要重新下载全量 SWE-reBench：

```bash
# 新机宿主机
rsync -avP <OLD_SERVER>:/mnt/pipeline-data/j00972288/project/uniagent_linear/tpr_experiments/shortfix/shortfix_candidates_task.parquet \
  "$WS/tpr_experiments/shortfix/"
```

尚未获得旧数据时，可先取得原始 Parquet，再运行本分支 `filter_swe_rebench_shortfix.py`；过滤器本身不负责下载数据。每条任务在 `extra_info.tools_kwargs.task.sandbox.image` 中存有镜像名，必须与 Docker daemon 里的镜像严格匹配。

已知 pelita 实例的逻辑镜像名为：

```text
swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
```

若使用旧机的 OpenYuanrong registry 镜像映射，必须先确认新机的 registry 权限、镜像架构及 digest，拉取后用 `docker tag` 标记成数据集要求的逻辑名。**不要因为本机存在 `uniagent:new` 就认为存在任务 sandbox 镜像**。

```bash
# 宿主机检查；Docker 内共享该 daemon
docker image inspect swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
docker image inspect uniagent:new
```

一个成功的 E2E 还需要工具 sidecar 镜像、可用 Docker daemon、Agent 工具调用网络，以及正确的 reward 环境。先运行下面的单任务 inference smoke，再消耗 8 条 rollout。

## 6. 从数据到单任务 Inference / Reward

```bash
# Docker 内
cd /workspace/uni-agent-cc
source /workspace/.venvs/uniagent-cc/bin/activate
export PYTHONPATH=/workspace/vllm-ascend:/workspace/uni-agent-cc/verl:/workspace/uni-agent-cc:${PYTHONPATH:-}
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export DATA_PATH=/workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet
export TASK_CONFIG=/workspace/verl-tpr-e2e/examples/tpr/task_config_swe_rebench_docker.yaml
mkdir -p /workspace/tpr_experiments/shortfix/rollout_smoke1

# 注意：下面是推理 + Reward smoke，不是 PPO E2E。
python examples/inference/parallel_infer_verl.py \
  --data-path "$DATA_PATH" \
  --model-path "$MODEL_PATH" \
  --task-config "$TASK_CONFIG" \
  --n 1 --limit 1 \
  --response-length 40960 \
  --log-dir /workspace/tpr_experiments/shortfix/rollout_smoke1 \
  --result-path /workspace/tpr_experiments/shortfix/rollout_smoke1/result.json
```

`--limit 1` 只选择数据中的第一条；**它不保证是 pelita-875**。需要 pelita 定点实验时，按 `integration/REBENCH_E2E.md` 使用 `REBENCH_INSTANCE_IDS` 生成固定的任务子集，或先将独立 Parquet 设为 DATA_PATH。检查 result.json 的 `num_scored_sessions`、reward 和日志里的实际 `instance_id`。

## 7. 单机训练 E2E：先 Full 后 TPR

当前仓库 launcher：`tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh`。这是 **Online UniAgent + Rollout + Transfer Queue + 原生 Actor optimizer** 路径；与离线 F/B 回放不同。默认 N=8、TP2、CP2，在四卡时 DP1。先选择四卡物理设备 0..3，对 `ASPP__pelita-875` 一条独立 prompt 做 1 step。

```bash
# Docker 内。在源和 sandbox/模型均齐备、hard-cap check 实际通过后执行。
cd /workspace/verl-tpr-e2e
export TASK_CONFIG=/workspace/verl-tpr-e2e/examples/tpr/task_config_swe_rebench_docker.yaml
export MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K
export TRAIN_FILE=/workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet
export REBENCH_INSTANCE_IDS=ASPP__pelita-875
export UNIAGENT_SESSION_PATCHER=/workspace/bridge/patches/apply_uniagent_session_hardcap.py
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export N_GPUS_PER_NODE=4 TRAIN_TP=2 TRAIN_CP=2
export TRAIN_PROMPT_BSZ=1 GEN_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1
export N=8 TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1
export TPR_TREE_POLICY=qwen_turn
export CAPTURE_ACTOR=False
SCRIPT=tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh

# 静态预检，不启动 Ray/NPU；跳过缺失补丁的变体仅供预检
SESSION_HARD_CAP=False PRECHECK_ONLY=True TPR_ENABLED=False \
  RUN_NAME=pelita-precheck bash "$SCRIPT"

# Full，真实端到端
SESSION_HARD_CAP=True TPR_ENABLED=False RUN_NAME=pelita-full-step1 \
  bash "$SCRIPT" 2>&1 | tee /workspace/tpr_experiments/e2e/pelita-full-step1.log

# TPR，只有 Full smoke 通过且运行时 TPR 导入检查通过后再跑
SESSION_HARD_CAP=True TPR_ENABLED=True RUN_NAME=pelita-tpr-step1 \
  bash "$SCRIPT" 2>&1 | tee /workspace/tpr_experiments/e2e/pelita-tpr-step1.log
```

每个 `RUN_NAME` 必须是新的名字，不能把旧 checkpoint 或 captures 当新跑出来的结果。检查 `/workspace/tpr_experiments/e2e/<RUN_NAME>/` 内日志，确认：
- rollout 真实完成、任务 reward 运行；
- Transfer Queue 中有有效 Actor mini-batch；
- Full/TPR 分支分别进入正确的训练代码；
- `optimizer_step` 实际执行；错误和截断轨迹没有被误算成成功；
- 同等数据与设置下记录 wall time、样本量、token 长度、显存；独立 rollout 不可直接用来归因 TPR 的训练加速。

### 必须保留的运行时限制

现有 launcher 在任务启动前执行本地 `ray stop --force`，**不要在已经运行共享多节点 Ray head 的节点上直接启动**。它内部写死 `NNODES=1`，在正式多机实验前须重新设计 Ray placement、actor/rollout 独立设备分配、权重同步和 Docker sandbox 所在节点；不能简单在四台机器各运行一次脚本就得到“三机推理、一机训练”。

## 8. 验收记录模板（每个机器/镜像都要留）

```text
host / accelerator:
uniagent:new digest:
Docker mount root:
CANN / driver / torch / torch_npu / vllm_ascend versions:
uni-agent-cc SHA:
uni-agent-cc/verl runtime SHA:
verl-tpr-e2e SHA:
bridge SHA / hardcap patch status:
MODEL_PATH / config.json sha256:
TRAIN_FILE sha256 / task instance_id:
sandbox image name / digest:
UniAgent single-task inference + reward: PASS/FAIL
Full online 1 step optimizer: PASS/FAIL
TPR online 1 step optimizer: PASS/FAIL
Next: 10 step + controlled identical-capture performance replay
```

完整数据选择、hint H1/H2/H3 和 E2E 的已有说明见 [REBENCH_E2E.md](../../tests/models/mcore/tpr/integration/REBENCH_E2E.md)；生产 TPR 逻辑详见 [TPR 开发文档](TPR开发文档.md)。

## 9. 当前未验证的项目（不能用“脚本存在”替代 PASS）

1. 当前实际新机是否为 Ascend NPU，`uniagent:new` 是否自带可用的 torch-npu/vLLM-Ascend，镜像内虚拟环境是否完整。
2. 公开 `RichardFido/uni-agent` 新 checkout 与 `verl-tpr-e2e` 运行时 VERL 的 V1/TQ API 兼容性。
3. 旧 Bridge hard-cap patch 对新 UniAgent 版本的适用性。
4. 新增加的本地 Docker task YAML、Claude Code sidecar 镜像、pelita reward sandbox 是否配套。
5. 旧机自定义的 Qwen3-4B-128K、归一化 Parquet 和任务 sandbox 镜像在新机是否可访问。
6. **四机 16 卡 3 推理 + 1 训练尚未实现和验收**；本文件只确保从零部署和单机路径可核对，不伪装已有多机 E2E 结果。
