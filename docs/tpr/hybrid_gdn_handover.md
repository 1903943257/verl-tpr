# TPR × GDN 技术方案与工作交接

> 更新日期：2026-11-26  
> 当前状态：实验分支，核心状态复用已实现，Hybrid 多层正确性及 CP 联合验收待闭环。

## 1. 工作背景与代码入口

在已有 Full Attention TPR（Tree Prefix Reuse）基础上，扩展对 Qwen3.5 等 GDN + Full Attention 混合架构的支持，复用多条训练轨迹的共享前缀计算，并适配 Context Parallel（CP）。

- **开发分支**：[verl-tpr / feat/tpr-hybrid-gdn-restore](https://github.com/1903943257/verl-tpr/tree/feat/tpr-hybrid-gdn-restore)
- **GDN 核心代码**：[verl/experimental/tpr/gdn](https://github.com/1903943257/verl-tpr/tree/feat/tpr-hybrid-gdn-restore/verl/experimental/tpr/gdn)
- **TPR 主框架**：[verl/workers/engine/megatron/tpr](https://github.com/1903943257/verl-tpr/tree/feat/tpr-hybrid-gdn-restore/verl/workers/engine/megatron/tpr)
- **测试目录**：[tests/models/mcore/tpr/linear](https://github.com/1903943257/verl-tpr/tree/feat/tpr-hybrid-gdn-restore/tests/models/mcore/tpr/linear)
- **最近测试代码提交**：[f4a3dbc6](https://github.com/1903943257/verl-tpr/commit/f4a3dbc6ec9ec2e318ec2a6eead1f468dada20b8)，增加 4/24 层参数化测试和完整梯度诊断。
- **单层诊断修复**：[2a866221](https://github.com/1903943257/verl-tpr/commit/2a86622117675fcb8ff6fb985047b00434c2c2fc)，修复 GDR 诊断 wrapper 的参数透传。

主要工作分为 Stateful GDN 算子适配、TPR × GDN 状态复用、GDN × CP 分布式执行三部分。

## 2. TPR × GDN 核心方案

### 2.1 GDN 前缀状态复用

GDN 主要包含 CausalConv 和 Gated Delta Rule（GDR）两类有状态计算：

- **Conv State**：保存因果卷积的历史窗口信息，语义上需要最近 W-1 个有效输入 token；当前算子接口的物理状态宽度为 W。
- **Recurrent State**：保存 GDR 递归矩阵状态，典型形状为 `[B, H, K, V]`，当前适配中使用 FP32。

GDN 的后续 Segment 只需要继承直接父节点的最终状态，不需要访问所有祖先的历史 token。

| 对比项 | Full Attention | GDN |
| --- | --- | --- |
| 前缀复用对象 | K/V | Conv State + Recurrent State |
| 读取范围 | 所有祖先节点的 KV | 直接父节点的最终状态 |
| 单个边界状态大小 | 随 Segment 长度增长 | 固定模型维度下与 Segment 长度无关 |
| 反向传播 | dK、dV | dConvState、dRecurrentState |

GDN 状态接口可抽象为：

```python
output, final_state = gdn(segment, initial_state=parent_state)
# final_state = (conv_state, recurrent_state)
```

### 2.2 DFS 执行与梯度回传

沿用现有 TPR SegmentExecutor 的 Push / Visit / Pop 调度，不重新实现树调度器。

```text
                  Prefix P
                  /      \
              Suffix1   Suffix2

Forward:    State(P) -> Suffix1 / Suffix2
Backward:   dState(S1) + dState(S2) -> Prefix P
```

1. **Push**：无梯度执行共享前缀 P，保存各层的 graph-free Conv State 和 Recurrent State。
2. **Visit**：每个分支从父节点状态独立开始，通过 `detach().requires_grad_(True)` 建立分支局部的 Autograd Leaf Anchor。分支反向后收集状态梯度。
3. **Accumulation**：在父节点状态对象中累加所有子分支的状态梯度，保证不同分支不会污染缓存状态。
4. **Pop**：重新执行 P 的有梯度前向，将累积的状态梯度注入重新生成的最终状态，同时对 P 自身的逻辑 Loss 进行反向。所得父状态梯度继续向上汇聚，最后释放当前状态。

```python
# Pop 的概念示例：FA KV 梯度可在同一次 backward 中一并注入。
torch.autograd.backward(
    (prefix_loss, final_conv_state, final_recurrent_state),
    grad_tensors=(None, accumulated_dconv, accumulated_drecurrent),
)
```

### 2.3 Hybrid 模型集成

Qwen3.5-0.8B 的完整模型由 18 层 GDN 和 6 层 Full Attention 组成。现有实现保留两类状态容器，共用同一套执行器、构树和逻辑 Loss 机制：

```text
SegmentExecutor
├── KVStack
│   └── Full Attention: KV / dKV
└── GDNPrefixState
    ├── Conv State / dConvState
    └── Recurrent State / dRecurrentState
```

通过 ModelSpec 替换将 GDN 接入 `TPRGatedDeltaNet`。有 TPR Context 时走 Stateful GDN 路径，无 Context 时回退至原生 GDN forward。GDN 模块位于 experimental 目录，通过 `TPR_ENABLE_GDN=1` 显式启用，尽量避免影响现有 Dense TPR 路径。

## 3. GDN × CP 方案

### 3.1 当前方案：Headwise A2A CP

复用 MindSpeed 原生 GDN CP 的 All-to-All 通信方式。将序列分片转换为 Head/Channel 分片，使各 rank 在 GDN 核心计算阶段持有当前 Segment 的完整时间维度，但只处理部分 Head/Channel。

以 CP=2 为例：

```text
CP layout:
  Rank 0: Sequence Shard 0 × All Heads
  Rank 1: Sequence Shard 1 × All Heads
                  |
            CP -> HP A2A
                  |
HP layout:
  Rank 0: Full Segment × Head Shard 0
  Rank 1: Full Segment × Head Shard 1
                  |
         Stateful Conv + GDR
                  |
          保存各 rank 本地状态
                  |
            HP -> CP A2A
                  |
        恢复 Zigzag Sequence Layout
```

实际执行链路：

```text
in_proj
  -> tensor_a2a_cp2hp
  -> Stateful CausalConv / GDR（自然序列顺序）
  -> gated_norm
  -> tensor_a2a_hp2cp
  -> out_proj
```

**TPR 与 CP 的结合方式：**Prefix 和 Suffix 各自执行普通 GDN CP。状态保存在 CP→HP 后的 rank-local Head/Channel Layout 中。在 CP Group 和 Head Mapping 一致时，后缀直接读取本 rank 的父节点状态，不需要额外进行 State AllGather 或跨 rank 的递归状态接力。各 rank 分别汇聚本地状态梯度；原有 A2A autograd 和框架梯度同步仍需保持正确。

在 Hybrid 模型中，GDN 使用 A2A CP，Full Attention 使用 TPR Ring CP。层间统一保持原生 Zigzag Sequence Layout，两类 PrefixState 的内部 placement 不同。

当前代码已包含 CP2 执行路径，但仅支持非 packed、对齐 Segment（全局长度可被 4 整除）、TP=SP=1。完整 Hybrid CP2 正确性仍需验收。

### 3.2 后续方向：Chunkwise CP

沿 Sequence 维切分 GDN，通过跨 rank 的递归边界状态和 CausalConv Halo 完成前向，并反向传递边界状态梯度。相比 Headwise A2A，该方案有望减少特征重分布通信，但需解决递归依赖、并行度、卷积边界以及状态梯度接力问题。

该方向尚未集成到当前分支，建议在 A2A CP 方案验收完成后单独研究，不作为当前交付前提。

## 4. 主要工程问题

### 4.1 Stateful GDR 算子支持

原有 GDR 已具有状态相关接口：

```text
Forward:   initial_state -> output, final_state
Backward:  h0, dht -> dh0（以及其他输入/参数梯度）
```

其中 `h0` 为初始递归状态，`dht` 为最终状态的梯度，`dh0` 为初始状态的梯度。

历史 MindSpeed/MindSpeed-Ops 路径中，API 或 launcher 对非空 `initial_state` 存在训练调用限制。此前通过实验补丁开放该路径并进行 Stateful 前后向验证。

当前 TPR 直接调用 `mindspeed_ops.api.triton.chunk_gated_delta_rule` 和 `mindspeed_ops.api.triton.convolution`，不依赖历史测试中的临时算子绑定。

### 4.2 CausalConv 状态梯度

除了 GDR，CausalConv 同样必须支持初始状态梯度和最终状态梯度注入。历史实现存在 `BT < W` 时 initial_state 梯度计算错误的问题，已在实验分支中引入修复。

历史 MindSpeed-Ops 实验版本：

| Commit | 内容 |
| --- | --- |
| `51c0811` | 开放 Stateful GDR Training |
| `ffa1e90` | 修复 CausalConv initial_state 梯度错误 |
| `e4fd22d` | 合并上述修复的 `tpr-gdn-stateful` 实验版本 |

上游仓库：[Ascend/MindSpeed-Ops](https://gitcode.com/Ascend/MindSpeed-Ops.git)。上述提交属于历史实验记录，不能保证全部可从上游直接拉取。换机器时应核对实际算子源码、补丁和导入路径，不能默认新版 Ops 已支持完整 Stateful Training。

### 4.3 Padding / THD

GDN 最终状态必须对应最后一个真实 token，而不是 padding 后的物理位置。即使 padding 不参与 Loss，也可能改变 Conv State 或 Recurrent State。

后续适配需做到：

- 区分真实及物理 Segment 长度，保留 actual/padded `cu_seqlens`。
- 在 A2A 后恢复自然序列顺序，按有效 token 执行 Stateful Conv/GDR，再将输出恢复至通信布局；或使用等价的有效状态边界机制。
- Packed Sequence 之间不得串状态；普通 Conv1D 直接处理拼接序列可能污染跨序列边界。
- 所有 CP rank 保持相同 DFS 调度及 collective 顺序。

当前分支尚未支持 GDN THD/Packed 和非对齐 CP2 Segment。

### 4.4 数值误差与状态梯度精度

Full 与 Split 的 GEMM Shape、BF16 计算顺序、Native/Stateful 算子 Backend 可能不同。历史实验曾通过控制 `out_proj`、`mlp_fc2` 的计算形状定位数值差异。

需要分别比较 Native Full、Connected Split、TPR，不能将 Native 与 TPR 之间的全部误差直接归因于状态复用，也不能未经验证将其全部归因于 BF16。

GDN 当前使用分支独立 Anchor，Conv 状态梯度按状态 dtype 累加，Recurrent State 为 FP32。可进一步检查 Conv 梯度累加精度及反向顺序差异，但目前应先完成诊断，不直接放宽验收阈值。

## 5. 核心代码位置

GDN 独立实现目录：`verl/experimental/tpr/gdn/`

| 文件 | 功能 |
| --- | --- |
| `model.py` | TPRGatedDeltaNet，CP1/CP2 dispatch，Stateful Conv/GDR 接口 |
| `prefix_state.py` | GDNLayerState、GDNPrefixState、Anchor、梯度累加及状态释放 |
| `context.py` | GDNTPRAttentionContext，初始状态传入和最终状态收集 |
| `state.py` | `forward_gdn_cp_with_state`，CP2 A2A 及 rank-local 状态分片 |
| `module_spec.py` | Hybrid/GDN 模型结构适配 |
| `tree.py` | GDN 相关构树辅助逻辑 |

主框架目录：`verl/workers/engine/megatron/tpr/`

| 文件 | 功能 |
| --- | --- |
| `executor.py` | Push / Visit / Pop、KV/GDN 状态生命周期及梯度回传 |
| `adapter.py` | ModelSpec 替换、TPR opt-in 和训练框架入口 |

测试目录：`tests/models/mcore/tpr/linear/`

| 文件 | 功能 |
| --- | --- |
| `test_gdn_push_branch_pop_npu.py` | 单层 GDN 状态接力与梯度正确性 |
| `test_qwen35_hybrid_push_branch_pop_npu.py` | 4/24 层 Hybrid 三组对照 |
| `_gdn_npu_test_utils.py` | 模型构建、NPU 环境和梯度指标 |
| `_causal_conv_boundary_diagnostic.py` | Conv 前后向边界诊断 |
| `_first_layer_projection_control.py` | GEMM Shape 控制诊断辅助 |

## 6. 当前正确性结果

### 6.1 单层 GDN

此前单层 Stateful GDN 实验已完成核心正确性验证，覆盖 Full、Connected Split、TPR，以及状态梯度回传和分支汇聚。

注意：单层实验中的 Full 基线也使用 TPR Context 下的同一套 Stateful GDN 算子，而不是独立的 Native MindSpeed GDN。它证明的是 Stateful 路径下的分段与树形复用正确性，不能替代 Native Full 与 Hybrid TPR 的整体对照。

### 6.2 24 层 Hybrid（最新已知运行结果）

实验设置：Qwen3.5-0.8B 模型结构、随机初始化、BF16、CP=1、18 GDN + 6 FA。共享前缀 P 和两个后缀 S1/S2 均为 64 tokens。

| 指标 | Native Full | Connected Split | TPR |
| --- | ---: | ---: | ---: |
| Loss | 13.148965836 | 13.122335434 | 13.122335434 |

Logprob 对照：

| Segment | Native vs Connected / TPR：max abs | relative L2 | 超出容差 | Connected vs TPR |
| --- | ---: | ---: | ---: | ---: |
| P | 0.48114395 | 1.434% | 8/65 | 0 |
| S1 | 0.60118675 | 1.488% | 6/63 | 0 |
| S2 | 0.38703442 | 1.270% | 4/63 | 0 |

Connected Split vs TPR 的 48 份状态边界梯度（36 份 GDN、12 份 FA）：

| 指标 | 最大 relative L2 |
| --- | ---: |
| Conv State Gradient | 1.208% |
| Recurrent State Gradient | 1.401% |
| FA KV Gradient | 0.752% |

**结论：**Connected Split 与 TPR 的 Loss/Logprob 完全一致，状态梯度误差均低于原定 2% 的 relative L2 阈值；这表明当前 case 的 TPR 状态复用没有引入可观测的额外前向误差。但 Native Full 与分段执行的 Logprob 超出原阈值（atol=0.08，rtol=0.02）。原测试提前失败，未执行正式参数梯度和状态梯度 cosine 检查，因此完整模型仍未通过验收。

注意：上述状态梯度误差不是 Native Full vs TPR 的参数梯度误差，两类对照不可混淆。

### 6.3 四层 Hybrid（下一步优先任务）

已在恢复分支加入 3 GDN + 1 FA 的四层模型测试，保留 24 层原实验。两种模型均执行 Native Full / Connected Split / TPR 三组对照，并输出：

- Loss、Logprob max abs、relative L2、超出容差的元素数。
- 全模型参数 GradNorm、Norm Ratio、relative L2、Cosine、误差最大的参数。
- GDN/FA 状态梯度的整体及逐层指标。

测试会在三组前后向执行完成后统一评估数值断言，避免因首个 Logprob 失败而丢失后续梯度诊断。原验收阈值不变。最新修改尚未完成 NPU 实机验证，四层结果待补充。

## 7. 环境与复现命令

优先使用此前已经配置 MindSpeed-Ops 的 NPU 环境，不建议直接升级整套依赖。历史基线包括 Megatron-Core `55ac7082`、MindSpeed `376e9cc3`（存在本地修改）及 MindSpeed-Ops `tpr-gdn-stateful` 实验版本；恢复分支与实际服务器环境需要重新核对，历史版本号不构成当前完整安装锁定清单。

### 7.1 拉取恢复分支

```bash
cd /workspace
git clone -b feat/tpr-hybrid-gdn-restore --single-branch \
    https://github.com/1903943257/verl-tpr.git \
    verl-tpr-hybrid-gdn-restore
cd verl-tpr-hybrid-gdn-restore

git log -3 --oneline
export PYTHONPATH="$PWD:$PYTHONPATH"
export TPR_ENABLE_GDN=1
# 根据 npu-smi info 选择空闲设备
export ASCEND_RT_VISIBLE_DEVICES=0
```

检查实际导入路径：

```bash
python - <<'PY'
import torch, torch_npu, triton
import mindspeed, mindspeed_ops, verl
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('triton:', triton.__version__)
print('MindSpeed:', mindspeed.__file__)
print('MindSpeed-Ops:', mindspeed_ops.__file__)
print('VERL:', verl.__file__)
from mindspeed_ops.api.triton.convolution import causal_conv1d
from mindspeed_ops.api.triton.chunk_gated_delta_rule import chunk_gated_delta_rule
print('Stateful operator imports OK')
PY
```

导入成功不代表算子前后向已通过；应保留单层 NPU 测试作为环境验证门槛。

### 7.2 单层 GDN

```bash
set -o pipefail
torchrun --master_addr=127.0.0.1 --master_port=29561 \
    --nproc_per_node=1 -m pytest -s -v \
    tests/models/mcore/tpr/linear/test_gdn_push_branch_pop_npu.py \
    2>&1 | tee gdn_single_correctness.log
```

### 7.3 四层 / 24 层 Hybrid

```bash
# 优先跑四层
set -o pipefail
torchrun --master_addr=127.0.0.1 --master_port=29563 \
    --nproc_per_node=1 -m pytest -s -v \
    tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py \
    -k hybrid4 2>&1 | tee hybrid4_correctness.log

# 四层验证后再跑完整模型
torchrun --master_addr=127.0.0.1 --master_port=29563 \
    --nproc_per_node=1 -m pytest -s -v \
    tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py \
    -k hybrid24 2>&1 | tee hybrid24_correctness.log
```

## 8. 剩余工作与建议顺序

| 优先级 | 工作 | 当前状态 |
| --- | --- | --- |
| P0 | 单层 Stateful GDN | 历史核心正确性已通过；新环境需复验 |
| P0 | 四层 Hybrid CP1 | 测试已提交，待运行 |
| P0 | 24 层 Hybrid CP1 | 已实现，Native Full 对照未通过 |
| P1 | Hybrid CP2（GDN A2A + FA Ring） | 已有非 packed 对齐路径，联合验收待完成 |
| P1 | THD、非对齐 Padding | 待适配 |
| P2 | GDN TP/SP、Activation Checkpointing | 待适配和验收 |
| P2 | Grouped Leaf / TreeBatch | 当前 GDN 路径尚不支持 |
| P2 | 真实 RL E2E、性能及显存验证 | 待完成 |
| P3 | Chunkwise CP | 可选优化，尚未集成 |

四层实验优先依据 Native vs Connected 和 Connected vs TPR 的差异进行定位：前者有误差时检查 Native/Stateful Backend 和 Full/Split 计算形状；后者有误差时检查状态接力、分支隔离、梯度累加及重算。四层通过而 24 层失败时，再做逐层定位，区分数值累积和实现错误。

**交接重点：**已有 GDN 状态复用、DFS 梯度回传及 A2A CP2 基础实现，不需要从头设计。当前主要任务是完成正确性闭环，然后逐步接入生产训练能力。MindSpeed-Ops 的实验补丁、服务器本地未提交修改及真实依赖版本需另行保存、交接和复核；仅克隆 VERL 分支并不保证可以直接复现。
