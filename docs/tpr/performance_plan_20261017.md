# TPR 阶段性验收、清理与 NPU 采集

Last updated: 10/17/2026.

## 冻结基线与范围

用户报告的验收基线：main `ce6387c5af2213de00562c6752fbfd63aa96df7c`
对 PR7 `2dc82406a7abf1e0a78694b7810522bbe4f1d3a6`，Qwen3-4B BF16，8 Ascend NPU，
真实 Actor PPO capture，梯度工程容差 2%。本机没有访问这些服务器结果或执行 NPU 重放。

| 验收 | 用户报告结果 |
| --- | --- |
| TP2×CP4 | loss gap = 0；logprob max_abs = 0 |
| TP2×CP4 全参数梯度 | rel L2 = 0.01027872；cosine = 0.99994723 |
| TP2×CP4 抽样梯度 / gradient norm | rel L2 = 0.01581897；norm 相对差 = 0.000298 |
| TP2×CP4 更新 | SGD 后抽样参数一致 |
| TP2×CP2×DP2，Step 6 / 30K | loss / rank loss gap = 0；抽样梯度 rel L2 = 0.01450909 |
| DP2 副本 / 比较器 | gradient max_abs = 0；PASS |
| Linux CPU CI | 471 项通过 |

现有结果（服务器路径）：

```text
/workspace/tpr_log/pr7_dp2_30k_retry2/old/step6_tpr_tp2cp2dp2.json
/workspace/tpr_log/pr7_dp2_30k_retry2/new/step6_tpr_tp2cp2dp2.json
/workspace/tpr_log/pr7_dp2_30k_retry2/comparison.json
```

**未验收**：DP2 全参数梯度、CP metrics 的独立数值参考、完整连续 AdamW、端到端吞吐。
此轮没有真实算子 Profile，不宣称任何加速比或实际峰值显存收益。
累计改动仍须对冻结基线比较，不能把 2% 当作每个补丁分别可消耗的漂移预算。

## 此轮清理

生产代码只合并等价 RoPE 参数分支、删除 Push 已压紧 KV 的重复 detach 容器、
以有序字典代替 objective 的重复去重列表，并把错误消息长度列表延后到错误分支。
保留调用次序、Tensor 所有权、目标排列、CP padding、梯度接力和原生 GEMM 形状。
生产净减少 17 行；测试与独立采集工具单独计量，不以删除回归覆盖换行数。

测试入口默认 `qwen_turn`，去掉无效 Exact 分支及历史 pre-refactor 导入路径。
MindSpeed 参数清洗、RoPE CP-group sentinel、完整梯度门禁等正确性相关处理保留。
生产 Ring / Attention 后端 / Fixed Tile 状态均未改；采集标记仅安装在独立测试进程。

## 四个方向：先验证瓶颈，再单独改动

### Root Ring

`verl/workers/engine/megatron/tpr/ring.py` 中已经有 `ordinary_ring_cp_attention`，但当前生产 Root
仍使用分块自定义路径。无 padding、CP 大小 C、祖先段数 A 时，静态调度 FA 调用数为：

| 路径 | 每层每次 forward 或 backward 的 FA 次数 |
| --- | --- |
| 当前 Root | 2C + 1（CP2: 5；CP4: 9） |
| 默认有祖先段 | 4CA + 2C + 1 |
| prefix full coalesce | 2CA + 2C + 1 |
| 额外满足 query coalesce 条件 | CA + 2C + 1 |
| 原生 MindSpeed Root 候选 | C（CP2: 2；CP4: 4） |

这是源码调度推导，不是实测时间（原生路径参考本机 MindSpeed `376e9cc`，服务器版本仍需记录）；
padding 会改变调度。内部段通常还包括 Push forward、
Pop 重算 forward 和 Pop backward，不能只看一次 Root forward。
候选快路径须核对空 prefix、CP group/rank 顺序、RoPE 后 Q/K、GQA、dtype/scale、
输出布局及梯度归属；先无 padding，再覆盖尾部虚拟 padding 和全 padding rank。
不能通过改 QKV/MLP 的 GEMM 形状来获得“快路径”。此次不启用该候选。

### Ring 通信与调度

两种 Root 路径的理论 KV / dKV 通信跳数并不会随 FA 次数减少：每 KV 段、每 rank，
forward 为 C−1 个 KV hop，backward 为 C−1 个 KV 加 C−1 个 dKV hop，合计 3(C−1)。
每个 packed 消息含 K/V；祖先段数会扩大消息数。
当前自定义 backward 的 dKV 发送后很快 wait，原生实现有跨下一块 FA 延后等待的机会；
需要 HCCL Trace 证明实际暴露等待和重叠，不能用当前计算 stream Event 当成网络耗时。
另记录 online merge 的 FP32 cast / exp / cat、padding mask 分配和各段 buffer 创建。
暂不调整 coalesce 默认值、消息顺序、通信组或流。

### Push LM Head

Push 当前调用完整模型并丢弃 logits。原生 Megatron 的部分版本支持
`output_processor` / `output_processor_context`，在 LM Head 前返回 decoder 输出；
仓内已有接口探测逻辑，但必须核查服务器实际安装版本，不能仅凭本机源码推定可用。
未来只对 Push 探测并使用原生接口；旧版本保留完整 forward，不 monkeypatch 模型算法。
保留最终 norm、SP 布局和原有 decoder GEMM；不要为了丢弃的 hidden states 额外 gather。
先从 Profile 确认 Push 的 LM Head GEMM、通信和 logits 生命周期占比。

### KV 压实

设 BF16、batch=1、当前 rank 物理段长 S、本地 Q heads=q、KV heads=k、head dim=d。
紧凑 KV 为 `4*S*k*d` 字节；融合 QKV 为 `2*S*(q+2*k)*d` 字节。
若 V view 钉住融合 QKV、RoPE 后 K 独立存储，则压实理论可释放
`2*S*(q+k)*d` 字节/已完成层；有 output gate 时还需计入 gate backing storage。
必须实际读取唯一 storage 元数据，不能把逻辑 Tensor 大小相加当作占用。

仅作尺度示例：q=16、k=4、d=128、S=7680、36 层时，紧凑 KV 为 15 MiB/层，
上述别名假设下旧占用为 52.5 MiB/层，累计差约 1.32 GiB；
这不代表真实 Turn 段长、全局峰值或 allocator reserved 的实测下降。
原 QKV / V 的局部变量仍可能存活到 attention 返回，紧接压实函数后的采样不等于已释放；
同时观察下一层入口、Push 返回、Pop 重算和 backward。
原生 Root 还可能保存远端 KV，不能默认比自定义 Root 更省显存。

## 独立采集入口

工具：`tests/models/mcore/tpr/parallel/profile_tpr_dp2_step6.py`，默认不启用 Profiler。
从同一份新工具代码启动，但 `--repo` 分别指向两个干净目标 checkout，
复用各自原有 runtime / placement / engine / PPO 实现。
`--reference-json` 使用对应已验收 JSON，校验 Step 6、30720 cap、原始 8 行、
输入 SHA、原 PPO token 分母和 DP partition；不重新裁剪、合成或重采样。

Git 仍在宿主机操作。预先在宿主机确认目标 SHA 与生产目录干净；容器内 Git 不可用时，
工具把 `--revision` 标记为 **host_declared_not_verified**，并保存实际加载文件的 SHA256，
不能把这个标签当作容器内 Git 验证成功。新工具自身也保存文件哈希。

示例（`tooling`、`main_repo`、`pr7_repo` 替换为容器实际可见路径；不要复制 Markdown 链接作 URL）：

```bash
tooling=/workspace/verl-tpr-tooling
main_repo=/workspace/verl-main-baseline
pr7_repo=/workspace/verl-pr7-baseline
out=/workspace/tpr_experiments/profiles/pr7_dp2_step6
refs=/workspace/tpr_log/pr7_dp2_30k_retry2

torchrun --standalone --nproc_per_node=8 \
  "$tooling/tests/models/mcore/tpr/parallel/profile_tpr_dp2_step6.py" \
  --repo "$main_repo" --label main \
  --revision ce6387c5af2213de00562c6752fbfd63aa96df7c \
  --reference-json "$refs/old/step6_tpr_tp2cp2dp2.json" \
  --output-root "$out/main/trace" --profile --warmups 3 --repeats 1

torchrun --standalone --nproc_per_node=8 \
  "$tooling/tests/models/mcore/tpr/parallel/profile_tpr_dp2_step6.py" \
  --repo "$pr7_repo" --label pr7 \
  --revision 2dc82406a7abf1e0a78694b7810522bbe4f1d3a6 \
  --reference-json "$refs/new/step6_tpr_tp2cp2dp2.json" \
  --output-root "$out/pr7/trace" --profile --warmups 3 --repeats 1
```

每 rank 独立目录，已有输出拒绝覆盖。Level1、CPU+NPU、`profile_memory=True`、
`record_shapes=True`、`with_stack=True`、`data_simplification=False`；不排除 HCCL。
保留原始 native 输出、Text Trace/CSV、阶段时间/事件、KV storage 元数据、配置、版本、
声明 SHA 和实际文件哈希。具体能力与导出状态写入清单；不能把未导出项报告为已采集。
观察器只在测试进程临时安装 scope / module hooks，退出时恢复，不改变结果和 Tensor 返回。
树编号与 segment 的 parent / prefix / length 一同记录；CPU scope 包括树构造、计划摘要和预检、
objective 绑定、forest 执行、梯度 finalize 及 CP loss 聚合。
阶段 Event 是当前 compute stream 的时间；通信线上的真实重叠以原始 HCCL Trace 为准。

性能测量另起无 Profiler 进程：同样命令**去掉 `--profile`**、改到新输出目录
`main/timing` 或 `pr7/timing`，使用 `--warmups 3 --repeats 10`。
此模式不安装观察 hooks；记录同步 forward+backward 的 max-rank wall time、median / p90。
建议交错 main / PR7 独立进程重复，记录温度/频率/负载；Profiler 内耗时仅归因，不作收益数字。
工具不更新权重，不构成 AdamW 连续训练或 DP2 全参数梯度的新验收。
默认可见卡为 `8,9,10,11,12,13,14,15`；如容器使用其它映射，可显式设置
`ASCEND_RT_VISIBLE_DEVICES` 为 8 个唯一非负整数 ID，两边必须使用相同映射。
原生 shapes/memory/stack 采集本身也可能改变 Tensor 生命周期；
Profiler 峰值不可与无 Profiler 峰值混作压实收益，显存收益须另在同等无 Profiler 条件下确认。

核查：FA / GEMM / RoPE / LM Head 时间线及形状；HCCL bytes / hop / wait / overlap；
Push/Pop/leaf/反向事件；KV 唯一 storage、基线和峰值 allocated/reserved；CPU enqueue 时间。
旧 `profiling/test_tpr_qwen3_ring_cp_profile_npu.py` 使用 legacy scheduler/executor，
不是当前生产 Actor 路径的替代证明。

## 本机验证（不是新的 NPU 验收）

```bash
python -B tests/models/mcore/tpr/run_cpu_regressions.py
git diff --check
```

Windows / PyTorch 2.10 CPU / TensorDict 0.14.3：**516 passed, 1 skipped**；
跳过项为本机 torch 构建无法创建 Gloo device，Linux CI 不据此跳过。
独立 Ring/Offload unittest 共 33 项（1 skipped），291 个 Python 文件语法、
15 项架构合同和两份 old/new shell 语法检查通过。
新观察器的 15 项 mock 测试及独立 driver 的 17 项 CPU 合同已纳入上述 runner；
mock API 不能验证真实 torch_npu 导出或 HCCL 时序。

上述真实 NPU 导出、清理后 main/PR7 重放及收益测量仍待服务器运行；提交状态以 PR head 为准。
AI assistance: OpenAI Codex；提交/合入前需人工逐行审阅。
