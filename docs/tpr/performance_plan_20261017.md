# TPR 主线：正确性、精简与可验证的性能优化

更新：2026-10-18。适用仓库：`1903943257/verl-tpr` PR7。
原则：优先复用 VERL / Megatron / MindSpeed；不改变 Full 基线的 GEMM 形状，不开启 Fixed Tile；
每个优化独立提交、回归、测量，不把 Profiler 本身当作项目目标。

## 已验证的边界与尚未验收的部分

2026-10-16/17 冻结验收：main `ce6387c5af2213de00562c6752fbfd63aa96df7c`
对 PR7 旧头 `2dc82406a7abf1e0a78694b7810522bbe4f1d3a6`，
Qwen3-4B BF16、真实 Agent PPO Capture、8×Ascend NPU。

- TP2×CP4 两条 sibling：loss / logprob 完全一致；
  full-gradient relative L2 = 0.01027872，cosine = 0.99994723；2% 工程门禁下阶段接受，
  **原始 1% 梯度门禁未通过**。
- TP2×CP2×DP2 的 Step6 / 30K：loss gap = 0；
  sampled-gradient relative L2 = 0.01450909，副本抽样完全一致。
- 以上均为**固定权重单步 F/B + gradient finalize**；
  不是 AdamW 连续训练，也不代表新 PR7 头已经完成 NPU 回归。
- 仍缺：DP2 全参数梯度、CP 独立数值参考、多步更新后漂移、
  完整训练吞吐、重计算下的显存和梯度正确性。

## 已清理的过度设计

不再维护独立 `profile_tpr_dp2_step6.py`、`_npu_profile_capture.py` 及其大量 mock 测试。
一次 8-rank Level1 + stack/shapes/memory 原生 Profile 曾生成约 **118 GB**，
且解析耗时超过一小时，无法作为轻量性能诊断工具。

**唯一默认入口是既有 PPO replay**：

- `tests/models/mcore/tpr/parallel/run_tpr_tp2cp2dp2_one_step.sh`
- `TPR_EXEC_PROFILE=0`：可比的无诊断总耗时、loss 与峰值 allocated。
- `TPR_EXEC_PROFILE=1`：复用现有 `SegmentEventProfiler`；
  仅记录每个 rank 的 Push/Leaf/Pop NPU Event、执行前后 allocated/reserved、
  局部峰值、Tree Build / Executor Setup 等 Host 耗时。
  输出每 rank 一个小型 `stepN_rankR_segments.json`，**不调用 torch_npu.profiler**。
  此模式有埋点开销，**不作为 speedup 计算依据**。
- 事件显存不能简单相加；局部 peak 不等于长期 KV 真实保留量；
  CPU Host 时间不等于 NPU Kernel 时间，通信重叠也不能从 compute-stream Event 推断。
- 全局峰值在 Event 局部 reset 后需用 `SegmentEventProfiler` 保存的最大值恢复；
  不得误报更低显存。

Docker 示例（Git 只在宿主机操作；先核对原始 manifest 的实际路径）：

```bash
cd /workspace/verl-tpr-review
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
TPR_EXEC_PROFILE=1 TPR_BENCH_CAPTURE_LOGPROBS=0 \
TPR_BENCH_WARMUPS=1 TPR_BENCH_REPEATS=3 TPR_BENCH_MASTER_PORT=29798 \
bash tests/models/mcore/tpr/parallel/run_tpr_tp2cp2dp2_one_step.sh \
  /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1/capture_manifest.json \
  /workspace/tpr_experiments/performance/step6_light_v1 6
```

正式 Full/TPR 性能比较必须 `TPR_EXEC_PROFILE=0` 且关闭额外日志采集，
相同权重、样本、采样字段、并行参数、设备、重算策略和预热条件；
报告 max-rank median / p90 与峰值显存，不能用单 rank Host 时间代表全局吞吐。
上述轻量命令仍**不包含 optimizer step**。

## 优先级与完成条件

| 优先级 | 方向 | 下一项具体工作 | 完成门禁 |
| --- | --- | --- | --- |
| P0 | 精度与语义 | PR7 当前头重跑 TP2×CP4 sibling full gradients 与 TP2×CP2×DP2 Step6；修正不必要的对象、分支和状态，不改数学行为 | loss/logprob、完整梯度、DP replica 与前版冻结结果对齐；2% 是累计而非逐补丁预算 |
| P0 | 真正的 optimizer step | 相同离线 Capture、相同初始权重下，Native Full 与 TPR 各运行至少两步 AdamW（含 optimizer state、clip、weight update） | 每步 loss、grad norm、权重更新及 held-out Full 验证有独立记录 |
| P0 | Recompute | 完成 Full Uniform(1) 对 TPR Push/Leaf/Pop 与 KV 生命周期的验证，先 TP2×CP4，再 DP2；兼顾 30K/40K OOM | 开关前后 objective/梯度门禁 + 峰值显存、总步耗时；不能只证明 F/B 能运行 |
| P1 | 代码精简 | 模块级核对执行入口、Tree/Objective/KV、CP/Ring 与 Megatron 原生能力，逐项删冗余；保留 Padding/Turn Tree，确保旧测试覆盖不被误删 | 每批小 diff + CPU/8NPU 回归，未使用能力不混入生产热路径 |
| P1 | DP 生产化 | 当前 DP2 仍是实验门禁；确认真实 GRPO minibatch、DP Placement、world loss 分母、gradient finalize 和副本一致性 | 真正 VERL Actor 在线/离线 DP2 入口运行，完整参数梯度及更新通过 |
| P1 | 性能归因 | 用轻量 per-segment 时间/显存确定 Root Ring、Push/Pop、PPO chunk/gradient finalize 谁占主要成本 | 真实数据上 Main/PR7 无埋点对比，明确性能排名与回归风险 |
| P1 | Root Ring | 先核对能否直接复用 MindSpeed 的原生 Root，保留内部段正确的矩形因果与 Ring；不重新实现新通信后端 | CP2/CP4、GQA、多段 Prefix、Padding、前反向一致，再测收益 |
| P1 | Push / KV | 避免 Push 中被丢弃的 LM Head logits；验证 compact KV 的 storage 别名、Pop 释放时机 | loss/梯度不变，实测峰值和分段耗时下降；不靠理论显存估算宣称收益 |
| P1 | 完整 E2E | 使用统一捕获轨迹作离线精度对照；另做原生 Rollout→GRPO→Actor 多步真实训练 | 逐步 reward、loss、weight、显存、吞吐，无异常 OOM/重复计数 |
| P2 | TP/SP 与跨模型 | 原生 TP2×SP 在 CP1 的收益评估；扩展不同 Chat Template 的 Turn Tree，Qwen3-1.7B/4B 之外再测 Llama；Linear/GDN 与 CP 组合单独立项 | 明确各自支持边界、独立验收，不拖慢 Dense Qwen PPO 主线 |

## 四个待验证的性能方向

1. **Root Ring**：自定义 Root 的多块 FA 调度可能有冗余；先以代码和轻量分段耗时确认，再评估原生 MindSpeed Root。
2. **Ring HCCL**：通信次数、阻塞和重叠只在 Root 优化仍不足时进一步研究，不默认启用昂贵原生 Trace。
3. **Push LM Head**：共享前缀无需输出 logits，优先试已有 Megatron 输出处理接口，避免 monkeypatch。
4. **KV 压实**：核对 QKV backing storage、每层 KV 的唯一 storage bytes、Pop 后释放和 allocator 全局峰值，不把逻辑大小当物理节省。

## 维护与验收约束

- 代码与 Git 操作只在宿主机；Docker 工作树统一 `/workspace/verl-tpr-review`。
- `torchrun --master_addr=127.0.0.1 --master_port=<port>`，禁止 `--standalone`。
- 路径中的 Full/TPR 已验收 JSON、真实 Captures、模型权重均不可被新工具清理脚本自动删除。
- CPU 回归：`python -B tests/models/mcore/tpr/run_cpu_regressions.py`。
- 生产代码 PR / 优化结论必须配套真正 NPU 测试；CI CPU PASS 不等于 NPU 正确。
