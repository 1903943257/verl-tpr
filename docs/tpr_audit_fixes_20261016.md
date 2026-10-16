# TPR 本地审计修复（2026-10-16）

Last updated: 10/16/2026.

基线：PR #7 `0de0542d84b9f2a655ec62530c32c63311ea91a2`。
本地开发分支：`fix/tpr-audit-20261016`；提交目标为现有 PR #7。
不直接更新远端 main，PR 保持 Draft，等待人工审阅与服务器验收。

## 已修改

| 问题 | 处理 |
| --- | --- |
| FP32 原生 vocab-parallel CE 原地覆盖 logits，破坏 entropy 和诊断 | entropy 使用独立存储，诊断在 CE 前计算；保留原生 PPO 数学和归一化 |
| TP 诊断把全局 target 当作本地词表下标 | 按 TP rank 取 target 所有权，汇总全局 target logit 和稳定 logsumexp；只在诊断开启时通信 |
| 合法 minibatch 中存在全零监督独立树 | 绑定返回零 counts；Forest 跳过整棵无监督树，不改变原始 batch 分母；全空 Forest 仍拒绝训练 |
| 冻结 embedding/主干时，对不需要梯度的 KV 调用 backward | 过滤无 autograd 图的回传根；保留可训练头和祖先锚点的梯度链 |
| 随机 Dropout 令 Push/Pop 重算不一致 | 对非零配置及 training Dropout 明确拒绝；未实现 RNG 重放，不宣称支持随机层 |
| 无监督段对整词表做两次 FP32 转换/求和 | 用零宽视图构造连接到原始 logits 图的零标量，不读取 NaN，不改变 CP 反传参与性 |
| Push 的 KV view 一直钉住各层完整 QKV backing storage | 每层保存 KV 时，在显式 no-grad 模式下压紧；Pop 保留原始图；复用同一压紧函数 |
| 首次 dKV 接力无条件复制 anchor.grad | 内部消费并转移梯度缓冲区；跨 K/V、层及缓存别名时复制隔离；公共累加默认仍防御性复制 |
| CP 日志只反映当前 rank 的 PPO 指标 | 一次小字典交换后加总 SUM 和已加权 MEAN；空监督 rank 也参与；普通诊断仍保持列表语义 |
| 仅 TP 计划预检，CP 可能执行不同 collective 顺序 | 对整个 TP×CP 原生组交换计划摘要；发现差异时所有组内 peer 一起拒绝 |
| rank-local 非有限 loss 导致其它 peer 继续进入 collective | 树结束时在 TP×CP 组上检查有限性，再共同决定是否终止；不声称能恢复任意中途异常 |
| 对不支持的注意力语义静默忽略 | TPR 明确拒绝外部 mask、非因果/cross attention、滑窗、sink softmax 和额外位置 bias；无 TPR context 的原生路径不变 |
| CPU 数值回归未进入此 fork 的 CI | 增加显式隔离 CPU runner 及 PR/main/TPR 开发分支触发；原结构检查保留 |

`tpr/forest_trees`、`physical_segments`、`unique_tree_tokens` 仍表示规划出的拓扑规模，
不是跳过零监督树后的实际执行计数。

## 本地验证与范围

```bash
python -B tests/models/mcore/tpr/run_cpu_regressions.py
python -B -m unittest -q \
  tests.models.mcore.tpr.unit.test_ring_coalescing_schedule \
  tests.models.mcore.tpr.unit.test_ring_streaming_transport \
  tests.models.mcore.tpr.unit.test_offload_b_gate \
  tests.models.mcore.tpr.unit.test_activation_offload_adapter
git diff --check
```

执行结果：隔离 CPU 回归 **470 passed, 1 skipped**；独立 unittest **32 passed, 1 skipped**；
286 个 Python 文件语法检查、15 项独立架构检查和两份 replay shell 语法检查通过。
对应提交的 GitHub CI 状态以 PR Checks 为准；本地验证不代表 GitHub CI 已通过。
提交 `93e2368e` 的 [TPR GitHub Actions](https://github.com/1903943257/verl-tpr/actions/runs/38900850183)
随后在 Linux / TensorDict `0.10.0` 上完成 **471 passed**，包含真实双进程 Gloo 测试；
结构检查也通过。这仍不覆盖 NPU/HCCL。

隔离 runner 只跳过 Ray/Engine 包初始化，并提供未安装的 Megatron 接口桩；
实际 TPR、PyTorch autograd、TensorDict 和 loss 对照均是真实执行。
它不是完整 Megatron、原生 TP/NPU kernel 或优化器多步一致性的证明。

当前 Windows 验证环境：PyTorch `2.10.0+cpu`、TensorDict `0.14.3`。
新增 Gloo 双进程测试在该 Windows torch 构建中因无法创建 Gloo device 跳过；
Linux CI 不会对此错误跳过。HCCL 的 object collective 仍需服务器验证。
CI 使用项目允许的 TensorDict `0.10.0`；本地该版本的独立安装因网络下载停滞取消，
未将其列为已通过的验证。现有 AReaL 环境未修改。

## 明确保留的后续工作

- 超长 Turn/unary 段尚无限长切块；原生 activation recompute 仍不支持。单段训练显存峰值依旧可能 OOM。
- Push 仍调用完整模型并产生随后丢弃的 LM-head logits；需要原生模型前向接口支持，未用临时 monkeypatch 绕过。
- 同树内纯零监督分支仍执行；本轮只安全跳过整棵全局无监督树，不能用 CP-local counts 随意剪枝。
- Ring padding 仍可能生成二次方 mask；prefix coalescing 默认值和 Ring 协议未改，避免未经 NPU 门禁的可见性/通信顺序变化。
- 未批量重写 objective 的逐行设备校验、整棵树 hash 或深层 Turn 压缩；这些性能问题需单独基准及边界测试。
- experimental DP2 中整个 DP-local minibatch 零监督仍不支持；生产范围仍为原先的 DP1。
- 任意单 rank 在模型内部异常、collective 之前输入校验失败的全局协调，尚未统一实现。

服务器验收应固定 capture SHA、checkpoint、TP/CP/DP 拓扑、温度、熵系数、分母、dtype
和 chunk/coalesce 配置；对照 PR #7 基线与本分支的逐 token log-prob、全参数梯度、
优化器更新和多步训练，并分别记录暖身后时间及峰值显存。
现有 `compare_old_new_tpr_tp2cp4_ppo.py --require-full-grads` 可用于完整梯度检查；
DP2 脚本的 sampled gradient witness 不能替代它。现有旧版 wrapper 的导入路径固定为
历史 pre-refactor 布局，比较两个新布局 checkout 时应分别运行各自的 offline runner，
不要直接把 PR #7 当作该 wrapper 的历史 `OLD_ROOT`。

AI assistance: OpenAI Codex；合入/提交前仍需人工逐行审阅及服务器门禁。
