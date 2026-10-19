# PR7 最小化迁移：保留正确性修复

Last updated: 10/18/2026.

基底：最新 PR7 `662211d518be568197fe5702b4566651e4b411d9`。
本地分支：`fix/tpr-contract-cleanup-20261018`。本轮仅作本地提交，不推送远端。
顺序固定为：**正确性 → 代码精简与框架复用 → 定位真实瓶颈 → 小范围优化 → 完整训练验证**。

## 保留的必要修复

- 生产 TPR 删除 GDN 状态/分支；实验 Context 独立，原生非 TPR VERL 模型兼容性不变。
- Selected BF16/FP16 query 先升 FP32，再调用原生 logprob/entropy；同 query 的多 target/
  temperature 在 FP32 汇总梯度后一次回铸，CE/entropy 存储独立，不转换整份物理 logits。
  Checkpoint 不拆开同 query 组，因此 chunk size 对单组是 soft cap。
- 正确解包温度 NonTensorStack，保留 plain dict/NonTensorData 兼容；温度、字段行和 masks
  每 batch 准备一次。复用原生 PPO，不增加另一套 loss 或通用缓存框架。
- DP 成本复用生产 Turn 边界、prompt 端点和 minimum；Exact 参考单独标注。
  **新 planner 可能改变 DP 分区**，重验需核对实际分区；旧 Exact 成本和旧 NPU 结果不能直接继承。
- 全局 objective owner 的祖先闭包决定剪枝；CP-local 零 refs 仍参与必要的 Ring backward。
  父 query 预测子首 token 的监督保留；全 forest active SP 段执行前预检，不静默补齐。
- 深树压缩改迭代 DFS；模型准备按 batch 复用，继续检查 live dropout 设置。
  异常时释放 executor 栈持有的 KV/dKV 并禁止继续；外部 anchors/traceback 引用不在释放保证内。

## 不恢复重型 Profile

四个独立采集/专用 mock 文件及其 runner 注册保持删除。不恢复旧 CLI、observer、module hooks、
全 rank 算子/stack/shapes/memory Trace 或解析导出框架；只为旧 driver 服务的 partitions
透传/可选 API 和专用测试也不迁回。

保留新基底已有的 replay + SegmentEventProfiler：默认 `TPR_EXEC_PROFILE=0`；需要分段归因时
才设为 `1`，每 rank 输出小型 JSON，不调用 `torch_npu.profiler`，保留全局 peak 恢复逻辑。
不新建测试入口。命令和解释沿用[主线路线图](performance_plan_20261017.md)。
Event 模式不用于 speedup 结论，也不能证明 HCCL 重叠；收益用同条件无诊断重复测量确认。

## 验证边界

CPU：`python -B tests/models/mcore/tpr/run_cpu_regressions.py`。
本轮：**594 passed、1 skipped**；289 个 Python 文件 AST、两份相关 shell 语法及
`git diff --check` 通过。减少的 36 项是已删 Profile 工具的专用覆盖和不再需要的 pin 接口测试。
CPU 使用真实 torch autograd/TensorDict 和隔离的缺失 Megatron 接口，不代表真实 NPU/Megatron。
本机 Windows Gloo、真实 ModuleSpec、既有两个 experimental hybrid 失败的边界不变。

新 FP32 契约、checkpoint 分组、全局剪枝和 DP 分配需在真实 NPU 重新验收；
历史 2% BF16 容差是累计预算，不是每个补丁的预算。DP2 全参数梯度、CP metrics 独立参考、
连续 AdamW/optimizer state、完整训练和吞吐仍未验收。未改变 Ring 算法、Attention 后端、
Fixed Tile 或 Native GEMM 形状；本轮不声称净减码或性能收益。AI 协助修改，合并前需人工 review。
