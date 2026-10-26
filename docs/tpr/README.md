# TPR开发文档

**新服务器从零开始**：[uniagent:new Docker → UniAgent SWE-reBench Full/TPR E2E](GETTING_STARTED_UNIAGENT_NEW.md)（包含宿主机 Git、Docker 创建、NPU 镜像验证、模型/数据、Runtime VERL、Reward、单机 E2E；多机尚未验收）


**主文档**：[TPR开发文档](TPR开发文档.md)  
**实验汇总**：[TPR实验结果](TPR实验结果.md)（正确性与性能历史快照，后续更新可替换）

这是一份用于项目交付和代码评审的开发文档，基于 `main` 的 `93b24dff`（2026-10-19）。主要说明 TPR（Tree Prefix Reuse）实现了什么、原生 VERL/Megatron 的修改范围、各个源码文件职责、真实训练调用链、Tree/Segment/KV/Attention/PPO/TP/CP 的实现边界及当前验收范围。

**专项方案**：

- [TPR Ring CP 方案](ring_design.md)：通信、Shard/Padding、块可见性和 dKV 返回
- [TPR Recompute 方案](recompute_design.md)：Tree Pop 与层级 checkpoint、KV 梯度及显存管理

已有专项资料（实验记录和协议性补充，不替代主文档）：

- [生产架构与验证契约](architecture.md)
- [性能与优化计划](performance_plan_20261017.md)
- [实验性 Recompute 记录](recompute_prototype.md)
- [PR7 契约修复](contract_fixes_20261018.md)
- [历史重构清单](refactor_inventory.md)
