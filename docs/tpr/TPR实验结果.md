# TPR实验结果

> **项目**：Tree Prefix Reuse（TPR）  
> **更新日期**：2026-10-19；**文档维护基线**：[`verl-tpr/main@25478d3`](https://github.com/1903943257/verl-tpr/commit/25478d3c5e1e585fcbf328c000b11e667013af10)。各阶段实验仍按各自历史代码、数据和测量口径解读。  
> **文档性质**：已有实验的**结果索引与临时性能快照**。所引历史测量日期、代码版本和测试边界不能自动升级为当前 main 的生产验收。未在仓库留存数值输出的实验，记录实现及复现实验入口，**不补造结果**。

## 1. 结果摘要与验收状态

| 类别 | 已有证据 | 当前应如何解释 |
| --- | --- | --- |
| 构树 / PPO / KV / Ring CPU 合同 | 主线具备回归与差分单测；历史 PR7 清理报告 `594 passed, 1 skipped` | 这是**当时分支 CPU**结果，不是当前 main 的 NPU 验收 |
| 真实 Qwen3-4B 两 sibling（TP2×CP4） | 旧 main / PR7 冻结对照：loss / logprob 相同，**完整参数梯度 rel-L2 = 1.027872%**，cos=0.99994723 | 通过当时 2% 阶段门禁；**未通过 1% 门禁**；不视为当前主线重新实测 |
| Qwen3-4B Step6 / 30K 三路径数值对照 | 同数据 Full CP4、TPR CP4、TPR CP2DP2 的 PPO loss 可比；**稀疏**梯度误差仍有 45.58%–50.27% | 不能用小 loss gap 或 CP/DP 副本一致判定 Full/TPR 算法梯度正确 |
| 真实 PPO 20 Step Full↔TPR（**历史汇总，未核原始 artifact**） | 历史报告：Loss 平均绝对差 **0.000794**、最大 **0.005623**；Logprob MAE 平均 **0.1891**、最大 **0.2744**；GradNorm 平均相对差 **3.76%**、最大 **15.70%** | 只能说明既有统计中 **Loss 未持续扩大、Logprob 中期抬升、GradNorm 有尖峰**；原始 `multistep_comparison.json` / rank snapshots 未随仓归档，不作为当前 main 多步数值验收 |
| AReaL DTA 对照实现 | 已有 **HF Full↔AReaL DTA**、**Megatron Full↔TPR** 和跨框架基线脚本 | 实验逻辑与输出项已在仓；**没有在本仓文档发现可直接引用的同次四路径运行数值**，需导入原始 artifact 才能填写 |
| 历史 Qwen3-1.7B CP1 长 Prefix，N8 | P8192/S1024（8:1）**3.002×**；P4096/S1024（4:1）**2.234×** | 2026-09 合成 Prefix/Suffix 受控 F/B，不是 PPO；原始日志尚未进仓；§3.1 |
| 更早约 0.566B GPT，CP1 | P16K/S1K/N8 **4.30×**；P16K/S2K/N8 **3.71×** | 不同模型与历史实现，不和 Qwen3 合并统计；§3.2 |
| 历史 Qwen3-1.7B RingCP4 | P16K/S4K：Prefix FA 不合并 **1.078×**；Q+KV Merge **1.249×** | 同 CP/P/S/N 的优化对照；Merge 默认关闭；FA 调度调用数不等于 kernel trace；§3.3 |
| 30K 固定数据性能 | Full CP4→TPR CP4 **1.0944×**；Full CP4→TPR CP2DP2 **1.3268×** | 均为 8 NPU 的**固定权重 F/B + 手工梯度 finalize**；不含 optimizer/rollout，DP 版本改变了 CP 拓扑 |
| 40K 固定数据性能 | 不开 recompute：Full **23.1378 s**、TPR **21.8022 s**，**1.0613×** | 仍是单步离线 F/B；开 recompute 数据有 GC 诊断差异，不能当正式性能收益 |
| 30K DP2 Recompute | ON/OFF PPO loss gap `3.73e-9`；**完整 BF16 梯度 shard** rel-L2 **0.9223%**；峰值 allocated **-59.65%** | 使用**实验 FP32 梯度同步**，不证明默认 BF16 生产同步或连续 GRPO |

**三条底线**：（1）“CPU PASS”“脚本完成”“数值门禁 PASS”不是同一种结论；（2）**完整梯度 rel-L2、抽样梯度 rel-L2、grad norm** 必须分开；（3）性能必须标注 **F/B、F/B+finalize、Actor train_batch（含 optimizer）、Rollout+GRPO 端到端** 的测量边界，不能交叉使用加速比。

## 2. 正确性实验

### 2.1 TPR 与 AReaL DTA 对照（已实现，待补结果快照）

**实验目标**：在相同 Qwen3-1.7B checkpoint、相同真实 TQ token / PPO 数据下，区分**树形/分段计算数值漂移**与**HF / Megatron 后端差异**。

| 比较组 | 参考 | 被测 | 可以回答的问题 |
| --- | --- | --- | --- |
| A（同 HF 后端） | HF Full | AReaL DTA | DTA 的分段 backward/Pop 相对自身 Full 引入多少偏差 |
| B（同 Megatron 后端） | Megatron Full | TPR Forest | TPR 的 Segment/Prefix 复用相对自身 Full 引入多少偏差 |
| C（框架底噪） | HF Full | Megatron Full | HF 与 Megatron 本身在同一数据/权重下的 logprob 差异 |
| D（仅观察） | HF DTA / HF Full | Megatron TPR | 原始跨框架差异；**不能直接归因于 TPR 算法** |

**运行输入和算法**：

- 真实 TQ **8 行 × 192 tokens**，每行取 P=128、S=64（共 512 个 Response token），Qwen3-1.7B。
- AReaL 的 backward 引擎代码通过测试专用适配保留，包含梯度根、KV fork/Pop 和梯度接力；**不是生产 TPR 实现**。
- DTA 实验可以使用真实记录的 `advantages` / `old_log_probs`；若未提供，脚本允许确定性代理值，必须标注 **PPO_PROXY**，不能称真实 PPO 训练。`TPR_PPO_REQUIRE_RECORDED=1` 可强制要求真实字段。
- [总入口](../../tests/models/mcore/tpr/correctness/run_qwen17_dta_tpr_triplet.sh) 默认 `TPR_DTA_BWD_OBJECTIVE=ppo_unclipped`、`TPR_DTA_BWD_BLOCK=64`；**这不等于默认开启 clipping 的完整 PPO 对照**。
- HF 端比较**所有参数**梯度（同 HF 命名空间），Megatron 端比较**逐参数采样**梯度（同 Megatron 命名空间）；不同框架的梯度未经参数映射，**不得直接比较 rel-L2**。

**源码与产物**：

| 组件 | 路径 | 输出 |
| --- | --- | --- |
| HF Full / DTA 实际 backward | [`test_qwen3_1_7b_areal_backward_ppo_npu.py`](../../tests/models/mcore/tpr/correctness/test_qwen3_1_7b_areal_backward_ppo_npu.py) | `hf_full_dta.pt`、`P1 DTA_BACKWARD SUMMARY` |
| AReaL DTA 原算法测试适配 | [`_qwen17_areal_native_dta_backward.py`](../../tests/models/mcore/tpr/correctness/_qwen17_areal_native_dta_backward.py) | 保留实际 fork/Pop/bwd，适配 HF logits/labels |
| Megatron Full / TPR 实际 backward | [`test_qwen3_1_7b_tpr_dta_triplet_npu.py`](../../tests/models/mcore/tpr/correctness/test_qwen3_1_7b_tpr_dta_triplet_npu.py) | `megatron_full_tpr.pt`、`P1 TPR_TRIPLET SUMMARY` |
| 四路径汇总比较 | [`compare_qwen17_dta_tpr_triplet.py`](../../tests/models/mcore/tpr/correctness/compare_qwen17_dta_tpr_triplet.py) | `P1 TRIPLET LOGPROB/GRAD/STEP`、`comparison.log` |

**当前结果记录**：对照实现和命令已经进入仓库，但上述两份 `.pt`、`comparison.log` 为运行目录产物；此次核对的主线文档**没有一组可追溯的同次四路径数值**。因此本节不填写 DTA/TPR 的相对误差百分比。尤其汇总脚本的 `P1 TRIPLET RESULT status=PASS` 表示**输入和比较流程完成**，同时写有 `numerical_parity=DIAGNOSTIC_ONLY`；不能解释为数值误差达标。HF DTA 脚本的 `RESULT status=PASS` 同样明确 `numerical_parity=UNVERIFIED`。

复现（需真实路径，Docker 内有 NPU、Qwen3-1.7B 和测试依赖）：

```bash
export TPR_REAL_TQ_BATCH=/path/to/real_tq.pt
export TPR_QWEN_1_7B_PATH=/path/to/Qwen3-1.7B
bash tests/models/mcore/tpr/correctness/run_qwen17_dta_tpr_triplet.sh
```

要形成可对外复核的结果，请将同一次运行的 `comparison.log`、HF/Megatron artifacts 的 SHA、`old_source`、`objective`、Attention 后端与梯度统计范围写入本节。

### 2.2 TP2×RingCP4：两 sibling 完整梯度阶段验收（历史快照）

2026-10-16/17 的冻结对照条件：Qwen3-4B BF16，8×Ascend NPU，真实 Actor PPO capture，两 sibling，TP2×CP4；对比旧 main 与 PR7 旧头，固定权重、单次 F/B + 原生梯度 finalize。

| 指标 | 记录 |
| --- | ---: |
| PPO loss / logprob | 旧头与基线一致（原始记录用语：完全一致） |
| **完整参数**梯度相对 L2 | **0.01027872（1.027872%）** |
| 完整参数梯度 cosine | **0.99994723** |
| 当时工程门禁 | **2% 阶段接受**；原始 **1% 门禁未通过** |

来源：[performance_plan_20261017.md](performance_plan_20261017.md)。该对照不是当前 `main@93b24dff` 的重新验收，也不是 Full-vs-TPR 理论精确性的证明。

### 2.3 Step 6 / 30K：Full、TPR 与 DP2 拓扑差异

2026-10-15 固定源 Qwen3-4B-128K、8 NPU、同一 30,720-token-cap capture、36,933 个受监督 PPO token、BF16，`HCCL_DETERMINISTIC=true`；F/B 后手工梯度 finalize，不执行 AdamW。

| 比较 | PPO Loss | **稀疏**梯度 rel-L2 | **稀疏**梯度 cosine |
| --- | ---: | ---: | ---: |
| Full TP2CP4DP1 | 0.151394129 | 参考 | 参考 |
| TPR TP2CP4DP1 vs Full | 0.151653767 | **0.455813** | 0.898021 |
| TPR TP2CP2DP2 vs Full | 0.151566282 | **0.502732** | 0.870341 |
| TPR CP2DP2 vs TPR CP4 | — | **0.416250** | 0.910348 |

以上是每个参数选 8 个位置、共 **4,640 个**的不加权 gradient witness；**不是所有参数的 rel-L2**，且会相对高估小规模 Norm 张量的重要性。三个路径各自的 CP/DP 副本抽样一致，**不意味着 Full-vs-TPR 算法梯度通过**。后续已确定需区分 BF16 分段执行、clip 分支、梯度累积和 CP/DP 数值影响。

来源：[2026-10-14/15 30K 三路径完整记录](../tpr_tp2cp2dp2_one_step_20261014.md)（含相同 capture SHA 和参数边界）。这组冻结数据不能与 2.2 的不同源、不同基线和不同梯度统计口径混为同一通过率。

### 2.4 单卡数值归因与分段执行对照

当前主线已有四条受控路径：

- **Full**：CP1 square-CANN，完整 Sequence，不启用 TPR Context；这是**受控 Full 算子参考**，不自动等于 VERL Native Engine。
- **One-node**：独立单节点，无共享、无切分。
- **Split-no-share**：按与共享版相同的 Segment 边界分段，但不同轨迹之间不复用 KV。
- **Shared TPR**：共享 Root、两个 sibling 分支，包含 Prefix dKV。

此设计分别隔离 **Full→单节点后端差异**、**单节点→分段执行差异**、**无共享分段→共享 KV 差异**；代码见 [CP1 对照](../../tests/models/mcore/tpr/parallel/run_real_actor_cp1_short_full_tpr.sh) 及 [Attention 归因](../../tests/models/mcore/tpr/parallel/run_cp1_attention_attribution.sh)。固定数据会裁剪原始真实轨迹的上下文，因此是**相同保留标签的受控数值比较，不等价于原始完整在线 PPO**。当前文档没有为这四条路径提供一套已归档的完整参数实测数值；不可把脚本存在写成梯度 parity PASS。

其他历史 Qwen3-1.7B 短轨迹案例（2026-10-09）：真实 TQ 8×(P128/S64)，原始 BF16 训练观测到 **31/512** PPO clipped 分支不同，selected-parameter gradient rel-L2 **0.523553**、cos **0.89627931**；在**人为对齐的 BF16 Tile** 数值诊断中，clip 分支为 0/512、selected-gradient rel-L2 **0.012100**。来源：[QWEN17_NUMERICS_AND_GEMM.md](../../tests/models/mcore/tpr/profiling/QWEN17_NUMERICS_AND_GEMM.md)。这是**历史因果归因实验**，不是当前主线启用 Fixed Tile 的建议；正式 Full/TPR 对照**不得强行开启 Fixed Tile**。

### 2.5 Recompute 数值验收（实验）

已归档 2026-10-18 的 Qwen3-4B **TP2×RingCP2×DP2**、8 NPU、原始 Actor Step 6、30K OFF/ON 固定数据对照：

| 指标 | OFF vs ON |
| --- | ---: |
| Global PPO loss 绝对差 | **3.7252903e-9** |
| **完整 BF16 参数梯度 shards** 相对 L2 | **0.00922255517（0.9223%）** |
| CP×DP 副本样本门禁 | PASS |
| CPU FP32-master 单次 AdamW 样本更新 max abs gap | **3.05175781e-5** |

采用实验性**FP32 逐参数梯度归约**，不是默认 BF16 生产通信。单次实验性 AdamW 验证不能推出多步在线 GRPO 收敛。来源：[recompute_prototype.md](recompute_prototype.md)。

### 2.6 真实 PPO 20 Step：Loss / Logprob / GradNorm 历史漂移（仅归档，不新增实验）

> **证据级别：历史对话中已报告的统计摘要，未在当前仓库复核原始 20-Step artifact。** 相关离线测试和比较器存在，但当前文档分支没有附上对应的 `multistep_comparison.json`、`full_step*_rank*.pt`、`tpr_step*_rank*.pt` 或同次日志，因此下表为**待原始文件复核的历史数值**，不能把“支持连续更新的脚本存在”当成“本次重新证明运行成功”。本节按历史资料保存，不计划为此再补实验。

**历史记录中的设置**：Qwen3-4B-128K，BF16，TP2×RingCP4，8×Ascend NPU；真实 GRPO/Actor 捕获的 PPO 数据（group_size=8、40K 截断）；Full 和 TPR 使用同一初始权重、各自的 CPU FP32-master AdamW 状态；数据和 historical `old_log_probs` / advantages 固定，非每 Step 重新采样的 on-policy 训练。以上是历史报告口径，**具体 20 步源 Capture / SHA、选取行数、Recompute 状态及完整运行配置仍须以原始 artifact 为准**。

| Step | PPO Loss 绝对差（Full vs TPR） | Response Logprob MAE | 全局 GradNorm 相对差 |
| ---: | ---: | ---: | ---: |
| 1 | 0.000111 | 0.0331 | 3.97% |
| 2 | 0.000417 | 0.0541 | 2.75% |
| 3 | 0.000444 | 0.0885 | 6.11% |
| 4 | 0.000130 | 0.1407 | 1.45% |
| 5 | 0.001587 | 0.2387 | 3.92% |
| 10 | 0.000678 | 0.2185 | 1.93% |
| 13 | 0.005623 | 0.2744 | 9.10% |
| 14 | 0.000641 | 0.2623 | 15.70% |
| 20 | 0 | 0.1733 | 0% |
| **历史 20 Step 汇总** | **平均 0.000794；最大 0.005623（Step 13）** | **平均 0.1891；最大 0.2744（Step 13）** | **平均 3.76%；最大 15.70%（Step 14）** |

**解读及边界**：

1. **PPO Loss**：历史汇总中多数 Step 的绝对差很小，Step 13 的差值较高，但未呈现持续单调放大；**这不是 Reward/在线收敛的保证**。
2. **Logprob**：MAE 从 Step 1 的 **0.0331** 抬升至 Step 13 的 **0.2744**，Step 20 回落至 **0.1733**；说明中期存在明显数值漂移，不能由小 Loss gap 推断 Logprob parity。
3. **GradNorm**：平均相对差 **3.76%**，Step 14 峰值 **15.70%**；这是**全局梯度范数的标量差异**，**不是完整梯度向量 rel-L2，也不是梯度方向 cosine**。小 GradNorm gap 不表示两个梯度向量一致。
4. 历史记录还注明 Step **11/12/16/19/20** 的两路 PPO Loss / 有效梯度为零；若从原始快照证实，相关 Step 的 GradNorm gap=0 **不构成梯度正确性的正向证据**。即便新梯度为零，AdamW 也可能受历史动量影响继续更新权重。
5. **实验口径**：固定录制轨迹、旧策略 Logprob 和 Advantages 的离线多步回放，不是逐步重新 Rollout 的 on-policy GRPO；也不能自动证明当前 `main@93b24dff`、新 Ring / Recompute / DP 组合下的数值行为。

**已有代码入口（可查证对比定义，非新的实测）**：
[`run_real_actor_tp2cp4_multistep.sh`](../../tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_multistep.sh) →
[`test_real_actor_tp2cp4_multistep_npu.py`](../../tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_multistep_npu.py) →
[`compare_real_actor_tp2cp4_multistep.py`](../../tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_multistep.py)。
比较器定义了 `loss_gap`、`logprob.mean_abs`、`grad_norm_relative_gap`，并另外支持梯度向量、FP32 master 参数累计更新和副本审计。**本节只有上表三个标量指标的历史摘要，未归档完整梯度 cosine / 参数更新 rel-L2 / 共同 Full oracle 评估数据。**

**阶段性处理**：本轮将该历史多步漂移作为**已观察到、存在证据边界的风险**记录，暂不补做 20 Step；项目其余单步正确性结论维持各自原有门禁，不将本节提升为“当前代码多步严格 parity PASS”。

## 3. 已有性能测量（**可替换历史快照**）

**分组原则**：§3.1–3.3 为历史 Qwen3-1.7B / 0.566B GPT **人工共享 Prefix/Suffix** 的受控模型前反向 Profile；§3.4–3.6 为 Qwen3-4B **真实 Actor 捕获的离线 PPO**，包含 PPO Loss 与梯度 Finalize，不包含 Megatron 生产 Optimizer 或在线 Rollout。不同模型、TP/CP/DP、P/S/N、以及 Merge ON/OFF 的加速比都**不能混算平均值**。加速比为对应行 Full 时间 / TPR 时间；历史日期不代表最新主线复测。


### 3.1 Qwen3-1.7B：历史长 Prefix、CP1（2026-09）

真实 Qwen3-1.7B 权重（约 1.721B 参数），人工构造 **1 个共享 Prefix + N=8 个 Suffix**；以 8 条完整 `P+S` 轨迹的 Full 训练前反向为参考。CP1 不使用 Ring。以下摘录 **2026-09-15、09-16、09-20** 的历史 Profile 日志，**不是 VERL Actor PPO 完整训练**。

| Prefix P | Suffix S | P:S | Full（ms） | TPR（ms） | 加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 512 | 4:1 | 3,167.4 | 2,566.6 | **1.234×** |
| 4,096 | 4,096 | 1:1 | 6,363.3 | 4,623.6 | **1.376×** |
| 6,144 | 2,048 | 3:1 | 6,337.7 | 3,563.7 | **1.778×** |
| **8,192** | **1,024** | **8:1** | **6,216.3** | **2,070.6** | **3.002×** |
| 4,096 | 1,024 | 4:1 | 3,121.8 | 1,397.7 | **2.234×** |

对 `P8192/S1024/N8`：Full 的逻辑 Token 数为 `8×(8192+1024)=73728`，共享方案去重的逻辑 Token 数为 `8192+8×1024=16384`，**理论 Token 共享比例 = 77.78%**；对应前反向加速 **3.002×**。77.78% 不是实际耗时节省比例，且没有计入 Push/Pop 重算的重复算力；注意力复杂度、BF16 GEMM 形状、词表 Head、调度等都会影响时间。各行总长度不同，不能视为严格等总工作量的单变量 P:S 消融。

**可追溯性**：数值来自已发现的历史原始 Profile 日志摘录，**完整 `.log` 及运行 Commit/输入 SHA 尚未随仓保存**；不是本次在当前 `main` 上重跑的结果。仓库现有 [真实 Qwen CP1 Profile](../../tests/models/mcore/tpr/profiling/test_tpr_qwen3_profile_npu.py) 和 [共享 Profiling Harness](../../tests/models/mcore/tpr/profiling/test_tpr_engine_profile_npu.py) 可用于查阅方法，但不应认定它们与 9 月原始日志的代码版本完全相同。

### 3.2 早期约 0.566B GPT：长 Prefix、CP1

另一套**不同模型与历史实现**的 Dense GPT 合成基线，固定 `P=16384`，前反向测量如下：

| Prefix P | Suffix S | N | Full（ms） | TPR（ms） | 加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 1,024 | 2 | 6,244.5 | 3,896.5 | **1.60×** |
| 16,384 | 1,024 | 8 | 24,148.7 | 5,607.9 | **4.30×** |
| 16,384 | 2,048 | 8 | 27,841.4 | 7,503.6 | **3.71×** |

同样的 P/S，N 从 2 增到 8 时共享的累计收益扩大；N8、P16K/S1K 达 **4.30×**。**该 0.566B GPT 不是 Qwen3-1.7B/4B**，也不是当前默认 Profiling Harness 的受支持模型；**不要合并不同模型的加速比求平均**。这是独立的历史快照，原始 `.log` 尚未作为仓库 artifact 归档。

### 3.3 Qwen3-1.7B：CP4 Ring 与 Prefix 合并（2026-09）

同样采用真实 Qwen3-1.7B 权重和 **N=8** 受控 Prefix/Suffix，但启用 **Ring CP4**；Full 为独立完整轨迹，TPR 为共享 Prefix。2026-09-18～20 的历史结果：

| Prefix P | Suffix S | P:S | Prefix FA 优化 | Full（ms） | TPR（ms） | 加速比 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 16,384 | 4,096 | 4:1 | 不合并 Prefix FA | 7,989.2 | 7,411.9 | **1.078×** |
| 16,384 | 4,096 | 4:1 | Prefix Q+KV Merge | 8,001.3 | 6,406.9 | **1.249×** |
| 16,384 | 16,384 | 1:1 | Prefix KV Merge | 13,790.7 | 13,517.8 | **1.020×** |
| 16,384 | 16,384 | 1:1 | Prefix Q+KV Merge | 13,763.2 | 12,950.2 | **1.063×** |

**同一 CP4 / P16K / S4K 的 Merge 对照**：将 Prefix Q+KV Merge 开启后，TPR 耗时从 **7,411.9 降至 6,406.9 ms**（减少 **13.56%**），相对 Full 加速从 **1.078× 提高至 1.249×**。两次 Full 时间略有差异，不是完全固定分母。

关联的**调度层前向 FA 调用数**由 **6,104 → 3,416**（每 Rank，28 层、8 个分支并包含 Push/Pop，减少 **44.0%**）。该调用数可由 [Ring Prefix Coalescing](../../tests/models/mcore/tpr/profiling/RING_PREFIX_COALESCING.md) 的 **shape-only/dispatch 统计合同**支持，**不是本仓已归档的 NPU 硬件 kernel trace 实测**；不能直接把 44% 的调用数降低解释为等比例硬件算力节约，Ring 通信次数也不会因此同步减少。

`P16K/S16K (1:1)` 的 Q+KV Merge 只有 **1.063×**。这与较低共享 Prefix 比例、分块 FA/通信等成本共同相关，但缺少完整的设备级归因记录。

**CP1 与 CP4 不能直接消融**：CP1 的 `P4K/S1K/N8` 为 **2.234×**；CP4 的 `P16K/S4K/N8` 即使 Q+KV Merge 为 **1.249×**。两组虽均为 4:1、CP4 全局长度为 CP1 四倍且 Rank-local 长度近似，**全局工作量、通信、算子切分与 Profile 路径不同**，不能把差额全部归因于 Ring CP 开销。应优先使用**同一 CP4 P/S/N 的 Merge OFF/ON**分析 FA 调度成本。

`TPR_RING_COALESCE_PREFIX_FULL/QUERY/PADDED` 均为**默认关闭的可选优化**；以上 Merge ON 不代表生产默认速度。历史延迟来自 2026-09 Profile 日志摘录（完整日志尚未进仓），优化方法和调度数见 [Coalescing 文档](../../tests/models/mcore/tpr/profiling/RING_PREFIX_COALESCING.md)，参考运行入口为 [真实 Qwen Ring CP Profile](../../tests/models/mcore/tpr/profiling/test_tpr_qwen3_ring_cp_profile_npu.py)。后续补充原始日志时应附 SHA、配置及实际 FA trace。

### 3.4 真实 Actor PPO 30K：同一数据、三种并行组合

2026-10-15 完成的三路径实验，Qwen3-4B-128K、8×Ascend NPU、原始 Step6 30K 派生数据、36,933 监督 PPO token、1 次 warmup + 3 次测量、BF16，**F/B + 手工梯度 finalize**，不包含 optimizer、rollout 或完整 GRPO。

| 模式 | 并行 | 中位耗时（s） | 峰值 allocated（GiB） | 相对 Full 加速 |
| --- | --- | ---: | ---: | ---: |
| Native Full | TP2CP4DP1 | **16.454548** | 34.638256 | 1.000× |
| TPR | TP2CP4DP1 | **15.035786** | 24.657299 | **1.094359×** |
| TPR + DTA-style DP placement | TP2CP2DP2 | **12.401572** | 41.070560 | **1.326811×** |

独立结论：

- **只开启 TPR、固定 TP2CP4**：F/B+finalize 加速 **9.44%**，按 Full 时间计算省约 1.419 s；峰值显存 **-28.81%**。
- **TPR 从 CP4 切为 CP2DP2**：相对 TPR CP4 **1.212410×**，但峰值显存从 24.66 GiB 增至 41.07 GiB（+66.57%）。
- **Full CP4 → TPR CP2DP2** 的 **1.326811×** 同时包含 TPR + 拓扑变化，**不是纯 TPR 算法收益**。
- 当前基准包含**手工 finalize**、没有生产 Megatron DDP bucket overlap；该性能加速尚不能代表完整生产 Actor 更新时间。并且 §2.3 的稀疏梯度数值门禁**未通过**，性能结果只能作为暂存工程测量。

来源：[30K 完整实验](../tpr_tp2cp2dp2_one_step_20261014.md)。

### 3.5 真实 Actor PPO 40K：相同 Step 6，原生 Full 与 TPR

2026-10-16 记录：Qwen3-4B、8 NPUs、同输入 SHA、TP2CP4、1 warmup + 3 measured，F/B 固定形状对照：

| 执行模式 | Recompute | 中位耗时（s） | 峰值 allocated（GiB） | 相对同状态 Full |
| --- | --- | ---: | ---: | ---: |
| Native Megatron Full | OFF | **23.137812** | 43.616 | 1.000× |
| TPR | OFF | **21.802193** | 31.173 | **1.06126×** |
| Native Megatron Full | ON | **32.829754** | 13.746 | 1.000× |
| TPR | ON，附加 GC/weakref 诊断 | **29.913969** | 14.865 | **1.09747×（暂定）** |

**主要采用无 recompute 的 1.06126× 作为这份历史快照的 TPR 同拓扑参考。** ON 版使用了 GC/weakref 诊断，需关闭诊断后重测稳态；不能作为最终线上收益。开启 recompute 时的不同显存变化也不能与 OFF 实验混为相同计算配置。

来源：[recompute_prototype.md：40K 固定形状](recompute_prototype.md)。

### 3.6 真实 Actor PPO 30K：Recompute 的时间/显存交换

Qwen3-4B、TP2CP2DP2、8 NPU、真实 Step6、同 capture、FP32 梯度归约，**TPR OFF/ON** 的固定数据实验：

| TPR 状态 | F/B+finalize 中位（s） | 峰值 allocated（GiB） |
| --- | ---: | ---: |
| Recompute OFF | **12.362386** | **40.653** |
| Recompute ON | **16.534103** | **16.403** |
| ON 相对 OFF | **耗时 +33.74%** | **显存 -24.250 GiB / -59.65%** |

此结果证明该**实验配置**下的时间换显存；计时**不含单独 optimizer 更新**。不直接预测 TP2CP4、默认 BF16 同步或线上 rollout 的收益。

同一归档还有 40K **Recompute ON-only**：median 28.582731 s、峰值 allocated 20.419 GiB；由于 **OFF 40K OOM / 无合格成对基线**，不能报告 40K OFF/ON 相对加速或显存节省百分比。

来源：[recompute_prototype.md：DP2 结果](recompute_prototype.md)。

### 3.7 尚缺严格同配置对照的性能维度

当前代码具有多种测试入口，但这些不应自动展示为已完成结果：

| 维度 | 已有入口 / 现状 | 后续报告时必填 |
| --- | --- | --- |
| Qwen3-4B 固定 N8、约 9K、8:1 CP1 | 已有的是 §3.1 **Qwen3-1.7B** 的 8:1 高收益历史；Qwen3-4B 同类实验尚未完成 | 同权重与样本、P/S/N、Full/TPR 多次无埋点耗时 |
| Qwen3 同 P/S/N/TP 的 CP1、CP2、CP4 | §3.1 和 §3.3 的 CP1/CP4 不同形状，**CP2 完整独立曲线也未找到** | 同模型/同全局长度/同输入，不把 CP2DP2 与 CP4 的差额仅归于 CP |
| Qwen3.5 Stage 4.5 AllGather CP2 | [Stage 4.5 性能规范](../../tests/models/mcore/tpr/parallel/STAGE45_PERFORMANCE.md) 自身数据 pending，且模型/后端不同于历史 Qwen3 Ring | AllGather 与 Ring 独立报告、P/S/N、真实通信时间 |
| Ring 合并的设备级消耗归因 | §3.3 仅有历史耗时与调度层 FA 调用数，缺少与历史版本匹配的 NPU Kernel/通信 trace | Prefix/Current FA F/B、Ring P2P、launch overhead、peak memory |
| 完整 Actor 更新（含 optimizer） | [多步对比工具](../../tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_multistep.sh) | 原生 Full/TPR 的训练 Step wall time、更新和 gradnorm |
| Rollout + GRPO + Actor E2E | 主线已有 native rollout 和 Actor 更新能力，**此文未收录可比 TPR E2E 吞吐基准** | trajectories/s、tokens/s、rollout/train 占比 |
| Padding/Shard、Root Ring 和 Prefix Q/K/V merge | [主线架构与验证说明](architecture.md) 中存在 opt-in 诊断 | CPU/NPU 数值门禁、FA 调用数、实际时间收益 |

## 4. 结果更新协议

**固定不变的比较口径**：

1. **数据与权重**：模型/参数 dtype、checkpoint hash、capture SHA、行 UID、P/S、N、树数、Segment 数、真实共享率、旧 logprob/advantages/temperature。
2. **算子与并行**：TP/CP/DP/SP、Ring backend、recompute、logprob chunk、BF16/FP32 梯度同步、HCCL deterministic、是否启用实验 merge。
3. **正确性**：响应 token logprob mean/max/rel-L2、PPO loss gap、有效 clip flip；梯度统计必须注明 ALL_PARAMETERS / FULL_NORM_ONLY / SAMPLED_PER_PARAMETER；梯度副本一致性单独呈现。
4. **性能**：warmup/repeats、rank-max critical median、峰值 allocated/reserved、是否包含 finalize/optimizer/rollout、是否启用 profiler/capture、OOM 情况；不能拿不同拓扑的比例当纯 TPR 加速。
5. **结果状态**：`历史快照` / `同当前 main 重验` / `诊断通过、算法门禁失败` / `未执行`。新数据附输出文件或日志、Commit SHA 和完整配置，**替换对应表格，不覆盖旧原始 artifact**。

当前先保留历史 **CP1 3.002× / CP4 Ring 1.078× / CP4 Q+KV Merge 1.249×** 的条件性证据，不为补齐 8:1 立即重跑长实验。后续有同模型、同 P/S/N、同 TP 的 CP1/2/4 数据时，再单独报告跨 CP 代价并保留原日志来源；不能覆盖不同模型的历史快照。数值漂移问题优先采用四臂分段隔离和 AReaL DTA **同后端相对 Full** 的结果，不将不同后端的 raw gap 当作算法误差。

## 5. 相关文档与快捷入口

- [TPR开发文档](TPR开发文档.md)：架构、依赖、组件边界、真实训练调用链。
- [TPR Ring CP 方案](ring_design.md)：基础 Ring F/B、Shard/Pad、Root 和合并优化。
- [TPR Recompute 方案](recompute_design.md)：层级 checkpoint 机制和历史 OFF/ON 证据。
- [TPR 主线架构](architecture.md)：Ring 执行、数值归因与 NPU 门禁。
- [历史性能/优化计划](performance_plan_20261017.md)：阶段性正确性验收与生产化计划。
- [30K 三路径记录](../tpr_tp2cp2dp2_one_step_20261014.md)：同源 Full / TPR CP4 / TPR CP2DP2 详情。
- [Recompute 原始实验记录](recompute_prototype.md)：DP2 OFF/ON、40K 结果与内存问题。
- [Qwen3-1.7B GEMM/精度记录](../../tests/models/mcore/tpr/profiling/QWEN17_NUMERICS_AND_GEMM.md)：早期数值差异及隔离实验。
- [Qwen3 CP1 受控 Profile](../../tests/models/mcore/tpr/profiling/test_tpr_qwen3_profile_npu.py)、[Qwen3 Ring CP Profile](../../tests/models/mcore/tpr/profiling/test_tpr_qwen3_ring_cp_profile_npu.py)：人工 Prefix/Suffix 性能分析入口。
- [Ring Prefix Coalescing](../../tests/models/mcore/tpr/profiling/RING_PREFIX_COALESCING.md)：Prefix KV/Q 合并原理、FA 调度数与验证合同。
- [CPU 回归总入口](../../tests/models/mcore/tpr/run_cpu_regressions.py)：结构验证；**不是 NPU 算法正确性结论**。
