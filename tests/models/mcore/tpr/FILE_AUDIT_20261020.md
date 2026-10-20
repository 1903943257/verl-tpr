# TPR 测试逐文件清理台账（2026-10-20）

> 此表记录 **仓库路径、测试职责和初步保留建议**，不是每个文件都已运行或逐行完成依赖审查。对未纳入本次源码/引用精查的文件标记为 `REVIEW-DELETE`，其含义是“下一批核对调用者/数值证据后删除”，**不是授权直接删除**。已经删除的文件另见 `PRUNE_LOG.md`。

## 本轮清理数据

- 现存 TPR 文件：**308**；其中 `test_*.py`：**177**。
- Linear/GDN/Qwen3.5 Hybrid 已隔离：**34 个测试文件**，共 58 个原有文件移入 `experimental/linear/`，另增 3 个 package initializer。
- 默认 CPU 引导保持明确的 **38 个选定测试文件**，移动 GDN 后无悬挂路径（静态校验）。
- 本次新增删除：历史 10K TP/CP/DP 链 6 个、实验性旧 DP 门禁 1 个、失效 Fixed Tile helper 1 个；同时移除 112 行 Fix Tile/FC2 fixed-M 诊断干预。
- 测试整理分支没有改动 `verl/` 生产代码；NPU 数值复测仍须在用户 Docker 执行。

### 标记

- `KEEP-P0`：正式入口、CPU 白名单、Ring/TP/DP/真实 PPO/recompute 等持续验收依赖。
- `KEEP-P1`：当前未列入短门禁但有独立契约价值；先保留，必要时合并相近测试。
- `KEEP-DOC`：辅助代码或历史文档不适用；这些项目按代码风险单独处理。
- `REVIEW-DELETE`：旧阶段实验、数据诊断或多重比较；先检查是否被脚本/报告引用以及基线是否有替代，再执行删除。
- `EXPERIMENTAL`：Linear/GDN/Qwen3.5 Hybrid 支线，不属于 Dense FA 主线验收。

## 所有现存文件（逐文件）

| 相对路径 | 建议 | 具体职责 / 保留理由 |
| --- | --- | --- |
| `PRUNE_LOG.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `README.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `TEST_MATRIX.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `__init__.py` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `correctness/README.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `correctness/__init__.py` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `correctness/_qwen17_areal_dta_reference.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/_qwen17_areal_native_dta_backward.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/_qwen17_areal_training_forward_reference.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/_qwen17_dta_style_reference.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/_qwen17_ppo_acceptance.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `correctness/_qwen17_weak_e2e_sampling.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/compare_qwen17_dta_tpr_triplet.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/run_fixed_8row_swe_tq_parity.sh` | KEEP-P1 | 真实轨迹 Capture/Manifest 处理 |
| `correctness/run_qwen17_areal_full_backward_ppo.sh` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `correctness/run_qwen17_dta_tpr_triplet.sh` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/run_qwen17_hf_dta_reference.sh` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/run_qwen17_hf_dta_training_forward.sh` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/test_qwen3_1_7b_areal_backward_ppo_npu.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `correctness/test_qwen3_1_7b_hf_dta_reference_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/test_qwen3_1_7b_hf_dta_training_forward_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/test_qwen3_1_7b_real_tq_ppo_npu.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `correctness/test_qwen3_1_7b_shape_rootcause_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/test_qwen3_1_7b_split_equivalence_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/test_qwen3_1_7b_tpr_dta_triplet_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `correctness/test_qwen3_dotproduct_vs_rectangular_npu.py` | KEEP-P1 | 真实模型兼容性、参数和数值/性能对照 |
| `correctness/test_tpr_correctness_report_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `correctness/test_tpr_kv_checkpoint_replay_npu.py` | KEEP-P0 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `correctness/test_tpr_native_recompute_contract_npu.py` | KEEP-P1 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `correctness/test_tpr_qwen3_compatibility_npu.py` | KEEP-P0 | 真实模型兼容性、参数和数值/性能对照 |
| `correctness/test_tpr_qwen3_reference_twenty_step_repeatability_npu.py` | REVIEW-DELETE | 真实模型兼容性、参数和数值/性能对照 |
| `correctness/test_tpr_qwen3_three_way_twenty_step_npu.py` | REVIEW-DELETE | 真实模型兼容性、参数和数值/性能对照 |
| `correctness/test_tpr_qwen3_twenty_step_correctness_npu.py` | REVIEW-DELETE | 真实模型兼容性、参数和数值/性能对照 |
| `correctness/test_tpr_twenty_step_correctness_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `equivalence/__init__.py` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `equivalence/test_fixed_topology_scheduler_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `equivalence/test_megatron_engine_tpr_entry_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `equivalence/test_rectangular_attention_npu.py` | KEEP-P0 | Attention/算子数值和 GEMM 精度 |
| `equivalence/test_segment_push_pop_npu.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `equivalence/test_tpr_attention_equivalence_npu.py` | KEEP-P1 | Attention/算子数值和 GEMM 精度 |
| `equivalence/test_tpr_engine_reference_equivalence_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `equivalence/test_tpr_gpt_model_equivalence_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `equivalence/test_tpr_gpt_optimizer_step_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `equivalence/test_tpr_transformer_block_equivalence_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `experimental/__init__.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/STAGE33_QWEN35.md` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/__init__.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/_causal_conv_boundary_diagnostic.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/_first_layer_projection_control.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/_gdn_state_test_utils.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/STAGE45_PERFORMANCE.md` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/STAGE45_STATEFUL_OPS.md` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/STAGE4_GDN.md` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/__init__.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_cp_gdn_test_utils.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_fa_core_replay.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_fa_fp32_reference.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_first_gdn_zigzag_control.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_full_hybrid_drift_diagnostic.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_full_linear_shape_control.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_gdn_layer_replay.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_hybrid_divergence_probe.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_native_ring_tnd_layout.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_ordinary_ring_control.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_prefix_reuse_metrics.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_propagation_closure.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_ring_merge_probe.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_stage45_sync_trace.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_stateful_ops_probe.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_trajectory_fa_trace.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/_whole_fa_allgather.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/run_stateful_ops_ab.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_conv_dh0_bounds_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_cp_gdn_equivalence_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_fa_transport_matrix_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_full_qwen35_tree_cp_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_native_ring_tnd_layout.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_ordinary_ring_adapter.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_ordinary_ring_schedule_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_pure_gdn_tree_cp_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_qwen35_prefix_reuse_perf_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_qwen35_trajectory_transport_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_small_hybrid_tree_cp_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_stateful_gdn_cp_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/parallel/test_trajectory_fa_trace_contract.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/test_gdn_push_branch_pop_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/test_linear_prefix_state_forward_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/test_qwen35_first_layer_shape_probe_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/test_qwen35_hybrid_push_branch_pop_npu.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/__init__.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_fa_fp32_reference.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_first_gdn_zigzag_control.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_first_layer_projection_control.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_full_hybrid_drift_metadata.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_gdn_branch_executor.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_gdn_context.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_gdn_cp_state_shapes.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_gdn_prefix_state.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_hybrid_segment_executor.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_prefix_reuse_metrics.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_propagation_closure.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_ring_merge_probe.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_stage45_sync_trace.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_stateful_ops_probe.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `experimental/linear/unit/test_whole_fa_allgather.py` | EXPERIMENTAL | GDN / Qwen3.5 Hybrid-Linear 及其测试工具，单独保留，不参与 Dense Full-Attention 默认门禁 |
| `integration/run_uniagent_20step_three_experiments.sh` | REVIEW-DELETE | 组件边界、测试工具或历史断言 |
| `integration/run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `integration/test_real_actor_update_minibatch_contract.py` | KEEP-P1 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `integration/test_real_tq_phase4_training_keys.py` | REVIEW-DELETE | 真实轨迹 Capture/Manifest 处理 |
| `integration/test_real_tq_trajectory_tree.py` | REVIEW-DELETE | Segment/Tree/Prefix/KV 生命周期 |
| `integration/test_real_tq_tree_plan_builder.py` | REVIEW-DELETE | Segment/Tree/Prefix/KV 生命周期 |
| `integration/test_tpr_forest_ppo_objective_cpu.py` | KEEP-P1 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/ORDINARY_RING.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `parallel/README.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `parallel/__init__.py` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `parallel/_qwen3_tp_checkpoint.py` | KEEP-P0 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `parallel/_ring_block_probe.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/_tpr_cp_test_utils.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/analyze_cp1_attention_attribution.py` | REVIEW-DELETE | Attention/算子数值和 GEMM 精度 |
| `parallel/analyze_real_actor_ppo_adjoint.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/analyze_tpr_dp2_gradient_parity.py` | REVIEW-DELETE | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `parallel/audit_all_actor_success_lengths.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/audit_saved_cp_grad_replicas_cpu.py` | REVIEW-DELETE | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/audit_saved_ppo_loss_cpu.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/compare_cp4_split_no_share_cpu.py` | REVIEW-DELETE | 组件边界、测试工具或历史断言 |
| `parallel/compare_native_cp_multistep.py` | REVIEW-DELETE | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/compare_old_new_tpr_tp2cp4_ppo.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/compare_real_actor_cp1_short_full_tpr.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/compare_real_actor_tp2cp4_multistep.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/compare_real_actor_tp2cp4_offline_ppo.py` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/compare_real_tq_multisibling_prefix.py` | REVIEW-DELETE | Segment/Tree/Prefix/KV 生命周期 |
| `parallel/compare_real_tq_tpr_full_multistep.py` | REVIEW-DELETE | Native/Full 多步基线及漂移对照 |
| `parallel/compare_real_tq_tpr_p512s512.py` | REVIEW-DELETE | 真实轨迹 Capture/Manifest 处理 |
| `parallel/compare_tpr_30k_three_way.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `parallel/compare_tpr_dp2_recompute.py` | KEEP-P0 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `parallel/compare_tpr_recompute_matrix.py` | REVIEW-DELETE | 重计算/Checkpoint 显存与梯度正确性边界 |
| `parallel/compare_tpr_recompute_off_on.py` | KEEP-P0 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `parallel/compare_tpr_tp2cp2dp2_one_step.py` | KEEP-P0 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `parallel/cp1_attention_attribution.py` | KEEP-P1 | Attention/算子数值和 GEMM 精度 |
| `parallel/grad_sync_precision.py` | KEEP-P0 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `parallel/gradient_witness.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `parallel/inspect_real_actor_turn_tree.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/inspect_real_tq_siblings.py` | REVIEW-DELETE | 真实轨迹 Capture/Manifest 处理 |
| `parallel/multistep_artifact_recovery.py` | KEEP-P1 | Attention/算子数值和 GEMM 精度 |
| `parallel/native_full_megatron_schedule.py` | KEEP-P0 | Native/Full 多步基线及漂移对照 |
| `parallel/offline_session_selection.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `parallel/prepare_real_actor_20step_manifest.py` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/prepare_tpr_30k_one_step.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `parallel/prepare_tpr_cp1_short_capture.py` | REVIEW-DELETE | 真实轨迹 Capture/Manifest 处理 |
| `parallel/probe_full_45k.py` | REVIEW-DELETE | Native/Full 多步基线及漂移对照 |
| `parallel/run_cp1_attention_attribution.sh` | REVIEW-DELETE | Attention/算子数值和 GEMM 精度 |
| `parallel/run_existing_20step_offline_full_tpr.sh` | REVIEW-DELETE | Native/Full 多步基线及漂移对照 |
| `parallel/run_full_native_ring_step1_length_probe.sh` | REVIEW-DELETE | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/run_native_cp_drift.sh` | REVIEW-DELETE | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/run_native_cp_multistep.sh` | REVIEW-DELETE | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/run_old_new_tpr_tp2cp2dp2_ppo.sh` | REVIEW-DELETE | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `parallel/run_old_new_tpr_tp2cp4_ppo.sh` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/run_real_actor_cp1_short_full_tpr.sh` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/run_real_actor_tp2cp4_multistep.sh` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/run_real_actor_tp2cp4_offline_ppo.sh` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/run_real_tq_multisibling_prefix.sh` | REVIEW-DELETE | Segment/Tree/Prefix/KV 生命周期 |
| `parallel/run_real_tq_tpr_full_multistep.sh` | REVIEW-DELETE | Native/Full 多步基线及漂移对照 |
| `parallel/run_real_tq_tpr_p512s512_triplet.sh` | REVIEW-DELETE | 真实轨迹 Capture/Manifest 处理 |
| `parallel/run_tpr_dp2_recompute_validation.sh` | KEEP-P0 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `parallel/run_tpr_dp2_vs_full_30k_one_step.sh` | REVIEW-DELETE | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `parallel/run_tpr_fixed_shape_40k_benchmark.sh` | KEEP-P0 | 一次性定位和探针辅助 |
| `parallel/run_tpr_tp2cp2dp2_one_step.sh` | KEEP-P0 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `parallel/run_tpr_tp2cp4dp1_30k_one_step.sh` | REVIEW-DELETE | 组件边界、测试工具或历史断言 |
| `parallel/run_verl_engine_full_replay.sh` | REVIEW-DELETE | Native/Full 多步基线及漂移对照 |
| `parallel/test_allgather_cp_attention_contract.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_allgather_cp_attention_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_audit_saved_ppo_loss_cpu.py` | KEEP-P1 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/test_cp1_attention_attribution_cpu.py` | KEEP-P1 | Attention/算子数值和 GEMM 精度 |
| `parallel/test_cp1_complete_gradients_cpu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `parallel/test_cp1_short_capture_cpu.py` | KEEP-P1 | 真实轨迹 Capture/Manifest 处理 |
| `parallel/test_cp4_multistep_replica_audit_cpu.py` | KEEP-P1 | Native/Full 多步基线及漂移对照 |
| `parallel/test_cp4_three_arm_comparison_cpu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `parallel/test_cp_backend_contract.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_cp_padding_contract.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_cp_segment_execution_contract.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_engine_cp_runtime_contract.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_hccl_bf16_replica_probe_npu.py` | KEEP-P1 | 一次性定位和探针辅助 |
| `parallel/test_hybrid_cp_attention_contract.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_hybrid_cp_attention_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_megatron_engine_tpr_cp_entry_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_multistep_artifact_recovery_cpu.py` | KEEP-P1 | Attention/算子数值和 GEMM 精度 |
| `parallel/test_native_cp1_cp2_cp4_drift_npu.py` | REVIEW-DELETE | Native/Full 多步基线及漂移对照 |
| `parallel/test_native_cp_multistep_npu.py` | REVIEW-DELETE | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_native_full_dynamic_microbatches_cpu.py` | KEEP-P1 | Native/Full 多步基线及漂移对照 |
| `parallel/test_native_full_schedule_cpu.py` | KEEP-P1 | Native/Full 多步基线及漂移对照 |
| `parallel/test_native_ring_output_oracle_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_real_actor_cp1_short_full_tpr_npu.py` | KEEP-P1 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/test_real_actor_tp2cp4_multistep_npu.py` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/test_real_actor_tp2cp4_offline_ppo_npu.py` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `parallel/test_real_tq_multisibling_prefix_npu.py` | REVIEW-DELETE | Segment/Tree/Prefix/KV 生命周期 |
| `parallel/test_real_tq_tpr_full_multistep_npu.py` | REVIEW-DELETE | Native/Full 多步基线及漂移对照 |
| `parallel/test_real_tq_tpr_p512s512_triplet_npu.py` | REVIEW-DELETE | 真实轨迹 Capture/Manifest 处理 |
| `parallel/test_ring_coalescing_tree_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_ring_cp_attention_contract.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_ring_cp_attention_npu.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_ring_numerics_cpu.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_tpr_cp_equivalence_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_tpr_cp_hybrid_equivalence_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_tpr_cp_segment_push_pop_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_tpr_fixed_shape_40k_benchmark_npu.py` | KEEP-P0 | 一次性定位和探针辅助 |
| `parallel/test_tpr_qwen3_cp_equivalence_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_tpr_tp2_npu.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `parallel/test_tpr_tp2_sp_npu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `parallel/test_tpr_tp2cp2dp2_one_step_npu.py` | KEEP-P0 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `parallel/test_ulysses_cp_attention_contract.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_ulysses_cp_attention_npu.py` | KEEP-P1 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `parallel/test_verl_engine_full_replay_npu.py` | KEEP-P1 | Native/Full 多步基线及漂移对照 |
| `parallel/tpr_dp2_loss_consistency.py` | KEEP-P0 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `parallel/tpr_dp2_recompute_update.py` | KEEP-P0 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `profiling/ACTIVATION_OFFLOAD_PHASE_A.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `profiling/ACTIVATION_OFFLOAD_PHASE_B.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `profiling/NATIVE_RECOMPUTE_AUDIT.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `profiling/QWEN17_NUMERICS_AND_GEMM.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `profiling/README.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `profiling/RING_PREFIX_COALESCING.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `profiling/RING_STREAMING_KV.md` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `profiling/__init__.py` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `profiling/_offload_b_gate.py` | KEEP-P1 | Offload/显存生命周期专项 |
| `profiling/_qwen3_profile_target.py` | KEEP-P1 | 真实模型兼容性、参数和数值/性能对照 |
| `profiling/collect_activation_offload_phase_b.py` | KEEP-P1 | Offload/显存生命周期专项 |
| `profiling/run_ref_vs_tpr_offload.sh` | KEEP-P1 | Offload/显存生命周期专项 |
| `profiling/summarize_activation_offload_phase_b.py` | KEEP-P1 | Offload/显存生命周期专项 |
| `profiling/summarize_ref_vs_tpr_offload.py` | KEEP-P1 | Offload/显存生命周期专项 |
| `profiling/test_activation_offload_phase_a_npu.py` | REVIEW-DELETE | Offload/显存生命周期专项 |
| `profiling/test_activation_offload_phase_b_npu.py` | REVIEW-DELETE | Offload/显存生命周期专项 |
| `profiling/test_qwen17_dense_fp32_shared_gmm_backward_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `profiling/test_qwen17_fp32_main_grad_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `profiling/test_qwen17_mindspeed_shared_gmm_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `profiling/test_qwen3_1_7b_gemm_tile_bench_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `profiling/test_qwen3_1_7b_grouped_autograd_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `profiling/test_qwen3_1_7b_mindspeed_gmm_real_npu.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `profiling/test_tpr_engine_profile_npu.py` | REVIEW-DELETE | 性能基准与 profiling |
| `profiling/test_tpr_gpt_long_context_smoke_npu.py` | REVIEW-DELETE | 组件边界、测试工具或历史断言 |
| `profiling/test_tpr_memory_lifecycle_audit_npu.py` | REVIEW-DELETE | Offload/显存生命周期专项 |
| `profiling/test_tpr_qwen3_capacity_npu.py` | REVIEW-DELETE | 真实模型兼容性、参数和数值/性能对照 |
| `profiling/test_tpr_qwen3_profile_npu.py` | REVIEW-DELETE | 真实模型兼容性、参数和数值/性能对照 |
| `profiling/test_tpr_qwen3_ring_cp_profile_npu.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `run_correctness.sh` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `run_cpu_regressions.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `run_performance.sh` | KEEP-P0 | 性能基准与 profiling |
| `run_recompute.sh` | KEEP-P0 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `run_suite.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `run_unit.sh` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `suite_catalog.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/__init__.py` | KEEP-DOC | 组件边界、测试工具或历史断言 |
| `unit/test_30k_capture_derivation.py` | KEEP-P1 | 真实轨迹 Capture/Manifest 处理 |
| `unit/test_activation_offload_adapter.py` | KEEP-P1 | Offload/显存生命周期专项 |
| `unit/test_actor_20step_manifest.py` | KEEP-P1 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `unit/test_actor_capture_multistep.py` | KEEP-P1 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `unit/test_actor_capture_patcher.py` | KEEP-P1 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `unit/test_context.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_control_plane_gloo_cpu.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_dp2_gradient_parity_report.py` | KEEP-P1 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `unit/test_dp2_hccl_dtype_gate.py` | KEEP-P1 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `unit/test_dp_placement.py` | KEEP-P0 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `unit/test_engine_adapter.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `unit/test_fixed_topology_scheduler.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_kv_stack.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_module_spec.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `unit/test_objective_adapter.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `unit/test_offload_b_gate.py` | KEEP-P1 | Offload/显存生命周期专项 |
| `unit/test_ppo_batch_cpu.py` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `unit/test_ppo_executor_cpu.py` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `unit/test_ppo_objective_cpu.py` | KEEP-P0 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `unit/test_prefix_state.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_production_architecture.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_qwen17_areal_exact_reference.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `unit/test_qwen17_areal_native_backward.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `unit/test_qwen17_areal_training_forward.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `unit/test_qwen17_dta_style_reference.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `unit/test_qwen17_dta_tpr_triplet.py` | REVIEW-DELETE | Qwen3-1.7B / AReaL-DTA 早期数值参考 |
| `unit/test_qwen17_ppo_acceptance.py` | REVIEW-DELETE | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `unit/test_real_actor_ppo_adjoint_cpu.py` | KEEP-P1 | 真实 Actor PPO/capture、固定标签与 Loss/梯度一致性 |
| `unit/test_rectangular_attention_contract.py` | KEEP-P0 | Attention/算子数值和 GEMM 精度 |
| `unit/test_replay_bootstrap_cpu.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_ring_coalescing_schedule.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `unit/test_ring_streaming_transport.py` | KEEP-P0 | CP Ring/Ulysses/AllGather 通信、形状及梯度 |
| `unit/test_rope.py` | KEEP-P0 | Attention/算子数值和 GEMM 精度 |
| `unit/test_segment_executor.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_segment_external_objective.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_segment_plan.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_sequence_shard_cpu.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_shard.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_suite_registry_cpu.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_tp2_cp4_gate.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `unit/test_tp_validation.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_tpr_30k_three_way_comparison.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `unit/test_tpr_batch_runner.py` | KEEP-P0 | 组件边界、测试工具或历史断言 |
| `unit/test_tpr_dp2_gradient_parity_breakdown.py` | KEEP-P1 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `unit/test_tpr_dp2_loss_consistency.py` | KEEP-P1 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `unit/test_tpr_dp2_recompute_compare_cpu.py` | KEEP-P1 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `unit/test_tpr_full_recompute_cpu.py` | KEEP-P0 | 重计算/Checkpoint 显存与梯度正确性边界 |
| `unit/test_tpr_offline_grad_sync_precision.py` | KEEP-P1 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `unit/test_tpr_phase4_patcher.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `unit/test_tpr_production_dp2_gate_cpu.py` | KEEP-P0 | TP/CP/DP2 分片、梯度规约和副本一致性 |
| `unit/test_tpr_self_attention.py` | KEEP-P0 | Attention/算子数值和 GEMM 精度 |
| `unit/test_tpr_zero_objective_cpu.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `unit/test_trajectory_tree.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_trajectory_tree_turn.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_tree_plan_builder.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_turn_tree_cpu.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |
| `unit/test_uniagent_cc_tpr_patcher.py` | KEEP-P1 | 组件边界、测试工具或历史断言 |
| `unit/test_unified_kv_state_cpu.py` | KEEP-P0 | Segment/Tree/Prefix/KV 生命周期 |

## 下一批删除前必须核对

1. 禁止仅凭未出现在 `suite_catalog.py` 就删除：不少 NPU 对照脚本被 shell wrapper 间接调用。
2. 查找 `from .xxx`、`from ..xxx`、`python -m ...`、pytest 测试路径和数据准备工具调用；先迁移使用者再删。
3. 一次性数值/性能诊断先把独立参考条件与关键指标写回实验报告，再去掉实现。
4. 固定旧捕获字段不能借助删测试偷换真实 PPO 协议；保留 30K/40K 实测和 Recompute OFF/ON 检查。
5. 清理后必须运行 CPU 与至少一次真实 NPU 主线门禁，不得把静态路径核对写成运行 PASS。
