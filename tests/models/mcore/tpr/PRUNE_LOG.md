# TPR first-pass physical test pruning (2026-10-20)

This file records **deletions on the test organization branch only**. No
production runtime changes, regression thresholds, Attention implementation or
PPO math are included. Each removed file can be recovered from the main
history with `git show <old-commit>:<path>`.

## 16 retired files (10 pytest files)

- 9 retired files from the Qwen3-1.7B weak-E2E experiment island:
  `correctness/_qwen17_weak_e2e_{token_capture,trace}.py`,
  `correctness/{analyze_qwen17_weak_e2e_samples.py,run_qwen17_real_tq_weak_e2e.sh,test_qwen3_1_7b_real_tq_weak_e2e_npu.py}`,
  and `unit/test_qwen17_weak_e2e_{analysis,sampling,token_capture,trace}.py`.
  These investigate an earlier Qwen3-1.7B weak-E2E path rather than the
  current Qwen3-4B production PPO acceptance.
- Fixed Tile CPU experiment:
  `unit/test_qwen17_fixed_tile_probe.py`. The earlier equal-M workaround
  must not be re-enabled; it changes the physical GEMM and was rejected as a
  correctness control.
- Six-FA clamp diagnostic interventions:
  `parallel/test_six_fa_clamp_npu.py` and
  `parallel/test_six_fa_backward_clamp_npu.py`.
  These replace intermediate values for diagnosis, not production training.
- P0 native GEMM and QKV replay diagnostic pairs:
  `parallel/test_native_gemm_m_replay_npu.py`,
  `parallel/run_native_gemm_m_replay.sh`,
  `parallel/test_native_attention_qkv_replay_npu.py`,
  `parallel/run_native_attention_qkv_replay.sh`.
  The original native CP drift and live Ring/TP/DP gates are retained.

**Retained dependency helpers:** `correctness/_qwen17_weak_e2e_sampling.py`
remains because the native CP drift probe imports `exact_sample_indices`.
`correctness/_qwen17_fixed_tile_probe.py` remains because a separate
historical Qwen3-1.7B PPO control imports it conditionally. Neither helper is
part of a maintained production acceptance suite. Do not silently remove
these imports or enable Fixed Tile in the production path.

## Acceptance requirements before merge

1. Run `run_unit.sh core` on the target checkout and compare results with
   current main; no all-skip results.
2. Check no surviving code references removed helpers with
   `git grep -n -E '_qwen17_weak_e2e_|_qwen17_fixed_tile_probe|test_six_fa_(backward_)?clamp_npu|test_native_(gemm_m|attention_qkv)_replay_npu'`
   (allow references in this log and historical docs only).
3. Run the maintained Attention, Ring and TP2 NPU gates on the target NPU
   environment before merging.
4. Retain legacy numerical artifacts outside source tree if they are needed
   to reproduce an older report.

This is deliberately a first batch. The long tail of one-off
`parallel/` launchers should be removed only after their data and import
dependencies are audited, not by indiscriminate filename globbing.


## 2026-10-20 第二批清理

- 移动（非删除）原有 58 个 Linear/GDN/Qwen3.5 Hybrid 文件至 `experimental/linear/`，并调整既存跨包引用，增加 3 个 package initializer。
- 删除 10K parity 闭环：`parallel/{tpr_10k_canonical_witness.py,compare_tpr_10k_parallel_parity.py,prepare_tpr_10k_parity_capture.py,run_tpr_10k_parallel_parity.sh,test_tpr_10k_parallel_parity_npu.py}` 与 `unit/test_tpr_10k_parallel_parity_cpu.py`（6 个文件）。
- 删除 `unit/test_tpr_offline_dp2_gate.py`：这是已与生产 DP2 路由分离的旧 legacy 路由门禁；现有生产 DP2 CPU 入口仍保留。
- 删除 `correctness/_qwen17_fixed_tile_probe.py`：由旧 Phase5 PPO 测试中移除了 Fixed Tile/FC2 fixed-M test-only 干预（112 行）；旧变量现在 fail closed。
- 这一批共删 **8 个文件**（其中 **3 个 `test_*.py`**）。
- 已验证 CPU isolated 选定测试路径仍全部存在；尚未声称新增清理后的 NPU 回归 PASS。
