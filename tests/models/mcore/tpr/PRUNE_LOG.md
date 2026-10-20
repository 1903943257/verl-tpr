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
