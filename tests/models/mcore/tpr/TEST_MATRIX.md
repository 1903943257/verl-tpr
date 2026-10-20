# Production TPR test matrix

Current topology focus: Ascend NPU, Qwen3, Ring CP, native Megatron TP, real Agent Actor PPO / GRPO capture, TP2×CP2×DP2, recompute and 30K/40K forward+backward performance.

This table describes **gates**, not a statement that every NPU test has already passed after pruning.

| Area | Retained reference | Acceptance scope |
| --- | --- | --- |
| CPU core | `run_unit.sh core` | Tree ownership, Prefix/KV lifecycle, PPO masks/logprob, Ring contracts, DP placement and deterministic report contracts |
| Rectangular Attention | `run_correctness.sh cp1` | Single-rank causal forward/backward, nonzero Prefix KV gradients and Push/Pop |
| Ring CP | `run_correctness.sh ring` | 2-rank Ring Attention and gradient parity |
| Native TP | `run_correctness.sh tp2` | Native Megatron TP2 model/gradient checks |
| Real Qwen compatibility | `run_correctness.sh qwen` | Model-level Qwen3 vs TPR numerical reference |
| CP4 real PPO | `run_correctness.sh actor-cp4` | Immutable real Actor minibatch, true PPO labels, full/split/shared mode (optional), full gradients |
| Multi-step optimizer | `parallel/run_real_actor_tp2cp4_multistep.sh` | Same weights and original capture, independent Full/TPR training trajectories |
| DP2 Actor | `parallel/run_tpr_tp2cp2dp2_one_step.sh` | TP2×RingCP2×DP2 8-rank PPO forward/backward/DP gradient finalization |
| Recompute | `run_recompute.sh actor-cp4` or `run_recompute.sh dp2` | OFF/ON matched capture, full gradients and optimizer delta where enabled |
| Real Full baseline | `parallel/run_verl_engine_full_replay.sh` | Independent native VERL/Megatron engine reference |
| 30K/40K perf | `run_performance.sh fixed40k` and explicit 30K scripts | Same model/data/topology, wall-clock F/B, allocated/reserved peak memory |

The CPU-only MindSpeed `''` parser regression remains in `unit/test_replay_bootstrap_cpu.py`, and the real offline runner shares the same sanitizer for TP2×CP4 and TP2×CP2×DP2.

`suite_catalog.py` has 10 suites/11 selected gates. Offline eight-rank Actor/recompute runners are **asset-gated manual experiments**, not part of this generic registry. `npu-recompute-negative` is an intentionally failing diagnostic: never claim it is a positive acceptance test.

## Evidence / merge policy

1. `run_suite.py check` must pass (registry paths, relative imports and literal shell target paths).
2. `run_unit.sh core` must execute and pass, no all-skipped acceptance.
3. Attention, Ring, native TP2 and real Actor DP2/CP4 numerical tests must pass on their explicitly selected hardware before release.
4. Performance results must identify model/checkpoint, input SHA, P:S/N, TP/CP/DP, recompute, precision flags, trial counts, memory definition and measurement scope.
5. Never activate the retired Fixed Tile or FC2 fixed-M interventions to artificially align backends.

**Historical status:** User's 2026-10-20 CPU gate (642 passed) belonged to the preceding 143-test checkout. The maintained core now contains **52** test modules after eight test files were merged without deleting their 28 test functions. After this cleanup, a fresh CPU and NPU verification is required. A prior 30K DP2 result is historical evidence, not a substitute for running the new checkout.
