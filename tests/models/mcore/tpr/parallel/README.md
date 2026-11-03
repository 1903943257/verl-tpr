# Parallel, Ring CP and real Actor PPO tests

The mainline keeps Ring Attention numerical contracts, CP padding/shard tests, native TP2 integration, Qwen3 Ring CP equivalence, real Actor PPO and DP2/recompute gates.

Key real 8-NPU launchers:

- `run_real_actor_tp2cp4_offline_ppo.sh` — full-vs-TPR fixed recorded Actor PPO.
- `run_real_actor_tp2cp4_multistep.sh` — independent fixed-capture multi-step replay.
- `run_verl_engine_full_replay.sh` — real native VERL baseline.
- `run_tpr_tp2cp2dp2_one_step.sh` and `run_tpr_dp2_recompute_validation.sh` — Ring CP2/DP2 and recompute.
- `run_tpr_dp2_vs_full_30k_one_step.sh`, `run_tpr_tp2cp4dp1_30k_one_step.sh` — controlled 30K comparison.
- `run_tpr_fixed_shape_40k_benchmark.sh` — fixed-shape 40K F/B benchmark.
- `run_cp4_qkv_padding_10step_5way.sh` — five-arm CP4 numerical comparison with a Native Full repeatability baseline; see [the correctness closeout](CP4_RING_QKV_CORRECTNESS_20261103.md).

These shell scripts require pre-generated capture assets and a model checkpoint; they do not synthesize replacement data at launch. The five-way CP4 runner specifically consumes a pre-generated, non-aligned synthetic Actor-PPO capture for numerical replay; it is not an on-policy E2E run. General device gates use `../run_correctness.sh`. Do not treat passing CPU shape contracts as distributed gradient parity.

## Cleanup boundary

The maintained parallel tree keeps native Ring/CP, real Actor PPO, DP2, recompute, full-engine and 30K/40K comparisons. Auxiliary one-off ratio sweeps, CP4 Level1 trace launchers, and historical K1/K2 microbench wrappers were removed. Three still-required helper modules were retained because the CPU or NPU contract tests import them; their source remains in git history. Keep importing shared fixtures from existing retained modules, and route maintained smoke/performance workflows via `../run_correctness.sh`, `../run_recompute.sh`, and `../run_performance.sh`.


## CP4 Ring / QKV numerical closeout (2026-11-03)

The five-arm fixed-capture comparison and its accepted first-phase numerical boundary are documented in [CP4_RING_QKV_CORRECTNESS_20261103.md](CP4_RING_QKV_CORRECTNESS_20261103.md). In the historical run directory, the arm folder named none means **TPR-CP-Ring** (TPR SegmentExecutor with KV/Query coalescing disabled), not TPR disabled. The maintained launcher now uses explicit arm names: tpr_cp_ring, tpr_cp_ring_kv_merge, and tpr_cp_ring_qkv_merge.
