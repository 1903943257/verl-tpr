# Parallel, Ring CP and real Actor PPO tests

The mainline keeps Ring Attention numerical contracts, CP padding/shard tests, native TP2 integration, Qwen3 Ring CP equivalence, real Actor PPO and DP2/recompute gates.

Key real 8-NPU launchers:

- `run_real_actor_tp2cp4_offline_ppo.sh` — full-vs-TPR fixed recorded Actor PPO.
- `run_real_actor_tp2cp4_multistep.sh` — independent fixed-capture multi-step replay.
- `run_verl_engine_full_replay.sh` — real native VERL baseline.
- `run_tpr_tp2cp2dp2_one_step.sh` and `run_tpr_dp2_recompute_validation.sh` — Ring CP2/DP2 and recompute.
- `run_tpr_dp2_vs_full_30k_one_step.sh`, `run_tpr_tp2cp4dp1_30k_one_step.sh` — controlled 30K comparison.
- `run_tpr_fixed_shape_40k_benchmark.sh` — fixed-shape 40K F/B benchmark.

These shell scripts require a real capture and model checkpoint and will not create synthetic substitute data. General device gates use `../run_correctness.sh`. Do not treat passing CPU shape contracts as distributed gradient parity.
