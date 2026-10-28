# Real-model correctness

Active tests:

- `test_tpr_qwen3_compatibility_npu.py`: real Qwen3 checkpoint numerical parity; auxiliary fixtures live in `equivalence/_engine_reference_fixture.py` and `profiling/_engine_profile_fixture.py`.
- `test_tpr_kv_checkpoint_replay_npu.py`: checkpoint replay with explicit KV roots, positive primitive.
- `test_tpr_native_recompute_contract_npu.py`: negative control for unadapted checkpointing (an expected failure, not a green gate).

Previous Qwen3-1.7B/AReaL-DTA comparative research and synthetic twenty-step experiments were retired from the maintained test tree; earlier versions are available in Git history. Use `parallel/` for current real Actor PPO, DP2 and Ring tests.
