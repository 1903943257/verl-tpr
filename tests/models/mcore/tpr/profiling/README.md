# TPR profiling

Active NPU profiling entry: `test_tpr_qwen3_ring_cp_profile_npu.py`, invoked by `../run_performance.sh ring`. The shared real-Qwen profile helpers live in `_qwen3_profile_target.py` and `_engine_profile_fixture.py`.

For reproducible real Actor performance, prefer `../parallel/run_tpr_fixed_shape_40k_benchmark.sh` or explicit 30K scripts rather than a standalone synthetic benchmark. Results must distinguish F/B-only timing from optimizer/online E2E.

Existing `*.md` files describing older Qwen3-1.7B GEMM, activation-offload and Ring probe results are historical research notes; their removed launchers are **not** maintained commands.
