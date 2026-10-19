# TPR profiling is not a correctness suite

This folder contains capacity probes, memory lifecycle studies, GEMM
diagnostics, Ring CP profiles and activation offload experiments.
Run these only with a fixed model, capture, topology, flags, number of
warmups/repetitions, and a comparable baseline.

A `profiling/test_*.py` file being named as a pytest does not mean its
output certifies forward/backward parity. In particular:
- `test_tpr_qwen3_ring_cp_profile_npu.py` requires the opt-in
  `TPR_RUN_QWEN_RING_CP_PROFILE=1`.
- `NATIVE_RECOMPUTE_AUDIT.md` is a historical analysis of the
  *config-only* native checkpoint contract, not a negative statement about
  the later KV-aware prototype.
- All-gather/native and custom Ring reference paths must be identified
  precisely in result tables.

See `../TEST_MATRIX.md` for the acceptance order. Numerical correctness
and full parameter gradients precede any claimed memory/speedup result.
