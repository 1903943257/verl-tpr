# TPR correctness and reference tests

Do not interpret every test here as a mainline production gate. This folder
contains real Qwen3, historical HF/AReaL DTA and PPO comparisons, multi-step
experiments, plus the Recompute native-contract probes.

## Maintained selection

Start with `../TEST_MATRIX.md`, and run
`python tests/models/mcore/tpr/run_suite.py show npu-qwen` to inspect
the curated Qwen3 entry. Each device test still owns its environment variables,
checkpoint paths and test-data prerequisite. A skipped result is not a pass.

## Recompute: two different semantics

- `test_tpr_native_recompute_contract_npu.py`: intentionally negative
  controls showing *config-only* native checkpointing cannot expose
  internal KV roots or restore an external replay ContextVar.
  It is **not** a passing acceptance suite.
- `test_tpr_kv_checkpoint_replay_npu.py`: positive low-level contract
  for explicitly exported KV, fresh replay context and gradient roots.
  It does **not** validate the complete Qwen3/Ring optimizer update.

Real full-model OFF/ON recompute validation happens through the capture-aware
scripts under `parallel/`. See `docs/tpr/recompute_prototype.md`.

## Historical numerical controls

`_qwen17_*.py` helpers and their `test_qwen3_1_7b_*.py` drivers include
frozen HF/AReaL/DTA baseline comparisons. Preserve their numerical
interpretation; do not mix historical fixed-tile probes into live default
training, and never silently relax tolerance to claim regression success.

The `*_twenty_step*_npu.py` scripts use fixed data in their own protocols;
they must not be described as online GRPO Reward-convergence validation.

The early Qwen3-1.7B weak-E2E experiment and Fixed Tile diagnostic helpers
were deleted as a dependency island on 2026-10-20. Historical commits remain
available for reproduction; do not enable Fixed Tile as a correctness fix.
