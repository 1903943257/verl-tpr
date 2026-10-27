# TPR tests — production acceptance

The test folder is intentionally limited to maintained **Dense Full Attention / Ring CP / real Qwen3 Actor PPO** contracts. Older Qwen3-1.7B/AReaL-DTA numerical investigations, Tiny GPT synthetic model suites, alternative CP research backends, and historical one-off diagnostic launchers were retired from this branch. Their previous implementations remain in git history.

After the final Dense-TPR-oriented prune, the test tree has **65 mainline `test_*.py` modules**; the 32 experimental GDN/Qwen3.5 test modules and their standalone research fixtures/docs were removed from this delivery branch. Their source is retained in git history (and outside this cleanup branch). Mainline tests remain in `unit/`, `equivalence/`, `parallel/`, `correctness/`, `integration/`, and `profiling/`. This fork-level test suite is **not** copied into the small upstream TPR recipe.

## Four supported entry points

Run from the repository root:

```bash
bash tests/models/mcore/tpr/run_unit.sh core
bash tests/models/mcore/tpr/run_correctness.sh ring
bash tests/models/mcore/tpr/run_recompute.sh cpu
bash tests/models/mcore/tpr/run_performance.sh ring
```

The underlying dispatcher `run_suite.py` maintains **10 suites / 11 gates**, checks local relative Python imports and literal shell targets, and reports mainline file counts (experimental is now zero):

```bash
python tests/models/mcore/tpr/run_suite.py check
python tests/models/mcore/tpr/run_suite.py inventory
python tests/models/mcore/tpr/run_suite.py list
```

The shared `_mindspeed_first_pytest.py` bootstrap imports MindSpeed before collecting Megatron tests, keeping pytest short flags away from MindSpeed's argument parser. For device suites, the dispatcher uses `torchrun --master_addr=localhost --master_port=${TPR_TEST_MASTER_PORT:-29541}` (no `--standalone`). The isolated CPU runner stubs unavailable Megatron interfaces. A green CPU run means the selected CPU contracts passed, **not** that every NPU gate has passed. Real NPU tests require model/capture assets and appropriate Ascend hardware. A fully skipped test is not accepted as a green gate.

## Maintained real-capture workflows

Use the launchers below only when the corresponding comparison is needed:

| File under `parallel/` | Purpose |
| --- | --- |
| `run_real_actor_tp2cp4_offline_ppo.sh` | Fixed real Actor PPO, TP2×RingCP4 |
| `run_real_actor_tp2cp4_multistep.sh` | Matched multi-step Full vs TPR replay |
| `run_verl_engine_full_replay.sh` | Real native VERL/Megatron Full baseline |
| `run_tpr_tp2cp2dp2_one_step.sh` | Real Actor PPO, TP2×RingCP2×DP2 |
| `run_tpr_dp2_recompute_validation.sh` | DP2 recompute OFF/ON and full-gradient verification |
| `run_tpr_dp2_vs_full_30k_one_step.sh` | 30K native Full/TPR comparison |
| `run_tpr_tp2cp4dp1_30k_one_step.sh` | Matched 30K CP4 TPR |
| `run_tpr_fixed_shape_40k_benchmark.sh` | Identical-shape 40K training-side performance |

The `integration/` scripts run Uni-Agent capture/online experiments; do not confuse these with standalone training-side F/B performance.

**No Fixed Tile workaround** is allowed. Numerical claims require the same input hash, checkpoint, PPO labels, precision flags, parallel topology, optimizer scope and recompute configuration; do not label forward/backward-only measurements as online end-to-end speedups.

## Second-pass consolidation (2026-10-20)

Eight small CPU test files were consolidated into seven existing test modules. Their **28 test functions remain**; the CPU entry list and CP4 offline preflight were updated to point to the retained files. No production implementation was modified. After syncing with `main@9493dfa2`, the core suite had **65** test files. A subsequent recipe-focused cleanup removed the separate 32-module experimental GDN/Linear test tree without changing core CPU/NPU test assertions. The original real-Actor/DP2/recompute/30K/40K/Full launchers remain, and the newer TreeBatch-specific launchers from `main` are preserved for those workflows. The two `integration/` launchers remain paired for online Native capture.

Current `main` includes opt-in CP1 TreeBatch leaf micro-batching. `TPR_LEAF_BATCH_SIZE=1` remains the default; full-model numerical parity for `K>1` is unresolved and must not be presented as production acceptance. See `parallel/LEAF_MICROBATCH.md` and `parallel/LEAF_MICROBATCH_STAGE_CLOSEOUT.md`.

These are **source-level changes, not CPU/NPU acceptance evidence**. Re-run `run_suite.py check`, `run_unit.sh core`, and the hardware-specific NPU gates on the new checkout before claiming numerical or performance validation.

## Cleanup policy

- Only keep stable numerical contracts, production integration, real Actor capture comparators and reproducible benchmarks in the core area.
- Keep reusable synthetic **tensor** micro-tests for Ring/Attention/Mask/DP logic; synthetic **model** end-to-end test suites were removed. Real Qwen3 fixtures live in non-`test_` modules.
- Avoid adding another launcher to reproduce an existing `run_*.sh`; add a mode to an existing maintained script instead.
- A subsequent parallel-runner prune removed 19 historical scripts/helpers/docs (extra CP2 ratio sweeps, per-case analysis, old Ring Level1 trace harness and small-K leaf diagnostics). Retained: CP1 N8 Full/TPR K sweeps, Ring merge validation, CP4 real Actor replay, TP2×CP2×DP2, Recompute OFF/ON and fixed40K benchmarks. No registered gate, core pytest module or its assertions were deleted.
- The `parallel` inventory is now 78 tracked files, including 25 pytest modules. Old exploratory runner recipes remain available in git history.
- Historical result descriptions are retained where useful under `profiling/*.md`, but references to the deleted old scripts there are historical, not executable instructions.
- Review [TEST_MATRIX.md](TEST_MATRIX.md) for active numerical acceptance boundaries.

The previous large inventory/audit was removed because it listed deleted files. Git history is the authoritative record of the deleted experiments.
