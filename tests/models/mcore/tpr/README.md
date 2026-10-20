# TPR tests — production acceptance

The test folder is intentionally limited to maintained **Dense Full Attention / Ring CP / real Qwen3 Actor PPO** contracts. Older Qwen3-1.7B/AReaL-DTA numerical investigations, Tiny GPT synthetic model suites, alternative CP research backends, and historical one-off diagnostic launchers were retired from this branch. Their previous implementations remain in git history.

As of 2026-10-20: **52 mainline `test_*.py` files + 32 optional `experimental/linear` test files**. The latter are GDN/Qwen3.5 research, never a production Dense FA gate. Mainline tests are grouped in `unit/`, `equivalence/`, `parallel/`, `correctness/`, `integration/`, and `profiling/`.

## Four supported entry points

Run from the repository root:

```bash
bash tests/models/mcore/tpr/run_unit.sh core
bash tests/models/mcore/tpr/run_correctness.sh ring
bash tests/models/mcore/tpr/run_recompute.sh cpu
bash tests/models/mcore/tpr/run_performance.sh ring
```

The underlying dispatcher `run_suite.py` maintains **10 suites / 11 gates**, checks local relative Python imports and literal shell targets, and reports `core` versus `experimental` file counts:

```bash
python tests/models/mcore/tpr/run_suite.py check
python tests/models/mcore/tpr/run_suite.py inventory
python tests/models/mcore/tpr/run_suite.py list
```

The isolated CPU runner stubs unavailable Megatron interfaces. A green CPU run means the selected CPU contracts passed, **not** that every NPU gate has passed. Real NPU tests require model/capture assets and appropriate Ascend hardware. A fully skipped test is not accepted as a green gate.

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

Eight small CPU test files were consolidated into seven existing test modules. Their **28 test functions remain**; the CPU entry list and CP4 offline preflight were updated to point to the retained files. No production implementation was modified. The core suite now has **52** test files; `experimental/linear` remains **32**. The eight `parallel/run_*.sh` launchers are retained because each owns a distinct maintained real-Actor/DP2/recompute/30K/40K/Full replay workflow; the two `integration/` launchers remain paired for online Native capture.

These are **source-level changes, not CPU/NPU acceptance evidence**. Re-run `run_suite.py check`, `run_unit.sh core`, and the hardware-specific NPU gates on the new checkout before claiming numerical or performance validation.

## Cleanup policy

- Only keep stable numerical contracts, production integration, real Actor capture comparators and reproducible benchmarks in the core area.
- Keep reusable synthetic **tensor** micro-tests for Ring/Attention/Mask/DP logic; synthetic **model** end-to-end test suites were removed. Real Qwen3 fixtures live in non-`test_` modules.
- Avoid adding another launcher to reproduce an existing `run_*.sh`; add a mode to an existing maintained script instead.
- Historical result descriptions are retained where useful under `profiling/*.md`, but references to the deleted old scripts there are historical, not executable instructions.
- Review [TEST_MATRIX.md](TEST_MATRIX.md) for active numerical acceptance boundaries.

The previous large inventory/audit was removed because it listed deleted files. Git history is the authoritative record of the deleted experiments.
