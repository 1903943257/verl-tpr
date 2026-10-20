# TPR parallel tests and artifact tools

This folder is **not one pytest suite**. It combines primary CP/TP/DP
correctness, native baselines, offline actor capture comparators, launchers,
and research diagnostics. Start with [the TPR test matrix](../TEST_MATRIX.md) and the maintained
`../run_correctness.sh` / `../run_recompute.sh` / `../run_performance.sh`
entry scripts (internally using [the suite launcher](../run_suite.py)). Do not run
`pytest parallel/` as a single acceptance gate.

| Prefix | Role | Execution |
| --- | --- | --- |
| `test_*_contract.py`, `test_*_cpu.py` | CPU layout/ownership/gradient contracts | `python -m pytest` or isolated CPU selection |
| `test_*_npu.py` | NPU numerical and distributed probes | Exact `torchrun` topology; many require opt-in env |
| `run_*.sh` | Topology, data and artifact-aware launchers | `bash` from repository root with explicit inputs |
| `compare_*.py` | Post-process saved rank artifacts | Direct Python script, **not** pytest |
| `prepare_*.py`, `inspect_*.py` | Build/check captures; no training | Direct Python, preserve original inputs |
| `analyze_*.py`, `audit_*.py`, `probe_*.py` | Diagnostics; often historical | Not production acceptance |
| `_*probe.py`, `_*control.py`, `_*reference.py` | Imported implementation/reference helpers | Do not execute as independent tests |

## Active correctness dependencies

- Ring numerical contract: `test_ring_cp_attention_contract.py` (CPU),
  `test_ring_cp_attention_npu.py` (NPU 2/4 ranks).
- TP: `test_tpr_tp2_npu.py` (TP2); SP remains a separate acceptance scope.
- DP2: `test_tpr_tp2cp2dp2_one_step_npu.py`; offline wrapper
  `run_tpr_dp2_recompute_validation.sh`.
- Fixed actor PPO: `run_real_actor_tp2cp4_offline_ppo.sh`, with corresponding
  `compare_real_actor_tp2cp4_offline_ppo.py`.
- Recompute OFF/ON: `compare_tpr_recompute_off_on.py` and
  `compare_tpr_recompute_matrix.py`; the latter imports
  `compare_real_actor_tp2cp4_offline_ppo.py` by sibling module name.

**Import ownership:** these CLI tools are not independent modules. In
particular `compare_tpr_recompute_matrix.py` uses a bare sibling import.
Moving only one side is a breaking change. First group by behavior and make
the selected commands explicit, then move *whole dependency clusters* only
when all shell and Python import sites are audited.

## Result naming

Use a new artifact directory per invocation. Never overwrite source captures.
Record: source capture SHA, model checkpoint and precision, recompute OFF/ON,
TP/CP/DP, Ring backend, logical PPO tokens, all-rank loss/gradients and
peak allocated/reserved. Comparisons involving an unshared SegmentExecutor
baseline must not be relabelled native Megatron Full.

The 2026-10-20 pruning batch retired native GEMM/QKV replay diagnostic
pairs and six-FA value-clamp tests; they were not production correctness
entrances. See `../PRUNE_LOG.md` for restore paths.
