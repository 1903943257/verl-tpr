# TPR testing: one entry point, explicit acceptance boundaries

All commands are relative to the **verl-tpr repository root**. Do not copy tests
from the obsolete `bridge` checkout into this tree. Production runtime code
lives in `verl/workers/engine/megatron/tpr/` and
`verl/models/mcore/tpr/`; the tests also include frozen historical and
experimental references.

## Four maintained commands

These are the **only recommended user-facing entry scripts**. The existing
smaller pytest modules and historical comparison utilities are their
implementation details; more of those will be retired after cross-backend
numerical acceptance.

```bash
bash tests/models/mcore/tpr/run_unit.sh core
bash tests/models/mcore/tpr/run_correctness.sh cp1
bash tests/models/mcore/tpr/run_recompute.sh cpu
bash tests/models/mcore/tpr/run_performance.sh ring
```

NPU modes require a populated model and capture directory in the NPU-enabled
VERL runtime (for example the configured Docker mount). On a plain host these
are **commands to inspect**, not an assertion that Ascend libraries are installed.
Use `run_unit.sh all` only when the full Python environment includes Megatron.
Use `run_recompute.sh actor-cp4 CAPTURE.pt NEW_OUTPUT_ROOT` or
`run_recompute.sh dp2 MANIFEST NEW_OUTPUT_ROOT [STEP]` for real OFF/ON.
Use `run_performance.sh fixed40k MANIFEST NEW_OUTPUT_ROOT` for measured
forward/backward. The performance case deliberately does not claim an
optimizer-step or online E2E speedup.

The catalog `run_suite.py` remains the **internal** dispatcher and can
display the precise gated commands. It rejects all-skipped selected tests
rather than treating their exit code as an acceptance result.

The 2026-10-20 cleanup removes 18 dated weak-E2E / Fixed Tile / clamp /
Native GEMM-QKV diagnostic files. See [PRUNE_LOG.md](PRUNE_LOG.md).
This is the first deletion batch, not the final 30–50-file target.

## Start here

```bash
python tests/models/mcore/tpr/run_suite.py list
python tests/models/mcore/tpr/run_suite.py inventory
python tests/models/mcore/tpr/run_suite.py check
python tests/models/mcore/tpr/run_suite.py show cpu-core
python tests/models/mcore/tpr/run_suite.py run cpu-core
```

The registry (`suite_catalog.py`) is the **curated gate shortlist**, not a claim
that all files under this directory are maintained or passing. `check` is
dependency-free and validates the selected file references; it does not run
numerical tests. `inventory` enumerates physical pytest files and identifies
the scope of remaining consolidation.

For real NPU gates, first inspect the command, then explicitly opt in:

```bash
python tests/models/mcore/tpr/run_suite.py show npu-ring
python tests/models/mcore/tpr/run_suite.py run npu-ring --ack-npu
```

Set `ASCEND_RT_VISIBLE_DEVICES` and confirm the installed MindSpeed/Megatron
version and fixture topology **before** running a distributed gate. The small
suite launcher uses fixed, local rendezvous and invokes one curated pytest file
at a time; it is not a replacement for dedicated 8-rank real-actor wrappers.

**Important:** an exit code of zero with all tests skipped is *not* acceptance.
Inspect the pytest summary, skip reasons, every rank's output, numerical
thresholds, gradient witnesses, and recorded model/data hashes. No native NPU
tests were executed merely by reorganizing this directory.

## Physical directories and ownership

| Directory | What belongs there | Qualification |
| --- | --- | --- |
| `unit/` | CPU shape, graph, scheduler, PPO, DP, engine contracts | Some require actual Megatron; isolated runner covers only its explicit whitelist |
| `equivalence/` | Narrow NPU operator/model numerical equivalence | Device gate, not complete PPO |
| `correctness/` | Real-model, PPO, multi-step and historical DTA comparisons | Heavy and often asset-gated |
| `integration/` | Uni-Agent actor integration and real-TQ preparation contracts | Requires external capture or application environment |
| `parallel/` | CP/Ring/TP/DP parity **plus legacy launchers and analysis scripts** | Mixed historical folder; do not run blindly |
| `profiling/` | Latency, memory, capacity, offload and backend probes | Not correctness acceptance |
| `linear/` | Experimental GDN/Hybrid-linear model coverage | Outside production Dense FA gate |

Do not move a helper in `parallel/` simply because its name looks like a
diagnostic: many comparison tools import sibling files by filename and shell
launchers embed fixed repository-relative paths. Consolidation must preserve
these dependencies or update all of their callers in the same change.

## Gate priority

Read [TEST_MATRIX.md](TEST_MATRIX.md) for explicit feature-to-test mapping,
configuration, negative controls, stage ownership, and the separate 8-rank
offline PPO procedures.

- **P0 CPU:** `cpu-core` on every TPR runtime refactor; plain
  `run_cpu_regressions.py` remains the historical, stable entry.
- **P1 NPU primitive:** `npu-core` for rectangular Attention and Prefix
  backward; `npu-ring` and `npu-tp` for parallel changes.
- **P2 model and real PPO:** `npu-qwen`, followed by the dataset-specific
  scripts linked in TEST_MATRIX, with identical capture/checkpoint/optimizer.
- **P3 recompute:** `cpu-recompute`, `npu-recompute-primitive`, then
  CP1/CP2/TP2xCP4 whole-model OFF/ON and full-gradient comparisons.
- **P4 performance:** only after P0–P3, measured without diagnostic probes.

The `npu-recompute-negative` suite is deliberately expected to fail: it
demonstrates why unadapted native full checkpointing cannot export TPR KV roots.
It must **never** be counted as a passing production acceptance case. The
positive KV-replay primitive alone also does not establish Qwen3/Ring PPO
correctness.

## Test maintenance policy

1. Newly added production coverage must identify one mainline gate and provide
   an explicit pass condition. Keep negative controls and profiler probes out of
   passing acceptance suites.
2. Prefer an existing numerical reference and a same-input OFF/ON comparison
   over another bespoke experiment driver. Reuse the original VERL PPO loss.
3. Preserve historical tests until a replacement covers the same contract and
   independent evidence exists. A file move must update every direct import and
   shell path, not silently strand the old launcher.
4. Every performance result must record model, NPU, TP/CP/DP, data/hash, kernel
   flags, warmups/repeats, recompute state, peak allocated/reserved and measured
   scope (operator / forward+backward / optimizer step / online E2E).
5. Never turn on Fixed Tile as a test workaround. Do not loosen tolerances just
   to absorb backend gradient noise; report the independent baseline noise.

Legacy host/docker instructions previously embedded in this README are
historical only; current commands should use the active `verl-tpr` checkout.
See `docs/tpr/architecture.md` and `docs/tpr/recompute_prototype.md` for
runtime ownership and the recompute feature gates.
