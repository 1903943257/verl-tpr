# TPR Compact Runtime Inventory (2026-10-15)

Counts are physical Python lines, including comments and blanks, using
`str.splitlines()`. Baseline: `fa708bea` on
`refactor/tpr-ppo-engine-ring-turn-20261015`. No test files are included.

## Production model and engine

| File | Baseline | Current |
| --- | ---: | ---: |
| `models/mcore/tpr/__init__.py` | 101 | 101 |
| `models/mcore/tpr/attention.py` | 329 | 298 |
| `models/mcore/tpr/context.py` | 226 | 226 |
| `models/mcore/tpr/rectangular_attention.py` | 209 | 209 |
| `workers/engine/megatron/tpr/__init__.py` | 1 | 1 |
| `workers/engine/megatron/tpr/adapter.py` | 612 | 547 |
| `workers/engine/megatron/tpr/executor.py` | 865 | 600 |
| `workers/engine/megatron/tpr/objective.py` | 373 | 358 |
| `workers/engine/megatron/tpr/placement.py` | 327 | 327 |
| `workers/engine/megatron/tpr/ring.py` | 1731 | 1470 |
| `workers/engine/megatron/tpr/state.py` | 628 | 317 |
| `workers/engine/megatron/tpr/tree.py` | 914 | 383 |
| **Production total** | **6316** | **4837** |

Paths are relative to `verl/`. Production shrinks by **1,479 lines (23.4%)**,
not just whitespace changes. The 327-line placement module is an offline
planner; it is included in both totals, not hidden by relocation.

## Design reductions

- **One topology:** temporary Turn trie directly creates SegmentPlan and PPO
  refs. Delete persistent TrajectoryTree/TrajectoryNode/SegmentRef copies.
  Preserve complete-message sharing, internal terminals, duplicate-row weights,
  and parent-query ownership of the first child token.
- **One traversal:** SegmentExecutor.run performs iterative DFS. Delete the
  separate scheduler, event plan validation, state enum, physical-execution
  trace, and forward/backward/tree result wrappers. Collect detached losses
  in the same postorder.
- **One KV relay:** one ordered path dictionary and per-segment leaf anchors.
  CP1 concatenates these leaves through autograd; Ring consumes them directly.
  Delete the concat-detach/manual-slice gradient path and generic production
  PrefixState/PrefixStateStack hierarchy. Keep compact storage, generation
  guards, gradient accumulation and release-before-recompute.
- **One shard representation:** SequenceShard holds logical length, physical
  ranges, CP coordinates and padded length. Delete the production Protocol,
  contiguous/range subclass hierarchy and padded wrapper.
- **One forest synchronization:** a function replaces the single-use batch
  runner and result object. Native no_sync surrounds the whole forest; native
  finalize runs once, never after a failed tree.
- **One Ring backend:** experimental resolution now reuses production
  RingCPBackend rather than maintaining another implementation.

PPO normalization, CP compensation, empty-rank connected zeros, gradient
scaling, native metric SUM/MEAN handling, qkvmerge and physical/logical padding
remain part of the runtime. The native VERL engine entry and non-TPR model
construction paths are unchanged.

## API boundary and historical references

Production integrations import engine types directly from
`verl.workers.engine.megatron.tpr` submodules. Production SegmentPlan/Spec no
longer contain CE terms, total_loss_weight, topology_only or Push/Pop events.
Use `executor.run()` to obtain the additive scalar loss; there is no production
FixedTopologyScheduler or TreeScheduleResult.

`build_tree_execution_plans` retains Turn-only defaults and rejects other
policies; its old `trees=` injection is removed. TreeExecutionPlan directly
owns uid/tree_index/member_rows/segment_plan/objective_refs; `tree` is only a
metadata alias to self. TP agreement hashes use an explicit v2 domain and
include physical order and logical PPO refs; do not mix runtime versions
across peers.

The model package's lazy historical plan/event exports resolve to
`experimental/tpr/legacy/plan.py`. Those CE plans must be paired with the
legacy executor, not the production PPO executor. Old CE tests and benchmarks
explicitly import their legacy runtime. Production never imports it.

The experimental subtree contains 29 Python files and **6,970 lines**, versus
26 files / 6,489 lines at baseline. It gains thin state/shard adapters and a
349-line frozen CE plan reference; it loses the duplicate Ring backend.
Thus the **combined production + experimental code also shrinks**:
12,805 -> 11,807 lines (**998 lines / 7.8%**). Relocated compatibility code is
not claimed as a physical deletion. Historical code remains as a numerical
reference until device replay acceptance, not an alternate production path.

## Comparison with the local AReaL DTA checkout

DTA's six Python files total 1,565 lines. Its
`dta_engine.py + token_trie.py + trie.py` total 1,196 lines; the roughly
corresponding TPR `tree.py + executor.py + state.py` now total **1,300**.
This is a scope comparison, not a claim of feature equivalence. TPR additionally
contains VERL/Megatron PPO integration, native attention adaptation and Ascend
Ring transport/FlashAttention/online-softmax/backward.

## Verification and remaining device gate

The selected unified CPU matrix passes **411 tests** (production contracts,
new differential tests, and explicit historical references). Syntax and
removed-production-symbol import auditing passes across 650 Python files.

CPU differential tests cover randomized Turn topology/PPO refs against the
frozen implementation, full-row versus branched parameter gradients, scaled
loss and empty-rank behavior, compact PPO fields/loss/logit gradients, KV
lifetime/stale anchors, shard selection/gradients, padding and Ring contracts.
Historical CE/reference tests retain their assertions with explicit legacy
imports; they do not stand in for production numerical tests.

The local runner bypasses unrelated Ray initialization and supplies minimal
Megatron interfaces for attention orchestration tests. It does not execute
real Megatron collectives or NPU kernels. Additional native loss/balancer
integration tests require dependencies missing locally (including Ray and
codetiming); no full-suite pass is claimed.

An additional experimental Hybrid/GDN run has two unresolved failures in
`unit/test_hybrid_segment_executor.py::test_hybrid_multilevel_relay_matches_connected_graph`:
`[True]` writes final GDN state despite disabled capture; `[False]` differs
by approximately 2.38e-7 in float32 loss summation against a 1e-12 tolerance.
The assertions are unchanged. No isolated baseline was run to establish
whether the latter is pre-existing or influenced by shared-state changes.
GDN is outside production Dense PPO; these failures must not be described as
passing coverage or silently removed.

With the full VERL dependencies installed, run the new production
differential cases directly (the local isolated bootstrap is not required):

```bash
python -m pytest -q \
  tests/models/mcore/tpr/unit/test_turn_tree_cpu.py \
  tests/models/mcore/tpr/unit/test_ppo_executor_cpu.py \
  tests/models/mcore/tpr/unit/test_ppo_objective_cpu.py \
  tests/models/mcore/tpr/unit/test_ppo_batch_cpu.py \
  tests/models/mcore/tpr/unit/test_unified_kv_state_cpu.py \
  tests/models/mcore/tpr/unit/test_sequence_shard_cpu.py
```

AST comparison against baseline confirms unchanged Ring circulation,
reduction, input normalization, FlashAttention forward/backward, softmax merge,
source scheduling, physical masks and custom autograd. Syntax parsing and
`git diff --check` are also checked.

Before deployment, use the same captured minibatch and checkpoint for old/new
TP1CP1, TP2CP2 and TP2CP4 replay. Compare logical logprobs, native PPO loss,
parameter gradients/optimizer update, and peak allocated/reserved memory.
The CP1 anchor graph intentionally changes, so CPU agreement alone is not
device certification. Do not enable Fixed Tile as part of this change.
