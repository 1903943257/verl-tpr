# TPR Refactor Source Inventory (2026-10-15)

This report enumerates the **actual current Git tree**, not proposed filenames.
Counts use Python files split on newline (including blank/comment lines), and
are not effective non-comment source LOC. Numbers reflect the current
refactor/tpr-ppo-engine-ring-turn-20261015 branch.

## Production model and engine

| File | Lines |
| --- | ---: |
| `verl/models/mcore/tpr/__init__.py` | 102 |
| `verl/models/mcore/tpr/attention.py` | 330 |
| `verl/models/mcore/tpr/context.py` | 227 |
| `verl/models/mcore/tpr/rectangular_attention.py` | 210 |
| `verl/workers/engine/megatron/tpr/__init__.py` | 2 |
| `verl/workers/engine/megatron/tpr/adapter.py` | 613 |
| `verl/workers/engine/megatron/tpr/executor.py` | 866 |
| `verl/workers/engine/megatron/tpr/objective.py` | 374 |
| `verl/workers/engine/megatron/tpr/placement.py` | 328 |
| `verl/workers/engine/megatron/tpr/ring.py` | 1732 |
| `verl/workers/engine/megatron/tpr/state.py` | 629 |
| `verl/workers/engine/megatron/tpr/tree.py` | 915 |
| **Model package subtotal** | **869** |
| **Engine package subtotal** | **5459** |
| **Production total** | **6328** |

The core Engine contains seven implementation files, plus its package
initializer. Model-side code contains three implementation files plus an
initializer. Ring CP, CP padding layout, and shared Prefix KV all have one
canonical implementation. Production PPO does **not** depend on the old
CE/Exact, AllGather, Hybrid, or Ulysses factories.

## Optional / historical code (not in production)

The `verl/experimental/tpr/` subtree contains 26 Python files and 6,515
lines: historical full-CE/Exact reference code, old multi-backend CP
factory, optional GDN, AllGather, Hybrid, Ulysses, Offload, and actor capture.
This is **not** counted as the optimized PPO runtime.

These old implementations are kept only because historical correctness tests
still import them. They are not acceptable as production alternatives, and
cannot be counted as additional production Ring/Engine implementations.

## Remaining acceptance work

- GitHub Actions syntax and architecture structural checks pass.
- Actor PPO replay, CP2/CP4 Ring backward and optimizer comparisons on Ascend
  are **not** certified by syntax/structural checks.
- Removing optional historical code will require retiring or remapping the
  corresponding legacy tests. Never conceal removed test coverage.
- Keep qkvmerge and CP padded physical-logical mapping. Do not enable Fixed Tile.
