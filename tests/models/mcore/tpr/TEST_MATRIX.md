# TPR test matrix and consolidation plan (2026-10-19)

This is the canonical **test-selection** document for `verl-tpr`. It is not a
retrospective PASS report. Source paths are deliberately stable while the many
`parallel/` script-to-script imports are audited. For the executable, curated
subset use `python tests/models/mcore/tpr/run_suite.py list`.

## Baseline inventory at branch creation

The source tree contains **189 pytest-named files**: unit 79, parallel 61,
correctness 17, profiling 14, equivalence 9, integration 5 and linear 4.
In addition, `parallel/` alone holds approximately 49 launchers/comparison
tools and 32 other Python helpers, which explains why a blanket
`pytest tests/models/mcore/tpr/parallel` is not a meaningful pass/fail gate.

The curated registry intentionally selects **11 entry points in 10 suites**.
Unselected tests are *not deleted, passed, failed or deprecated by default*;
they remain discoverable through `run_suite.py inventory` and the original
paths. The initial reorganization is a **non-breaking test governance pass**,
not a completed physical migration of every historical file.

## Physical cleanup and four main entries

The repository now uses `run_unit.sh`, `run_correctness.sh`,
`run_recompute.sh` and `run_performance.sh` as the four user-facing
entry points. The `run_suite.py` registry is a dispatcher, not another
mainline testing workflow. The first batch **removed 16 historical files,
including 10 pytest files** (see [PRUNE_LOG.md](PRUNE_LOG.md)). The original
189-test count above is a *pre-cleanup baseline*, not a current count.

A green acceptance requires actual executed, passing tests: all-skipped
pytest output fails the dispatcher. A performance report additionally
requires numerical validation and consistent input/hash/topology conditions.

## Acceptance ladder

| Stage | Core question | Primary command/test | Required evidence |
| --- | --- | --- | --- |
| CPU foundation | Scheduling, tree ownership, PPO/optimizer input, KV lifetime | `run_suite.py run cpu-core` | Selected unit contracts actually run (not skipped) |
| Primitive FA | Full vs rectangular causal Attention fwd/bwd | `run_suite.py run npu-core --ack-npu` | Loss/output, Q/K/V & Prefix dK/dV |
| Ring CP | Shard/padding/CP Ring numerical equivalence | `run_suite.py run npu-ring --ack-npu` | Physical vs logical ownership, per-rank allreduce and dKV |
| Native TP | Native Megatron TP2 with TPR | `run_suite.py run npu-tp --ack-npu` | Full gradients across TP ranks |
| Qwen3 | Real-model compatibility | `run_suite.py run npu-qwen --ack-npu` | Model inputs, logprob, loss, parameter gradients |
| Actor PPO | One to five optimizer steps on fixed real capture | See offline wrapper below | Fixed capture/hash, full grads, FP32 master and optimizer states |
| Recompute primitive | Hidden/K/V explicit checkpoint outputs | `run_suite.py run npu-recompute-primitive --ack-npu` | Replay context, KV anchors, leaf and root gradients |
| Recompute full model | CP1 then CP2 then TP2xCP4 OFF/ON | See dedicated wrapper below | Full grads, optimizer delta, collective order, peak memory |
| Performance | Training-side fwd+bwd + optimizer and (separately) online E2E | `run_suite.py show npu-performance` | Memory/speed and numerics under matched conditions |

For the `cpu-recompute` suite, install real Megatron/MindSpeed Python modules:
the fake-block test exercises a mocked transformer under the *actual* package
type, not the stubbed isolated CPU bootstrap.

## Real-actor PPO and recompute: manual asset-gated workflows

The following intentionally remain **outside the generic suite runner**:
their launchers control artifact output directories, fixed captures, model
paths and world layouts. Use the existing wrappers, not a bare pytest command.

### Fixed capture TP2 x CP4 actor PPO

```bash
TPR_OFFLINE_RECOMPUTE=0 \
TPR_OFFLINE_RUN_MODES=tpr \
TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_off_unique \
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh

TPR_OFFLINE_RECOMPUTE=1 \
TPR_OFFLINE_RUN_MODES=tpr \
TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_on_unique \
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh

python tests/models/mcore/tpr/parallel/compare_tpr_recompute_off_on.py \
  /workspace/tpr_off_unique /workspace/tpr_on_unique \
  --require-full-grads
```

**Both runs must save full gradients** using the launcher-supported
`TPR_OFFLINE_SAVE_FULL_GRADS=1`, share the same capture and initial checkpoint,
and confirm identical PPO labels and tree policy. These comparisons must not
be called native Megatron Full parity: the baseline here is an unshared
SegmentExecutor/Ring execution.

### TP2 x CP2 x DP2 recompute validation

```bash
bash tests/models/mcore/tpr/parallel/run_tpr_dp2_recompute_validation.sh \
  /workspace/real_actor_40k_manifest.json \
  /workspace/tpr_dp2_recompute_new_results 6
```

This wrapper checks an existing 20-step, eight-session manifest, requires
Qwen3-4B assets and NPUs 8..15 by default, and creates output directories
exclusively. It uses the **offline** FP32 gradient synchronization control
unless explicitly changed. BF16 production DP synchronization is a distinct
gate. Do not reuse a prior output directory or call an ON-only test an OFF/ON
comparison.

### Negative control, not a regression gate

`correctness/test_tpr_native_recompute_contract_npu.py` is the original
naive-checkpoint compatibility probe and deliberately makes assertions that
are not satisfied without KV-aware adaptation. It is never included in
positive suites. The explicit-KV-output probe is
`correctness/test_tpr_kv_checkpoint_replay_npu.py`.

## Historical / diagnostic boundaries

The following are useful investigations, **not** substitutes for mainline
acceptance:

- `correctness/_qwen17_*reference.py` and `test_qwen3_1_7b_*_npu.py`:
  legacy HF/AReaL/DTA precision comparisons. Keep the baselines frozen.
- `parallel/compare_*.py`, `parallel/analyze_*.py`,
  `parallel/probe_*.py`, `parallel/prepare_*.py`:
  offline artifacts, diagnosis and extraction tools, not pytests.
- `parallel/run_*.sh`: launcher-specific topology, fixed capture formats,
  and environment; no blanket execution.
- `profiling/`: measured performance and memory, not general correctness.
- `linear/`: research GDN and hybrid models, outside current production Dense.
- `parallel/test_native_*_npu.py`: native baseline controls, not TPR parity
  unless paired with explicit same-input TPR runs.

## Physical follow-up: dependency-safe migration

The current `parallel/` folder contains many sibling imports, for example
`compare_tpr_recompute_matrix.py` imports
`compare_real_actor_tp2cp4_offline_ppo.py` by module basename. Moving these
files individually breaks command-line execution and the wrappers calling
them. Before each physical move, record imports, shell invocations, dataset
prerequisites and source hashes, then migrate a whole dependency island in
one commit with compatibility checks. Do not create duplicate copies of tests
or delete previous numerical references merely to reduce the file count.

The first cleanup added one maintained gate registry, zero-dependency path
checks, and removed a verified historical-only batch. Further deletion must
preserve surviving module imports, CLI callers and numerical coverage.


## 2026-10-20 status

- 30K TP2×CP2×DP2 PPO 单步已由用户在 NPU 环境验收通过；此前的 MindSpeed 参数空键问题已做 CPU 回归。
- 旧版 10K parity 专项和仅适用于 legacy Megatron DP 的旧门禁已删除。
- Linear/GDN/Qwen3.5 Hybrid 被集中到 `experimental/linear/`（58 个原有文件），不作为 Dense TPR 主线门禁。
- 现存 `test_*.py` 180 个，其中 experimental 34，主线目录 146。
- 后续删减以 [FILE_AUDIT_20261020.md](FILE_AUDIT_20261020.md) 的逐文件表为依据。
