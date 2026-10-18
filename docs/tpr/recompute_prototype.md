# Experimental TPR Full Uniform(1) Recompute

**Branch:** `feat/tpr-full-recompute-uniform1-refactor`

**Status:** implementation prototype only. The Qwen3 real-model, Ascend NPU,
MindSpeed TP/Ring backward and memory-benefit gates have **not** been run.
No production recompute support is claimed. The original native-contract
negative probes in `test_tpr_native_recompute_contract_npu.py` are intentionally
not changed.

## Design

Based on main `ce6387c`; the production TPR implementation is now in
`verl/workers/engine/megatron/tpr/{adapter,executor,state,ring}.py`.
The regular TPR Push/Visit/Pop schedule is unchanged.

- Push runs graph-free and exports detached prefix KV, as before.
- Visit/Pop call Megatron's Full Uniform(1) TransformerBlock path.
- The TPR model instance substitutes only the block's
  `_checkpointed_forward`; no global Megatron or MindSpeed monkeypatch.
- The adapter calls the installed `tensor_parallel.checkpoint` for
  each native TransformerLayer, and exposes `(hidden_states, new_K, new_V)`
  as checkpoint outputs. This preserves KV roots in `SegmentExecutor.pop`.
- The captured outer `TPRAttentionContext` is read-only to the replay.
  Every invocation creates a fresh collector for new KV; the Ring backend,
  ancestor KV, anchor grads and RoPE are shared references.
- Engine routing accepts recompute only after checking that the adapter is
  installed on all TransformerBlocks.

## Explicit restrictions

Dense FA only; `recompute_granularity=full`,
`recompute_method=uniform`, `recompute_num_layers=1`.
No GDN, selective checkpoint, FP8/FP4, simultaneous Swap Attention,
Megatron CPU offload, distributed checkpointed activations, MTP or PP.
The adapter also rejects nonempty transformer context/packed-sequence
arguments rather than silently dropping them.

Use the existing VERL override fields when running an isolated experiment:

```text
++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
```

The production launcher should not be modified until all gates pass.

## Verification order

1. CPU: `python -m pytest -q tests/models/mcore/tpr/unit/test_tpr_full_recompute_cpu.py`
2. NPU checkpoint primitive with real Megatron:
   `TPR_RUN_KV_CHECKPOINT_REPLAY=1 torchrun --nproc_per_node=1 -m pytest -s -v tests/models/mcore/tpr/correctness/test_tpr_kv_checkpoint_replay_npu.py`
3. NPU CP1, one/two-layer Qwen3 tree with 2 siblings: compare full loss,
   every parameter gradient, and prefix K/V anchor gradients OFF vs ON.
4. Ring CP2 then TP2xCP4: repeat correctness, monitor replay collective
   sequence, deadlocks, and profile per-rank allocated/reserved peak.
5. Offline real-actor PPO 10K then 30K/40K; compare TPR OFF/ON, then
   compare native/TPR with the same recompute granularity.

The new native NPU contract probes only *explicit KV checkpoint outputs*, not
full-model accuracy. A successful CPU primitive repro is not the NPU gate.

## Known engineering follow-ups

- Check the installed MindSpeed repatched TransformerBlock checkpoint
  signature against the pinned MCore version; the adapter validates essential
  parameters at install but rejects unrecognized active options during run.
- Confirm Ring replay communications are identical on every CP rank,
  including ranks with no owned PPO tokens.
- Check that checkpoint replay reconstructs K/V using the correct local
  sequence shards and preserves CP gradients into ancestor anchors.
- Validate the PP=1 block wrapping path with vanilla MBridge as actually
  installed on the training server.
- Measure how much peak memory is due to permanently live prefix KV versus
  layer activations; recompute cannot eliminate prefix-state residency.

## Log organization on this development branch

All test console logs for this branch belong under `/workspace/tpr_log/recompute/`.
The offline TP2xCP4 wrapper automatically sets `TPR_TREE_POLICY=qwen_turn`
and writes `offline_tp2cp4_tpr.log`, `offline_tp2cp4_full.log`, and the
comparison log there. `TPR_OFFLINE_OUTPUT_DIR` still controls captured tensors
and comparison artifacts, not the console log location. The offline script is
**OFF by default**: with `TPR_OFFLINE_RECOMPUTE=0`, it sets
`model.config.recompute_granularity = None`. With `=1`, it installs
the KV-aware checkpoint adapter and selects Full Uniform(1). A PASS from
the old OFF baseline is not a recompute correctness claim.

```bash
mkdir -p /workspace/tpr_log/recompute
cd /workspace/verl-tpr-recompute
set -o pipefail
python -m pytest -q -x tests/models/mcore/tpr/unit/test_tpr_full_recompute_cpu.py \
  2>&1 | tee /workspace/tpr_log/recompute/cpu_contract.log

TPR_RUN_KV_CHECKPOINT_REPLAY=1 torchrun --nproc_per_node=1 \
  --master_addr=127.0.0.1 --master_port=29661 \
  -m pytest -s -v -x \
  tests/models/mcore/tpr/correctness/test_tpr_kv_checkpoint_replay_npu.py \
  2>&1 | tee /workspace/tpr_log/recompute/npu_checkpoint_contract.log
```

## Real actor Qwen3-4B TP2×CP4 OFF/ON toggle

The existing eight-NPU offline actor PPO launcher supports
`TPR_OFFLINE_RECOMPUTE=0` (default) or `1` for **TPR mode**.
ON configures `full/uniform/1`, calls
`install_tpr_full_recompute(model, model.config)` on the real Qwen model,
and verifies that every TransformerBlock shares the configured model state
and is patched. The saved rank snapshots include recompute mode and patch
count. The OFF control remains the same TPR source capture and tree policy.

The two runs must have different `TPR_OFFLINE_OUTPUT_DIR` values because
snapshots intentionally fail on existing output files. Console logs always
go to `/workspace/tpr_log/recompute/`.

```bash
cd /workspace/verl-tpr-recompute
mkdir -p /workspace/tpr_log/recompute
set -o pipefail
export TPR_OFFLINE_RUN_MODES=tpr
export TPR_OFFLINE_ROWS=4,6
export TPR_TREE_POLICY=qwen_turn

TPR_OFFLINE_RECOMPUTE=0 \
TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_recompute_off_v1 \
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh \
  2>&1 | tee /workspace/tpr_log/recompute/off_wrapper.log

TPR_OFFLINE_RECOMPUTE=1 \
TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_recompute_on_v1 \
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh \
  2>&1 | tee /workspace/tpr_log/recompute/on_wrapper.log

python tests/models/mcore/tpr/parallel/compare_tpr_recompute_off_on.py \
  /workspace/tpr_recompute_off_v1 /workspace/tpr_recompute_on_v1 \
  2>&1 | tee /workspace/tpr_log/recompute/off_on_compare.log
```

For a **full** parameter-gradient test, set `TPR_OFFLINE_SAVE_FULL_GRADS=1`
for **both** runs and include `--require-full-grads` in the comparison
command. The existing baseline replay without that flag contains gradient
samples and an exact scalar global norm, but not complete tensors.

The default runner uses `TPR_OFFLINE_RECOMPUTE=0`; the ON path has **not**
yet passed the NPU/Ring backward gate. The first ON execution is a gate for
adapter compatibility, HCCL replay, prefix gradient correctness and peak
memory. A result from the existing OFF baseline is not a recompute result.

## 2026-10-16: first TP2×RingCP4 real Actor PPO ON/OFF replay

**User-reported server run** (not independently repeated from the connector).
Captured rows `(4, 6)`, 4,616 PPO tokens, Qwen3-4B (36 layers);
eight ranks TP2×CP4; same initial sampled parameter weights.
OFF: `/workspace/tpr_recompute_baseline_v2`.
ON: `/workspace/tpr_recompute_on_v1`.
The ON snapshots confirm `enabled=True, granularity=full, method=uniform,
num_layers=1, installed_blocks=1`; all eight NPU ranks passed.

| Statistic | OFF | ON |
| --- | ---: | ---: |
| PPO loss | -1.7545651197433472 | -1.7545651197433472 |
| Logprob relative L2 / max_abs | 0 / 0 | 0 / 0 |
| Global BF16 gradient norm | 8.456138610839844 | 8.454642295837402 |
| Critical forward+backward (s) | 7.447827 | 8.577214 |
| Worst peak allocated (GiB) | 27.231803 | 14.538741 |
| Worst peak reserved (GiB) | 30.148438 | 16.574219 |

The ON result saves **12.693 GiB (46.61%) peak allocated** and
**13.574 GiB (45.02%) peak reserved**, at the cost of **1.129s
(15.16%)** longer critical forward+backward. Global BF16 gradient norm
relative gap is about **0.01770%**, but **sampled parameter gradient
relative L2 = 0.015527614 (1.55%)**. Post-SGD sampled weight L2 is
zero, which is not a sufficient gradient-parity witness for BF16
small-learning-rate updates.

**Gate still pending:** `complete_grad=None`. The observed losses
and logprobs are exactly equal, but full BF16 gradient relative L2,
cosine, top tensor contributors and ancestor prefix-anchor gradient
parity have not been independently established. Run OFF and ON with
`TPR_OFFLINE_SAVE_FULL_GRADS=1` and compare using
`compare_tpr_recompute_off_on.py --require-full-grads`. All console
logs must remain under `/workspace/tpr_log/recompute/`.

No changes should be made to `main` on the basis of this single replay.

## 2026-10-16: complete TPR OFF/ON gradient gate

**User-reported NPU run** with rows (4, 6), 4,616 PPO tokens, Qwen3-4B
TP2×CP4, `TPR_OFFLINE_SAVE_FULL_GRADS=1`:
- Equal loss -1.7545651197433472 and equal logical logprobs.
- Whole-model BF16 gradient relative L2: **0.010247685665679882**;
  cosine **0.9999474913094778**; exact global norms OFF
  **8.456650538670567**, ON **8.456029583016793**.
- Sampled gradient relative L2: 0.015617284133223113.
- Critical fwd/bwd OFF **7.092767982976511s**, ON
  **8.858999602030963s** (ON/OFF **1.2490187**).
- Worst peak allocated OFF **27.290441 GiB**, ON **14.536106 GiB**
  (**46.7% lower**); reserved OFF **30.148438 GiB**, ON **17.673828 GiB**.
- Largest absolute gradient differences concentrate in some FC2 and QKV
  tensors. A 1.025% whole-model gradient L2 gap remains; this is not a
  bitwise/exact backward parity claim.

## Controlled 2×2: unshared Full / shared TPR × recompute OFF / ON

The recorded replay now supports `TPR_OFFLINE_RECOMPUTE=1` with
`TPR_OFFLINE_RUN_MODES=full`, `tpr`, or `both`. For both modes,
full/uniform/1 is enabled on the **same TPR-capable TransformerBlock**
and installs the same explicit-KV checkpoint adapter. The *Full* case
uses isolated trajectory trees and the same SegmentExecutor/Ring
backend; it is **not** genuine native Megatron/VERL Full execution.
It is a controlled test of prefix sharing vs no sharing with the same
layer-recompute implementation. Use a separate Native Megatron test to
quantify native checkpoint gains independently.

To fill the missing Full OFF/ON cells while reusing previously saved
TPR OFF/ON snapshots:

```bash
cd /workspace/verl-tpr-recompute
mkdir -p /workspace/tpr_log/recompute
set -o pipefail
export TPR_OFFLINE_RUN_MODES=full
export TPR_OFFLINE_ROWS=4,6
export TPR_OFFLINE_SAVE_FULL_GRADS=0

TPR_OFFLINE_RECOMPUTE=0 \
TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_recompute_full_off_v1 \
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh \
  2>&1 | tee /workspace/tpr_log/recompute/full_off_wrapper.log

TPR_OFFLINE_RECOMPUTE=1 \
TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_recompute_full_on_v1 \
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh \
  2>&1 | tee /workspace/tpr_log/recompute/full_on_wrapper.log

python tests/models/mcore/tpr/parallel/compare_tpr_recompute_matrix.py \
  /workspace/tpr_recompute_full_off_v1 \
  /workspace/tpr_recompute_full_on_v1 \
  /workspace/tpr_recompute_baseline_v2 \
  /workspace/tpr_recompute_on_v1 \
  2>&1 | tee /workspace/tpr_log/recompute/recompute_matrix.log
```

The four-way tool checks matching source capture, rows, model, initial
parameter witnesses and per-mode tree and Ring configuration. It prints
latency, allocated/reserved peaks, loss, sampled-gradient gaps and
per-mode Recompute OFF→ON savings. Optional
`--full-gradients` computes complete BF16 gradients when **all four
cases** have full gradient manifests.

Alternatively run both checkpointed cases in fresh processes with
`TPR_OFFLINE_RECOMPUTE=1 TPR_OFFLINE_RUN_MODES=both` and a new unique
`TPR_OFFLINE_OUTPUT_DIR`. The standard Full-vs-TPR comparison in that
mode is `--diagnostic-only --require-recompute`; legacy threshold gates
are reported but not enforced. All console logs go under
`/workspace/tpr_log/recompute/`.

## 2026-10-16: observed controlled Full/TPR × recompute 2×2 matrix

**User-reported NPU results** for Qwen3-4B, TP2×CP4, captured rows (4, 6),
4,616 logical PPO tokens. These are four independent eight-rank replay runs.
`full` remains **unshared SegmentExecutor/Ring**, **not genuine native Megatron Full**.

| Case | PPO loss | Global BF16 gradient norm | Critical fwd/bwd (s) | Worst allocated (GiB) | Worst reserved (GiB) | Trees / Segments |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full OFF | -1.7548251152038574 | 7.976451873779297 | 6.635661877 | 43.084094 | 47.197266 | 2 / 2 |
| Full ON | -1.7548251152038574 | 7.974936485290527 | 7.877788798 | 17.997396 | 22.335938 | 2 / 2 |
| TPR OFF | -1.7545651197433472 | 8.456138610839844 | 7.447827103 | 27.231803 | 30.148438 | 1 / 3 |
| TPR ON | -1.7545651197433472 | 8.454642295837402 | 8.577214051 | 14.538741 | 16.574219 | 1 / 3 |

**Isolation within each execution mode:**
- Full OFF vs ON: equal loss and exact reported response logprobs,
  sampled parameter-gradient relative L2 **0.014425218**;
  recompute saves **58.2273% allocated**, **52.6754% reserved**,
  but runtime increases **18.7190%**.
- TPR OFF vs ON: equal loss and exact reported response logprobs,
  sampled parameter-gradient relative L2 **0.015527614**;
  recompute saves **46.6112% allocated**, **45.0246% reserved**,
  but runtime increases **15.1640%**.
- With recompute ON in both, TPR vs unshared Full saves
  **19.2175% allocated** and **25.7957% reserved**, but TPR
  runs **8.8784% longer**.
- With recompute OFF in both, TPR saves **36.7938% allocated**,
  **36.1225% reserved**, but runs **12.2394% longer**.

**Numerical parity warning (must not ignore):** Full vs shared TPR has
loss gap **0.00025999546**, response logprob relative L2
**0.0573565426** (5.74%), and sampled parameter-gradient relative
L2 **0.547133764 ON** / **0.547227756 OFF** (~54.7%). The
approximately unchanged Full-vs-TPR drift under recompute indicates
that **recompute is not the dominant new source of the cross-mode
difference**. In contrast, *within* Full and within TPR, OFF/ON
logprobs match exactly. Avoid interpreting a small scalar PPO loss gap
as a gradient-equivalence pass. Diagnose tree execution/segment
boundary, logprob ownership/temperature/clipping, and CP Ring
numerics under the *same logical PPO labels* before production gating.

**Follow-ups:** run separately *genuine native Megatron/VERL Full*
OFF/ON with matched capture, model, topology, recompute policy and
measurement window; the current `run_verl_engine_full_replay.sh`
still explicitly rejects native activation recompute. For the
controlled four cases, optionally enable full BF16-gradient capture
for each run and compare per-parameter gradients/cosines. No main
change or claim of native Full recompute validation follows from
this result.

## 2026-10-16: matched 40K 8-Session Step 6 Full/TPR Recompute ON

The original warm-shape benchmark
`test_tpr_fixed_shape_40k_benchmark_npu.py` now accepts
`TPR_BENCH_RECOMPUTE=0|1` (default OFF) on **both** independent launch
modes, without changing capture selection, Megatron mesh, PPO objective,
warmups/repeats, or existing non-recompute file names.

- Native Full ON: real `native_attention=True`, Megatron
  `get_forward_backward_func()` schedule, native
  `full/uniform/1` checkpoint method; zero TPR patches.
- TPR ON: real `TPRSelfAttention`, production Ring/SegmentExecutor,
  `full/uniform/1` plus installed explicit-KV checkpoint adapter on
  each TransformerBlock. The existing engine-builder clears recompute
  during configuration; this benchmark intentionally re-enables and
  patches **after** the builder returns.
- ON output suffix `_recompute_on_fixed_shape.json` ensures OFF
  artifacts `_fixed_shape.json` remain untouched. Each output records
  the actual recompute config and TPR adapter install count.
- The native model factory consults
  `TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS` at construction. The benchmark
  requires that inherited variable to be unset/zero, then controls it
  exclusively via `TPR_BENCH_RECOMPUTE` (Native ON selects 1;
  TPR ON selects 0 native layers and installs its separate adapter).

Run after updating the Docker worktree:

```bash
cd /workspace/verl-tpr-recompute
mkdir -p /workspace/tpr_log/recompute
export TPR_OFFLINE_CAPTURE_MANIFEST=/workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_BENCH_SOURCE_STEP=6
export TPR_BENCH_OUTPUT_DIR=/workspace/tpr_recompute_fixed40k_step6_v1
export TPR_BENCH_WARMUPS=1
export TPR_BENCH_REPEATS=3
export TPR_BENCH_CAPTURE_LOGPROBS=1
export TPR_BENCH_RECOMPUTE=1
export TPR_TREE_POLICY=qwen_turn
export TPR_EXEC_PROFILE=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
unset TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS
set -o pipefail

TPR_OFFLINE_MODE=full torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29795 \
  -m pytest -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
  2>&1 | tee /workspace/tpr_log/recompute/fixed40k_step6_full_recompute_on.log

TPR_OFFLINE_MODE=tpr torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29795 \
  -m pytest -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
  2>&1 | tee /workspace/tpr_log/recompute/fixed40k_step6_tpr_recompute_on.log
```

Source/protocol and model remain the same as the previously verified
Step 6 OFF benchmark. **The two ON runs still require real NPU validation**.
A failure is not an implicit PASS or automatic fallback to OFF.

## 2026-10-16: eight-trajectory Step 6 TPR ON memory accumulation / OOM

**User-supplied NPU failure log** from the first 40K 8-session test:
- Native Full + `full/uniform/1` completed: median fwd/bwd
  **32.829754 s** (n=3), worst peak allocated **13.746 GiB**.
- TPR + explicit-KV checkpoint adapter: warmup iteration 0
  **32.618061 s**, peak **24.860 GiB**; measured iteration 1
  **30.508493 s**, peak **38.376 GiB**; measured iteration 2
  **30.039366 s**, peak **51.893 GiB**; next iteration failed with
  NPU OOM (NPU 6, 60.95 GiB total, 53.74 GiB allocated).
  Peak grew ~**13.5 GiB per additional iteration**. TPR ON is
  **not an accepted performance result** until multi-iteration memory
  stabilizes.
- Immediate allocation failure occurred at
  `SegmentExecutor._compute_loss()` for a CP rank with **zero**
  owned PPO objective tokens:
  `logits.float().sum() * 0.0` requested **1.63 GiB**.
  This wide BF16→FP32 logits cast is unnecessary to construct a
  differentiable zero, and is a failure trigger, **not proven root
  cause** of the steady allocation growth.

**Mitigation and diagnostic, this branch only:**
- Replace both zero-objective sites by
  `logits[0, 0, 0].float() * 0.0`. This preserves a connected
  scalar zero and Ring backward participation without the
  full-vocabulary FP32 copy. A CPU contract test validates zero
  gradient for the whole logits tensor.
- Add `TPR_BENCH_MEM_AUDIT=1` to the fixed-shape benchmark:
  after each fully timed forward/backward, synchronizes, prints
  world-maximum **live allocated before/after `gc.collect()`**, and
  keeps GC strictly **outside** the measured F/B duration.
  This experiment intentionally changes cross-iteration memory lifetime.
  If before-GC grows but after-GC drops toward the stable baseline,
  inspect reference cycles / replay-local closures. If both grow,
  inspect live graph/KV/CP backend references and NPU allocator state.
  GC diagnostics are not a production memory management solution.

```bash
cd /workspace/verl-tpr-recompute
mkdir -p /workspace/tpr_log/recompute
export TPR_OFFLINE_CAPTURE_MANIFEST=/workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json
export TPR_BENCH_SOURCE_STEP=6
export TPR_BENCH_OUTPUT_DIR=/workspace/tpr_recompute_fixed40k_step6_gcprobe_v1
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_BENCH_RECOMPUTE=1
export TPR_BENCH_MEM_AUDIT=1
export TPR_BENCH_WARMUPS=1
export TPR_BENCH_REPEATS=3
export TPR_BENCH_CAPTURE_LOGPROBS=1
export TPR_TREE_POLICY=qwen_turn
export TPR_EXEC_PROFILE=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
unset TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS
set -o pipefail
TPR_OFFLINE_MODE=tpr torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29795 \
  -m pytest -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
  2>&1 | tee /workspace/tpr_log/recompute/step6_tpr_recompute_gcprobe_v1.log
```

Never compare a GC-audited critical section's repeated-step timings
with historical steady-state medians without noting the altered
between-iteration GC policy. Do not yet add to `main`.

## 2026-10-16: repeated 40K leak diagnosis and checkpoint closure fix

The GC probe confirmed a **real live allocation increase** on TPR
Recompute ON at the end of each repetition (max across 8 NPU ranks):

| Iteration | Peak allocated GiB | Live before GC GiB | Live after GC GiB |
| ---: | ---: | ---: | ---: |
| 0 (warmup) | 24.868 | 21.134 | 21.134 |
| 1 | 38.380 | 34.650 | 34.650 |
| 2 | 51.896 | 48.165 | 48.165 |

Exactly ~13.516 GiB additional **live** memory per iteration,
unaffected by diagnostic Python garbage collection. Recompute ON
Native Full remained stable and passed. The reduced zero-loss logits
upcast did **not** remove this accumulation.

**Root-cause hypothesis grounded in the actual code:** Megatron
`tensor_parallel.random.CheckpointFunction.forward()` sets
`ctx.run_function = run_function`. The TPR `run_layer` closure used to
capture the complete outer `TPRAttentionContext`; outer holds
`new_key_values` containing the **outputs of that checkpoint node**.
This forms a Python/C++ autograd cycle:
`Checkpoint ctx → run_layer → outer → (key, value).grad_fn → Checkpoint ctx`.
It can retain NPU activation memory that Python GC cannot reclaim.

**Branch-only fix:** checkpoint `run_layer` captures
`weakref.ref(outer)`, dereferences it only during forward/replay and
fails explicitly if the surrounding SegmentExecutor context is not
alive. The executor deliberately keeps its `context` local through
the complete `torch.autograd.backward()`, so its legitimate lifetime
is maintained. A CPU test statically asserts `outer` is no longer
in the function's closure variables and verifies K/V gradient
propagation. A separate CPU test no longer calls the broken
`torch.autograd.profiler` in the deployed PyTorch/NPU environment:
instead it rejects non-scalar calls to `.float()` in the zero-loss path.
All test tensors explicitly specify `device="cpu"`.

**Verification (not yet re-run on the user's NPU server):**

```bash
cd /workspace/verl-tpr-recompute
mkdir -p /workspace/tpr_log/recompute
set -o pipefail
python -m pytest -q -x \
  tests/models/mcore/tpr/unit/test_tpr_zero_objective_cpu.py \
  tests/models/mcore/tpr/unit/test_tpr_full_recompute_cpu.py \
  2>&1 | tee /workspace/tpr_log/recompute/cpu_recompute_cycle_fix.log

export TPR_OFFLINE_CAPTURE_MANIFEST=/workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json
export TPR_BENCH_SOURCE_STEP=6
export TPR_BENCH_OUTPUT_DIR=/workspace/tpr_recompute_fixed40k_step6_weakref_v1
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_BENCH_RECOMPUTE=1
export TPR_BENCH_MEM_AUDIT=1
export TPR_RECOMPUTE_REF_AUDIT=1
export TPR_BENCH_WARMUPS=1
export TPR_BENCH_REPEATS=3
export TPR_BENCH_CAPTURE_LOGPROBS=1
export TPR_TREE_POLICY=qwen_turn
export TPR_EXEC_PROFILE=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
unset TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS
TPR_OFFLINE_MODE=tpr torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29795 -m pytest -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
  2>&1 | tee /workspace/tpr_log/recompute/step6_tpr_weakref_mem_audit.log
```

The diagnostic reports `live_recompute_refs` and allocated memory before
and after GC. **A CPU syntax/functional PASS alone does not validate
NPU memory stabilization.** If the repeated NPU run passes with flat
live memory, next rerun with both `TPR_BENCH_MEM_AUDIT=0` and
`TPR_RECOMPUTE_REF_AUDIT=0` to obtain fair timing.

## 2026-10-16: 40K 8-Session leak fixed, stable TPR checkpoint replay

User-reported TP2×CP4 8-trajectory 40K Step 6 fixed-shape replay
after the checkpoint closure fix (commit `fad3506`) and dataclass
weak-reference slot fix (`41bdd46`).

**Recompute ON TPR, 1 warmup + 3 measured repetitions:**

| Iteration | Critical fwd/bwd seconds | Peak allocated GiB | Live allocated before GC GiB | Live allocated after GC GiB | Weakly live checkpoint closures/contexts |
| --- | ---: | ---: | ---: | ---: | --- |
| warmup 0 | 32.374836 | 14.862 | 7.621 | 7.621 | 0 / 0 |
| 1 | 29.944529 | 14.865 | 7.620 | 7.620 | 0 / 0 |
| 2 | 29.913969 | 14.865 | 7.620 | 7.620 | 0 / 0 |
| 3 | 29.772104 | 14.865 | 7.620 | 7.620 | 0 / 0 |

Median TPR ON **29.913969 seconds**. PPO loss unchanged at
`0.0999025851`; SHA prefix `81c1375b1e575e5b`.
Saved file:
`/workspace/tpr_recompute_fixed40k_step6_weakref_v2/step6_tpr_recompute_on_fixed_shape.json`.

**Comparable 40K Step 6 reference data** (same input SHA, model,
mesh, PPO data, 1 warmup + 3 measured):

| Mode | Recompute | Median seconds | Peak allocated GiB | PPO loss |
| --- | --- | ---: | ---: | ---: |
| Native Megatron Full | OFF | 23.137812 | 43.616 | 0.0998954326 |
| TPR | OFF | 21.802193 | 31.173 | 0.0999025851 |
| Native Megatron Full | ON | 32.829754 | 13.746 | 0.0998954326 |
| TPR | ON + GC/weakref diagnostics | 29.913969 | 14.865 | 0.0999025851 |

With recompute OFF, Full/TPR median runtime ratio is **1.06126×**.
With recompute ON, the ratio is **1.09747×** (TPR takes **8.88% less
time**), but the ON TPR result uses diagnostic GC between measured
iterations: it is provisional until audited-off steady-state re-run.
ON TPR memory is **1.119 GiB higher** than Native Full ON; versus TPR
OFF, TPR ON saves about **52.31%** peak allocated. Native Full ON versus
Full OFF saves about **68.48%**.

**Leak conclusion:** prior diagnostic grew from live allocations
21.134 → 34.650 → 48.165 GiB after each iteration (~13.516
GiB per iteration), unaffected by `gc.collect()`, and OOMed.
Following the weak-reference change to the Checkpoint run function,
live memory remains **7.620 GiB** and peak **14.865 GiB** across all
three measured iterations, with no live checkpoint closures or
attention contexts according to the opt-in weak-reference counters.
This strongly corroborates the strong-outer-context closure cycle as
root cause; no extra Python GC is needed to reduce live allocations.
This is a server-reported experimental result, not a locally rerun NPU test.

**Final confirmation without diagnostic side effects**, fresh output
folder; do not overwrite earlier snapshots:

```bash
cd /workspace/verl-tpr-recompute
mkdir -p /workspace/tpr_log/recompute
export TPR_OFFLINE_CAPTURE_MANIFEST=/workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json
export TPR_BENCH_SOURCE_STEP=6
export TPR_BENCH_OUTPUT_DIR=/workspace/tpr_recompute_fixed40k_step6_final_v1
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_BENCH_RECOMPUTE=1
export TPR_BENCH_MEM_AUDIT=0
export TPR_RECOMPUTE_REF_AUDIT=0
export TPR_BENCH_WARMUPS=1
export TPR_BENCH_REPEATS=3
export TPR_BENCH_CAPTURE_LOGPROBS=1
export TPR_TREE_POLICY=qwen_turn
export TPR_EXEC_PROFILE=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
unset TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS
set -o pipefail
TPR_OFFLINE_MODE=tpr torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29795 \
  -m pytest -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
  2>&1 | tee /workspace/tpr_log/recompute/step6_tpr_recompute_final.log
```

After stabilization in no-audit mode, verify Full/TPR logprob
alignment and full parameter-gradient parity separately; small scalar
PPO loss differences do not establish training-gradient equivalence.


## TP2 CP2 DP2 recompute compatibility experiment (8 Ascend NPUs)

Status: code committed, server NPU validation pending. Reuses the existing
Qwen3-4B real-Actor PPO one-step TP2 RingCP2 DP2 replay and original
session-first8 Step6 captures and DTA/DFS minimax placement. No changes to
production TP/CP/DP, Ring kernels or optimizer paths.

Two independent eight-rank jobs load the same HF weights, replay the same
original PPO fields, and run TPR Recompute OFF and ON (full/uniform/1).
Every TransformerBlock is checked for successful KV-aware installation.
The test repeats frozen-weight F/B with CP×DP gradient finalization. After
timing, all BF16 gradients from TP0/TP1 CP0/DP0 are saved, then a single
CPU FP32-master AdamW update uses the worldwide global gradnorm and clipping.
Each TP-local CP×DP group receives updated BF16 parameters via broadcast.
A separate CPU comparator reads every saved parameter gradient and checks
whole-gradient relative L2 and max-abs, global PPO loss, exact gradient
replica witness, and post-update sampled weights. OFF and ON use distinct
fresh processes and output subdirectories; no partial result is called PASS.

Host:
    cd /mnt/pipeline-data/j00972288/project/uniagent_linear/verl-tpr-recompute
    git pull --ff-only

Docker:
    cd /workspace/verl-tpr-recompute
    mkdir -p /workspace/tpr_log/recompute
    python -m pytest -q -x tests/models/mcore/tpr/unit/test_tpr_dp2_recompute_compare_cpu.py 2>&1 | tee /workspace/tpr_log/recompute/dp2_compare_cpu.log
    bash tests/models/mcore/tpr/parallel/run_tpr_dp2_recompute_validation.sh /workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json /workspace/tpr_experiments/recompute_dp2_step6_pair_v1 6

The runner uses torchrun --master_addr=127.0.0.1 --master_port=29797,
never --standalone. It stores logs under its dedicated output/logs, and
all gradient snapshots inside off/ and on/. Preserve at least two complete
copies of model BF16 gradients on disk. Offline optimizer is separate from
timed F/B, and is NOT the production VERL Optimizer or online GRPO loop.

40K CP2/DP2 previously OOMed on some configurations. If the OFF baseline
fails on 40K, the same-input ON/OFF parity comparison is inconclusive.
Use the same separately derived 30K manifest for BOTH jobs if necessary
and label the cap correctly; do not mix 30K with 40K.

The report thresholds default to global PPO loss absolute gap <=1e-3,
full-gradient rel-L2 <=5%, and sampled updated-weight max-abs <=0.02.
The per-mode sampled gradient replica gate is strict/exact. Complete
post-update full-parameter parity and production optimizer throughput
are NOT established by sampled weight checks. CI/CPU tests do not
substitute for real 8-NPU HCCL execution.


### DP2 40K OFF OOM and BF16 replica non-exactness (follow-up)

The 40K Step6 TP2/CP2/DP2 Recompute ON actually completed one warmup and
three measured F/B passes, with ~28.46s median F/B and stable ~20.419 GiB
peak allocated. The strict BF16 CP×DP gradient replica witness then failed:
maximum sample gap 3.05175781e-5 (one BF16 rounding quantum near the
affected small gradient) and spread/representative sample norm ~0.003818.
This is an after-finalize *non-bitwise identical* weight-gradient issue;
it is not an activation checkpoint crash or evidence of a new memory leak.
OFF at 40K separately OOMed, so the 40K paired parity gate is incomplete.
The exact OFF OOM allocation site still requires its original traceback.

The isolated offline helper grad_sync_precision.py already documents BF16
HCCL SUM/4 causing divergent weight-gradient copies on Ascend 910B. The
dedicated recompute DP2 validation shell now defaults to explicit FP32
per-parameter all_reduce staging, while keeping the strict exact replica
witness enabled. This tests KV-aware recompute correctness in a controlled
precision setting; it does NOT claim native BF16 grad synchronization
is repaired or that production uses FP32.

For the **40K ON-only capacity + single AdamW gate** (fresh output):

    TPR_DP2_VALIDATION_MODES=on \
      bash tests/models/mcore/tpr/parallel/run_tpr_dp2_recompute_validation.sh \
      /workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json \
      /workspace/tpr_experiments/recompute_dp2_step6_on_fp32_v1 6

For the **matched 30K OFF/ON complete gradient + AdamW comparison**,
derive the new capture using the existing explicit truncation utility,
preserving all original supervised token labels that remain:

    python tests/models/mcore/tpr/parallel/prepare_tpr_30k_one_step.py \
      /workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json \
      /workspace/tpr_experiments/recompute_dp2_step6_30k_capture_v1 \
      --step 6 --cap 30720

    bash tests/models/mcore/tpr/parallel/run_tpr_dp2_recompute_validation.sh \
      /workspace/tpr_experiments/recompute_dp2_step6_30k_capture_v1/capture_manifest.json \
      /workspace/tpr_experiments/recompute_dp2_step6_30k_fp32_pair_v1 6

To reproduce BF16 native-gradient discrepancies separately, explicitly set
TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16; failure of the exact replica witness is
a real numerical consistency problem, not a passing test. Do not use
diagnostic-only policy to claim recompute correctness.
