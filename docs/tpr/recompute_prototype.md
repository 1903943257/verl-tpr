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
