# TP2 + native SP + TPR Ring CP acceptance (E2E branch)

Scope: only TP2+SP+RingCP2/4. Do not change DP placement or gradients.
This branch **does not claim SP+Ring is production validated** until NPU
tests and a real actor optimizer step pass. The opt-in is
`TPR_ENABLE_TP_SP_CP=1`. E2E uses `TRAIN_SP=True` explicitly.

## 0. Sync branch and CPU preflight

All commands below run **inside the Docker container**, in the E2E checkout.

```bash
cd /workspace/verl-tpr-e2e
git status --short
git fetch tpr feat/tpr-rebench-e2e-20261026
git merge --ff-only tpr/feat/tpr-rebench-e2e-20261026

python -m pytest -q tests/models/mcore/tpr/parallel/test_tp_sp_ring_contract_cpu.py
python tests/models/mcore/tpr/run_suite.py check
```

Pass: CPU physical CP-local shape contract for odd/unpadded segments at CP2
and CP4, opt-in validation, existing suite registry intact.

## 1. TP2+SP+CP1 established comparator (2 NPUs)

```bash
cd /workspace/verl-tpr-e2e
ASCEND_RT_VISIBLE_DEVICES=0,1 \
TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B \
TPR_RUN_TP2_SP=1 \
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29860 \
  -m pytest -v -s -x tests/models/mcore/tpr/parallel/test_tpr_tp2_sp_npu.py
```

Pass: native SP input, gathered QKV, partitioned output and controlled
Full/TPR loss/logit/gradient assertions. This does not test CP.

## 2. TP2+SP+RingCP2 shape, KV, backward smoke (4 NPUs)

```bash
cd /workspace/verl-tpr-e2e
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B \
TPR_RUN_TP2_SP_CP2=1 TPR_ENABLE_TP_SP_CP=1 \
torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29861 \
  -m pytest -v -s -x tests/models/mcore/tpr/parallel/test_tpr_tp2_sp_cp2_npu.py
```

Pass: **TPR_TP2_SP_CP2 PASS** on rank 0; input lengths equal half the
physical CP-local length; QKV lengths equal the complete CP-local length;
Push/Visit/Pop completes with finite, nonzero QKV parameter gradients.
**This is a smoke check, not Full-vs-TPR parity**.

## 3. TP2+SP+RingCP4 recorded PPO: SP-off/on comparison (8 NPUs)

Requires an existing real actor-update mini-batch captured with the original
PPO advantages, logprobs and masks. Use a valid path on the current server.

For SP-off control:

```bash
cd /workspace/verl-tpr-e2e
export TPR_REAL_ACTOR_MINIBATCH=/workspace/tpr_experiments/YOUR_CAPTURE/actor_update_rank0_batch0.pt
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TPR_OFFLINE_SP=0 TPR_OFFLINE_RECOMPUTE=0 TPR_OFFLINE_RUN_MODES=both \
  bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh
```

For SP-on with **identical** capture / rows / RNG / checkpoint:

```bash
TPR_OFFLINE_SP=1 TPR_OFFLINE_RECOMPUTE=0 TPR_OFFLINE_RUN_MODES=both \
  bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh
```

The script sets `TPR_ENABLE_TP_SP_CP=1` for SP-on; SP-off/on
artifacts go to separate default output directories. The script's
comparison distinguishes Full unshared (same TPR Ring backend) versus
TPR shared. **It does not claim native MindSpeed Full parity.**

Check these before production activation:

- No collective hang, SP layout assertion, invalid gradient or OOM.
- The same capture SHA, selected rows, model weights and PPO denominator in all jobs.
- Compare Full-vs-TPR loss, logprobs, gradient norms and sampled gradients with
  the *SP-off* reference. Keep both numerical comparisons, not just PASS/FAIL.
- SP-on vs SP-off must not materially worsen existing BF16/segmentation drift;
  investigate any new systematic mismatch. No new tolerance is assumed.
- Record peak per-rank allocated GiB and median NPU F/B; SP can trade memory
  for extra TP communications.

On 8x32GiB cards the historical two-trajectory 4B capture may OOM.
If it does, choose a shorter **identical** real capture or run one row as a
memory/layout probe; a one-row test cannot establish prefix reuse benefit.

## 4. Full E2E one-step actor update (not PRECHECK_ONLY)

```bash
cd /workspace/verl-tpr-e2e
PRECHECK_ONLY=True TPR_ENABLED=True TRAIN_TP=2 TRAIN_CP=4 \
  TRAIN_SP=True TPR_ENABLE_TP_SP_CP=1 N_GPUS_PER_NODE=8 \
  bash tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh
```

After preflight, run with `PRECHECK_ONLY=False`, `TOTAL_TRAINING_STEPS=1`,
`ACTOR_RECOMPUTE=False` initially, the known runnable dataset/task, a
short 1-row job and a **new** `RUN_NAME`. Increase context only after the
small case passes. Capture evidence of a nonzero actor loss/gradient and
a completed optimizer update; successful rollout/reward alone is insufficient.
Compare against an otherwise identical Full run using `TRAIN_SP=True`.
Full and TPR online rollouts are stochastic, so they are not fixed-data parity.

## Promotion criteria

Only after the CPU gates, TP2SPCP2 smoke, CP4 fixed-data PPO parity and
real E2E actor optimizer step pass should the experiment-only
`TPR_ENABLE_TP_SP_CP` gate be removed. Do not remove it based only on the
absence of a Python exception or a PRECHECK_ONLY pass.
