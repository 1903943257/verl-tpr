# 2K, ten-update production training comparison

To reuse four completed routes and add Base CP1 repeat plus TPR CP2 QKV merge,
see [the six-route supplement](TRAINING_SIXWAY.md).

For one additional Base CP1 run against the saved historical control, see
[the Base CP1 repeat](BASE_CP1_REPEAT.md). It reuses the original fixture and
does not rerun the other three routes.

Run after the single-backward production-path gate, using the same branch
and normal Ascend training container. Pull on the host; Git is optional
inside Docker.

```bash
export TPR_TRAIN_MODEL_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_TRAIN_DEVICES=0,1
export TPR_TRAIN10_OUTPUT=/workspace/tpr_experiments/p1024s1024_train10_1
bash tests/models/mcore/tpr/parallel/run_training_multistep.sh
```

The output directory must be new. The launcher runs Base CP1, Base CP2,
TPR CP1, TPR CP2 sequentially. Each route loads once and performs **ten actual
native optimizer updates**. It uses the same production engine, bridge,
DDP, native schedules/TPR Forest, fused wgrad and FP32 gradient finalize
as [the single-backward gate](TRAINING_PATH_PRECISION.md).

If a later route fails, keep the same checkpoint and controls, set
`TPR_TRAIN10_RESUME=1`, and rerun the launcher. Completed routes are validated
and skipped; incomplete route folders are archived inside the experiment
directory, and that route restarts all ten updates from the initial state.
This is route-level recovery, not an optimizer checkpoint at an intermediate
step. Resume requires Base CP1 to have completed and its references to exist.

## Controlled training

* TP=PP=DP=1; CP=1/2. BF16 parameters and activations, FP32 native main_grad,
  FP32 DistributedOptimizer masters and Adam moments. No CPU/custom optimizer.
* Native `optimizer_zero_grad`, `forward_backward_batch`, `optimizer_step`
  and `lr_scheduler_step` execute each update. Clipping remains inside the
  native optimizer, with threshold 1.0. Default LR is a **constant 1e-6**,
  warmup zero, weight decay 0.01. `TPR_TRAIN10_LR` changes LR for all routes.
* The HF weights initialize a fresh optimizer for each route. No checkpoint
  optimizer state or previously completed updates are loaded. Initial master
  shards must exactly equal the BF16 model converted to FP32; otherwise the
  observer refuses to infer the initial master values.
* Ten different deterministic minibatches are fixed in `actor_fixture.pt`.
  Each has two trajectories sharing P1024, with different S1024 responses:
  2K **per trajectory**, two logical rows, three physical TPR segments.
* Initial Base CP1 native inference supplies old logprobs for all ten
  batches and an eleventh, held-out probe. The four routes reuse those
  exact tokens, masks, old logprobs, advantages and order. Each batch uses
  the existing unmodified vanilla token-mean PPO objective. Old logprobs
  never change with an individual route's updates.
* Dropout/recomputation off, microbatch one, entropy/KL off. Same controls
  as the previous gate: delayed prefix dKV FP32 off unless explicitly set,
  root-native-ring/coalescing/local FP32 operator experiments off.
  `TPR_TRAIN_GRAD_ACCUM_FUSION` and `TPR_TRAIN_VANILLA_MBRIDGE` retain their
  previous defaults of 1. Do not change any control between routes.

This is fixed-data objective/optimizer drift, not an on-policy rollout or
convergence benchmark. From update 2 onward, gradients are evaluated at
already-diverged weights. The previous single-backward gate provides the
same-initial-state gradient comparison.

## Measurements and cost

Every update logs loss, PPO metrics, native pre-clip gradient norm, the
clipping coefficient derived from that norm, actual learning rates used,
peak allocated NPU memory and synchronized zero/F-B/optimizer/scheduler
timings. Native route profiling runs **only on update 1**; updates 2–10
have no Python call profiler. The first step therefore includes profiler
and cold-kernel overhead. Each rank retains a separate provenance audit.

At updates **1, 5 and 10**, take up to 1024 fixed global coordinates from
**every named parameter**, identical for CP1 and CP2. These sampled FP32
master-update comparisons show a trend, not a full-gradient/full-update
error bound. At update **10 only**, scan **every optimizer-owned FP32 master
element** and compare the complete cumulative updates to both Base routes.

The full metric is `||delta_candidate - delta_reference|| / ||delta_reference||`,
where `delta = FP32_master_after_10 - FP32_master_initial`. Comparing full
weights instead would hide differences behind their common initial state;
comparing BF16 model weights could hide updates below BF16 rounding.

Only two full Base update references are saved temporarily: approximately
14 GB for 1.7B / 33 GB for 4B. TPR update blocks are compared online and not
saved. Successful final aggregation removes only manifest-listed Base
blocks; set `TPR_TRAIN10_KEEP_UPDATES=1` to keep them. No full per-step
gradient dumps or per-step weight hashes. A BF16 initial host copy is kept
per rank (about 3.4 GB for 1.7B / 8 GB for 4B), with bounded FP32 CPU blocks.
There are still substantial final CPU scans; `TRAIN10_FINAL_SCAN` marks them.

Unlike the previous no-update gate, true Adam updates allocate native
moment states: roughly **8 bytes per owned parameter** beyond a lazy-state
optimizer's pre-update memory. That is about 13 GiB for 1.7B CP1 / 6.5 GiB
per CP2 rank, or 30 GiB for 4B CP1 / 15 GiB per CP2 rank, before temporary
optimizer buffers. Use 1.7B when memory is limited. No silent offload or
precision fallback is installed.

## Held-out output evaluation

At steps **0, 1, 5, 10**, every route evaluates the same held-out probe
through **native whole-sequence forward-only inference**, not TPR segment
execution. TPR attention falls back to its native parent when no TPR
context is active; the observer rejects a leaked training context.

The route retains its CP size. Thus Base CP1 versus TPR CP1 and Base CP2
versus TPR CP2 use matched evaluation layouts. **Cross-CP output gaps still
include native evaluation-layout effects**; this is not a common-CP1
checkpoint evaluator. Reports include raw logprob gaps and gaps in each
route's output change relative to its own step 0. Subtracting the initial
gap helps interpretation but does not make CP1 and CP2 evaluators identical.

## Read the short report

```bash
cat "$TPR_TRAIN10_OUTPUT/summary.md"
grep -hE 'TRAIN10_(STEP|DONE|COMPARE|ALL_DONE)' \
  "$TPR_TRAIN10_OUTPUT"/*/run.log "$TPR_TRAIN10_OUTPUT/summary.log"
```

`comparison.json` contains scalar comparisons/trends and final-update error
ratios to native Base CP1/CP2. Detailed per-step metrics/times are in each
`<mode>/result.json`; raw small probe/sample tensors are `.pt` files, not
thousands of JSON output rows. Native transformer configs/provenance remain
in `runtime_audit_rank*.json`. Package versions, exact input digest and full
initial-weight digest must match across all four routes.

Use the native CP1/CP2 drift as the reference scale, and inspect the sampled
trend plus matched-CP output changes. A near-zero native denominator is
reported as an undefined ratio. No arbitrary automatic correctness threshold
or claim about untested long sequences/topologies is added.
