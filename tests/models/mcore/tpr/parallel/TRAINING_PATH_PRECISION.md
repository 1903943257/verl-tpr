# 1K+1K production training precision controls

This experiment uses the existing `MindspeedEngineWithLMHead` for **Base CP1,
Base CP2, TPR CP1 and TPR CP2**, in four separate processes. It does not use
`_qwen3_tp_checkpoint`, the square CANN reference adapter, the explicit native
Ring API bridge, a custom loss implementation, a custom gradient finalize,
or post-hoc BF16-to-FP32 gradient promotion.

## Run

Pull this branch on the host, then run inside the normal training container:

```bash
export TPR_TRAIN_MODEL_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_TRAIN_DEVICES=0,1
export TPR_TRAIN_OUTPUT=/workspace/tpr_experiments/p1024s1024_training1
bash tests/models/mcore/tpr/parallel/run_training_path_precision.sh
```

Use an existing local dense Qwen3 checkpoint with its tokenizer. The model
architecture comes from the checkpoint through the real training bridge;
there is no reduced/random substitute. Qwen3-1.7B is recommended to limit
memory. Qwen3-4B is supported by the same entry, but requires more memory.
The output directory must be new. Git is optional inside the container.

The default bridge is the existing Ascend production choice, legacy vanilla
`mbridge` (`TPR_TRAIN_VANILLA_MBRIDGE=1`). Set it to `0` only if the target
training environment uses Megatron-Bridge. All four modes use the same choice.

## Data and objective

The production tree builder compresses a single, nonbranching trajectory
into one whole segment. To exercise actual production sharing without
patching the planner, the input has **two trajectories**, each P1024+S1024,
with exactly the same 1024-token prompt and different 1024-token responses.
The ordinary `token_lcp` planner must create one prefix plus two suffixes,
all length 1024. The gate rejects a degenerate whole-sequence TPR run.

The Base modes process the same two rows with microbatch size one. TPR
shares the prefix and processes leaves one at a time. The first response
token is supervised by the last prefix query and is included in every mode.

Base CP1's ordinary production inference supplies the old logprobs once.
Those old logprobs, deterministic nonzero advantages, token IDs, masks and
temperature are then frozen for every mode. Backward uses the **unmodified
VERL vanilla token-mean PPO loss**, with clipping enabled, entropy/KL off.
These are synthetic actor inputs, not a rollout-quality or PPO convergence
experiment. Reported next-token CE is an observation, not the backward loss.

## Actual training path and precision

The engine constructs the real bridge/model, BF16 parameter wrapper,
Megatron DDP, DistributedOptimizer and native training hooks. Base training
uses the native pipeline schedule. TPR uses the existing Forest adapter.
Both invoke the registered native `finalize_model_grads` exactly once.
The experiment stops after finalize, **before clipping or optimizer update**.
The optimizer is genuinely constructed; no Adam moments are allocated by
an update. TP=PP=DP=1, CP=1/2, dropout and activation recomputation are off.

All modes use the same settings:

| Item | Default and runtime checks |
|---|---|
| Parameters/activations | BF16 |
| Linear wgrad fusion | Enabled; every native linear must inherit it |
| Fused wgrad destination | Real FP32 `parameter.main_grad`; native backward completion flags required |
| DDP gradient reduction | FP32 |
| Optimizer | Real Megatron DistributedOptimizer |
| Delayed prefix dKV experiment | Off, matching production default |
| Local FP32 projection/attention experiments | Not installed |

`TPR_TRAIN_GRAD_ACCUM_FUSION=0` runs the same four routes with the native
nonfused wgrad path. It still has FP32 DDP buffers, but per-call wgrad can
round to BF16 before the DDP hook adds it to `main_grad`. Run this as a
**separate output directory**; do not mix its references with fused results.
`TPR_EXPERIMENT_DELAYED_DKV_FP32=1` is also a separate explicit ablation.
Neither option installs a replacement operator or gradient hook.

Read-only hooks observe the actual core-attention instances registered in
the model, including MindSpeed TE implementations whose entry is simply
named `forward`. Profiling must also observe a native fused-attention call
inside those native core modules; a similarly named function or an unrelated
operator call elsewhere does not qualify an adapter as the Base path.
Base CP2 must additionally enter MindSpeed Ring through its own core modules.
No attention implementation is replaced and no adapter fallback is permitted.
Backward may run on autograd threads outside Python's profiler; therefore
native per-weight `grad_added_to_main_grad` flags and the actual FP32
buffers, not a Python call count alone, validate fused wgrad completion.

Every rank writes `<mode>/runtime_audit_rankN.json` before training and
updates it immediately after forward/backward, **before route validation**.
It includes the actual core classes, forward source files, input shapes,
per-core call counts, observed native entrypoints, gradient configuration
and package versions. A failed forward or route check records its error
there and prints `TRAIN_PRECISION_AUDIT_FAILED` with the path. This report
is available even when no gradient reference/result was produced.

If a run failed the older function-name-only check, pull the fix and use a
fresh output directory. The old failed directory is preserved; its gradients
were not accepted as a reference. Retain the same checkpoint and seed.

TPR currently requests `force_all_reduce=True` in its existing native
finalizer. Base DistributedOptimizer ordinarily uses reduce-scatter.
**Only native optimizer-owned parameter ranges are read** in either mode.
Reading every rank's full `main_grad` after Base reduce-scatter would mix
valid reduced slices with stale tails. The observer verifies that owned
slices cover every parameter exactly once, excluding buffer padding.
It never modifies or communicates training gradients.

## Outputs and acceptance

Terminal output includes one completion line per mode and aggregate pair
metrics. `comparison.json` contains:

- Base CP1 vs Base CP2: the native numerical-change reference.
- Base CP1 vs TPR CP1 and Base CP1 vs TPR CP2: total differences.
- Base CP2 vs each TPR mode, including the matched CP2 comparison.
- TPR-to-native gradient-error ratios using the same Base CP1 denominator.
- Actual main-grad/fusion/DDP configuration, core-forward provenance,
  call counts, package versions, full initial-weight SHA256 and actor-input
  SHA256, PPO loss, response logprobs/CE and per-rank peak allocated memory.

There is no hardcoded FP32-style closeness gate. Acceptance compares TPR's
error with native CP's error on the same workload. A zero native error
produces a null ratio and requires inspection rather than division by zero.
This first run has one input seed; it does not certify every workload.

Only the two Base gradient references are temporarily saved in bounded
FP32 blocks (about 14 GB disk for 1.7B or 33 GB for 4B). Candidate gradients
are compared online, with CPU NumPy FP64 metric accumulation. There are no
per-parameter gradient console dumps or layer traces. Reference blocks are
removed only after all four modes succeed; failures retain them. Set
`TPR_TRAIN_KEEP_GRADS=1` to retain successful references too.

## Local validation limits

CPU tests exercise the real token-LCP planner and prompt boundary, bounded
gradient comparisons against a FP64 oracle, optimizer shard ownership,
rejection of BF16 gradient substitutes, missing/duplicate coverage and
native fusion completion flags. They do not execute the NPU training stack.
Run the launcher in the target container for that validation.
