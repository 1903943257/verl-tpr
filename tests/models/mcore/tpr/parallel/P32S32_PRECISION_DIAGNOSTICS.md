# P32/S32 BF16 forward attribution and CP2 comparison

For the follow-up **matched first-projection shape probe and native MindSpeed
CP2 Ring baseline**, see [PROJECTION_NATIVE_RING_CHECKS.md](PROJECTION_NATIVE_RING_CHECKS.md).

This experiment deliberately stays at **32 prefix + 32 suffix tokens**. All
attention precision modes use the same loaded real Qwen checkpoint, BF16 model,
seed 20261104, synthetic tokens, global positions and fixed linear loss as the
previous quick run. No long sequence, external dataset or optimizer is needed.

## Run the complete sequence

From the repository root in the existing Ascend/MindSpeed container:

```bash
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_P32S32_DEVICES=0,1
export TPR_P32S32_DIAG_OUTPUT=/workspace/tpr_experiments/p32s32_precision_diag1
bash tests/models/mcore/tpr/parallel/run_p32s32_precision_diagnostics.sh
```

The output directory must be new. Keep the same model as the earlier BF16 run
when interpreting the difference. `TPR_P32S32_PROFILE=1.7b` supports the smaller
real checkpoint, but must be accompanied by the corresponding checkpoint path.

The launcher runs sequentially:

1. One NPU: Full / connected split / production Replay, for each of the three
   attention implementations below. A single BF16 model is reused without
   updates; all non-attention fusion settings remain unchanged across modes.
2. One NPU: complete 64-token BF16 Full reference.
3. Two NPUs: native MindSpeed Ring Full, then TPR Ring Full and P32/S32 Replay
   in separate processes. The native reference is now included by default.

Models from different processes/stages are not kept on the NPU simultaneously.
P32/S32 math attention has only tiny attention matrices; the FP32 ablation
promotes attention internals only, not the whole model. FP32 loss/logprob
reductions are retained in all modes.

The one-NPU comparisons need two temporary CPU BF16 gradient snapshots (about
16 GB host RAM for 4B, plus CPU forward snapshots and bounded metric workspace).
They write no raw gradient/activation tensors. The CP1-to-CP2 process boundary
uses three temporary gradient reference sets with the native control, about 24 GB disk for 4B. Those are
deleted after successful comparisons by default; `TPR_P32S32_KEEP_GRADS=1`
retains them for debugging. Only manifest-listed generated files are removed.

## What the attention controls mean

| Mode | Model and QKV boundary | Attention implementation | Output boundary |
| --- | --- | --- | --- |
| `cann` | BF16 | Existing production CANN rectangular adapter | BF16 |
| `math_bf16` | BF16 | Explicit GQA; BF16 QK/PV results, FP32 softmax, BF16 probabilities | BF16 |
| `math_fp32` | BF16 | Already-rounded Q/K/V promoted; FP32 QK/softmax/PV | BF16 before projection |

The previous whole-FP32 test changed parameter/activation precision, replaced
CANN attention with explicit math, and disabled some fusion implementations at
the same time. Its good result therefore does not isolate any one of those
changes. In particular, it does not establish that CANN has a bug or that a
specific BF16 operation caused the original error.

These controls separate a backend change from an explicit attention precision
change more closely. CANN's internal arithmetic is not claimed to match the
`math_bf16` control. `math_fp32` cannot recover precision already lost in BF16
QKV projection, Norm, RoPE or upstream residuals. It tests a possible local
precision boundary, not a production kernel/performance recommendation.

If CANN and explicit BF16 math differ substantially, backend/arithmetic-policy
attribution remains necessary; it is not by itself evidence of a CANN defect.
If explicit BF16 math has a large split error and the FP32 attention control
reduces it, attention precision is a stronger candidate. If the first forward
divergence is already at QKV/Norm/RoPE, increasing attention precision cannot
restore those inputs; inspect that preceding operation with matched inputs.

## Reports

`attention/report.json` contains all three attention-mode summaries. Each mode
also has its own `attention/<mode>/report.json` and:

- `full_vs_split_connected_layers.json`
- `full_vs_split_replay_layers.json`
- `full_<other-mode>_vs_<mode>_layers.json` for cross-mode Full comparisons.

Forward records cover embedding, layer input, input Norm, raw QKV, Q/K Norm,
post-RoPE Q, full K/V, attention core, projection, pre-MLP Norm, MLP FC1/FC2
where exposed by the model, MLP output, layer output, final Norm and logits.
Suffix slices are matched by global positions. Graph-free Push and Replay
records remain separate. There are no backward activation hooks.

Summaries give both the first nonzero difference and first recorded stage
above `atol=1e-5, rtol=1e-4`, in Full forward order. BF16 tensors are copied
to CPU and promoted exactly for metric calculation. These are diagnostic
thresholds, not acceptance criteria. A later layer's difference can be inherited
from earlier stages; its appearance alone does not identify that layer as the
cause. Raw activation tensors are not serialized.

For every Full/connected/Replay pair the reports include complete-parameter
gradient relative L2/cosine/max-absolute errors, fixed linear loss gaps, next-token
logprob errors, and next-token cross-entropy gaps. Query 31 still predicts token
32; only query 63 is excluded. Cross entropy/logprobs are observations, not the
backward objective. This is **not** a PPO/optimizer training run; BF16 describes
computation precision rather than a training metric.

`cp2/comparison.json` compares CP1 Full, native MindSpeed CP2 Full, TPR CP2 Full,
and TPR CP2 Replay. The `full_cp2` name still means **production TPR Ring on one
complete segment**; `native_cp2` identifies the new native API control described
in the follow-up guide. Positions and gradient normalization are matched.
Comparisons use rank 0's finalized weight
gradients; they do not certify cross-replica equality or an optimizer step.

The default CP weight-gradient collective is BF16. For an independent FP32
collective ablation, set `TPR_P32S32_GRAD_SYNC_DTYPE=fp32` and use a new output
directory. This only promotes each gradient for the collective, then casts it
back. Model parameters and stored gradients remain BF16. The actual precision
is recorded; this isolated direct-`parameter.grad` harness does not install a
production optimizer/FP32 `main_grad` wrapper.

`TPR_EXPERIMENT_DELAYED_DKV_FP32` is preserved across all stages and defaults
to `1`, matching the last quick BF16 run. Set it explicitly to `0` only for a
separate delayed-dKV ablation. Local attention promotion and CP gradient-reduce
precision are independent settings.

## Run only one part

For forward/attention attribution on one NPU:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export TPR_SPLIT_BF16_OUTPUT=/workspace/tpr_experiments/p32s32_attention_diag1
TPR_SPLIT_BF16_DIAGNOSTICS=1 TPR_SPLIT_BF16_PREFIX=32 TPR_SPLIT_BF16_SUFFIX=32 \
bash tests/models/mcore/tpr/parallel/run_split_bf16_quick.sh
```

For only the BF16 CP1/CP2 Ring comparison:

```bash
export TPR_P32S32_OUTPUT=/workspace/tpr_experiments/p32s32_cp2_diag1
export TPR_P32S32_DEVICES=0,1
bash tests/models/mcore/tpr/parallel/run_split_p32s32_cp2_parity.sh
```

Git access inside Docker is optional. `TPR_SOURCE_HEAD` may be supplied by the
host; otherwise an unavailable source revision is recorded without failing.
No real NPU integration result is implied by CPU tests of these diagnostics.
