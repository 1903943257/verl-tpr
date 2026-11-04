# Full / connected split / replay split FP32 attribution

For the subsequent **P32/S32 BF16 layer-forward attribution, local attention
FP32 ablation, and BF16 CP2 Ring comparison**, see
[P32S32_PRECISION_DIAGNOSTICS.md](P32S32_PRECISION_DIAGNOSTICS.md).

The matched first-layer projection and native MindSpeed Ring follow-up is in
[PROJECTION_NATIVE_RING_CHECKS.md](PROJECTION_NATIVE_RING_CHECKS.md).

Run this from the existing delayed-dKV experiment branch. It uses one real
Qwen3 checkpoint, one synthetic trajectory, and three executions with unchanged
weights. Default length is 32 prefix + 32 suffix tokens. No external capture or
dataset is needed.

| Arm | Forward | Prefix graph |
| --- | --- | --- |
| `full` | One whole-sequence forward | Ordinary autograd |
| `split_connected` | Prefix then suffix | Live prefix KV graph |
| `split_replay` | Production `SegmentExecutor` Push/Leaf/Pop | Detached KV and Replay |

All arms use FP32 parameters, buffers, observed activations, and gradients.
Both square and rectangular attention use the **same explicit FP32 GQA math**
with a global-position causal mask; CANN fusion attention is bypassed. Autocast
and HF32 matmul are disabled/checked. This is a precision/semantic reference,
not a throughput benchmark or a claim to exercise the BF16 CANN kernel.

The fixed linear loss uses every query except the last, including the query
immediately before the split. Its upstream logit gradient and global denominator
are identical across arms. No PPO clipping, optimizer, Ring, SP, DP, checkpoint
recomputation, or cross-trajectory sharing is included.

## Run on one NPU

### Quick BF16 P32/S32 comparison (summary only)

Use the **same checkpoint/profile and device** as the FP32 experiment:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_SPLIT_BF16_OUTPUT=/workspace/tpr_experiments/split_bf16_p32s32_quick1
bash tests/models/mcore/tpr/parallel/run_split_bf16_quick.sh
```

This runs `full`, `split_connected`, and production `split_replay` with BF16
parameters/QKV/parameter gradients and the existing **CANN attention adapter**.
It reuses the FP32 experiment's execution code, seed 20261104, token generation,
global loss coefficients, prefix boundary query, and denominator. The linear
loss reduction stays FP32. Autocast is disabled; the model itself is BF16.
It uses direct `parameter.grad`, as in the short BF16 CP2 test, without a
production optimizer or FP32 `main_grad` wrapper. Existing BF16 fusion settings
are reported rather than represented as identical to the FP32 reference.

Only three aggregate gradient comparisons (`rel_l2`, cosine, max absolute
difference, norms, elements, and loss gap) are printed. Only `report.json` and
the sibling console log are written: no parameter lists, gradient shards, layer
traces, or per-tensor progress. Exact full-parameter comparisons still require
reading every gradient. Two temporary CPU BF16 gradient snapshots require
about **16 GB host RAM for 4B** (about 7 GB for 1.7B), plus one current tensor
and bounded FP64 metric workspace. There are no large gradient files to compare
offline. This is an error measurement, without automatic accuracy acceptance.

Defaults are P32/S32 and profile `4b`. `TPR_SPLIT_BF16_PREFIX`,
`TPR_SPLIT_BF16_SUFFIX`, and `TPR_SPLIT_BF16_PROFILE` override them; if unset,
the corresponding existing `TPR_SPLIT_FP32_*` settings are inherited.
`TPR_EXPERIMENT_DELAYED_DKV_FP32` defaults to `1` in this launcher to match the
FP32 launcher and can be set to `0` for a separate legacy-dKV run with a new
output directory. Its actual value is recorded. Q/K/V and parameter gradients
remain BF16 with either setting.

Compare the reported token hash/profile with the previous FP32 report. A large
BF16 split error here shows that short length alone does not prevent the issue.
A small one only establishes this short synthetic case; long/real-data tests
remain necessary. Unlike the FP32 run, this uses CANN attention and BF16 fusion
settings, so this experiment alone cannot separate arithmetic precision from
attention backend effects.

### FP32 detailed reference

Git is optional in both FP32 and BF16 experiment containers. If the source is
mounted from a host worktree whose `.git` pointer cannot resolve in Docker,
the report records `source_head: null`, `source_head_origin: "unavailable"`,
and the captured Git error; this does not fail the numerical experiment or
print a fatal Git error to the console. Missing Git and Git timeouts are also
handled. A host may explicitly pass its revision through the container's
`TPR_SOURCE_HEAD` environment variable; it is labeled `"environment"` and is
not claimed to have been verified against the mounted files. No Git operation
inside the container is needed in that case.

If an older BF16 runner failed only at `git rev-parse HEAD` **after all three
`SPLIT_BF16_ARM_DONE` messages**, the already-written `report.json` contains
the complete gradient/loss comparisons and can be used without repeating the
NPU run. Only the final environment metadata and console summary are missing.

Use the same Ascend/MindSpeed/Megatron environment as the existing experiments:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # choose an available physical device
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_SPLIT_FP32_OUTPUT=/workspace/tpr_experiments/split_fp32_p32s32_run1
bash tests/models/mcore/tpr/parallel/run_split_fp32_attribution.sh
```

The output path must be new. Full FP32 Qwen3-4B weights and gradients require
roughly 32 GB device memory before activations/workspace. Three complete FP32
gradient dumps require roughly 48 GB disk, plus traces. The same model is reused
between arms and its weights are checked for in-place mutation; gradients are
streamed to disk, not held for all arms on the NPU simultaneously.

`TPR_SPLIT_FP32_PREFIX` / `TPR_SPLIT_FP32_SUFFIX` change the split, with a maximum
total of 512 tokens. `TPR_SPLIT_FP32_PROFILE=1.7b` permits the existing real
Qwen3-1.7B checkpoint profile for a smaller run. Synthetic inputs test execution
equivalence; a successful short run does not certify long real PPO trajectories.

## Outputs and interpretation

- `report.json`: losses, three pairwise full-parameter gradient comparisons,
  actual dtype/precision settings, token IDs/hash, checkpoint, and source SHA.
- `full_vs_split_connected_layers.json`, `full_vs_split_replay_layers.json`:
  aligned forward/gradient errors for embedding, each layer input, QKV, Q/K
  norms, post-RoPE Q and attention K/V, math attention output, attention output,
  MLP, layer output, final norm, and logits (where the real model exposes hooks).
- `<arm>/trace.pt`: CPU tensor snapshots. Graph-free Push and gradient-carrying
  Replay are separate records; `grad_enabled` identifies the distinction.
- `<arm>/manifest.json` and `grad_*.pt`: every trainable parameter's FP32 gradient.

Layer summaries report the first **recorded stage** above configurable
`TPR_SPLIT_FP32_ATOL` / `TPR_SPLIT_FP32_RTOL` (defaults 1e-5/1e-4). This is a
diagnostic threshold, not automatic model-equivalence acceptance. Suffix K/V
attention inputs include the prefix; their gradients mix partial contributions,
so those K/V input hooks are forward-only. QKV/module/layer gradients are recorded.

- Full and connected split differ: inspect the earliest forward stage and mask,
  positions, arithmetic, or model kernel behavior.
- Connected split agrees but Replay differs: inspect detach/gradient relay/Replay.
- These FP32 arms agree while the existing BF16 CANN arms differ: precision and
  kernel attribution remains necessary. This alone does not distinguish ordinary
  BF16 math/GEMM rounding from a CANN attention-specific effect.

The CPU tests use a small two-layer FP32 model with GQA, RoPE, norms and an MLP,
but the **actual production executor** for Replay. They also inject a dKV fault
to verify that the recorder detects backward divergence with unchanged forward.

```bash
python -m tests.models.mcore.tpr._mindspeed_first_pytest -q -x \
  tests/models/mcore/tpr/unit/test_split_fp32_attribution_cpu.py
```

That repository bootstrap requires the configured MindSpeed environment even
for the CPU tests. A CPU-only reviewer can bootstrap optional package initializers
as in the existing local audit; this does not validate real NPU integration.


## Recover an existing FP32 three-arm run (no NPU)

If all three `manifest.json`, `trace.pt` and `grad_*.pt` sets were
written, but the run failed during `torch.load(Path(...), mmap=True)`, do
**not** start a new 48GB run. On this experiment branch use:

```bash
python -m tests.models.mcore.tpr.parallel.compare_split_fp32_saved \
  /workspace/tpr_experiments/split_fp32_p32s32_run1 \
  --prefix 32 --suffix 32
```

The recovery reuses existing per-layer JSONs, reconstructs the fixed linear
loss from saved grad-enabled logits, streams the three all-parameter
comparisons on CPU, and writes `report.json`. Loss reconstruction may differ
slightly in final sum order; it is labelled as recovered. The path is passed
to `torch.load` as a string for the installed `torch_npu` serialization
wrapper. Do not delete the gradients before `report.json` exists.

## Separate BF16 CP1 vs Ring CP2 P32S32 parity

The **existing FP32 math-attention gradient dumps cannot be used as a
reference for BF16 Ring CP2**. Test matched BF16 across all arms instead:

```bash
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_P32S32_DEVICES=0,1
export TPR_P32S32_OUTPUT=/workspace/tpr_experiments/p32s32_cp2_bf16_run1
bash tests/models/mcore/tpr/parallel/run_split_p32s32_cp2_parity.sh
```

This runs sequentially:
- Full CP1, one ordinary 64-token square CANN forward/backward
- Full CP2, the **same** 64-token sequence on 2-NPU TPR Ring CP
- TPR CP2, a Prefix32/Suffix32 two-segment Push/Leaf/Pop replay on the
  same two ranks and weights

All modes use the same deterministic tokens/checkpoint and the same
global-position linear loss as the FP32 attribution test, computed with BF16
model parameters and FP32 loss arithmetic. The 63 next-token logprobs are
measured separately and are **not** the backward objective. All BF16
parameter gradients are compared after CP2 SUM/average, using BF16 by default.
`TPR_P32S32_GRAD_SYNC_DTYPE=fp32` explicitly selects FP32 collective staging
and a BF16 result instead. The actual setting is recorded.
The result `comparison.json` contains all three pairwise gradient rel-L2,
cosine, max-abs, loss gap, and next-token logprob error.

**Full CP1, Full TPR CP2, and the enabled native MindSpeed CP2 reference**
parameter gradients are temporarily written:
roughly 16GB total for 4B, or 24GB with the now-default native MindSpeed
reference, deleted after success unless
`TPR_P32S32_KEEP_GRADS=1`. The TPR CP2 gradients are compared directly, not
dumped as a third set. No optimizer update. CP1 square CANN and CP2 Ring may differ from
ordinary BF16 kernel rounding; report differences, do not assert bitwise
equivalence. This is a separate BF16 kernel/topology diagnostic, not another
arm of the original all-FP32 math-attention experiment.

CPU sanity gate (MindSpeed-first bootstrap in configured development env):

```bash
python -m tests.models.mcore.tpr._mindspeed_first_pytest -q -x \
  tests/models/mcore/tpr/unit/test_split_saved_and_cp2_loss_cpu.py
```
