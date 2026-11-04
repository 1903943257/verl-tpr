# Matched projection shape probe and native MindSpeed Ring comparison

Both checks remain at **P32/S32** and use the same real BF16 checkpoint and
seed 20261104 as the previous diagnostic. This launcher runs only the new
checks; it does not repeat all three full-model attention precision ablations.

```bash
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_P32S32_DEVICES=0,1
export TPR_P32S32_CHECKS_OUTPUT=/workspace/tpr_experiments/projection_native_ring1
bash tests/models/mcore/tpr/parallel/run_projection_native_ring_checks.sh
```

Run Git on the host as before. Git metadata is optional inside the container.
Use a new output directory. `TPR_P32S32_PROFILE=1.7b` is also supported with
the corresponding checkpoint, but keep the same model when comparing runs.

## First-layer projection probe

The first stage executes one ordinary 64-token Qwen forward/backward and hooks
the first layer's actual `linear_proj`. It captures the BF16 input X, output Y,
upstream dY, input gradient dX, and weight gradient dW into CPU memory. Dropout
and activation recomputation are disabled. The actual RowParallelLinear module
is used for the BF16 control; TP is 1. No activation/gradient tensors are saved.

All isolated executions use the **same captured X, unchanged W, and fixed dY**:

- `full`: one 64-row projection and backward.
- `split_connected`: two 32-row projections, concatenated before one backward.
- `split_sequential`: each 32-row projection backward separately, retaining dW
  between backward calls. This tests contribution accumulation separately.
- `repeat_full`: another identical-shape execution, providing a noise baseline.

Three precision modes are evaluated:

| Mode | Linear arithmetic | Tensor and gradient boundary |
| --- | --- | --- |
| `module_bf16` | Actual model projection module | BF16 X/W/Y/dY/dX/dW |
| `fp32_bf16_boundary` | FP32 `F.linear`, output cast to BF16 | BF16 X/W/Y/dY/dX/dW |
| `fp32_reference` | FP32 `F.linear` | Captured BF16 values promoted exactly; FP32 tensors and accumulation |

The local FP32 boundary mode deliberately preserves BF16 parameter-gradient
storage, so it cannot recover dW contributions lost when each branch casts its
gradient back before accumulation. The FP32 reference distinguishes that effect
from changing GEMM shape. This reference uses the same BF16-rounded weights;
it does not silently switch to different checkpoint weight values. HF32 is
disabled for FP32 controls.

`projection/report.json` contains forward/dX/dW errors for the shape and
backward comparisons, errors against one fixed FP32 Full reference, and
`captured_vs_isolated_full` for replay fidelity. If repeat-full or captured-vs-
isolated differences are significant, investigate that baseline before blaming
the 64-vs-32 row count. Input/W/dY hashes and the captured input stride are
recorded. There is no full-model local-FP32 intervention in this isolated probe.

## Four-group CP comparison

The updated CP launcher executes fresh processes in this order:

1. `full_cp1`: 64-token BF16 square CANN reference on one NPU.
2. `native_cp2`: 64-token BF16 Full using ordinary Megatron SelfAttention,
   with a thin explicit core bridge to MindSpeed's
   `ringattn_context_parallel` on two NPUs.
3. `full_cp2`: the existing complete-segment TPR Ring control, followed by
   `tpr_cp2`: the existing Prefix32/Suffix32 TPR Ring replay.

The native arm does **not** construct TPRSelfAttention, enter a TPR attention
context, or use SegmentExecutor. Paired causal CP positions are constructed
independently and checked against the existing global token contract in CPU
tests. Megatron handles normal QKV/Norm/RoPE/projection. Only the core API
adapter is supplied here, avoiding assumptions about which DotProductAttention
constructor/forward patches are installed by a particular MindSpeed build.
Streaming, block selection, online softmax, communication and attention backward
are owned by the actual MindSpeed function; they are not reimplemented here.

The explicit native configuration is **causal SBH Ring, CP2, one inner rank,
no cache, zero dropout and no dedicated send/recv overlap group**. It is not a
claim to reproduce every possible native training option, BNSD configuration,
DDP/main_grad setup, optimizer schedule or PPO objective. Both the model and
stored gradients remain BF16. The fixed linear loss, global 63-query denominator
and weight-gradient reduction precision are the same across controls.

Every layer must execute the native entry once. The actual call counts and
entrypoint are recorded in `cp2/native_cp2/result.json`. Missing/unsupported
MindSpeed APIs cause failure, with no fallback to TPR. The summary
`cp2/comparison.json` now contains all six pairwise comparisons between the
four groups, including:

- `full_cp1_vs_native_cp2`
- `native_cp2_vs_full_cp2`
- `native_cp2_vs_tpr_cp2`

Only compact gradient rel-L2, loss gap and maximum logprob gap are printed;
cosine, gradient max-absolute, gradient norms and CE metrics remain in JSON.
`full_cp2` continues to mean **TPR Ring Full**, while `native_cp2` explicitly
names the native MindSpeed control.

CP1/native/TPR whole-gradient references need about **24 GB temporary disk for
4B**. The split gradient is compared online, not dumped. References are deleted
after successful comparisons unless `TPR_P32S32_KEEP_GRADS=1`. Failures retain
the available artifacts. Each process loads only one model; the projection's
FP32 controls contain only one projection weight, not an FP32 model.

`TPR_P32S32_GRAD_SYNC_DTYPE=bf16` and delayed-dKV FP32 enabled are the defaults,
matching the last diagnostic. The latter does not participate in the native
Full arm, which has no detached prefix relay. Source revision may be supplied
using `TPR_SOURCE_HEAD`; unavailable Git metadata never invalidates the run.

To run only the projection probe, set `ASCEND_RT_VISIBLE_DEVICES` and
`TPR_PROJECTION_OUTPUT` and run `run_projection_shape_probe.sh`. For only the
four-group CP comparison, set `TPR_P32S32_OUTPUT` and `TPR_P32S32_DEVICES` and
run `run_split_p32s32_cp2_parity.sh`. `TPR_P32S32_NATIVE_RING=0` explicitly opts
out of the added native group when reproducing the earlier three-group control.

CPU tests exercise projection math, BF16 accumulation loss, native API routing,
position coverage and normalization. Actual Qwen/MindSpeed NPU integration
still requires the target Ascend environment.
