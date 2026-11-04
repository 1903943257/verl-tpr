# Full / connected split / replay split FP32 attribution

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
