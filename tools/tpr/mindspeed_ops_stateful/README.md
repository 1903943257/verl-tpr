# MindSpeed-Ops Stateful GDN migration patch

This patch kit restores the two changes used by the historical TPR GDN
stateful experiment, without replacing whole MindSpeed-Ops source files.

## Target verified by the user

- MindSpeed-Ops checkout: \`95392732dfd61c07479cc6a076d9cdf1ed17dc2e\`
- Device: Ascend 910B4
- Runtime reported by the user: CANN 8.5.2, torch/torch_npu 2.9.0,
  Triton-Ascend 3.2.0
- Active path: \`arch32\` (\`is_arch35=False\`)

The patcher is intentionally source-layout-aware and fails closed if the target
source no longer matches the inspected layout. It does not claim that this
software combination is officially supported.

## What it migrates

### 1. Stateful GDR gate

Historical reference: MindSpeed-Ops commit
\`51c0811ea5653f1929d8198ee05b5e18580a0d90\`.

- \`mindspeed_ops/api/triton/chunk_gated_delta_rule.py\`
- \`mindspeed_ops/arch32/triton/gdn/chunk_gated_delta_rule_fwd_h.py\`

It removes the two explicit \`initial_state is not None\` rejections. It does not
change the GDR forward/backward math or add a new kernel.

### 2. CausalConv initial-state backward

Historical reference: PR169 commit
\`ffa1e90a3e328327b93bf0d7550861954ecc78e1\`.

- \`mindspeed_ops/arch32/triton/convolution.py\`

It migrates the PR169 changes:
- compute initial-state \`dh0\` contributions only on \`i_t == 0\`;
- read \`dy\`/\`y\` using global \`i_t2\`, with a short-sequence mask;
- move initial-state's contribution to \`dw\` into one extra per-sequence
  reduction slice;
- allocate \`dh0\` without a time-tile dimension in FP32;
- remove the invalid time-tile \`dh0\` store and reduction.

This is the same fix that addresses the historical \`BT < W\` indexing and
\`dh0\` buffer/store range mismatch. It is not merely a one-line index change.

## Apply

Run on the MindSpeed-Ops checkout, passing its root directory:

\`\`\`bash
cd /workspace/MindSpeed-Ops

# First inspect whether the current files match the expected source layout.
bash /workspace/verl-tpr-hybrid-gdn-restore/tools/tpr/mindspeed_ops_stateful/apply.sh . --check

# Apply the patch. Original files are backed up as *.bak.tpr-stateful.
bash /workspace/verl-tpr-hybrid-gdn-restore/tools/tpr/mindspeed_ops_stateful/apply.sh .

# Verify source markers and Python syntax.
bash /workspace/verl-tpr-hybrid-gdn-restore/tools/tpr/mindspeed_ops_stateful/verify.sh .
\`\`\`

The apply script:
- is idempotent for already-patched files;
- refuses to overwrite locally modified target files;
- creates a backup before changing each file;
- refuses partial/unknown CausalConv layouts rather than guessing.

## After applying

Inspect the exact changes:

\`\`\`bash
git diff --check
git diff -- \\
  mindspeed_ops/api/triton/chunk_gated_delta_rule.py \\
  mindspeed_ops/arch32/triton/gdn/chunk_gated_delta_rule_fwd_h.py \\
  mindspeed_ops/arch32/triton/convolution.py
\`\`\`

Then run the historical Stateful GDR operator experiment and the CausalConv
backward tests, followed by the current VERL TPR single-GDN and Qwen3.5 Hybrid
NPU tests.

Static verification is not a substitute for NPU numerical verification.
The patch kit does not build \`mindspeed_ops_C\`, resolve CMake downloads, or
validate ABI compatibility between CANN, torch_npu and Triton-Ascend.
