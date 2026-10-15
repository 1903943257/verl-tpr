# TPR Ring-only CP branch

Branch: `refactor/tpr-cp-ring-only-qkvmerge-20261015`. Based on main at `e67aa93ca345aace1c621afceb49f608e2d280b6`.
This branch deliberately has no changes to main, no PR update, and no fallback to another CP Attention implementation.

## Production boundaries

- CP=1: existing local rectangular causal attention.
- CP>1: `verl/models/mcore/tpr/parallel/ring.py` only. It reuses MindSpeed RingP2P and keeps local-KV streaming, owner-gradient reverse Ring, segment-local FULL/CAUSAL/SKIP visibility, multi-ancestor Prefix and physical Padding.
- Prefix KV and Q merging are enabled by default **on this experimental branch only**; `TPR_RING_COALESCE_PREFIX_FULL=0` and `TPR_RING_COALESCE_PREFIX_QUERY=0` remain available for A/B numerical gates. Padded or ineligible blocks retain the original unmerged schedule.
- AllGather, Ulysses, and Hybrid Ulysses+Ring CP Attention policies are removed, not silently mapped to Ring. Select `tpr_cp_backend=ring` (or `context_parallel_algo=megatron_cp_algo`) for CP>1. Unsupported configurations fail early.
- CP2 stateful GDN is **not** Ulysses Attention. It lives in `gdn_cp.py` with state-tree test support in `gdn_tree.py`.
- CP Anchor state management lives in `kv_stack.py`, Engine CP glue in `megatron_cp.py`.
- The whole-sequence MindSpeed native Ring oracle lives under tests, not the production CP module. Test-only AllGather oracle is used only in historical whole-FA diagnostics, not the TPR engine.

## Follow-up numerical acceptance (NOT yet run on NPU in this branch)

1. CPU ring-only configuration/padding and streaming/coalescing dispatch:
   `pytest -q tests/models/mcore/tpr/parallel/test_ring_only_cp_policy.py tests/models/mcore/tpr/parallel/test_cp_padding_contract.py tests/models/mcore/tpr/parallel/test_engine_cp_runtime_contract.py tests/models/mcore/tpr/unit/test_ring_coalescing_schedule.py tests/models/mcore/tpr/unit/test_ring_streaming_transport.py`.
2. CP2 and CP4 Ring: `torchrun --standalone --nproc_per_node=2 -m pytest -x -s tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py`; repeat with `--nproc_per_node=4`.
3. `TPR_RUN_RING_COALESCING_TREE=1`: `test_ring_coalescing_tree_npu.py` on CP2/4; verify loss/logprob, all parameter grads, nonzero Prefix dK/dV and repeated backward.
4. NPU equivalence on actual Qwen3 and representative TP2/CP2/DP2 PPO trajectories with the **same capture**, compare merged vs explicitly unmerged and Full reference; do not change tolerance or use Fixed Tile.
5. Measure E2E throughput, FA count, peak allocated/reserved before considering merge ready for production/main.

Historical tests that exclusively exercised retired AllGather/Ulysses/Hybrid CP policies have been deleted on this branch only. Other old diagnostic scripts remain as historical tools and are not part of the Ring-only acceptance matrix. A passing syntax check or CPU scheduling test does **not** imply NPU numerical correctness.
