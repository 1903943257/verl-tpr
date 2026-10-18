# TPR production architecture (Dense PPO)

This document specifies the **intended production execution contract** of the
VERL-TPR implementation. The codebase targets Ascend NPU + Megatron; it is not
an alternate PPO implementation. See the [development baseline](../tpr_development_baseline.md)
for the reproducible imported starting point.

## Execution overview

```text
VERL Trainer / TrainingWorker
    | original real actor PPO mini-batch and native loss callback
MegatronEngine.forward_backward_batch
    | tpr_enabled after batch_num_tokens and dp_size are established
workers.engine.megatron.tpr.adapter
    | Turn-aware forest, CP/TP compatibility gates, one native finalize
    +-- tree.py       : Qwen3 ChatML turn radix, SegmentObjectiveRef ownership
    +-- objective.py  : segment logits -> VERL original PPO loss
    +-- executor.py   : DFS Push / Visit / Pop, recompute, Prefix dKV relay
    +-- state.py      : detached KV path stack and sharded gradient anchors
    +-- ring.py       : Prefix-aware MindSpeed RingP2P, online softmax, CP runtime
    +-- placement.py  : optional DP forest planning (not trainer dispatch)
    |
models.mcore.tpr.attention / context / rectangular_attention
    | native Megatron QKV, GQA, TP, RoPE; CANN rectangular CP1 or Ring CP>1
    +-- new KV consumed by descendants; differentiable anchors during replay
```

The model package `verl/models/mcore/tpr/` now physically contains only
`attention.py`, `context.py`, `rectangular_attention.py` and its lazy
`__init__.py`. The Engine owns all Segment plans, Prefix State, Shard/Padding,
ModuleSpec installation and TP group agreement. Optional GDN lives in
`experimental/tpr/gdn/`. Tree scheduling,
PPO ownership and distributed execution are Engine concerns and must not be
imported by model-side primitives. The first-party `context.py` transfers
per-forward prefix tensors, RoPE and new per-layer KV between the Executor and
native GPTModel layers using a ContextVar. It is **not** the persistent KV cache.

### Data ownership and gradient flow

1. One original TQ mini-batch is grouped into UID-scoped **Turn-aware trees**;
   only complete ChatML message / captured prompt boundaries can be reused.
2. A physical Segment is computed once per branch; logical PPO tokens retain
   their `(sample_row, response_offset, target_token_id, query_offset)` refs.
   A response token at absolute position `t` is predicted at query `t-1`,
   including when the query physically belongs to its parent Segment.
3. DFS `Push` computes and stores graph-free KV, `Visit` computes a leaf
   objective and propagates dKV to ancestors, and `Pop` releases cached KV,
   recomputes a graph-connected Segment, and relays accumulated descendant dKV.
4. `objective.py` calls the **original VERL PPO loss callback**, reusing the
   native tensor-parallel vocab log-prob and retaining token-mean normalization,
   temperature, old logprob, advantage and response masks. It keeps the PPO
   logprob chunk-checkpoint optimization for long trajectories.
5. Every CP rank must participate in Ring backward, including ranks with no
   locally owned supervised tokens (use a connected zero objective). Forest
   gradients are accumulated under Megatron's no_sync and **finalized once**.

### Parallelism and support boundaries

| Dimension | Formal production scope | Implementation owner |
| --- | --- | --- |
| TP | TP=1/2 | Native Megatron QKV/proj/vocab heads; TPR plan agreement |
| CP | CP=1 or CP=2/4 in tested combinations | CP1 CANN, CP>1 Prefix Ring; MindSpeed P2P |
| SP | Native SP for TP2+CP1 only, with aligned Segment lengths | Megatron; TP2+CP>1+SP is gated off |
| DP | DP=1 | Native VERL DP; DP2 currently **offline parity only** |
| PP, EP | 1 | Other sizes fail explicitly |
| Attention | Dense SelfAttention / GQA | GDN/MoE remain experimental |
| PPO | Vanilla, token-mean actor loss | VERL callback, not a reimplemented loss |
| Recompute/offload | Prefix Push/Pop replay, no native layer recompute | Experimental activation offload isolated |

Ring CP uses the MindSpeed two-range zigzag placement. Logical Segment tokens
are distinct from physically padded tokens (multiple of `2*CP`): a physical
query/KV must not receive an unintended padded loss or attention score.
`ring.py` now owns Ring communication, the physical/logical SequenceShard,
zigzag layout, padding metadata and block validity masks. The old model-side shard implementation has been deleted; all imports now use Engine Ring.
Padding numerical correctness still needs the NPU CP2/CP4 parity gate.

### Optional experiments

`verl/experimental/tpr/` contains AllGather, Ulysses, Hybrid, GDN and activation
offload experiments. Historical CE/Exact reference implementations are held only under
`experimental/tpr/legacy/`, and their tests now import them explicitly.
**Production PPO must never import these experimental backends.** The
model-side `parallel/` directory and duplicate Ring/CP factories are deleted;
Engine `ring.py` is the sole maintained Ring kernel.

### Reusing native components

TPR replaces the SelfAttention ModuleSpec only where needed to accept external
Prefix KV. It does not independently implement tensor-parallel all-reduces,
optimizer steps, policy clipping, process-group lifecycle or normal distributed
synchronization. Future support for TP4, TP+SP+CP, non-Qwen turn formats or DP2
needs separate explicit numeric and training acceptance, not feature flags that
silently disable checks.

### Source reduction guarantees

The Engine owns all Dense training state and Segment plans. Model-side code
contains only Attention/Context/CANN. Historical CE/Exact code is kept as
an explicitly experimental reference, not a second production implementation.
The old model-side compatibility files are physically deleted.

The query/KV coalescing optimization (qkvmerge) is intentionally retained.
The Ring implementation still includes _can_coalesce_prefix_query,
_prefix_full_slices and its online-softmax path. It should become the default
only after numerical and throughput parity. Ring layout, validity mask and
logical/physical Padding share one canonical implementation, but padded CP
backward still requires an NPU parity gate.

### Prefix FA call-count diagnostics (opt-in)

Core Ring accepts physical right-padding and masks invalid tokens. By default,
the Prefix KV coalescing fast path requires an unpadded Prefix. For experiments,
`TPR_RING_COALESCE_PREFIX_PADDED=1` in combination with
`TPR_RING_COALESCE_PREFIX_FULL=1` allows the **valid** zero-copy portion of
each source rank's padded Prefix KV to participate in one FULL FA call.
Communication buffers remain physically padded; gradient slices map back into
their original source buffers. With all three flags enabled,
`TPR_RING_COALESCE_PREFIX_QUERY=1` now merges the physical local Query for
Prefix FULL FA even if the Current Segment is padded. The Query path masks
virtual-Query upstream gradients and dQ; Prefix dK/dV updates only valid
source KV slices. Ranks with an entirely padded local Query chunk fall back
to the two-range Query schedule, and a Prefix source with no real tokens
skips its FULL FA without changing Ring transport. Neither optimization
changes Current causal FA. All Merge optimizations remain **off by default**
and padded Query+KV Merge is experimental until CP2/CP4 NPU correctness passes.

Set `TPR_RING_BLOCK_STATS=1` to print JSON `TPR_RING_BLOCK_STATS` records on
CP rank zero, for the first transformer layer of each executed Segment.
Forward and Backward are reported separately, including the effective
`fa_prefix_kv_merged`, `fa_prefix_query_merged`,
`fa_prefix_unmerged_reference` and `fa_prefix_calls_saved`. A requested
flag with `fa_prefix_calls_saved=0` is **not** a Merge performance result.
Counters are untimed, do not synchronize the NPU, and do not retain remote KV.

CPU gate:

```bash
python -m pytest -q tests/models/mcore/tpr/parallel/test_ring_cp_attention_contract.py
```

NPU padded-KV-only gates with explicit localhost rendezvous:

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9 torchrun --master_addr=localhost --master_port=29505 \
  --nproc_per_node=2 -m pytest -xvs \
  tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py \
  -k padded_prefix_full_coalescing_off_on
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 torchrun --master_addr=localhost --master_port=29506 \
  --nproc_per_node=4 -m pytest -xvs \
  tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py \
  -k padded_prefix_full_coalescing_off_on
```

To exercise padded **joint Query+KV Merge** on a real Actor capture, set:

```bash
TPR_RING_BLOCK_STATS=1 \
TPR_RING_COALESCE_PREFIX_FULL=1 \
TPR_RING_COALESCE_PREFIX_PADDED=1 \
TPR_RING_COALESCE_PREFIX_QUERY=1 \
TPR_OFFLINE_RUN_MODES=tpr \
TPR_TREE_POLICY=qwen_turn \
TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_ring_qkv_padded_on \
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh
```

For CP4 and a single Prefix with all four source ranks owning real
KV, enabling only padded KV Merge changes a leaf's Prefix FA calls
from 16 to 8. Adding padded Query Merge reduces them further to 4
(total 25 -> 13 including nine unchanged Current FA calls).
Enable `TPR_RING_BLOCK_STATS=1` to **verify the actual call counts**,
not infer eligibility from environment variables. A low-level CPU contract
does not establish Ascend NPU numerical parity.

CP2/CP4 NPU Query+KV OFF/ON verification:

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9 torchrun --master_addr=localhost --master_port=29507 \
  --nproc_per_node=2 -m pytest -xvs \
  tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py \
  -k padded_prefix_query_kv_coalescing_off_on
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 torchrun --master_addr=localhost --master_port=29508 \
  --nproc_per_node=4 -m pytest -xvs \
  tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py \
  -k padded_prefix_query_kv_coalescing_off_on
```

For the existing eight-rank captured PPO replay, enable
`TPR_RING_BLOCK_STATS=1` and compare a baseline with
`TPR_RING_COALESCE_PREFIX_FULL=0` against an experimental run with
`TPR_RING_COALESCE_PREFIX_FULL=1`,
`TPR_RING_COALESCE_PREFIX_PADDED=1` and
`TPR_RING_COALESCE_PREFIX_QUERY=0`. Use separate output directories. The
existing BF16 reproducibility/gradient gates must be interpreted separately
from whether the number of actual FA launches decreased.

### Validation gate

Before a production merge, compare **old TPR vs new TPR on identical actor
captures**, with TP1CP1, TP2CP4 and the explicitly gated offline TP2CP2DP2
case where available. Check Turn forests and refs, logits/logprobs, PPO loss,
parameter grad relative-L2, optimizer multi-step update, and peak NPU memory.
Do not mix this architectural migration with the existing Full-vs-TPR BF16
numerical drift investigation or enable Fixed Tile. A passed AST/import smoke
test does **not** mean end-to-end PPO parity passed.

CPU structural check:

```bash
python -m pytest -q tests/models/mcore/tpr/unit/test_production_architecture.py
```

NPU integration tests require a configured Ascend 910 / MindSpeed / Megatron
stack and authentic actor captures. Record every skipped gate explicitly.

### Ring maintenance branch (2026-10-18)

Continue Ring implementation, QKV experiments and their acceptance tests on
`refactor/tpr-ring-layout-validation-main-20261016`. The canonical source is
`verl/workers/engine/megatron/tpr/ring.py`.

The gradient/contract fixes from `75b53c15` are adapted to this Engine-owned
implementation: FP32 dQ accumulation, shared CANN visibility parameters,
single-sequence TND statistics conversion and shared K/Q slice metadata for
V/dO. The prepared per-Segment layout, padded joint Query+KV merging, valid
KV gradient scatter, virtual Query masking, native whole-Ring reference and
effective FA counters are retained. FULL/QUERY/PADDED merging remains opt-in
with all three flags defaulting to zero.

The historical experimental AllGather/Ulysses references remain available on
this branch. The old Ring-only branch's removal of those test parameters does
not apply to these independent references. Deep-Prefix CPU and NPU regression
tests supplement the existing padded CP2/CP4 gates; CPU validation does not
establish CANN/HCCL numerical or throughput acceptance.


### Experimental Fast Root Ring (2026-10-18)

Set `TPR_ROOT_NATIVE_RING_PHASE` to `both` (default), `push`, or `pop`
when `TPR_ROOT_NATIVE_RING=1`. Push-only switches graph-free Root KV
creation; Pop-only switches the recomputed Root forward/backward. These
are diagnostic ablations, not independent production configurations.
Record the phase in the output directory name, and keep the same captured
Actor batch, qwen_turn topology, Prefix Merge settings and model weights.

Use `TPR_ROOT_NATIVE_RING=1` to opt in; default is `0`. Only the actual tree
Root (`segment_id == plan.root_id`, no parent) switches to MindSpeed's native
`ringattn_context_parallel()` via the existing `ordinary_ring_cp_attention()`.
An empty Prefix alone is NOT a Root signal. Internal and Leaf Segments retain
the existing TPR Prefix/Current Ring schedule.

Root Push keeps its graph-free cached K/V; Root Pop recomputes graph-connected
K/V and accepts relayed descendant dK/dV without changing tree ownership.
Native Root Ring uses only trailing physical padding. Virtual Query output
rows are zeroed before returning so even a nonzero padded upstream dO has
zero Root graph gradient. Native causal visibility excludes virtual Keys
from all real Queries. No new Ring transport or backward implementation.

CPU routing and padding contract:

```bash
python -m pytest -q tests/models/mcore/tpr/parallel/test_ring_cp_attention_contract.py -k native_root
```

NPU Root Push/Pop numerical checks (run sequentially):

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9 torchrun --master_addr=127.0.0.1 --master_port=29797 \
  --nproc_per_node=2 -m pytest -xvs \
  tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py \
  -k native_root_ring_matches_tpr_root_with_relayed_kv_gradients
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 torchrun --master_addr=127.0.0.1 --master_port=29797 \
  --nproc_per_node=4 -m pytest -xvs \
  tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py \
  -k native_root_ring_matches_tpr_root_with_relayed_kv_gradients
```

Next gate: same authentic Actor capture with `TPR_ROOT_NATIVE_RING=0` vs `1`,
keeping Prefix Merge settings fixed. Compare Root Push/Pop time, end-to-end
loss/logprob, full parameter gradients, actual FA calls, and memory. Kernel
parity does not establish complete PPO/optimizer acceptance.


### Controlled single-card Qwen3-4B PPO Full vs TPR (2026-10-18)

Run `bash tests/models/mcore/tpr/parallel/run_real_actor_cp1_short_full_tpr.sh`
from the synced Docker worktree with physical NPU 8 available. The script
uses fixed `127.0.0.1:29797`, a 2048-token cap per trajectory, and the
existing real Actor PPO capture rows (4,6). It produces three independent
checkpoint-loaded TP1/CP1/DP1 runs:

- `full`: whole-sequence GPT forward/backward **without** TPR context,
  using the controlled square-CANN CP1 core; **not** native VERL Engine.
- `one_node`: two independent TPR one-Root-leaf plans; no shared KV.
- `shared`: one `qwen_turn` tree, Root Push / two siblings / Root Pop.

The derivation keeps an identical trailing prompt window and selects an
original supervised response window around the first sibling divergence.
It never synthesizes PPO labels or alters values of the retained
`old_log_probs`, `advantages`, response/loss masks or temperatures.
**Left-truncating the prompt and shifting the response window changes
conditioning**: this is a deterministic controlled numerical comparison,
not equivalent to the original online PPO objective. The output capture
audit records original row IDs, offset, lengths and fingerprints. Scripts
reject missing branches or missing PPO supervision rather than claiming
reuse without an actual shared Root.

The comparator saves `comparison.json` with Full vs one-node, one-node vs
shared and Full vs shared token logprob/Loss metrics, canonical sampled
parameter gradients, complete Norm-only gradients and global grad norms.
Neither the 64-sample canonical witness nor the Norm-only vector is a
complete all-parameter gradient L2. No optimizer step is executed. Full,
one-node and shared use the same Qwen3-4B BF16 weights, VERL vanilla PPO
objective and optional chunked logprob computation; dropout=0 and model
recompute disabled. Root Native Ring is forced OFF: CP1 cannot validate
Native Ring correctness by itself.

CPU capture contract:

```bash
python -m pytest -q tests/models/mcore/tpr/parallel/test_cp1_short_capture_cpu.py
```

After CP1 is well characterized, use the established TP2CP1 and TP2CP4
replay matrices to isolate tensor/context parallelism and then return to
Fast Root Native Push-only drift on the original 4B Actor sibling capture.
