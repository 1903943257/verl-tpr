"""Ring-only context-parallel Attention backend for TPR.

AllGather/Ulysses/Hybrid attention backends are intentionally unsupported.
"""

from ..kv_stack import (
    ShardedPastKVAnchors,
    build_sharded_past_anchors,
    accumulate_sharded_past_anchor_gradients,
)
from .ring import (
    RING_CP_BACKEND,
    RingBlockKind,
    RingCPAttentionBackend,
    RingCPBackend,
    RingLocalKVBlock,
    classify_ring_block,
    make_ring_sequence_shard,
    ring_cp_attention,
    resolve_tpr_cp_backend,
    resolve_cp_group,
)

__all__ = [
    "RING_CP_BACKEND",
    "RingBlockKind",
    "RingCPAttentionBackend",
    "RingCPBackend",
    "RingLocalKVBlock",
    "ShardedPastKVAnchors",
    "classify_ring_block",
    "make_ring_sequence_shard",
    "ring_cp_attention",
    "resolve_tpr_cp_backend",
    "resolve_cp_group",
    "build_sharded_past_anchors",
    "accumulate_sharded_past_anchor_gradients"
]
