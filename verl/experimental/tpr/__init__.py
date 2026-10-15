"""Optional TPR CP/GDN and historical test/reference modules."""
from importlib import import_module

_EXPORTS = {
    "accumulate_sharded_past_anchor_gradients": "verl.experimental.tpr.legacy.cp.execution_context",
    "all_gather_sequence": "verl.experimental.tpr.allgather",
    "ALLGATHER_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "allgather_cp_rectangular_attention": "verl.experimental.tpr.allgather",
    "AllGatherCPAttentionBackend": "verl.experimental.tpr.legacy.cp.execution_context",
    "AllGatherCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "build_sharded_past_anchors": "verl.experimental.tpr.legacy.cp.execution_context",
    "classify_ring_block": "verl.workers.engine.megatron.tpr.ring",
    "HYBRID_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "hybrid_cp_rectangular_attention": "verl.experimental.tpr.hybrid",
    "HybridCPAttentionBackend": "verl.experimental.tpr.hybrid",
    "HybridCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "HybridCPTopology": "verl.experimental.tpr.hybrid",
    "HybridLocalKVBlock": "verl.experimental.tpr.hybrid",
    "LocalKVBlock": "verl.experimental.tpr.allgather",
    "make_anchored_allgather_cp_backend": "verl.experimental.tpr.legacy.cp.execution_context",
    "make_cached_allgather_cp_backend": "verl.experimental.tpr.legacy.cp.execution_context",
    "make_hybrid_sequence_shard": "verl.experimental.tpr.hybrid",
    "make_ring_sequence_shard": "verl.workers.engine.megatron.tpr.ring",
    "resolve_cp_group": "verl.experimental.tpr.legacy.cp.execution_context",
    "resolve_mindspeed_hybrid_topology": "verl.experimental.tpr.hybrid",
    "resolve_tpr_cp_backend": "verl.experimental.tpr.legacy.cp.backend",
    "ring_cp_attention": "verl.workers.engine.megatron.tpr.ring",
    "RING_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "RingBlockKind": "verl.workers.engine.megatron.tpr.ring",
    "RingCPAttentionBackend": "verl.workers.engine.megatron.tpr.ring",
    "RingCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "RingLocalKVBlock": "verl.workers.engine.megatron.tpr.ring",
    "ShardedPastKVAnchors": "verl.experimental.tpr.legacy.cp.execution_context",
    "TPRCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "ULYSSES_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "ulysses_cp_rectangular_attention": "verl.experimental.tpr.ulysses",
    "UlyssesCPAttentionBackend": "verl.experimental.tpr.ulysses",
    "UlyssesCPBackend": "verl.experimental.tpr.legacy.cp.backend",
}
_MODULES = {
    "allgather_attention": "verl.experimental.tpr.allgather",
    "backend": "verl.experimental.tpr.legacy.cp.backend",
    "engine_runtime": "verl.experimental.tpr.legacy.cp.engine_runtime",
    "execution_context": "verl.experimental.tpr.legacy.cp.execution_context",
    "gdn_state": "verl.experimental.tpr.gdn.state",
    "gdn_tree": "verl.experimental.tpr.gdn.tree",
    "hybrid_attention": "verl.experimental.tpr.hybrid",
    "ring_attention": "verl.workers.engine.megatron.tpr.ring",
    "ulysses_attention": "verl.experimental.tpr.ulysses",
}
__all__ = [
    "ALLGATHER_CP_BACKEND",
    "AllGatherCPAttentionBackend",
    "AllGatherCPBackend",
    "HYBRID_CP_BACKEND",
    "HybridCPAttentionBackend",
    "HybridCPBackend",
    "HybridCPTopology",
    "HybridLocalKVBlock",
    "LocalKVBlock",
    "RING_CP_BACKEND",
    "RingBlockKind",
    "RingCPAttentionBackend",
    "RingCPBackend",
    "RingLocalKVBlock",
    "ShardedPastKVAnchors",
    "TPRCPBackend",
    "ULYSSES_CP_BACKEND",
    "UlyssesCPAttentionBackend",
    "UlyssesCPBackend",
    "accumulate_sharded_past_anchor_gradients",
    "all_gather_sequence",
    "allgather_attention",
    "allgather_cp_rectangular_attention",
    "backend",
    "build_sharded_past_anchors",
    "classify_ring_block",
    "engine_runtime",
    "execution_context",
    "gdn_state",
    "gdn_tree",
    "hybrid_attention",
    "hybrid_cp_rectangular_attention",
    "make_anchored_allgather_cp_backend",
    "make_cached_allgather_cp_backend",
    "make_hybrid_sequence_shard",
    "make_ring_sequence_shard",
    "resolve_cp_group",
    "resolve_mindspeed_hybrid_topology",
    "resolve_tpr_cp_backend",
    "ring_attention",
    "ring_cp_attention",
    "ulysses_attention",
    "ulysses_cp_rectangular_attention"
]


def __getattr__(name):
    if name in _MODULES:
        value = import_module(_MODULES[name])
    elif name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
