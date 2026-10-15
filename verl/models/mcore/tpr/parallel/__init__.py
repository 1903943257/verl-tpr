"""Lazy TPR CP compatibility exports.

Importing model primitives must not eagerly initialize Engine, experimental CP
backends or optional MindSpeed kernels. Prefer explicit submodule imports in
production code; these exports only preserve existing public API/test imports.
"""
from importlib import import_module

_EXPORTS = {
    "accumulate_sharded_past_anchor_gradients": ".execution_context",
    "all_gather_sequence": ".allgather_attention",
    "ALLGATHER_CP_BACKEND": ".backend",
    "allgather_cp_rectangular_attention": ".allgather_attention",
    "AllGatherCPAttentionBackend": ".execution_context",
    "AllGatherCPBackend": ".backend",
    "build_sharded_past_anchors": ".execution_context",
    "classify_ring_block": ".ring_attention",
    "HYBRID_CP_BACKEND": ".backend",
    "hybrid_cp_rectangular_attention": ".hybrid_attention",
    "HybridCPAttentionBackend": ".hybrid_attention",
    "HybridCPBackend": ".backend",
    "HybridCPTopology": ".hybrid_attention",
    "HybridLocalKVBlock": ".hybrid_attention",
    "LocalKVBlock": ".allgather_attention",
    "make_anchored_allgather_cp_backend": ".execution_context",
    "make_cached_allgather_cp_backend": ".execution_context",
    "make_hybrid_sequence_shard": ".hybrid_attention",
    "make_ring_sequence_shard": ".ring_attention",
    "resolve_cp_group": ".execution_context",
    "resolve_mindspeed_hybrid_topology": ".hybrid_attention",
    "resolve_tpr_cp_backend": ".backend",
    "ring_cp_attention": ".ring_attention",
    "RING_CP_BACKEND": ".backend",
    "RingBlockKind": ".ring_attention",
    "RingCPAttentionBackend": ".ring_attention",
    "RingCPBackend": ".backend",
    "RingLocalKVBlock": ".ring_attention",
    "ShardedPastKVAnchors": ".execution_context",
    "TPRCPBackend": ".backend",
    "ULYSSES_CP_BACKEND": ".backend",
    "ulysses_cp_rectangular_attention": ".ulysses_attention",
    "UlyssesCPAttentionBackend": ".ulysses_attention",
    "UlyssesCPBackend": ".backend"
}

__all__ = [
    "ALLGATHER_CP_BACKEND",
    "HYBRID_CP_BACKEND",
    "RING_CP_BACKEND",
    "ULYSSES_CP_BACKEND",
    "AllGatherCPBackend",
    "AllGatherCPAttentionBackend",
    "HybridCPAttentionBackend",
    "HybridCPBackend",
    "HybridCPTopology",
    "HybridLocalKVBlock",
    "LocalKVBlock",
    "RingBlockKind",
    "RingCPAttentionBackend",
    "RingCPBackend",
    "RingLocalKVBlock",
    "ShardedPastKVAnchors",
    "TPRCPBackend",
    "UlyssesCPAttentionBackend",
    "UlyssesCPBackend",
    "accumulate_sharded_past_anchor_gradients",
    "all_gather_sequence",
    "allgather_cp_rectangular_attention",
    "build_sharded_past_anchors",
    "classify_ring_block",
    "hybrid_cp_rectangular_attention",
    "make_hybrid_sequence_shard",
    "make_ring_sequence_shard",
    "make_anchored_allgather_cp_backend",
    "make_cached_allgather_cp_backend",
    "resolve_cp_group",
    "resolve_mindspeed_hybrid_topology",
    "resolve_tpr_cp_backend",
    "ring_cp_attention",
    "ulysses_cp_rectangular_attention"
]


def __getattr__(name):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
