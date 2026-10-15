"""Model-side TPR Attention/Context/CANN and lazy backwards exports."""
from importlib import import_module

_EXPORTS = {
    "all_gather_sequence": "verl.experimental.tpr.allgather",
    "ALLGATHER_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "allgather_cp_rectangular_attention": "verl.experimental.tpr.allgather",
    "AllGatherCPAttentionBackend": "verl.experimental.tpr.legacy.cp.execution_context",
    "AllGatherCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "build_sharded_rotary_pos_emb": "verl.workers.engine.megatron.tpr.executor",
    "build_suffix_rotary_pos_emb": "verl.workers.engine.megatron.tpr.executor",
    "classify_ring_block": "verl.workers.engine.megatron.tpr.ring",
    "FixedTopologyScheduler": "verl.workers.engine.megatron.tpr.executor",
    "GDNLayerState": "verl.experimental.tpr.gdn.prefix_state",
    "GDNPrefixAnchors": "verl.experimental.tpr.gdn.prefix_state",
    "GDNPrefixState": "verl.experimental.tpr.gdn.prefix_state",
    "get_tpr_attention_context": "verl.models.mcore.tpr.context",
    "HYBRID_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "hybrid_cp_rectangular_attention": "verl.experimental.tpr.hybrid",
    "HybridCPAttentionBackend": "verl.experimental.tpr.hybrid",
    "HybridCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "HybridCPTopology": "verl.experimental.tpr.hybrid",
    "HybridLocalKVBlock": "verl.experimental.tpr.hybrid",
    "iter_valid_sequence_slices": "verl.workers.engine.megatron.tpr.ring",
    "KVPrefixAnchors": "verl.workers.engine.megatron.tpr.state",
    "KVPrefixState": "verl.workers.engine.megatron.tpr.state",
    "KVStack": "verl.workers.engine.megatron.tpr.state",
    "KVStackEntry": "verl.workers.engine.megatron.tpr.state",
    "LeafVisitResult": "verl.workers.engine.megatron.tpr.executor",
    "LocalKVBlock": "verl.experimental.tpr.allgather",
    "make_hybrid_sequence_shard": "verl.experimental.tpr.hybrid",
    "make_ring_sequence_shard": "verl.workers.engine.megatron.tpr.ring",
    "make_tpr_module_spec_provider": "verl.workers.engine.megatron.tpr.adapter",
    "PaddedSequenceShard": "verl.workers.engine.megatron.tpr.ring",
    "PastKVAnchors": "verl.workers.engine.megatron.tpr.state",
    "PastKVSlice": "verl.workers.engine.megatron.tpr.state",
    "physical_sequence_shard": "verl.workers.engine.megatron.tpr.ring",
    "PhysicalExecution": "verl.workers.engine.megatron.tpr.executor",
    "PhysicalExecutionKind": "verl.workers.engine.megatron.tpr.executor",
    "PopSegment": "verl.workers.engine.megatron.tpr.tree",
    "PrefixShard": "verl.workers.engine.megatron.tpr.ring",
    "PrefixState": "verl.workers.engine.megatron.tpr.state",
    "PrefixStateEntry": "verl.workers.engine.megatron.tpr.state",
    "PrefixStateStack": "verl.workers.engine.megatron.tpr.state",
    "PushSegment": "verl.workers.engine.megatron.tpr.tree",
    "RangeSequenceShard": "verl.workers.engine.megatron.tpr.ring",
    "rectangular_causal_attention": "verl.models.mcore.tpr.rectangular_attention",
    "replace_self_attention_with_tpr": "verl.workers.engine.megatron.tpr.adapter",
    "resolve_mindspeed_hybrid_topology": "verl.experimental.tpr.hybrid",
    "resolve_tpr_cp_backend": "verl.experimental.tpr.legacy.cp.backend",
    "ring_cp_attention": "verl.workers.engine.megatron.tpr.ring",
    "RING_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "RingBlockKind": "verl.workers.engine.megatron.tpr.ring",
    "RingCPAttentionBackend": "verl.workers.engine.megatron.tpr.ring",
    "RingCPBackend": "verl.workers.engine.megatron.tpr.ring",
    "RingLocalKVBlock": "verl.workers.engine.megatron.tpr.ring",
    "round_up_sequence_length": "verl.workers.engine.megatron.tpr.ring",
    "SchedulerState": "verl.workers.engine.megatron.tpr.executor",
    "SegmentBackwardResult": "verl.workers.engine.megatron.tpr.executor",
    "SegmentEvent": "verl.workers.engine.megatron.tpr.tree",
    "SegmentEventKind": "verl.workers.engine.megatron.tpr.tree",
    "SegmentExecutor": "verl.workers.engine.megatron.tpr.executor",
    "SegmentForwardResult": "verl.workers.engine.megatron.tpr.executor",
    "SegmentKV": "verl.workers.engine.megatron.tpr.state",
    "SegmentLossTerm": "verl.workers.engine.megatron.tpr.tree",
    "SegmentPlan": "verl.workers.engine.megatron.tpr.tree",
    "SegmentSpec": "verl.workers.engine.megatron.tpr.tree",
    "SequenceShard": "verl.workers.engine.megatron.tpr.ring",
    "ShardedPastKVAnchors": "verl.workers.engine.megatron.tpr.state",
    "TPR_REQUEST_KEY": "verl.experimental.tpr.legacy.engine_adapter",
    "TPRAttentionBackend": "verl.models.mcore.tpr.context",
    "TPRAttentionContext": "verl.models.mcore.tpr.context",
    "TPRCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "TPRForwardBackwardRequest": "verl.experimental.tpr.legacy.engine_adapter",
    "TPRSelfAttention": "verl.models.mcore.tpr.attention",
    "TreeScheduleResult": "verl.workers.engine.megatron.tpr.executor",
    "ULYSSES_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "ulysses_cp_rectangular_attention": "verl.experimental.tpr.ulysses",
    "UlyssesCPAttentionBackend": "verl.experimental.tpr.ulysses",
    "UlyssesCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "use_tpr_attention_context": "verl.models.mcore.tpr.context",
}
_MODULES = {
    "parallel": "verl.experimental.tpr",
}
__all__ = sorted(set(_EXPORTS) | set(_MODULES))


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
