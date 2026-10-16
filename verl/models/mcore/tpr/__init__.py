"""Model-side TPR Attention/Context/CANN and lazy backwards exports."""
from importlib import import_module

# Public TPR model/runtime exports; no optional/legacy symbols in __all__.
_PRODUCTION_EXPORTS = {
    "build_sharded_rotary_pos_emb": "verl.workers.engine.megatron.tpr.executor",
    "build_suffix_rotary_pos_emb": "verl.workers.engine.megatron.tpr.executor",
    "classify_ring_block": "verl.workers.engine.megatron.tpr.ring",
    "get_tpr_attention_context": "verl.models.mcore.tpr.context",
    "iter_valid_sequence_slices": "verl.workers.engine.megatron.tpr.ring",
    "KVPrefixAnchors": "verl.workers.engine.megatron.tpr.state",
    "KVPrefixState": "verl.workers.engine.megatron.tpr.state",
    "KVStack": "verl.workers.engine.megatron.tpr.state",
    "KVStackEntry": "verl.workers.engine.megatron.tpr.state",
    "make_ring_sequence_shard": "verl.workers.engine.megatron.tpr.ring",
    "make_tpr_module_spec_provider": "verl.workers.engine.megatron.tpr.adapter",
    "PastKVAnchors": "verl.workers.engine.megatron.tpr.state",
    "rectangular_causal_attention": "verl.models.mcore.tpr.rectangular_attention",
    "replace_self_attention_with_tpr": "verl.workers.engine.megatron.tpr.adapter",
    "ring_cp_attention": "verl.workers.engine.megatron.tpr.ring",
    "RingBlockKind": "verl.workers.engine.megatron.tpr.ring",
    "RingCPAttentionBackend": "verl.workers.engine.megatron.tpr.ring",
    "RingCPBackend": "verl.workers.engine.megatron.tpr.ring",
    "RingLocalKVBlock": "verl.workers.engine.megatron.tpr.ring",
    "round_up_sequence_length": "verl.workers.engine.megatron.tpr.ring",
    "SegmentExecutor": "verl.workers.engine.megatron.tpr.executor",
    "SequenceShard": "verl.workers.engine.megatron.tpr.ring",
    "TPRAttentionBackend": "verl.models.mcore.tpr.context",
    "TPRAttentionContext": "verl.models.mcore.tpr.context",
    "TPRSelfAttention": "verl.models.mcore.tpr.attention",
    "use_tpr_attention_context": "verl.models.mcore.tpr.context",
}
# Transitional explicit imports for historical tests; not part of public __all__.
_LEGACY_EXPORTS = {
    "all_gather_sequence": "verl.experimental.tpr.allgather",
    "ALLGATHER_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "allgather_cp_rectangular_attention": "verl.experimental.tpr.allgather",
    "AllGatherCPAttentionBackend": "verl.experimental.tpr.legacy.cp.execution_context",
    "AllGatherCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "FixedTopologyScheduler": "verl.experimental.tpr.legacy.fixed_topology_scheduler",
    "GDNLayerState": "verl.experimental.tpr.gdn.prefix_state",
    "GDNPrefixAnchors": "verl.experimental.tpr.gdn.prefix_state",
    "GDNPrefixState": "verl.experimental.tpr.gdn.prefix_state",
    "HYBRID_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "hybrid_cp_rectangular_attention": "verl.experimental.tpr.hybrid",
    "HybridCPAttentionBackend": "verl.experimental.tpr.hybrid",
    "HybridCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "HybridCPTopology": "verl.experimental.tpr.hybrid",
    "HybridLocalKVBlock": "verl.experimental.tpr.hybrid",
    "LeafVisitResult": "verl.experimental.tpr.legacy.segment_executor",
    "LocalKVBlock": "verl.experimental.tpr.allgather",
    "make_hybrid_sequence_shard": "verl.experimental.tpr.hybrid",
    "PaddedSequenceShard": "verl.experimental.tpr.shard",
    "PastKVSlice": "verl.experimental.tpr.legacy.state",
    "physical_sequence_shard": "verl.experimental.tpr.shard",
    "PhysicalExecution": "verl.experimental.tpr.legacy.fixed_topology_scheduler",
    "PhysicalExecutionKind": "verl.experimental.tpr.legacy.fixed_topology_scheduler",
    "PopSegment": "verl.experimental.tpr.legacy.plan",
    "PrefixShard": "verl.experimental.tpr.shard",
    "PrefixState": "verl.experimental.tpr.legacy.state",
    "PrefixStateEntry": "verl.experimental.tpr.legacy.state",
    "PrefixStateStack": "verl.experimental.tpr.legacy.state",
    "PushSegment": "verl.experimental.tpr.legacy.plan",
    "RangeSequenceShard": "verl.experimental.tpr.shard",
    "resolve_mindspeed_hybrid_topology": "verl.experimental.tpr.hybrid",
    "resolve_tpr_cp_backend": "verl.experimental.tpr.legacy.cp.backend",
    "RING_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "SchedulerState": "verl.experimental.tpr.legacy.fixed_topology_scheduler",
    "SegmentBackwardResult": "verl.experimental.tpr.legacy.segment_executor",
    "SegmentEvent": "verl.experimental.tpr.legacy.plan",
    "SegmentEventKind": "verl.experimental.tpr.legacy.plan",
    "SegmentForwardResult": "verl.experimental.tpr.legacy.segment_executor",
    "SegmentKV": "verl.experimental.tpr.legacy.state",
    "SegmentLossTerm": "verl.experimental.tpr.legacy.plan",
    "SegmentPlan": "verl.experimental.tpr.legacy.plan",
    "SegmentSpec": "verl.experimental.tpr.legacy.plan",
    "ShardedPastKVAnchors": "verl.experimental.tpr.legacy.state",
    "TPR_REQUEST_KEY": "verl.experimental.tpr.legacy.engine_adapter",
    "TPRCPBackend": "verl.experimental.tpr.legacy.cp.backend",
    "TPRForwardBackwardRequest": "verl.experimental.tpr.legacy.engine_adapter",
    "TreeScheduleResult": "verl.experimental.tpr.legacy.fixed_topology_scheduler",
    "ULYSSES_CP_BACKEND": "verl.experimental.tpr.legacy.cp.backend",
    "ulysses_cp_rectangular_attention": "verl.experimental.tpr.ulysses",
    "UlyssesCPAttentionBackend": "verl.experimental.tpr.ulysses",
    "UlyssesCPBackend": "verl.experimental.tpr.legacy.cp.backend",
}
_EXPORTS = {**_PRODUCTION_EXPORTS, **_LEGACY_EXPORTS}
_MODULES = {
    "parallel": "verl.experimental.tpr",
}
__all__ = sorted(_PRODUCTION_EXPORTS)


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
