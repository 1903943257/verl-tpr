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
__all__ = sorted(_PRODUCTION_EXPORTS)


def __getattr__(name):
    if name not in _PRODUCTION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PRODUCTION_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
