"""Legacy PrefixState interface; Dense KV state now lives in Engine state.py.

GDN continuation is loaded only when an experimental caller explicitly
requests one of its state types. Never import GDN from production PPO.
"""
from verl.workers.engine.megatron.tpr.state import (
    PrefixState,
    PrefixStateEntry,
    PrefixStateStack,
    KVPrefixAnchors,
    KVPrefixState,
    _validate_segment_id,
    _validate_layer_number,
    _compact_graph_free_tensor,
    _normalize_local_kv,
)

_GDN_SYMBOLS = frozenset(("GDNLayerState", "GDNPrefixAnchors", "GDNPrefixState", "_normalize_gdn_layer_states"))


def __getattr__(name):
    if name in _GDN_SYMBOLS:
        from verl.experimental.tpr.gdn import prefix_state
        value = getattr(prefix_state, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _GDN_SYMBOLS)
