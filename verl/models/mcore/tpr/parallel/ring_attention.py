"""Compatibility alias to the sole production Prefix Ring implementation.

Keep module identity intact so older tests monkeypatching Ring internals affect
the actual kernel; no second Ring implementation is maintained here.
"""
import sys as _sys
from verl.workers.engine.megatron.tpr import ring as _ring

_sys.modules[__name__] = _ring
