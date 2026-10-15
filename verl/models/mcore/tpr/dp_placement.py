"""Compatibility alias: active DP planning resides alongside Megatron TPR Engine.

DP2 planning is still an offline parity feature; not Trainer dispatch.
"""
import sys as _sys
from verl.workers.engine.megatron.tpr import placement as _placement

_sys.modules[__name__] = _placement
