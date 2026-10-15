"""Legacy RoPE entry points refer to the one Segment Executor implementation.

Production computes positional frequencies at the Engine Segment boundary.
"""
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("verl.workers.engine.megatron.tpr.executor")
