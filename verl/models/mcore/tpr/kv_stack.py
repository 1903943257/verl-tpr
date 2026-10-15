"""Legacy path aliases the one Engine-owned KVStack and gradient-anchor module."""
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("verl.workers.engine.megatron.tpr.state")
