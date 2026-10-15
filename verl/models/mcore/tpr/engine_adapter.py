"""Legacy import alias for engine_adapter.

Actual historical CE/Exact implementation resides in verl.experimental.tpr.legacy.
The production TPR engine never imports this path.
"""
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("verl.experimental.tpr.legacy.engine_adapter")
