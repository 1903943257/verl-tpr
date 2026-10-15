"""Compatibility alias: canonical implementation moved to verl.experimental.tpr.actor_capture."""
import sys as _sys
from importlib import import_module as _import_module
_sys.modules[__name__] = _import_module("verl.experimental.tpr.actor_capture")
