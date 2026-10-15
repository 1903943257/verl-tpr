"""Legacy CP compatibility alias; optional backend lives in experiments.

The production PPO path bypasses the multi-backend CP factory entirely.
"""
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("verl.experimental.tpr.legacy.cp.execution_context")
