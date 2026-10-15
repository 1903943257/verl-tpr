"""Compatibility module alias for the optional verl.experimental.tpr.gdn.tree extension.

The production PPO engine must import only its own Ring/Attention stack.
This alias preserves object identity for legacy monkeypatch-based tests.
"""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("verl.experimental.tpr.gdn.tree")
