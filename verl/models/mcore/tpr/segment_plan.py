"""Compatibility alias: Segment topology is defined in Engine tree.py.

Old tests may import this path, but PPO execution has one canonical Segment
layout and one token ownership mapping.
"""
import sys as _sys
from importlib import import_module as _import_module
_sys.modules[__name__] = _import_module("verl.workers.engine.megatron.tpr.tree")
