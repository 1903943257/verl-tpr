"""Backwards-compatible alias for Ring-owned logical/physical sequence layouts.

One implementation only: all SequenceShard and Padding behavior lives in the
Engine Ring module. New production code imports the Ring types directly.
Legacy tests and experimental CP backends may still import this module path.
"""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("verl.workers.engine.megatron.tpr.ring")
