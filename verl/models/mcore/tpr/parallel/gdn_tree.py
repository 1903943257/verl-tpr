"""Compatibility import for moved experimental TPR implementation.

New production TPR code must not import this module.
"""
from importlib import import_module as _import_module

_impl = _import_module("verl.experimental.tpr.gdn.tree")
for _symbol in dir(_impl):
    if not _symbol.startswith("_"):
        globals()[_symbol] = getattr(_impl, _symbol)


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
