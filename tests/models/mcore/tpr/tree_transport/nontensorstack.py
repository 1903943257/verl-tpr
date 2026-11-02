"""In-memory TensorDict / NonTensorStack packing and consumption helpers.

No TQ client, storage operation, serialization, or network transport is used.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import pack_session_trees, unpack_session_trees


def make_tree_fields(trees: Sequence[Mapping[str, Any]]):
    """Create the exact TensorDict field shape proposed for the TQ payload."""
    return pack_session_trees(trees)


def consume_tree_fields(fields: Any) -> list[dict[str, Any]]:
    """Read Session Tree dictionaries from the in-memory NonTensorStack."""
    return unpack_session_trees(fields)
