"""TQ-compatible NonTensorStack packing and async KV put helper."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .schema import TREE_PARTITION, pack_session_trees, unpack_session_trees


def make_tq_fields(trees: Sequence[dict[str, Any]]):
    return pack_session_trees(trees)


def read_tq_fields(fields: Any) -> list[dict[str, Any]]:
    return unpack_session_trees(fields)


async def put_session_trees(tq: Any, trees: Sequence[dict[str, Any]], *, partition_id: str = TREE_PARTITION):
    """Call the async_kv_batch_put interface proposed by the UniAgent colleague."""
    fields = make_tq_fields(trees)
    tree_ids = [tree["tree_id"] for tree in trees]
    await tq.async_kv_batch_put(keys=tree_ids, partition_id=partition_id, fields=fields)
    return fields
