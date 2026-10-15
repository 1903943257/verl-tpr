"""Test-only MindSpeed native whole-ring oracle; not a TPR CP backend."""
from __future__ import annotations

from typing import Any
from torch import Tensor
import torch.distributed as dist

from verl.models.mcore.tpr.shard import SequenceShard
from verl.models.mcore.tpr.parallel.ring import _normalize_inputs

def ordinary_ring_cp_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    current_shard: SequenceShard,
    cp_group: Any,
    softmax_scale: float | None = None,
    allow_trailing_padding: bool = False,
) -> Tensor:
    """Whole causal attention using paired MindSpeed native Ring.

    Explicit entry point: an empty prefix alone does NOT identify whole mode
    (a segmented execution's root also has no prefix). Keep that root on the
    existing TPR schedule until the prefix extension is migrated separately.

    MindSpeed owns streaming KV, causal grouping, output/stat correction,
    reverse backward and owner dKV accumulation. No extra gradient SUM/cast.
    """
    _, config = _normalize_inputs(
        query, current_key, current_value, prefix_blocks=(),
        current_shard=current_shard, cp_group=cp_group,
        softmax_scale=softmax_scale,
    )
    if current_shard.global_length != current_shard.padded_length:
        if not allow_trailing_padding:
            raise ValueError("ordinary whole Ring requires an unpadded sequence")
        # The SegmentExecutor's Ring sequence shard has only VIRTUAL right-
        # padding after the final real token. For causal attention every real
        # query is earlier than these virtual tokens, so padded keys cannot
        # contribute to any real query; the PPO mask excludes padded queries.
        # Never allow this option when Prefix blocks are present.
        if not (0 < current_shard.global_length < current_shard.padded_length):
            raise ValueError("native whole Ring requires strictly trailing padding")
    from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import (
        ringattn_context_parallel,
    )

    cp_para = {
        "causal": True,
        "cp_group": cp_group,
        "cp_size": config.cp_size,
        "rank": config.cp_rank,
        "cp_global_ranks": list(config.global_ranks),
        "cp_inner_ranks": [dist.get_rank()],
        "cp_outer_ranks": list(config.global_ranks),
        "cp_dkv_outer_ranks": list(config.global_ranks),
        "megatron_cp_in_bnsd": False,
        "cache_policy": None,
        "pse_type": 1,
    }
    # Native ordinary causal Ring consumes SBH, retaining GQA's smaller KV H.
    q, k, v = (x.flatten(2).contiguous() for x in (query, current_key, current_value))
    return ringattn_context_parallel(
        q, k, v, config.query_heads, cp_para,
        softmax_scale=config.softmax_scale, dropout_p=0.0,
    )
