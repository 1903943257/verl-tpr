"""Ring-only CP policy and QKV merge configuration contracts."""
import pytest
import torch

from verl.models.mcore.tpr.parallel import ring


def test_only_ring_backend_is_selectable(monkeypatch):
    monkeypatch.setattr(ring, "_group_world_size_and_rank", lambda _g: (2, 0))
    group = object()
    assert ring.resolve_cp_group(group) == (2, 0)
    for name in (None, "auto", "ring", "megatron_cp_algo"):
        backend = ring.resolve_tpr_cp_backend(
            name, cp_group=group, parallel_size=2, parallel_rank=0,
        )
        assert isinstance(backend, ring.RingCPBackend)
        assert backend.backend_name == "ring"
        assert backend.make_sequence_shard(127).padded_length == 128
    for removed in ("allgather", "kvallgather_cp_algo", "ulysses", "hybrid"):
        with pytest.raises(NotImplementedError, match="only supports Ring"):
            ring.resolve_tpr_cp_backend(
                removed, cp_group=group, parallel_size=2, parallel_rank=0,
            )


def test_qkv_merge_is_enabled_by_default_but_can_be_disabled(monkeypatch):
    monkeypatch.setattr(ring, "_group_world_size_and_rank", lambda _g: (2, 0))
    monkeypatch.setattr(ring, "_process_group_global_ranks", lambda _g: (0, 1))
    group = object()
    shard = ring.make_ring_sequence_shard(128, cp_rank=0, cp_size=2)
    q = torch.randn(64, 1, 8, 16)
    k = torch.randn(64, 1, 2, 16)
    v = torch.randn(64, 1, 2, 16)
    prefix = [ring.RingLocalKVBlock(0, shard, k, v)]
    monkeypatch.delenv("TPR_RING_COALESCE_PREFIX_FULL", raising=False)
    monkeypatch.delenv("TPR_RING_COALESCE_PREFIX_QUERY", raising=False)
    _, config = ring._normalize_inputs(
        q, k, v, prefix_blocks=prefix, current_shard=shard,
        cp_group=group, softmax_scale=None,
    )
    assert config.coalesce_prefix_full and config.coalesce_prefix_query
    assert ring._can_coalesce_prefix_query(q, config)
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_QUERY", "0")
    _, disabled = ring._normalize_inputs(
        q, k, v, prefix_blocks=prefix, current_shard=shard,
        cp_group=group, softmax_scale=None,
    )
    assert not ring._can_coalesce_prefix_query(q, disabled)
