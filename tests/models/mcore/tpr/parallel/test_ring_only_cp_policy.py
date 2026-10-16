"""Ring-only CP policy and QKV merge configuration contracts."""
import pytest
import torch
from types import SimpleNamespace

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


def test_cached_and_anchored_prefix_blocks_identical(monkeypatch):
    """One builder must preserve segment order, layers and original autograd tensors."""
    monkeypatch.setattr(ring, "_group_world_size_and_rank", lambda _g: (2, 0))
    group = object()
    shard_a = ring.make_ring_sequence_shard(64, cp_rank=0, cp_size=2)
    shard_b = ring.make_ring_sequence_shard(128, cp_rank=0, cp_size=2)
    state_map = {}
    for segment_id, shard in ((3, shard_a), (7, shard_b)):
        tensor_map = {
            layer: (
                torch.randn(shard.local_length, 1, 2, 8, requires_grad=True),
                torch.randn(shard.local_length, 1, 2, 8, requires_grad=True),
            )
            for layer in (1, 2)
        }
        state_map[segment_id] = SimpleNamespace(
            segment_id=segment_id, shard=shard, key_values=tensor_map,
        )
    stack = SimpleNamespace(
        segment_ids=(3, 7),
        prefix_length=192,
        get=lambda segment_id: SimpleNamespace(kv=state_map[segment_id]),
    )
    anchors = SimpleNamespace(
        entries=tuple(state_map.values()),
        global_prefix_length=192,
    )
    backend = ring.RingCPBackend(group, parallel_size=2, parallel_rank=0)
    current = ring.make_ring_sequence_shard(64, cp_rank=0, cp_size=2)
    cached = backend.make_attention_backend(
        stack, expected_layer_numbers=(1, 2),
        current_shard=current, past_anchors=None,
    )
    anchored = backend.make_attention_backend(
        stack, expected_layer_numbers=(1, 2),
        current_shard=current, past_anchors=anchors,
    )
    assert cached.global_prefix_length == anchored.global_prefix_length == 192
    assert cached.global_suffix_length == anchored.global_suffix_length == 64
    assert cached.local_suffix_length == anchored.local_suffix_length
    for layer in (1, 2):
        old_blocks = cached._prefix_blocks_by_layer[layer]
        new_blocks = anchored._prefix_blocks_by_layer[layer]
        assert [block.segment_id for block in old_blocks] == [3, 7]
        for old, new in zip(old_blocks, new_blocks, strict=True):
            assert old.shard == new.shard
            assert old.key is new.key
            assert old.value is new.value
            assert old.key.requires_grad and old.value.requires_grad


def test_prefix_block_builder_rejects_non_ring_shards():
    state = SimpleNamespace(
        segment_id=1,
        shard=ring.RangeSequenceShard(8, ((0, 8),), cp_rank=0, cp_size=1),
        key_values={1: (torch.zeros(8, 1, 1, 8), torch.zeros(8, 1, 1, 8))},
    )
    # Range placement is the required physical representation, while full
    # Ring layout validation remains in the actual Attention input contract.
    stack = SimpleNamespace(segment_ids=(1,), get=lambda _: SimpleNamespace(kv=state))
    assert ring._prefix_blocks_by_layer(stack, None, (1,))[1][0].shard == state.shard
    state.shard = object()
    with pytest.raises(TypeError, match="RangeSequenceShard"):
        ring._prefix_blocks_by_layer(stack, None, (1,))
