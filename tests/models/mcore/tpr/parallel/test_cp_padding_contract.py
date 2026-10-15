"""Ring-only logical/physical CP padding contracts."""
import torch
from verl.models.mcore.tpr import (
    KVPrefixState, PaddedSequenceShard, iter_valid_sequence_slices,
    make_ring_sequence_shard, round_up_sequence_length,
)


def test_round_up_sequence_length_uses_ring_alignment():
    assert round_up_sequence_length(7, 4) == 8
    assert round_up_sequence_length(14, 8) == 16


def test_ring_padding_preserves_logical_ownership_and_physical_zigzag():
    token_ids = torch.arange(7)
    shards = [make_ring_sequence_shard(7, cp_rank=i, cp_size=2) for i in range(2)]
    assert all(isinstance(s, PaddedSequenceShard) for s in shards)
    assert all(s.padded_length == 8 and s.local_length == 4 for s in shards)
    assert sum(s.valid_local_length for s in shards) == 7
    indices = torch.cat([s.global_indices() for s in shards])
    assert torch.equal(indices.sort().values, torch.tensor([0, 1, 2, 3, 4, 5, 6, 6]))
    for shard in shards:
        assert shard.select(token_ids).shape == (4,)
        assert sum(end - start for (start, end), _local in iter_valid_sequence_slices(shard)) == shard.valid_local_length


def test_padded_ring_kv_state_keeps_logical_segment_length():
    shard = make_ring_sequence_shard(7, cp_rank=1, cp_size=2)
    kv = {1: (torch.randn(4, 1, 2, 8), torch.randn(4, 1, 2, 8))}
    state = KVPrefixState(1, 7, kv, shard=shard)
    assert state.global_length == 7
    assert state.local_length == 4
