"""One concrete placement model preserves CP1/Ring padding and adjoints."""

import pytest
import torch

from verl.workers.engine.megatron.tpr.ring import (
    SequenceShard,
    _validate_ring_shard,
    iter_valid_sequence_slices,
    make_ring_sequence_shard,
)


@pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 9, 14, 16, 17, 33])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_ring_metadata_selects_the_reference_padded_zigzag_and_preserves_gradients(length, cp_size):
    owned = []
    for rank in range(cp_size):
        shard = make_ring_sequence_shard(length, cp_rank=rank, cp_size=cp_size)
        assert type(shard) is SequenceShard
        _validate_ring_shard(shard, cp_size=cp_size, cp_rank=rank, name="test")
        source = torch.randn(2, length, 3, dtype=torch.float64, requires_grad=True)
        physical_indices = shard.physical_global_indices()
        padded = torch.cat((source, source.new_zeros(2, shard.padded_length - length, 3)), dim=1)
        actual = shard.select(source, dim=-2)
        expected = padded.index_select(1, physical_indices)
        torch.testing.assert_close(actual, expected)
        weights = torch.randn_like(actual)
        actual_grad, = torch.autograd.grad(actual, source, weights)
        expected_grad, = torch.autograd.grad(expected, source, weights)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        assert shard.local_length == shard.padded_length // cp_size
        assert shard.global_indices().tolist() == physical_indices.clamp_max(length - 1).tolist()
        valid_count = 0
        for (start, end), local_slice in iter_valid_sequence_slices(shard):
            torch.testing.assert_close(actual[:, local_slice], source[:, start:end])
            valid_count += end - start
            owned.extend(range(start, end))
        assert valid_count == shard.valid_local_length
        for local, global_offset in enumerate(physical_indices.tolist()):
            if global_offset < length:
                assert shard.local_to_global(local) == global_offset
                assert shard.global_to_local(global_offset) == local
                assert shard.owns(global_offset)
            else:
                with pytest.raises(ValueError, match="padding token"):
                    shard.local_to_global(local)
    assert sorted(owned) == list(range(length))


def test_cp1_selection_is_a_view_with_the_same_offset_mapping():
    shard = SequenceShard.full(5)
    source = torch.randn(2, 5, requires_grad=True)
    selected = shard.select(source, dim=1)
    assert selected.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert shard.is_full
    assert shard.global_ranges == shard.physical_global_ranges == ((0, 5),)
    selected.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_padding_only_ranks_keep_autograd_connectivity_with_zero_input_gradient(rank):
    shard = make_ring_sequence_shard(1, cp_rank=rank, cp_size=4)
    source = torch.tensor([9.0], requires_grad=True)
    assert shard.valid_local_length == 0
    assert shard.global_ranges == ()
    assert iter_valid_sequence_slices(shard) == ()
    shard.select(source).sum().backward()
    torch.testing.assert_close(source.grad, torch.zeros_like(source))


@pytest.mark.parametrize("ranges", [(), ((0, 0),), ((0, 3), (2, 4)), ((0, 6),)])
def test_concrete_shard_validates_nonempty_ordered_physical_ranges(ranges):
    with pytest.raises(ValueError):
        SequenceShard(5, ranges)


def test_ring_validation_rejects_an_arbitrary_contiguous_layout():
    with pytest.raises(ValueError, match="MindSpeed causal Ring"):
        _validate_ring_shard(SequenceShard(8, ((0, 4),), 0, 2), cp_size=2, cp_rank=0, name="bad")


def test_experimental_constructors_share_selection_and_accept_concrete_ring_metadata():
    import pickle

    from verl.experimental.tpr.shard import (
        PaddedSequenceShard,
        PrefixShard,
        RangeSequenceShard,
        maybe_pad_sequence_shard,
        physical_sequence_shard,
    )

    contiguous = PrefixShard.contiguous(8, cp_rank=1, cp_size=2)
    assert contiguous.select(torch.arange(8)).tolist() == [4, 5, 6, 7]
    physical = RangeSequenceShard(8, ((0, 2), (6, 8)), cp_rank=0, cp_size=2)
    padded = maybe_pad_sequence_shard(7, physical)
    assert padded.select(torch.arange(7)).tolist() == [0, 1, 6, 0]
    assert physical_sequence_shard(padded) is physical
    restored = pickle.loads(pickle.dumps(padded))
    assert physical_sequence_shard(restored) == physical
    ring = make_ring_sequence_shard(7, cp_rank=0, cp_size=2)
    assert isinstance(ring, PaddedSequenceShard)
    assert isinstance(physical_sequence_shard(ring), RangeSequenceShard)
