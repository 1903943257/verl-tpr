# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from verl.models.mcore.tpr import KVPrefixState, SequenceShard
from verl.experimental.tpr.shard import PrefixShard, RangeSequenceShard

# Regression contracts formerly in test_sequence_shard_cpu.py
import pytest
import torch

from verl.workers.engine.megatron.tpr.ring import (
    SequenceShard as RingSequenceShard,
    _validate_ring_shard,
    iter_valid_sequence_slices,
    make_ring_sequence_shard,
)

def test_contiguous_shard_maps_global_and_local_offsets():
    shard = PrefixShard.contiguous(8, cp_rank=1, cp_size=2)
    source = torch.arange(16).reshape(8, 2)

    assert isinstance(shard, SequenceShard)
    assert shard.global_ranges == ((4, 8),)
    assert shard.select(source).tolist() == source[4:8].tolist()
    assert shard.global_indices().tolist() == [4, 5, 6, 7]
    assert shard.global_to_local(3) is None
    assert shard.global_to_local(6) == 2
    assert shard.local_to_global(2) == 6


def test_range_shard_preserves_declared_local_token_order():
    shard = RangeSequenceShard(8, ((0, 2), (6, 8)), cp_rank=0, cp_size=2)
    source = torch.arange(16).reshape(8, 2)

    assert isinstance(shard, SequenceShard)
    assert shard.local_length == 4
    assert shard.select(source).tolist() == torch.cat((source[:2], source[6:])).tolist()
    assert shard.global_indices().tolist() == [0, 1, 6, 7]
    assert [shard.global_to_local(offset) for offset in (0, 1, 2, 6, 7)] == [0, 1, None, 2, 3]
    assert [shard.local_to_global(offset) for offset in range(4)] == [0, 1, 6, 7]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RangeSequenceShard(8, (), cp_rank=0, cp_size=2),
        lambda: RangeSequenceShard(8, ((0, 4), (3, 8)), cp_rank=0, cp_size=2),
        lambda: RangeSequenceShard(8, ((0, 9),), cp_rank=0, cp_size=2),
        lambda: PrefixShard.contiguous(7, cp_rank=0, cp_size=2),
    ],
)
def test_invalid_shards_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_select_rejects_a_mismatched_global_sequence_length():
    shard = RangeSequenceShard(8, ((0, 2), (6, 8)), cp_rank=0, cp_size=2)

    with pytest.raises(ValueError, match="sequence dimension"):
        shard.select(torch.arange(7))


def test_kv_prefix_state_accepts_a_backend_neutral_sequence_shard():
    shard = RangeSequenceShard(8, ((0, 2), (6, 8)), cp_rank=0, cp_size=2)
    key = torch.zeros(4, 1, 2, 4)
    value = torch.ones_like(key)

    state = KVPrefixState(0, 8, {1: (key, value)}, shard=shard)
    anchors = state.make_anchors()

    assert state.shard is shard
    assert state.local_length == 4
    assert anchors.shard is shard


@pytest.mark.parametrize("shard", [
    PrefixShard.full(8),
    PrefixShard.contiguous(8, cp_rank=1, cp_size=2),
    RangeSequenceShard(8, ((0, 2), (6, 8)), cp_rank=0, cp_size=2),
])
def test_shard_selection_preserves_gradients_and_offset_round_trips(shard):
    source = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    selected = shard.select(source, dim=-1)
    indices = shard.global_indices()
    torch.testing.assert_close(selected, source.index_select(1, indices))
    selected.sum().backward()
    expected_grad = torch.zeros_like(source)
    expected_grad[:, indices] = 1
    torch.testing.assert_close(source.grad, expected_grad)
    for local, global_offset in enumerate(indices.tolist()):
        assert shard.owns(global_offset)
        assert shard.global_to_local(global_offset) == local
        assert shard.local_to_global(local) == global_offset
    if len(shard.global_ranges) == 1:
        assert selected.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()

# Tests consolidated from test_sequence_shard_cpu.py
@pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 9, 14, 16, 17, 33])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_ring_metadata_selects_the_reference_padded_zigzag_and_preserves_gradients(length, cp_size):
    owned = []
    for rank in range(cp_size):
        shard = make_ring_sequence_shard(length, cp_rank=rank, cp_size=cp_size)
        assert type(shard) is RingSequenceShard
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
    shard = RingSequenceShard.full(5)
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
        RingSequenceShard(5, ranges)


def test_ring_validation_rejects_an_arbitrary_contiguous_layout():
    with pytest.raises(ValueError, match="MindSpeed causal Ring"):
        _validate_ring_shard(RingSequenceShard(8, ((0, 4),), 0, 2), cp_size=2, cp_rank=0, name="bad")


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
