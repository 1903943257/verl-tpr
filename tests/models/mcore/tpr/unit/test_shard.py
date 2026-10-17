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
