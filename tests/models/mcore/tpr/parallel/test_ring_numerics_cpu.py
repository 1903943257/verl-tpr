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

"""CPU numerical checks for production Ring accumulation, independent of CANN/HCCL.

Only the FA kernels and transport are replaced. Symmetric Prefix shards let
every simulated source reuse the local KV; dQ still traverses the real forward,
online softmax, saved statistics, block schedule and backward implementation.
Owner dKV reduction and NPU kernel parity require their distributed tests.
"""

import pytest
import torch

import verl.workers.engine.megatron.tpr.ring as ring
from verl.workers.engine.megatron.tpr import ring_layout as ring_layout
from verl.workers.engine.megatron.tpr import ring_attention as ring_kernels


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _scores(query, key, softmax_scale, block_kind, attention_mask):
    scores = query.float().transpose(0, 1) @ key.float().permute(1, 2, 0) * softmax_scale
    if attention_mask is None and block_kind is ring.RingBlockKind.CAUSAL:
        attention_mask = torch.arange(key.shape[0])[None, :] > (
            key.shape[0] - query.shape[0] + torch.arange(query.shape[0])[:, None]
        )
    return scores if attention_mask is None else scores.masked_fill(attention_mask, -torch.inf)


def _pack_stats(stat):
    # CANN exposes [T,H,8] whose storage is [H,T,8], including repeated lanes.
    heads, tokens = stat.shape
    return stat[..., None].expand(heads, tokens, 8).reshape(tokens, heads, 8).contiguous()


def _fa_forward(query, key, value, *, query_heads, softmax_scale, block_kind, attention_mask=None):
    scores = _scores(query, key, softmax_scale, block_kind, attention_mask)
    maximum = scores.max(-1).values
    weights = (scores - maximum[..., None]).exp()
    total = weights.sum(-1)
    output = ((weights / total[..., None]) @ value.float().transpose(0, 1)).transpose(0, 1)
    return output.to(query.dtype), _pack_stats(maximum), _pack_stats(total)


def _fa_backward(
    query,
    key,
    value,
    grad_output,
    *,
    attention_output,
    softmax_max,
    softmax_sum,
    query_heads,
    softmax_scale,
    block_kind,
    attention_mask=None,
):
    scores = _scores(query, key, softmax_scale, block_kind, attention_mask)
    # Read the kernel's layout directly, without production flatten/slice helpers.
    maximum = softmax_max.view(query_heads, query.shape[0], 8)[..., 0]
    total = softmax_sum.view(query_heads, query.shape[0], 8)[..., 0]
    probabilities = (scores - maximum[..., None]).exp() / total[..., None]
    grad = grad_output.float().transpose(0, 1)
    output = attention_output.float().transpose(0, 1)
    ds = probabilities * (
        grad @ value.float().permute(1, 2, 0) - (grad * output).sum(-1, keepdim=True)
    ) * softmax_scale
    dq = (ds @ key.float().transpose(0, 1)).transpose(0, 1)
    dk = (ds.transpose(1, 2) @ query.float().transpose(0, 1)).transpose(0, 1)
    dv = (probabilities.transpose(1, 2) @ grad).transpose(0, 1)
    # FA returns gradients in the input dtype, even though its arithmetic is FP32.
    return dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype)


def _circulate_symmetric_sources(key, value, config, consume):
    for step in range(config.cp_size):
        consume((config.cp_rank - step) % config.cp_size, key, value)


def _replay_symmetric_sources(key, value, config, consume):
    dk, dv = torch.zeros_like(key), torch.zeros_like(value)
    for step in range(config.cp_size):
        consume((config.cp_rank + step + 1) % config.cp_size, key, value, dk, dv)
    return dk, dv


def _dense_reference(query, key, value, prefix_length, scale):
    query = query.double().detach().requires_grad_(True)
    scores = torch.einsum("qbhd,kbhd->hqk", query, key.double()) * scale
    excluded = torch.arange(key.shape[0])[None, :] > (
        prefix_length + torch.arange(query.shape[0])[:, None]
    )
    probabilities = scores.masked_fill(excluded, -torch.inf).softmax(-1)
    output = torch.einsum("hqk,kbhd->qbhd", probabilities, value.double())
    upstream = torch.zeros_like(output)
    upstream[0, :, :, 0] = 1
    gradient, = torch.autograd.grad(output, query, upstream)
    return output.detach(), gradient


@pytest.mark.parametrize("ancestors", [18, 26, 48])
@pytest.mark.parametrize("coalesce_full,coalesce_query", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_long_prefix_query_gradient_matches_dense(monkeypatch, ancestors, coalesce_full, coalesce_query, dtype):
    cp_size, heads, head_dim = 4, 2, 64
    length = 2 * cp_size
    scale = head_dim**-0.5
    shard = ring.make_ring_sequence_shard(length, cp_rank=0, cp_size=cp_size)
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_PADDED", "0")
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_FULL", str(int(coalesce_full)))
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_QUERY", str(int(coalesce_query)))
    monkeypatch.setattr(ring_layout, "_group_world_size_and_rank", lambda group: (cp_size, 0))
    monkeypatch.setattr(ring_layout, "_process_group_global_ranks", lambda group: tuple(range(cp_size)))
    monkeypatch.setattr(ring_kernels, "_block_attention_forward", _fa_forward)
    monkeypatch.setattr(ring_kernels, "_block_attention_backward", _fa_backward)
    monkeypatch.setattr(ring_kernels, "_circulate_kv", _circulate_symmetric_sources)
    monkeypatch.setattr(ring_kernels, "_reduce_ring_gradients_to_owner", _replay_symmetric_sources)

    global_query = torch.zeros(length, 1, heads, head_dim, dtype=dtype)
    prefix_key, prefix_value = torch.zeros_like(global_query), torch.zeros_like(global_query)
    signs = torch.tensor([-1.0] * cp_size + [1.0] * cp_size, dtype=dtype)
    prefix_key[:, 0, :, 0] = signs[:, None] * 8
    prefix_value[:, 0, :, 0] = signs[:, None] * torch.tensor([1.0, 2.0], dtype=dtype)
    local_query = shard.select(global_query).detach().requires_grad_(True)
    local_key, local_value = shard.select(prefix_key), shard.select(prefix_value)
    blocks = tuple(ring.RingLocalKVBlock(index, shard, local_key, local_value) for index in range(ancestors))
    current_key, current_value = torch.zeros_like(local_query), torch.zeros_like(local_query)

    actual = ring.ring_cp_attention(
        local_query,
        current_key,
        current_value,
        prefix_blocks=blocks,
        current_shard=shard,
        cp_group=object(),
        softmax_scale=scale,
    )
    upstream = torch.zeros_like(actual)
    upstream[0, 0, ::head_dim] = 1
    actual.backward(upstream)

    dense_key = torch.cat([prefix_key] * ancestors + [global_query])
    dense_value = torch.cat([prefix_value] * ancestors + [global_query])
    expected, expected_gradient = _dense_reference(
        global_query, dense_key, dense_value, ancestors * length, scale
    )
    assert actual.dtype == dtype
    assert local_query.grad.dtype == dtype
    torch.testing.assert_close(actual, shard.select(expected).flatten(2).to(dtype), rtol=0, atol=1e-7)
    expected_gradient = shard.select(expected_gradient).to(dtype)
    # These balanced fixtures have a known final BF16 rounding; FP32 permits
    # only normal FP32 summation error, rather than the former 6-16% BF16 drift.
    tolerance = {"rtol": 0, "atol": 0} if dtype == torch.bfloat16 else {"rtol": 1e-5, "atol": 1e-7}
    torch.testing.assert_close(local_query.grad, expected_gradient, **tolerance)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("coalesce_full,coalesce_query", [(False, False), (False, True), (True, False), (True, True)])
def test_padded_query_and_kv_gradients_match_dense(monkeypatch, rank, coalesce_full, coalesce_query):
    """Check real masking with virtual upstream Q gradients and uneven KV sources.

    The transport stub replays every source's actual shard. Its dKV result is
    this Query rank's contribution to the local owner, independently checked
    against dense autograd. Cross-rank dKV reduction still needs the NPU gate.
    """
    cp_size, heads, head_dim = 4, 2, 8
    prefix_lengths, current_length = (15, 7), 15
    lengths = (*prefix_lengths, current_length)
    scale = head_dim**-0.5
    generator = torch.Generator().manual_seed(831)
    global_query = torch.randn(current_length, 1, heads, head_dim, generator=generator)
    global_keys = [torch.randn(length, 1, heads, head_dim, generator=generator) for length in lengths]
    global_values = [torch.randn(length, 1, heads, head_dim, generator=generator) for length in lengths]
    shards = [ring.make_ring_sequence_shard(length, cp_rank=rank, cp_size=cp_size) for length in lengths]
    current_shard = shards[-1]
    local_query = current_shard.select(global_query).detach().requires_grad_()
    local_keys = [shard.select(key).detach().requires_grad_() for shard, key in zip(shards, global_keys, strict=True)]
    local_values = [shard.select(value).detach().requires_grad_() for shard, value in zip(shards, global_values, strict=True)]

    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_FULL", str(int(coalesce_full)))
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_QUERY", str(int(coalesce_query)))
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_PADDED", "1")
    monkeypatch.setattr(ring_layout, "_group_world_size_and_rank", lambda group: (cp_size, rank))
    monkeypatch.setattr(ring_layout, "_process_group_global_ranks", lambda group: tuple(range(cp_size)))
    monkeypatch.setattr(ring_kernels, "_block_attention_forward", _fa_forward)
    monkeypatch.setattr(ring_kernels, "_block_attention_backward", _fa_backward)
    segment_calls = {"forward": 0, "backward": 0}

    def source_kv(segment, source):
        shard = ring.make_ring_sequence_shard(lengths[segment], cp_rank=source, cp_size=cp_size)
        return shard.select(global_keys[segment]), shard.select(global_values[segment])

    def circulate(key, value, config, consume):
        segment = segment_calls["forward"]
        segment_calls["forward"] += 1
        for step in range(cp_size):
            source = (rank - step) % cp_size
            consume(source, *source_kv(segment, source))

    def replay(key, value, config, consume):
        segment = segment_calls["backward"]
        segment_calls["backward"] += 1
        owner_gradients = None
        for step in range(cp_size):
            source = (rank + step + 1) % cp_size
            source_key, source_value = source_kv(segment, source)
            dk, dv = torch.zeros_like(source_key), torch.zeros_like(source_value)
            consume(source, source_key, source_value, dk, dv)
            if source == rank:
                owner_gradients = dk, dv
        return owner_gradients

    monkeypatch.setattr(ring_kernels, "_circulate_kv", circulate)
    monkeypatch.setattr(ring_kernels, "_reduce_ring_gradients_to_owner", replay)
    blocks = tuple(ring.RingLocalKVBlock(index, shard, key, value) for index, (shard, key, value) in enumerate(
        zip(shards[:-1], local_keys[:-1], local_values[:-1], strict=True)
    ))
    backend = ring.RingCPAttentionBackend(
        global_prefix_length=sum(prefix_lengths), current_shard=current_shard,
        prefix_blocks_by_layer={1: blocks}, cp_group=object(),
    )
    actual = ring.ring_cp_attention(
        local_query, local_keys[-1], local_values[-1], prefix_blocks=blocks,
        current_shard=current_shard, cp_group=backend._ring_layout.cp_group,
        softmax_scale=scale, _layout=backend._ring_layout,
    )

    # Only this rank's real Query positions contribute to the dense dKV oracle.
    upstream = torch.zeros_like(global_query)
    for start, end in current_shard.global_ranges:
        upstream[start:end] = torch.randn(upstream[start:end].shape, generator=generator)
    local_upstream = current_shard.select(upstream)
    local_upstream[current_shard.valid_local_length:] = 3
    actual.backward(local_upstream.flatten(2))

    reference_query = global_query.double().requires_grad_()
    reference_keys = [key.double().requires_grad_() for key in global_keys]
    reference_values = [value.double().requires_grad_() for value in global_values]
    scores = torch.einsum("qbhd,kbhd->hqk", reference_query, torch.cat(reference_keys)) * scale
    excluded = torch.arange(sum(lengths))[None, :] > (
        sum(prefix_lengths) + torch.arange(current_length)[:, None]
    )
    probabilities = scores.masked_fill(excluded, -torch.inf).softmax(-1)
    expected = torch.einsum("hqk,kbhd->qbhd", probabilities, torch.cat(reference_values))
    gradients = torch.autograd.grad(expected, (reference_query, *reference_keys, *reference_values), upstream.double())

    torch.testing.assert_close(actual, current_shard.select(expected).flatten(2).float(), rtol=1e-5, atol=1e-6)
    actual_gradients = (local_query.grad, *[key.grad for key in local_keys], *[value.grad for value in local_values])
    gradient_shards = (current_shard, *shards, *shards)
    for observed, gradient, shard in zip(actual_gradients, gradients, gradient_shards, strict=True):
        torch.testing.assert_close(observed, shard.select(gradient).float(), rtol=1e-5, atol=1e-6)
        assert torch.count_nonzero(observed[shard.valid_local_length:]) == 0
    assert segment_calls == {"forward": len(lengths), "backward": len(lengths)}
