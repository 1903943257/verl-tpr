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

import verl.models.mcore.tpr.parallel.ring as ring


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
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_FULL", str(int(coalesce_full)))
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_QUERY", str(int(coalesce_query)))
    monkeypatch.setattr(ring, "_group_world_size_and_rank", lambda group: (cp_size, 0))
    monkeypatch.setattr(ring, "_process_group_global_ranks", lambda group: tuple(range(cp_size)))
    monkeypatch.setattr(ring, "_block_attention_forward", _fa_forward)
    monkeypatch.setattr(ring, "_block_attention_backward", _fa_backward)
    monkeypatch.setattr(ring, "_circulate_kv", _circulate_symmetric_sources)
    monkeypatch.setattr(ring, "_reduce_ring_gradients_to_owner", _replay_symmetric_sources)

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
