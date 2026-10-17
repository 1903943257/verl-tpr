"""Check the production PPO executor's branch gradients against full rows."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl.models.mcore.tpr.context import (
    TPRAttentionContext,
    get_tpr_attention_context,
    use_tpr_attention_context,
)
from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans


class _PrefixModel(nn.Module):
    """A causal model whose outputs depend on both cached K and V."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 3, dtype=torch.float64)
        self.head = nn.Linear(3, 7, dtype=torch.float64)
        self.rotary_pos_emb = lambda length, **_: torch.zeros(length, 1, 1, 2)

    def forward(self, *, input_ids, position_ids=None, attention_mask=None):
        context = get_tpr_attention_context()
        key = self.embedding(input_ids.remainder(16)).transpose(0, 1).unsqueeze(2)
        value = key.square()
        context.set_new_kv(1, key, value)
        hidden = (key + value).cumsum(0)
        past = context.get_past_kv(1)
        if past is not None:
            hidden = hidden + (past[0] + past[1]).sum(0, keepdim=True)
        return self.head(hidden.squeeze(2)).transpose(0, 1)


def _batch():
    # Includes a multi-level branch, duplicate rows, and an internal terminal.
    rows = [
        [1, 151645, 2, 151645, 3, 4],
        [1, 151645, 2, 151645, 5, 6],
        [1, 151645, 7, 8],
        [1, 151645, 2, 151645, 3, 4],
        [1, 151645, 2, 151645],
    ]
    return {
        "input_ids": [torch.tensor(row) for row in rows],
        "prompts": [torch.tensor(row[:2]) for row in rows],
        "responses": [torch.tensor(row[2:]) for row in rows],
        "response_mask": [torch.ones(len(row) - 2, dtype=torch.bool) for row in rows],
    }


@pytest.mark.parametrize("loss_scale", [1.0, 8.0])
@pytest.mark.parametrize("mask_branch_tokens", [False, True])
@pytest.mark.parametrize("freeze_embedding", [False, True])
def test_branch_replay_matches_full_rows(loss_scale, mask_branch_tokens, freeze_embedding):
    torch.manual_seed(37)
    model = _PrefixModel()
    model.embedding.requires_grad_(not freeze_embedding)
    full_model = copy.deepcopy(model)
    batch = _batch()
    if mask_branch_tokens:
        # Some segments then have no owned objective, but must relay dKV.
        for mask in batch["response_mask"]:
            mask[0] = False
    forest = build_tree_execution_plans(
        tuple(f"sample_tpr_{row}" for row in range(len(batch["input_ids"]))),
        batch,
        turn_min_segment_tokens=1,
    )
    denominator = forest.logical_loss_tokens
    full_losses = []
    for row, tokens in enumerate(batch["input_ids"]):
        with use_tpr_attention_context(TPRAttentionContext(0, len(tokens))):
            logits = full_model(input_ids=tokens.unsqueeze(0))
        for offset, active in enumerate(batch["response_mask"][row]):
            if active:
                full_losses.append(
                    logits[0, offset + 1, int(tokens[offset + 2]) % 7].square() * (row + 1)
                )
    full_loss = torch.stack(full_losses).sum() / denominator
    (full_loss * loss_scale).backward()

    tree_losses = []
    for tree in forest.trees:
        refs = {
            segment_id: tree.refs_for_segment(segment_id)
            for segment_id in tree.segment_plan.segments
        }

        def objective(segment, logits):
            terms = [
                logits[0, ref.query_offset, ref.target_token_id % 7].square() * (ref.sample_row + 1)
                for ref in refs[segment.segment_id]
            ]
            return torch.stack(terms).sum() / denominator, {}

        executor = SegmentExecutor(
            model, tree.segment_plan, expected_layer_numbers=(1,),
            segment_loss_fn=objective,
            segment_loss_term_counts={key: len(value) for key, value in refs.items()},
            loss_scale_func=lambda loss: loss * loss_scale,
        )
        profiled = []
        profiler = SimpleNamespace(
            begin=lambda segment, **fields: (segment.segment_id, fields["kind"]),
            end=profiled.append,
        )
        tree_losses.append(executor.run(event_profiler=profiler))
        pushed = [segment for segment, kind in profiled if kind == "push"]
        popped = [segment for segment, kind in profiled if kind == "pop"]
        leaves = [segment for segment, kind in profiled if kind == "visit_leaf"]
        assert set(pushed) == set(popped)
        assert len(popped) + len(leaves) == len(tree.segment_plan.segments)
        assert not executor.failed
        with pytest.raises(RuntimeError, match="single-use"):
            executor.run()
        executor.kv_stack.assert_empty()
    torch.testing.assert_close(sum(tree_losses), full_loss.detach())
    for actual, expected in zip(model.parameters(), full_model.parameters(), strict=True):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)
    assert get_tpr_attention_context() is None


def test_failed_replay_restores_context_and_cannot_resume():
    model = _PrefixModel()
    forest = build_tree_execution_plans(
        tuple(f"sample_tpr_{row}" for row in range(5)), _batch(), turn_min_segment_tokens=1,
    )
    tree = forest.trees[0]

    def fail(*_):
        raise RuntimeError("injected PPO failure")

    executor = SegmentExecutor(
        model, tree.segment_plan, expected_layer_numbers=(1,), segment_loss_fn=fail,
        segment_loss_term_counts={key: len(tree.refs_for_segment(key)) for key in tree.segment_plan.segments},
    )
    with pytest.raises(RuntimeError, match="injected PPO failure"):
        executor.run()
    assert executor.failed
    assert get_tpr_attention_context() is None
    with pytest.raises(RuntimeError, match="failed"):
        executor.push(tree.segment_plan.root_id)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("owned", [False, True])
def test_cp_loss_scaling_and_empty_rank_connected_zero(cp_size, owned):
    executor = object.__new__(SegmentExecutor)
    executor.cp_size = cp_size
    executor.loss_scale_func = lambda loss: loss * 8
    logits = torch.randn(1, 3, 7, requires_grad=True)
    loss = logits.square().sum() / 13
    scaled = executor._prepare_backward_loss(loss, logits=logits, has_owned_loss=owned)
    scaled.backward()
    expected = logits.detach() * (16 * cp_size / 13) if owned else torch.zeros_like(logits)
    torch.testing.assert_close(logits.grad, expected)


def test_push_caches_compact_detached_kv_in_state(monkeypatch):
    """Push transfers the context's detached compact tensors without rewrapping."""
    model = _PrefixModel()
    forest = build_tree_execution_plans(
        tuple(f"sample_tpr_{row}" for row in range(5)),
        _batch(),
        turn_min_segment_tokens=1,
    )
    tree = forest.trees[0]
    executor = SegmentExecutor(
        model,
        tree.segment_plan,
        expected_layer_numbers=(1,),
        segment_loss_fn=lambda segment, logits: (logits.sum(), {}),
        segment_loss_term_counts={
            segment_id: len(tree.refs_for_segment(segment_id))
            for segment_id in tree.segment_plan.segments
        },
    )
    forward, recorded = executor._forward, {}

    def capture_push(*args, **kwargs):
        context, logits = forward(*args, **kwargs)
        assert context.compact_new_kv and kwargs["no_grad"]
        recorded.update(context.new_key_values)
        return context, logits

    monkeypatch.setattr(executor, "_forward", capture_push)
    executor.push(tree.segment_plan.root_id)
    for layer, pair in executor.kv_stack.top().kv.key_values.items():
        for tensor, original in zip(pair, recorded[layer], strict=True):
            assert tensor is original
            assert tensor.grad_fn is None and not tensor.requires_grad
            assert tensor.is_contiguous()
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    executor.kv_stack.pop(tree.segment_plan.root_id).kv.release()
    executor.kv_stack.assert_empty()


def _executor(model, *, owns_loss=True):
    forest = build_tree_execution_plans(
        tuple(f"sample_tpr_{row}" for row in range(5)), _batch(), turn_min_segment_tokens=1,
    )
    tree = forest.trees[0]
    return SegmentExecutor(
        model, tree.segment_plan, expected_layer_numbers=(1,),
        segment_loss_fn=lambda segment, logits: (logits.square().sum(), {}),
        segment_loss_term_counts={
            segment_id: int(owns_loss) for segment_id in tree.segment_plan.segments
        },
    )


def test_all_frozen_zero_objective_can_finish_without_a_backward_root():
    model = _PrefixModel().requires_grad_(False)
    executor = _executor(model, owns_loss=False)
    assert executor.run().item() == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not executor.failed
    executor.kv_stack.assert_empty()


def test_empty_objective_does_not_cast_or_reduce_the_vocabulary():
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoVocabularyCast(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._to_copy.default:
                assert args[0].numel() <= 1, "empty objective cast the full vocabulary"
            if func in (torch.ops.aten.sum.default, torch.ops.aten.sum.dim_IntList):
                assert args[0].numel() == 0, "empty objective reduced the full vocabulary"
            return func(*args, **(kwargs or {}))

    executor = object.__new__(SegmentExecutor)
    executor.segment_loss_term_counts = {0: 0}
    executor.cp_size = 4
    executor.loss_scale_func = lambda loss: loss * 8
    logits = torch.full((1, 8, 65536), float("nan"), dtype=torch.bfloat16, requires_grad=True)
    backward_calls = []
    logits.register_hook(lambda grad: backward_calls.append(grad.shape))
    with NoVocabularyCast():
        loss = executor._compute_loss(SimpleNamespace(segment_id=0), logits)
        backward_loss = executor._prepare_backward_loss(loss, logits=logits, has_owned_loss=False)
    assert loss.item() == backward_loss.item() == 0
    backward_loss.backward()
    assert backward_calls == [logits.shape]
    assert logits.grad.dtype == logits.dtype
    assert torch.count_nonzero(logits.grad).item() == 0


@pytest.mark.parametrize("name", ["hidden_dropout", "embedding_dropout", "attention_dropout"])
def test_replay_rejects_configured_dropout(name):
    model = _PrefixModel()
    model.config = SimpleNamespace(**{name: 0.1})
    with pytest.raises(ValueError, match=f"requires {name}=0"):
        _executor(model)


@pytest.mark.parametrize("dropout", [nn.Dropout, nn.Dropout1d, nn.AlphaDropout])
def test_replay_rejects_training_dropout_modules(dropout):
    model = _PrefixModel()
    model.add_module("stochastic", dropout(0.1))
    with pytest.raises(ValueError, match="training Dropout p=0"):
        _executor(model)


def test_replay_rechecks_dropout_when_model_changes_to_training():
    model = _PrefixModel()
    model.add_module("stochastic", nn.Dropout(0.1))
    executor = _executor(model.eval())
    model.train()
    with pytest.raises(ValueError, match="training Dropout p=0"):
        executor.run()
