"""Qwen3.5-0.8B Hybrid TPR Push/Branch/Pop equivalence on Ascend.

This test runs the native materialized P+S1 / P+S2 reference, the current
production SegmentExecutor, and a connected split reference. It checks the
4-layer (3 GDN + 1 Full Attention) and 24-layer (18 GDN + 6 Full Attention)
models, parameter gradients, and the external KV / Conv / Recurrent boundary
VJPs. All three pairwise diagnostics are reported before failing the test.

Run from the verl repository root with the shared MindSpeed-first pytest
bootstrap (one process, CP=1)::

    torchrun --master_addr=127.0.0.1 --master_port=29563 --nproc_per_node=1 \
      -m pytest -s -v tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import sys

import pytest
import torch
import torch.nn.functional as F

from verl.utils.device import is_torch_npu_available

from ._gdn_npu_test_utils import (
    AllToAllProbe,
    VOCAB_SIZE,
    assert_hybrid_architecture,
    gradient_map_diagnostics,
    make_qwen35_model,
)


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@dataclass(frozen=True)
class _HybridPlan:
    segment_plan: object
    loss_terms: dict
    total_loss_weight: float


def _plan(length: int = 64) -> _HybridPlan:
    from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec

    prefix, suffix_a, suffix_b = (
        (torch.arange(length, dtype=torch.long) * step + start) % VOCAB_SIZE
        for step, start in ((17, 23), (19, 101), (23, 307))
    )
    root_terms = [(i, int(prefix[i + 1]), 2.0) for i in range(length - 1)]
    root_terms.extend(((length - 1, int(suffix_a[0]), 1.0), (length - 1, int(suffix_b[0]), 1.0)))
    terms = {
        0: tuple(root_terms),
        1: tuple((i, int(suffix_a[i + 1]), 1.0) for i in range(length - 1)),
        2: tuple((i, int(suffix_b[i + 1]), 1.0) for i in range(length - 1)),
    }
    segments = (
        SegmentSpec(0, None, prefix, 0, 0),
        SegmentSpec(1, 0, suffix_a, length, length),
        SegmentSpec(2, 0, suffix_b, length, length),
    )
    total_weight = float(sum(weight for rows in terms.values() for _, _, weight in rows))
    assert total_weight == float(2 * (2 * length - 1))
    return _HybridPlan(SegmentPlan(segments, root_id=0), terms, total_weight)


def _term_logprobs(logits: torch.Tensor, segment, plan: _HybridPlan) -> torch.Tensor:
    terms = plan.loss_terms[segment.segment_id]
    if not terms:
        return logits.new_empty((0,), dtype=torch.float32)
    offsets = torch.tensor([term[0] for term in terms], device=logits.device)
    targets = torch.tensor([term[1] for term in terms], device=logits.device)
    return -F.cross_entropy(logits[0].index_select(0, offsets).float(), targets, reduction="none")


def _loss_callback(plan: _HybridPlan):
    def callback(segment, logits):
        terms = plan.loss_terms[segment.segment_id]
        offsets = torch.tensor([term[0] for term in terms], device=logits.device)
        targets = torch.tensor([term[1] for term in terms], device=logits.device)
        weights = torch.tensor([term[2] for term in terms], device=logits.device)
        token_losses = F.cross_entropy(
            logits[0].index_select(0, offsets).float(), targets, reduction="none"
        )
        return (token_losses * weights).sum() / plan.total_loss_weight, {}

    return callback


def _parameter_grads(model) -> dict[str, torch.Tensor]:
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"missing parameter gradient: {name}"
            assert torch.isfinite(parameter.grad).all().item(), f"non-finite parameter gradient: {name}"
            result[name] = parameter.grad.detach().cpu().clone()
    return result


def _boundary_tensors(gdn, kv):
    result = {}
    for layer, state in gdn.items():
        result[f"gdn.{layer}.conv"] = state.conv_state
        result[f"gdn.{layer}.recurrent"] = state.recurrent_state
    for layer, (key, value) in kv.items():
        result[f"fa.{layer}.key"] = key
        result[f"fa.{layer}.value"] = value
    return result


@contextmanager
def _no_cp_probe(monkeypatch):
    import mindspeed.core.ssm.gated_delta_net as gdn
    import mindspeed.core.context_parallel.dot_product_attention as dpa

    ring_calls = []
    targets = [dpa]
    legacy = sys.modules.get("mindspeed.core.transformer.dot_product_attention")
    if legacy is not None and legacy is not dpa:
        targets.append(legacy)

    with monkeypatch.context() as patch, AllToAllProbe(gdn) as a2a:
        for module in targets:
            original = module.ringattn_context_parallel

            def traced(*args, _original=original, **kwargs):
                ring_calls.append(True)
                return _original(*args, **kwargs)

            patch.setattr(module, "ringattn_context_parallel", traced)
        yield
    assert not a2a.calls and not ring_calls, f"CP1 entered A2A/Ring: {a2a.calls}/{ring_calls}"


def _make_executor(model, plan: _HybridPlan, executor_cls=None):
    from verl.workers.engine.megatron.tpr.executor import SegmentExecutor, prepare_replay_model

    prepared = prepare_replay_model(model)
    cls = SegmentExecutor if executor_cls is None else executor_cls
    counts = {segment_id: len(rows) for segment_id, rows in plan.loss_terms.items()}
    return cls(
        model,
        plan.segment_plan,
        expected_layer_numbers=prepared.layer_numbers,
        prepared_model=prepared,
        segment_loss_fn=_loss_callback(plan),
        segment_loss_term_counts=counts,
    )


def _native_reference(model, plan: _HybridPlan, device):
    model.zero_grad(set_to_none=True)
    prefix_length = plan.segment_plan.get(0).length
    losses = []
    logprobs = {}
    for leaf_id in (1, 2):
        prefix = plan.segment_plan.get(0).token_ids
        suffix = plan.segment_plan.get(leaf_id).token_ids
        tokens = torch.cat((prefix, suffix)).to(device)
        positions = torch.arange(tokens.numel(), device=device).unsqueeze(0)
        logits = model(tokens.unsqueeze(0), positions, attention_mask=None)
        assert logits.shape == (1, tokens.numel(), VOCAB_SIZE)
        loss = F.cross_entropy(
            logits[0, :-1].float(), tokens[1:], reduction="sum"
        ) / plan.total_loss_weight
        losses.append(loss.detach().cpu())
        logprobs[leaf_id] = {
            0: _term_logprobs(logits[:, :prefix_length], plan.segment_plan.get(0), plan).detach().cpu(),
            leaf_id: _term_logprobs(
                logits[:, prefix_length:], plan.segment_plan.get(leaf_id), plan
            ).detach().cpu(),
        }
        loss.backward()
        del logits, loss
    root_logprobs = logprobs[1][0].clone()
    root_logprobs[-1] = logprobs[2][0][-1]
    return sum(losses), {0: root_logprobs, 1: logprobs[1][1], 2: logprobs[2][2]}, _parameter_grads(model)


def _run_tpr_executor(model, plan: _HybridPlan):
    from verl.workers.engine.megatron.tpr.executor import SegmentExecutor

    instances = []

    class ObservedExecutor(SegmentExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.forwards = []
            self.loss_calls = Counter()
            self.logprobs = {}
            self.boundary_gradients = {}
            self.saved_state = None
            instances.append(self)

        def _forward(self, segment, **kwargs):
            context, logits = super()._forward(segment, **kwargs)
            self._assert_collected_layers(context)
            self.forwards.append((segment.segment_id, kwargs["no_grad"]))
            return context, logits

        def _compute_loss(self, segment, logits):
            self.loss_calls[segment.segment_id] += 1
            self.logprobs[segment.segment_id] = _term_logprobs(logits, segment, plan).detach().cpu()
            return super()._compute_loss(segment, logits)

        def push(self, segment_id):
            result = super().push(segment_id)
            self.saved_state = self.gdn_states[segment_id]
            for tensor in _boundary_tensors(
                self.saved_state.layer_states, self.kv_stack.top().kv.key_values
            ).values():
                assert not tensor.requires_grad and tensor.grad_fn is None
            return result

        def pop(self, segment_id):
            tensors = _boundary_tensors(
                self.gdn_states[segment_id].gradients, self.kv_stack.top().gradients
            )
            self.boundary_gradients = {
                name: tensor.detach().cpu().clone() for name, tensor in tensors.items()
            }
            return super().pop(segment_id)

    model.zero_grad(set_to_none=True)
    executor = _make_executor(model, plan, ObservedExecutor)
    loss = executor.run()
    assert len(instances) == 1
    executor = instances[0]
    assert executor.forwards == [(0, True), (1, False), (2, False), (0, False)]
    assert executor.loss_calls == Counter({0: 1, 1: 1, 2: 1})
    assert len(executor.boundary_gradients) == model.config.num_layers * 2
    assert executor.saved_state.released
    assert not executor.gdn_states
    executor.kv_stack.assert_empty()
    return loss.detach().cpu(), executor.logprobs, _parameter_grads(model), executor.boundary_gradients


def _connected_split_reference(model, plan: _HybridPlan):
    from verl.experimental.tpr.gdn.prefix_state import GDNLayerState

    model.zero_grad(set_to_none=True)
    executor = _make_executor(model, plan)
    root_context, logits = executor._forward(
        plan.segment_plan.get(0), past_key_values={}, no_grad=False
    )
    loss = executor._compute_loss(plan.segment_plan.get(0), logits)
    logprobs = {
        0: _term_logprobs(logits, plan.segment_plan.get(0), plan).detach().cpu()
    }

    gdn_states = {
        layer: GDNLayerState(state.conv_state.clone(), state.recurrent_state.clone())
        for layer, state in root_context.new_gdn_states.items()
    }
    kv_states = {
        layer: (key.clone(), value.clone())
        for layer, (key, value) in root_context.new_key_values.items()
    }
    boundaries = _boundary_tensors(gdn_states, kv_states)
    for tensor in boundaries.values():
        tensor.retain_grad()

    for leaf_id in (1, 2):
        segment = plan.segment_plan.get(leaf_id)
        _, leaf_logits = executor._forward(
            segment,
            past_key_values=kv_states,
            initial_gdn_states=gdn_states,
            no_grad=False,
        )
        logprobs[leaf_id] = _term_logprobs(leaf_logits, segment, plan).detach().cpu()
        loss = loss + executor._compute_loss(segment, leaf_logits)

    loss.backward()
    boundary_gradients = {}
    for name, tensor in boundaries.items():
        assert tensor.grad is not None, f"missing connected boundary gradient: {name}"
        boundary_gradients[name] = tensor.grad.detach().cpu().clone()
    return _parameter_grads(model), boundary_gradients, loss.detach().cpu(), logprobs


def _assert_matching_boundary_keys(reference, actual):
    assert reference.keys() == actual.keys(), (
        f"boundary keys differ: missing={reference.keys() - actual.keys()}, "
        f"unexpected={actual.keys() - reference.keys()}"
    )


@pytest.mark.parametrize("num_layers", (4, 24), ids=("hybrid4", "hybrid24"))
def test_full_qwen35_hybrid_push_branch_pop_matches_materialized_paths(runtime, monkeypatch, num_layers):
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.experimental.tpr.gdn.model import TPRGatedDeltaNet

    torch.manual_seed(353301)
    reference_model = make_qwen35_model(runtime, cp_size=1, tpr=False, num_layers=num_layers)
    torch.manual_seed(353302)
    tpr_model = make_qwen35_model(runtime, cp_size=1, tpr=True, num_layers=num_layers)
    tpr_model.load_state_dict(reference_model.state_dict(), strict=True)

    gdn_layers, fa_layers = assert_hybrid_architecture(tpr_model)
    assert len(gdn_layers) == 3 * num_layers // 4 and len(fa_layers) == num_layers // 4
    assert tuple(layer.layer_number for layer in fa_layers) == tuple(range(4, num_layers + 1, 4))
    assert all(isinstance(layer.self_attention, TPRGatedDeltaNet) for layer in gdn_layers)
    assert all(isinstance(layer.self_attention, TPRSelfAttention) for layer in fa_layers)
    assert tpr_model.config.attention_dropout == tpr_model.config.hidden_dropout == 0

    plan = _plan(64)
    reference_loss, reference_logprobs, reference_grads = _native_reference(
        reference_model, plan, runtime.device
    )

    with _no_cp_probe(monkeypatch):
        tpr_loss, tpr_logprobs, tpr_grads, tpr_boundary = _run_tpr_executor(tpr_model, plan)

    connected_model = make_qwen35_model(runtime, cp_size=1, tpr=True, num_layers=num_layers)
    connected_model.load_state_dict(reference_model.state_dict(), strict=True)
    with _no_cp_probe(monkeypatch):
        connected_grads, connected_boundary, connected_loss, connected_logprobs = (
            _connected_split_reference(connected_model, plan)
        )

    # Do not stop on the first logprob mismatch: the state and parameter
    # gradients have already been collected, and identify a different seam.
    # Preserve all original acceptance thresholds.
    failures = []

    def check(label, fn):
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{label}: {exc}")
            print(f"GATE FAIL {label}: {exc}", flush=True)
        else:
            print(f"GATE PASS {label}", flush=True)

    def report_gradient(label, ref, actual, *, rtol=None, cosine_min=None):
        diagnostic = gradient_map_diagnostics(ref, actual)
        metrics = diagnostic.aggregate
        print(
            f"{label}: ref_norm={metrics.reference_norm:.9e}, "
            f"actual_norm={metrics.actual_norm:.9e}, norm_ratio={metrics.norm_ratio:.9f}, "
            f"absolute_l2={metrics.absolute_l2:.9e}, relative_l2={metrics.relative_l2:.9e}, "
            f"cosine={metrics.cosine:.9f}, "
            f"worst={diagnostic.worst_name}:{diagnostic.worst.relative_l2:.9e}",
            flush=True,
        )
        if rtol is not None and cosine_min is not None:
            if metrics.relative_l2 > rtol or metrics.cosine < cosine_min:
                message = (
                    f"{label}: relative_l2={metrics.relative_l2:.9e} (limit {rtol:.9e}), "
                    f"cosine={metrics.cosine:.9f} (minimum {cosine_min:.9f})"
                )
                failures.append(message)
                print(f"GATE FAIL {message}", flush=True)
            else:
                print(f"GATE PASS {label}", flush=True)

    print(f"\n===== HYBRID CP1 layers={num_layers} GDN={len(gdn_layers)} FA={len(fa_layers)} =====", flush=True)
    print("===== LOSS =====", flush=True)
    print(f"Native:    {reference_loss.item():.9f}", flush=True)
    print(f"Connected: {connected_loss.item():.9f}", flush=True)
    print(f"TPR:       {tpr_loss.item():.9f}", flush=True)
    check(
        "Native vs TPR/loss",
        lambda: torch.testing.assert_close(tpr_loss, reference_loss, atol=2e-2, rtol=2e-2),
    )

    pairs = (
        ("Native vs Connected", reference_logprobs, connected_logprobs, reference_grads, connected_grads),
        ("Connected vs TPR", connected_logprobs, tpr_logprobs, connected_grads, tpr_grads),
        ("Native vs TPR", reference_logprobs, tpr_logprobs, reference_grads, tpr_grads),
    )
    print("===== LOGPROB DIAGNOSTICS =====", flush=True)
    for segment_id in (0, 1, 2):
        for label, ref_logprobs, actual_logprobs, _, _ in pairs:
            ref = ref_logprobs[segment_id].detach().float()
            actual = actual_logprobs[segment_id].detach().float()
            diff = (actual - ref).abs()
            relative_l2 = (diff.norm() / ref.norm().clamp_min(1e-12)).item()
            mismatch = ~torch.isclose(actual, ref, atol=8e-2, rtol=2e-2)
            print(
                f"seg={segment_id} {label}: max_abs={diff.max().item():.8f}, "
                f"rel_l2={relative_l2:.8f}, mismatch={mismatch.sum().item()}/{ref.numel()}",
                flush=True,
            )
            if label != "Native vs Connected":
                check(
                    f"{label}/logprob/seg={segment_id}",
                    lambda: torch.testing.assert_close(actual, ref, atol=8e-2, rtol=2e-2),
                )

    print("===== PARAMETER GRADIENTS =====", flush=True)
    for label, _, _, ref_grads, actual_grads in pairs:
        # Native/Connected is a diagnostic, not an additional acceptance gate.
        thresholds = {
            "Native vs TPR": (0.10, 0.995),
            "Connected vs TPR": (0.02, 0.999),
        }
        rtol, cosine_min = thresholds.get(label, (None, None))
        report_gradient(f"{label}/parameters", ref_grads, actual_grads, rtol=rtol, cosine_min=cosine_min)

    print("===== STATE GRADIENTS (Connected vs TPR) =====", flush=True)
    check("boundary keys", lambda: _assert_matching_boundary_keys(connected_boundary, tpr_boundary))
    if tpr_boundary.keys() == connected_boundary.keys():
        for kind in ("gdn.", "fa."):
            names = tuple(name for name in tpr_boundary if name.startswith(kind))
            if names:
                report_gradient(
                    f"Connected vs TPR/{kind[:-1]}-states",
                    {name: connected_boundary[name] for name in names},
                    {name: tpr_boundary[name] for name in names},
                )
        for name in tpr_boundary:
            report_gradient(
                name,
                {name: connected_boundary[name]},
                {name: tpr_boundary[name]},
                rtol=0.02,
                cosine_min=0.999,
            )

    if failures:
        pytest.fail("Hybrid correctness gates failed (all gates evaluated):\n" + "\n".join(failures))

    print(
        "QWEN35_HYBRID_TPR_CP1_PASS "
        f"layers={len(tpr_model.decoder.layers)} GDN={len(gdn_layers)} FA={len(fa_layers)} "
        f"boundary_gradients={len(tpr_boundary)} loss={tpr_loss.item():.8f}",
        flush=True,
    )
