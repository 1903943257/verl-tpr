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

"""Push/pop execution of one TPR segment against a Megatron GPT model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass
from enum import Enum
import os
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from verl.models.mcore.tpr.context import KVPair, TPRAttentionContext, use_tpr_attention_context
from .state import KVStack
from .ring import RingCPBackend, PrefixShard, SequenceShard, resolve_cp_group
from .state import (
    ShardedPastKVAnchors,
    accumulate_sharded_past_anchor_gradients,
    build_sharded_past_anchors,
)
from .tree import SegmentEvent, SegmentEventKind, SegmentId, SegmentPlan, SegmentSpec


class _RotaryEmbedding(Protocol):
    def __call__(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: object | None = None,
    ) -> Tensor: ...


class _UnshardedContextParallelGroup:
    """Size-one sentinel that prevents RotaryEmbedding using its bound CP group."""

    @staticmethod
    def size() -> int:
        return 1


_UNSHARDED_CP_GROUP = _UnshardedContextParallelGroup()


@contextmanager
def disable_bound_context_parallel_sharding(rotary_embedding: _RotaryEmbedding):
    """Temporarily stop implicit CP slicing by a bound RotaryEmbedding group.

    GPTModel builds a rotary embedding in ``_preprocess`` even though TPR
    attention consumes the rank-local embedding stored in its execution
    context.  For padded, non-divisible sequence lengths that unused build can
    fail in MindSpeed before TPR attention is reached.  Override the embedding's
    bound group only for the model forward and restore it on every exit path.
    """

    if not hasattr(rotary_embedding, "cp_group"):
        yield
        return
    cp_group = rotary_embedding.cp_group
    rotary_embedding.cp_group = _UNSHARDED_CP_GROUP
    try:
        yield
    finally:
        rotary_embedding.cp_group = cp_group


def build_suffix_rotary_pos_emb(
    rotary_embedding: _RotaryEmbedding,
    *,
    prefix_length: int,
    suffix_length: int,
    disable_context_parallel_sharding: bool = False,
) -> Tensor:
    """Build standard 1D RoPE frequencies for positions ``[P, P+b)``.

    This delegates frequency generation to the model's existing Megatron
    ``RotaryEmbedding`` instance, preserving its rotary dimension, base,
    interpolation, interleaving, and scaling settings.  The MVP deliberately
    requests neither packed-sequence nor context-parallel sharding.
    """

    _validate_length("prefix_length", prefix_length, allow_zero=True)
    _validate_length("suffix_length", suffix_length, allow_zero=False)
    if not callable(rotary_embedding):
        raise TypeError("rotary_embedding must be callable")

    rotary_pos_emb = rotary_embedding(
        suffix_length,
        offset=prefix_length,
        packed_seq=False,
        cp_group=_UNSHARDED_CP_GROUP if disable_context_parallel_sharding else None,
    )
    if not isinstance(rotary_pos_emb, Tensor):
        raise TypeError(
            "TPR RoPE MVP requires standard RotaryEmbedding to return one tensor; "
            f"got {type(rotary_pos_emb).__name__}"
        )
    if rotary_pos_emb.ndim != 4:
        raise ValueError(
            "suffix rotary embedding must have shape [suffix, 1, 1, rotary_dim], "
            f"got {rotary_pos_emb.shape}"
        )
    if rotary_pos_emb.shape[0] != suffix_length:
        raise ValueError(
            f"suffix rotary embedding sequence length must be {suffix_length}, "
            f"got {rotary_pos_emb.shape[0]}"
        )
    if rotary_pos_emb.shape[1] != 1 or rotary_pos_emb.shape[2] != 1:
        raise ValueError(
            "TPR RoPE MVP requires singleton batch/head broadcast dimensions, "
            f"got {rotary_pos_emb.shape}"
        )
    if rotary_pos_emb.shape[3] <= 0:
        raise ValueError("suffix rotary embedding must have a positive rotary dimension")
    return rotary_pos_emb


def build_sharded_rotary_pos_emb(
    rotary_embedding: _RotaryEmbedding,
    *,
    position_start: int,
    shard: SequenceShard,
    disable_context_parallel_sharding: bool = False,
) -> Tensor:
    """Build RoPE in the local-token order described by ``shard``."""

    _validate_length("position_start", position_start, allow_zero=True)
    if not isinstance(shard, SequenceShard):
        raise TypeError(f"shard must implement SequenceShard, got {type(shard).__name__}")
    chunks = []
    for physical_start, physical_end in shard.physical_global_ranges:
        valid_end = min(physical_end, shard.global_length)
        valid_length = max(0, valid_end - physical_start)
        physical_length = physical_end - physical_start
        if valid_length:
            chunk = build_suffix_rotary_pos_emb(
                rotary_embedding,
                prefix_length=position_start + physical_start,
                suffix_length=valid_length,
                disable_context_parallel_sharding=disable_context_parallel_sharding,
            )
        else:
            chunk = build_suffix_rotary_pos_emb(
                rotary_embedding,
                prefix_length=position_start + shard.global_length - 1,
                suffix_length=1,
                disable_context_parallel_sharding=disable_context_parallel_sharding,
            )
        if valid_length < physical_length:
            padding = chunk[-1:].expand(physical_length - valid_length, *chunk.shape[1:])
            chunk = torch.cat((chunk if valid_length else chunk[:0], padding), dim=0)
        chunks.append(chunk)
    rotary_pos_emb = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    if rotary_pos_emb.shape[0] != shard.local_length:
        raise RuntimeError(
            f"sharded rotary embedding length must be {shard.local_length}, "
            f"got {rotary_pos_emb.shape[0]}"
        )
    return rotary_pos_emb


def _validate_length(name: str, value: int, *, allow_zero: bool) -> None:
    lower_bound = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class SegmentForwardResult:
    segment_id: SegmentId
    prefix_length: int
    suffix_length: int
    layer_count: int


@dataclass(frozen=True, slots=True)
class SegmentBackwardResult:
    segment_id: SegmentId
    loss_sum: Tensor
    normalized_loss: Tensor
    loss_term_count: int
    relayed_layer_count: int


@dataclass(frozen=True, slots=True)
class LeafVisitResult:
    """One physical leaf execution replacing an adjacent logical Push/Pop pair."""

    forward: SegmentForwardResult
    backward: SegmentBackwardResult


class SegmentExecutor:
    """Execute graph-free Push and gradient-carrying Pop for a SegmentPlan.

    The executor deliberately does not choose traversal order, synchronize
    gradients, or step an optimizer. Those responsibilities belong to the
    fixed-topology scheduler and the later Megatron entry adapter.
    """

    def __init__(
        self,
        model: nn.Module,
        plan: SegmentPlan,
        *,
        expected_layer_numbers: tuple[int, ...] | None = None,
        kv_stack: KVStack | None = None,
        loss_scale_func: Callable[[Tensor], Tensor] | None = None,
        cp_group: Any | None = None,
        cp_backend: RingCPBackend | str | None = None,
        segment_loss_fn: Callable[[SegmentSpec, Tensor], tuple[Tensor, dict[str, Any]]] | None = None,
        segment_loss_term_counts: Mapping[SegmentId, int] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")
        if not isinstance(plan, SegmentPlan):
            raise TypeError(f"plan must be SegmentPlan, got {type(plan).__name__}")
        if not hasattr(model, "rotary_pos_emb") or not callable(model.rotary_pos_emb):
            raise TypeError("model must expose a callable rotary_pos_emb")
        if expected_layer_numbers is None:
            expected_layer_numbers = _infer_layer_numbers(model)
        expected_layer_numbers = tuple(expected_layer_numbers)
        if not expected_layer_numbers:
            raise ValueError("expected_layer_numbers must not be empty")
        if any(
            not isinstance(layer, int) or isinstance(layer, bool) or layer <= 0
            for layer in expected_layer_numbers
        ):
            raise ValueError(f"expected_layer_numbers must contain positive integers, got {expected_layer_numbers}")
        if len(set(expected_layer_numbers)) != len(expected_layer_numbers):
            raise ValueError(f"expected_layer_numbers contains duplicates: {expected_layer_numbers}")

        if not plan.topology_only:
            raise ValueError("Production TPR requires a PPO topology-only SegmentPlan")
        if not callable(segment_loss_fn):
            raise TypeError("Production TPR requires a callable segment_loss_fn")
        counts = dict(segment_loss_term_counts or {})
        if not counts or any(
            segment_id not in plan.segments or count < 0
            for segment_id, count in counts.items()
        ):
            raise ValueError("invalid PPO segment objective counts")
        self.segment_loss_fn = segment_loss_fn
        self.segment_loss_term_counts = counts
        self.segment_loss_metrics: list[tuple[SegmentId, dict[str, Any]]] = []

        self.model = model
        self.plan = plan
        self.expected_layer_numbers = tuple(sorted(expected_layer_numbers))
        self.kv_stack = KVStack() if kv_stack is None else kv_stack
        self.loss_scale_func = loss_scale_func
        self.cp_group = cp_group
        if cp_group is None:
            if cp_backend is not None:
                raise ValueError("cp_backend requires a cp_group")
            self.cp_size, self.cp_rank = 1, 0
            self.cp_backend = None
        else:
            self.cp_size, self.cp_rank = resolve_cp_group(cp_group)
            if self.cp_size <= 1:
                raise ValueError(f"cp_group must contain more than one rank, got {self.cp_size}")
            if isinstance(cp_backend, RingCPBackend):
                if (cp_backend.parallel_size, cp_backend.parallel_rank) != (self.cp_size, self.cp_rank):
                    raise ValueError("RingCPBackend coordinates do not match initialized CP group")
                self.cp_backend = cp_backend
            elif cp_backend in (None, "ring", "megatron_cp_algo"):
                self.cp_backend = RingCPBackend(
                    cp_group, parallel_size=self.cp_size, parallel_rank=self.cp_rank,
                )
            else:
                raise NotImplementedError("Production TPR only supports Ring CP")
            for segment in plan.segments.values():
                try:
                    self.cp_backend.validate_segment_length(segment.length)
                except ValueError as exc:
                    raise ValueError(
                        f"segment {segment.segment_id} {exc}"
                    ) from exc
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def cp_enabled(self) -> bool:
        return self.cp_group is not None

    def push(self, segment_id: SegmentId) -> SegmentForwardResult:
        """Save graph-free per-layer Prefix KV without owning loss."""

        self._ensure_healthy()
        try:
            segment = self.plan.get(segment_id)
            context, logits = self._forward(
                segment,
                past_key_values={} if self.cp_enabled else self.kv_stack.build_past_key_values(),
                no_grad=True,
            )
            del logits
            self._assert_collected_layers(context)
            cached_key_values = _compact_kv_cache(context.new_key_values)
            del context
            self.kv_stack.push(segment, cached_key_values, shard=self._segment_shard(segment))
            return SegmentForwardResult(
                segment_id=segment_id,
                prefix_length=segment.prefix_length,
                suffix_length=segment.length,
                layer_count=len(cached_key_values),
            )
        except Exception:
            self._failed = True
            raise

    def pop(self, segment_id: SegmentId) -> SegmentBackwardResult:
        """Recompute current KV and relay gradients from descendant Segments."""

        self._ensure_healthy()
        try:
            entry = self.kv_stack.top()
            if entry.segment.segment_id != segment_id:
                raise RuntimeError(f"cannot pop segment {segment_id}: stack top is {entry.segment.segment_id}")

            relayed_gradients = dict(entry.gradients)
            if relayed_gradients and tuple(relayed_gradients) != self.expected_layer_numbers:
                raise RuntimeError(
                    f"segment {segment_id} relayed KV gradient layers must be {self.expected_layer_numbers}, "
                    f"got {tuple(relayed_gradients)}"
                )
            popped_entry = self.kv_stack.pop(segment_id)
            segment = popped_entry.segment
            del entry
            # The popped segment's cached KV is dead before recompute: its
            # relayed gradients and metadata were copied above, while the
            # replay consumes only the remaining ancestor path as past KV.
            # Release it for CP=1 as well as CP>1 so cached current KV does not
            # overlap with the graph-connected KV recreated by this forward.
            popped_entry.kv.release()
            if self.cp_enabled:
                anchors = build_sharded_past_anchors(self.kv_stack)
                past_key_values = {}
            else:
                anchors = self.kv_stack.build_past_anchors()
                past_key_values = anchors.key_values
            context, logits = self._forward(
                segment,
                past_key_values=past_key_values,
                no_grad=False,
                sharded_past_anchors=anchors if self.cp_enabled else None,
            )
            self._assert_collected_layers(context)

            loss_sum, normalized_loss = self._compute_loss(segment, logits)
            owned_loss_term_count = self._owned_loss_count(segment)
            roots: list[Tensor] = []
            root_gradients: list[Tensor | None] = []
            if owned_loss_term_count or self.cp_enabled:
                backward_loss = self._prepare_backward_loss(
                    normalized_loss,
                    logits=logits,
                    has_owned_loss=owned_loss_term_count > 0,
                )
                if not isinstance(backward_loss, Tensor) or backward_loss.numel() != 1:
                    raise TypeError("loss_scale_func must return a scalar tensor")
                roots.append(backward_loss)
                root_gradients.append(None)
            for layer_number in self.expected_layer_numbers:
                if layer_number not in relayed_gradients:
                    continue
                new_key, new_value = context.new_key_values[layer_number]
                key_grad, value_grad = relayed_gradients[layer_number]
                roots.extend((new_key, new_value))
                root_gradients.extend((key_grad, value_grad))
            if roots:
                torch.autograd.backward(roots, grad_tensors=root_gradients)
                self._accumulate_past_anchor_gradients(anchors)

            return SegmentBackwardResult(
                segment_id=segment_id,
                loss_sum=loss_sum.detach(),
                normalized_loss=normalized_loss.detach(),
                loss_term_count=owned_loss_term_count,
                relayed_layer_count=len(relayed_gradients),
            )
        except Exception:
            self._failed = True
            raise

    def visit_leaf(self, segment_id: SegmentId) -> LeafVisitResult:
        """Forward/backward one leaf without writing its new KV to the path stack."""

        self._ensure_healthy()
        try:
            segment = self.plan.get(segment_id)
            if self.plan.children_of(segment_id):
                raise ValueError(f"segment {segment_id} is not a leaf")

            expected_parent = self.kv_stack.top().segment.segment_id if len(self.kv_stack) else None
            if segment.parent_id != expected_parent:
                raise RuntimeError(
                    f"cannot visit leaf {segment_id}: expected parent on stack {segment.parent_id}, "
                    f"got {expected_parent}"
                )
            if segment.prefix_length != self.kv_stack.prefix_length:
                raise RuntimeError(
                    f"cannot visit leaf {segment_id}: stack prefix length is {self.kv_stack.prefix_length}, "
                    f"got {segment.prefix_length}"
                )

            if self.cp_enabled:
                anchors = build_sharded_past_anchors(self.kv_stack)
                past_key_values = {}
            else:
                anchors = self.kv_stack.build_past_anchors()
                past_key_values = anchors.key_values

            context, logits = self._forward(
                segment,
                past_key_values=past_key_values,
                no_grad=False,
                sharded_past_anchors=anchors if self.cp_enabled else None,
            )
            self._assert_collected_layers(context)

            loss_sum, normalized_loss = self._compute_loss(segment, logits)

            owned_loss_term_count = self._owned_loss_count(segment)

            # Every CP rank must traverse the same backward graph even when
            # this rank owns no loss term, otherwise Attention collectives hang.
            backward_loss = self._prepare_backward_loss(
                normalized_loss,
                logits=logits,
                has_owned_loss=owned_loss_term_count > 0,
            )
            if not isinstance(backward_loss, Tensor) or backward_loss.numel() != 1:
                raise TypeError("loss_scale_func must return a scalar tensor")


            torch.autograd.backward(backward_loss)

            self._accumulate_past_anchor_gradients(anchors)


            return LeafVisitResult(
                forward=SegmentForwardResult(
                    segment_id=segment_id,
                    prefix_length=segment.prefix_length,
                    suffix_length=segment.length,
                    layer_count=len(context.new_key_values),
                ),
                backward=SegmentBackwardResult(
                    segment_id=segment_id,
                    loss_sum=loss_sum.detach(),
                    normalized_loss=normalized_loss.detach(),
                    loss_term_count=owned_loss_term_count,
                    relayed_layer_count=0,
                ),
            )
        except Exception:
            self._failed = True
            raise

    def _forward(
        self,
        segment: SegmentSpec,
        *,
        past_key_values,
        no_grad: bool,
        sharded_past_anchors: ShardedPastKVAnchors | None = None,
    ) -> tuple[TPRAttentionContext, Tensor]:
        device = _model_device(self.model)
        shard = self._segment_shard(segment)
        local_token_ids = shard.select(segment.token_ids)
        if os.getenv("TPR_MEM_TRACE", "0") == "1":
            print(f"TPR_MEM FORWARD_BEGIN segment={segment.segment_id} parent={segment.parent_id} prefix={segment.prefix_length} length={segment.length} local_length={shard.local_length} no_grad={no_grad} cp={self.cp_size} alloc_gib={torch.npu.memory_allocated() / (1024 ** 3):.3f} reserved_gib={torch.npu.memory_reserved() / (1024 ** 3):.3f}", flush=True)
        input_ids = local_token_ids.to(device=device).unsqueeze(0)
        position_ids = shard.global_indices(device=device).add(segment.position_start).unsqueeze(0)
        attention_backend = None
        if self.cp_enabled:
            if self.cp_backend is None:
                raise RuntimeError("CP execution is missing its backend")
            attention_backend = self.cp_backend.make_attention_backend(
                self.kv_stack,
                expected_layer_numbers=self.expected_layer_numbers,
                current_shard=shard,
                past_anchors=sharded_past_anchors,
            )
        context = TPRAttentionContext(
            prefix_length=segment.prefix_length,
            suffix_length=segment.length,
            past_key_values=past_key_values,
            suffix_rotary_pos_emb=build_sharded_rotary_pos_emb(
                self.model.rotary_pos_emb,
                position_start=segment.position_start,
                shard=shard,
                disable_context_parallel_sharding=self.cp_enabled,
            ),
            attention_backend=attention_backend,
        )
        grad_context = torch.no_grad() if no_grad else torch.enable_grad()
        with grad_context:
            rope_context = (
                disable_bound_context_parallel_sharding(self.model.rotary_pos_emb)
                if self.cp_enabled
                else nullcontext()
            )
            with rope_context, use_tpr_attention_context(context):
                logits = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                )
        if os.getenv("TPR_MEM_TRACE", "0") == "1":
            print(f"TPR_MEM FORWARD_END segment={segment.segment_id} no_grad={no_grad} alloc_gib={torch.npu.memory_allocated() / (1024 ** 3):.3f} reserved_gib={torch.npu.memory_reserved() / (1024 ** 3):.3f}", flush=True)
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ValueError(f"model must return logits [batch, sequence, vocab], got {type(logits).__name__}")
        expected_shape = (1, shard.local_length)
        if tuple(logits.shape[:2]) != expected_shape:
            raise ValueError(f"model logits must start with shape {expected_shape}, got {tuple(logits.shape)}")
        return context, logits

    def _compute_loss(self, segment: SegmentSpec, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Delegate all policy math to VERL's original PPO callback."""
        if not self._owned_loss_count(segment):
            connected_zero = logits.float().sum() * 0.0
            return connected_zero, connected_zero
        loss, metrics = self.segment_loss_fn(segment, logits)
        if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.requires_grad:
            raise TypeError("PPO objective must return a differentiable scalar Tensor")
        self.segment_loss_metrics.append((segment.segment_id, metrics))
        return loss, loss

    def _owned_loss_count(self, segment: SegmentSpec) -> int:
        return self.segment_loss_term_counts.get(segment.segment_id, 0)

    def _prepare_backward_loss(
        self,
        normalized_loss: Tensor,
        *,
        logits: Tensor,
        has_owned_loss: bool,
    ) -> Tensor:
        backward_loss = normalized_loss * self.cp_size if has_owned_loss else logits.float().sum() * 0.0
        return backward_loss if self.loss_scale_func is None else self.loss_scale_func(backward_loss)

    def _segment_shard(self, segment: SegmentSpec) -> SequenceShard:
        if not self.cp_enabled:
            return PrefixShard.full(segment.length)
        if self.cp_backend is None:
            raise RuntimeError("CP execution is missing its backend")
        return self.cp_backend.make_sequence_shard(segment.length)

    def _accumulate_past_anchor_gradients(self, anchors) -> None:
        if self.cp_enabled:
            accumulate_sharded_past_anchor_gradients(self.kv_stack, anchors)
        elif anchors.key_values:
            self.kv_stack.accumulate_anchor_gradients(anchors)

    def _ensure_healthy(self) -> None:
        if self._failed:
            raise RuntimeError("SegmentExecutor is failed and cannot continue")

    def _assert_collected_layers(self, context: TPRAttentionContext) -> None:
        context.assert_new_kv_layers(self.expected_layer_numbers)


def _compact_kv_cache(key_values: Mapping[int, KVPair]) -> dict[int, KVPair]:
    """Copy no-grad KV views into compact, independently owned storage."""

    return {
        layer_number: (
            key.detach().clone(memory_format=torch.contiguous_format),
            value.detach().clone(memory_format=torch.contiguous_format),
        )
        for layer_number, (key, value) in key_values.items()
    }


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must contain at least one parameter") from exc


def _infer_layer_numbers(model: nn.Module) -> tuple[int, ...]:
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError("cannot infer TPR layer numbers; pass expected_layer_numbers explicitly")
    result = []
    for layer in layers:
        attention = getattr(layer, "self_attention", None)
        if getattr(attention, "tpr_state_kind", None) == "gdn":
            raise NotImplementedError("GDN TPR is experimental, not part of Dense PPO")
        layer_number = getattr(attention, "layer_number", None)
        if layer_number is None:
            raise ValueError("cannot infer layer_number from model.decoder.layers")
        result.append(layer_number)
    return tuple(result)


class SchedulerState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PhysicalExecutionKind(str, Enum):
    PUSH = "push"
    POP = "pop"
    VISIT_LEAF = "visit_leaf"


@dataclass(frozen=True, slots=True)
class PhysicalExecution:
    kind: PhysicalExecutionKind
    segment_id: int


@dataclass(frozen=True, slots=True)
class TreeScheduleResult:
    event_trace: tuple[SegmentEvent, ...]
    execution_trace: tuple[PhysicalExecution, ...]
    forward_results: tuple[SegmentForwardResult, ...]
    backward_results: tuple[SegmentBackwardResult, ...]
    direct_leaf_results: tuple[LeafVisitResult, ...]
    loss_sum: Tensor
    normalized_loss: Tensor
    peak_path_tokens: int

    @property
    def pushed_segment_count(self) -> int:
        return len(self.forward_results)

    @property
    def popped_segment_count(self) -> int:
        return len(self.backward_results)

    @property
    def direct_leaf_count(self) -> int:
        return len(self.direct_leaf_results)

    @property
    def executed_segment_count(self) -> int:
        return self.pushed_segment_count + self.direct_leaf_count


class FixedTopologyScheduler:
    """Execute one validated, immutable DFS Push/Pop event stream."""

    def __init__(
        self,
        plan: SegmentPlan,
        executor: SegmentExecutor,
        *,
        events: Sequence[SegmentEvent] | None = None,
        event_profiler=None,
        tree_index: int = 0,
    ) -> None:
        if not isinstance(plan, SegmentPlan):
            raise TypeError(f"plan must be SegmentPlan, got {type(plan).__name__}")
        if not isinstance(executor, SegmentExecutor):
            raise TypeError(f"executor must be SegmentExecutor, got {type(executor).__name__}")
        if executor.plan is not plan:
            raise ValueError("executor and scheduler must share the same SegmentPlan instance")
        if executor.failed:
            raise ValueError("executor is already failed")
        if len(executor.kv_stack):
            raise ValueError("executor KV stack must be empty before scheduling")

        candidate_events = tuple(plan.dfs_events()) if events is None else tuple(events)
        self.events = plan.validate_events(candidate_events)
        self.plan = plan
        self.executor = executor
        self.event_profiler = event_profiler
        self.tree_index = tree_index
        self.state = SchedulerState.READY

    def run(self) -> TreeScheduleResult:
        if self.state is not SchedulerState.READY:
            raise RuntimeError(f"scheduler can only run from READY, current state is {self.state.value}")
        self.state = SchedulerState.RUNNING
        forward_results: list[SegmentForwardResult] = []
        backward_results: list[SegmentBackwardResult] = []
        direct_leaf_results: list[LeafVisitResult] = []
        all_loss_results: list[SegmentBackwardResult] = []
        execution_trace: list[PhysicalExecution] = []
        peak_path_tokens = 0
        try:
            event_index = 0
            while event_index < len(self.events):
                event = self.events[event_index]
                trace_mem = os.getenv("TPR_MEM_TRACE", "0") == "1"
                if trace_mem:
                    segment = self.plan.get(event.segment_id)
                    kind = "VISIT_LEAF" if self._is_direct_leaf_pair(event_index) else event.kind.value
                    alloc = torch.npu.memory_allocated() / (1024 ** 3) if torch.npu.is_available() else 0.0
                    reserved = torch.npu.memory_reserved() / (1024 ** 3) if torch.npu.is_available() else 0.0
                    print(f"TPR_MEM EVENT_BEGIN index={event_index} kind={kind} segment={event.segment_id} parent={segment.parent_id} length={segment.length} prefix={segment.prefix_length} path_tokens={self.executor.kv_stack.prefix_length} depth={len(self.executor.kv_stack)} alloc_gib={alloc:.3f} reserved_gib={reserved:.3f}", flush=True)
                is_leaf = self._is_direct_leaf_pair(event_index)
                profiling = None
                if self.event_profiler is not None:
                    spec = self.plan.get(event.segment_id)
                    profiling = self.event_profiler.begin(
                        spec,
                        kind=("visit_leaf" if is_leaf else (
                            "push" if event.kind is SegmentEventKind.PUSH else "pop")),
                        tree_index=self.tree_index,
                        depth=self.plan.depth_of(event.segment_id),
                    )
                if is_leaf:
                    result = self.executor.visit_leaf(event.segment_id)
                    if profiling is not None:
                        self.event_profiler.end(profiling)
                    direct_leaf_results.append(result)
                    all_loss_results.append(result.backward)
                    execution_trace.append(PhysicalExecution(PhysicalExecutionKind.VISIT_LEAF, event.segment_id))
                    peak_path_tokens = max(
                        peak_path_tokens,
                        result.forward.prefix_length + result.forward.suffix_length,
                    )
                    if trace_mem:
                        alloc = torch.npu.memory_allocated() / (1024 ** 3) if torch.npu.is_available() else 0.0
                        reserved = torch.npu.memory_reserved() / (1024 ** 3) if torch.npu.is_available() else 0.0
                        print(f"TPR_MEM EVENT_END index={event_index} kind=VISIT_LEAF segment={event.segment_id} alloc_gib={alloc:.3f} reserved_gib={reserved:.3f}", flush=True)
                    event_index += 2
                    continue
                if event.kind is SegmentEventKind.PUSH:
                    forward_results.append(self.executor.push(event.segment_id))
                    execution_trace.append(PhysicalExecution(PhysicalExecutionKind.PUSH, event.segment_id))
                    peak_path_tokens = max(peak_path_tokens, self.executor.kv_stack.prefix_length)
                else:
                    result = self.executor.pop(event.segment_id)
                    backward_results.append(result)
                    all_loss_results.append(result)
                    execution_trace.append(PhysicalExecution(PhysicalExecutionKind.POP, event.segment_id))
                if profiling is not None:
                    self.event_profiler.end(profiling)
                if trace_mem:
                    alloc = torch.npu.memory_allocated() / (1024 ** 3) if torch.npu.is_available() else 0.0
                    reserved = torch.npu.memory_reserved() / (1024 ** 3) if torch.npu.is_available() else 0.0
                    print(f"TPR_MEM EVENT_END index={event_index} kind={event.kind.value} segment={event.segment_id} path_tokens={self.executor.kv_stack.prefix_length} depth={len(self.executor.kv_stack)} alloc_gib={alloc:.3f} reserved_gib={reserved:.3f}", flush=True)
                event_index += 1
            self.executor.kv_stack.assert_empty()
            if not all_loss_results:
                raise RuntimeError("schedule produced no backward results")

            loss_sum = all_loss_results[0].loss_sum
            normalized_loss = all_loss_results[0].normalized_loss
            for result in all_loss_results[1:]:
                loss_sum = loss_sum + result.loss_sum
                normalized_loss = normalized_loss + result.normalized_loss
            schedule_result = TreeScheduleResult(
                event_trace=self.events,
                execution_trace=tuple(execution_trace),
                forward_results=tuple(forward_results),
                backward_results=tuple(backward_results),
                direct_leaf_results=tuple(direct_leaf_results),
                loss_sum=loss_sum,
                normalized_loss=normalized_loss,
                peak_path_tokens=peak_path_tokens,
            )
        except Exception:
            self.state = SchedulerState.FAILED
            raise
        self.state = SchedulerState.COMPLETED
        return schedule_result

    def _is_direct_leaf_pair(self, event_index: int) -> bool:
        if event_index + 1 >= len(self.events):
            return False
        push_event = self.events[event_index]
        pop_event = self.events[event_index + 1]
        return (
            push_event.kind is SegmentEventKind.PUSH
            and pop_event.kind is SegmentEventKind.POP
            and push_event.segment_id == pop_event.segment_id
            and not self.plan.children_of(push_event.segment_id)
        )
