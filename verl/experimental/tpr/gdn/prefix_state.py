"""Optional GDN Prefix continuation and gradient buffers.

The production Dense TPR KV path never imports this implementation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from verl.workers.engine.megatron.tpr.tree import SegmentId
from verl.workers.engine.megatron.tpr.state import (
    PrefixState,
    KVPrefixAnchors,
    KVPrefixState,
    PrefixStateEntry,
    PrefixStateStack,
    _normalize_local_kv,
    _validate_segment_id,
    _validate_layer_number,
    _compact_graph_free_tensor,
)

@dataclass(frozen=True, slots=True)
class GDNLayerState:
    """Causal-convolution and recurrent continuation state for one GDN layer."""

    conv_state: Tensor
    recurrent_state: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.conv_state, Tensor) or not isinstance(self.recurrent_state, Tensor):
            raise TypeError("GDN conv_state and recurrent_state must be torch tensors")
        if self.conv_state.ndim != 3:
            raise ValueError(
                "GDN conv_state must have shape [batch, channels, history], "
                f"got {tuple(self.conv_state.shape)}"
            )
        if self.recurrent_state.ndim != 4:
            raise ValueError(
                "GDN recurrent_state must have shape [batch, heads, key_dim, value_dim], "
                f"got {tuple(self.recurrent_state.shape)}"
            )
        if any(dimension <= 0 for dimension in self.conv_state.shape):
            raise ValueError(f"GDN conv_state dimensions must be positive, got {tuple(self.conv_state.shape)}")
        if any(dimension <= 0 for dimension in self.recurrent_state.shape):
            raise ValueError(
                f"GDN recurrent_state dimensions must be positive, got {tuple(self.recurrent_state.shape)}"
            )
        if self.conv_state.shape[0] != self.recurrent_state.shape[0]:
            raise ValueError(
                "GDN continuation states must have the same batch size, "
                f"got conv={self.conv_state.shape[0]} and recurrent={self.recurrent_state.shape[0]}"
            )
        if self.conv_state.device != self.recurrent_state.device:
            raise ValueError(
                "GDN continuation states must be on the same device, "
                f"got conv={self.conv_state.device} and recurrent={self.recurrent_state.device}"
            )
        if not self.conv_state.is_floating_point() or not self.recurrent_state.is_floating_point():
            raise ValueError("GDN conv_state and recurrent_state must be floating-point tensors")


@dataclass(frozen=True, slots=True)
class GDNPrefixAnchors:
    """Autograd leaves for all GDN layer states saved by one segment."""

    segment_id: SegmentId
    generation: int
    layer_states: Mapping[int, GDNLayerState]


def _normalize_gdn_layer_states(
    layer_states: Mapping[int, GDNLayerState],
    *,
    graph_free: bool,
) -> dict[int, GDNLayerState]:
    if not isinstance(layer_states, Mapping) or not layer_states:
        raise ValueError("GDN layer_states must be a non-empty mapping")

    normalized: dict[int, GDNLayerState] = {}
    reference_device: torch.device | None = None
    reference_batch: int | None = None
    for layer_number, layer_state in layer_states.items():
        _validate_layer_number(layer_number)
        if not isinstance(layer_state, GDNLayerState):
            raise TypeError(
                f"layer {layer_number} state must be GDNLayerState, got {type(layer_state).__name__}"
            )
        conv_state = layer_state.conv_state
        recurrent_state = layer_state.recurrent_state
        if graph_free and (
            conv_state.requires_grad
            or recurrent_state.requires_grad
            or conv_state.grad_fn is not None
            or recurrent_state.grad_fn is not None
        ):
            raise ValueError(f"layer {layer_number} cached GDN state must be detached and graph-free")
        if reference_device is None:
            reference_device = conv_state.device
            reference_batch = conv_state.shape[0]
        elif conv_state.device != reference_device or conv_state.shape[0] != reference_batch:
            raise ValueError("all GDN layers in a prefix state must use the same device and batch size")
        normalized[layer_number] = GDNLayerState(
            _compact_graph_free_tensor(conv_state) if graph_free else conv_state,
            _compact_graph_free_tensor(recurrent_state) if graph_free else recurrent_state,
        )
    return dict(sorted(normalized.items()))


class GDNPrefixState:
    """Graph-free GDN continuation states and delayed gradients for one segment.

    Stage 3 deliberately stores no packed-sequence or context-parallel metadata.
    ``sequence_length`` belongs only to the segment lifecycle; GDN continuation
    itself consists solely of each layer's convolution and recurrent tensors.
    """

    state_kind = "gdn"

    def __init__(
        self,
        segment_id: SegmentId,
        sequence_length: int,
        layer_states: Mapping[int, GDNLayerState],
    ) -> None:
        _validate_segment_id(segment_id)
        if (
            not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length <= 0
        ):
            raise ValueError(f"sequence_length must be positive, got {sequence_length!r}")

        self._segment_id = segment_id
        self._sequence_length = sequence_length
        self._layer_states = _normalize_gdn_layer_states(layer_states, graph_free=True)
        self._layer_numbers = tuple(self._layer_states)
        self._gradients: dict[int, GDNLayerState] = {}
        self._released = False
        self._generation = 0

    @classmethod
    def save(
        cls,
        segment_id: SegmentId,
        sequence_length: int,
        layer_states: Mapping[int, GDNLayerState],
    ) -> GDNPrefixState:
        """Detach and compact state tensors produced by a graph-carrying Push."""

        normalized = _normalize_gdn_layer_states(layer_states, graph_free=False)
        saved_states = {
            layer_number: GDNLayerState(
                layer_state.conv_state.detach().clone(memory_format=torch.contiguous_format),
                layer_state.recurrent_state.detach().clone(memory_format=torch.contiguous_format),
            )
            for layer_number, layer_state in normalized.items()
        }
        return cls(segment_id, sequence_length, saved_states)

    @property
    def segment_id(self) -> SegmentId:
        return self._segment_id

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def layer_numbers(self) -> tuple[int, ...]:
        return self._layer_numbers

    @property
    def released(self) -> bool:
        return self._released

    @property
    def layer_states(self) -> Mapping[int, GDNLayerState]:
        self._ensure_live()
        return MappingProxyType(self._layer_states)

    @property
    def gradients(self) -> Mapping[int, GDNLayerState]:
        self._ensure_live()
        return MappingProxyType(self._gradients)

    def restore(self, layer_number: int) -> GDNLayerState:
        """Return one saved layer state for independent branch continuation."""

        self._ensure_live()
        _validate_layer_number(layer_number)
        try:
            return self._layer_states[layer_number]
        except KeyError as exc:
            raise KeyError(f"segment {self.segment_id} has no GDN state for layer {layer_number}") from exc

    def validate(self) -> None:
        self._ensure_live()
        normalized = _normalize_gdn_layer_states(self._layer_states, graph_free=True)
        if tuple(normalized) != self.layer_numbers:
            raise RuntimeError(
                f"segment {self.segment_id} GDN layers changed from {self.layer_numbers} to {tuple(normalized)}"
            )
        for layer_number, layer_state in normalized.items():
            stored_state = self._layer_states[layer_number]
            if (
                layer_state.conv_state is not stored_state.conv_state
                or layer_state.recurrent_state is not stored_state.recurrent_state
            ):
                raise RuntimeError(
                    f"segment {self.segment_id} layer {layer_number} GDN storage is no longer compact"
                )

    def make_anchors(self) -> GDNPrefixAnchors:
        """Create branch-local autograd leaves without mutating the saved state."""

        self._ensure_live()
        anchors = {
            layer_number: GDNLayerState(
                layer_state.conv_state.detach().requires_grad_(True),
                layer_state.recurrent_state.detach().requires_grad_(True),
            )
            for layer_number, layer_state in self._layer_states.items()
        }
        return GDNPrefixAnchors(
            segment_id=self.segment_id,
            generation=self._generation,
            layer_states=MappingProxyType(anchors),
        )

    def accumulate_layer_gradients(
        self,
        layer_number: int,
        conv_grad: Tensor,
        recurrent_grad: Tensor,
    ) -> None:
        self._ensure_live()
        _validate_layer_number(layer_number)
        try:
            saved_state = self._layer_states[layer_number]
        except KeyError as exc:
            raise KeyError(f"segment {self.segment_id} has no GDN state for layer {layer_number}") from exc
        gradient = GDNLayerState(conv_grad, recurrent_grad)
        for name, actual, expected in (
            ("conv", gradient.conv_state, saved_state.conv_state),
            ("recurrent", gradient.recurrent_state, saved_state.recurrent_state),
        ):
            if actual.shape != expected.shape:
                raise ValueError(
                    f"segment {self.segment_id} layer {layer_number} {name} gradient shape mismatch: "
                    f"expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
                )
            if actual.device != expected.device or actual.dtype != expected.dtype:
                raise ValueError(
                    f"segment {self.segment_id} layer {layer_number} {name} gradient "
                    "device/dtype mismatch"
                )

        if layer_number not in self._gradients:
            self._gradients[layer_number] = GDNLayerState(
                conv_grad.detach().clone(memory_format=torch.contiguous_format),
                recurrent_grad.detach().clone(memory_format=torch.contiguous_format),
            )
        else:
            accumulated = self._gradients[layer_number]
            accumulated.conv_state.add_(conv_grad.detach())
            accumulated.recurrent_state.add_(recurrent_grad.detach())

    def accumulate_anchor_gradients(self, anchors: GDNPrefixAnchors) -> None:
        self._ensure_live()
        if not isinstance(anchors, GDNPrefixAnchors):
            raise TypeError(f"anchors must be GDNPrefixAnchors, got {type(anchors).__name__}")
        if anchors.segment_id != self.segment_id or anchors.generation != self._generation:
            raise RuntimeError(f"GDN prefix anchors for segment {anchors.segment_id} are stale")
        if tuple(anchors.layer_states) != self.layer_numbers:
            raise RuntimeError(
                f"GDN prefix anchor layers must be {self.layer_numbers}, got {tuple(anchors.layer_states)}"
            )
        for layer_number, layer_state in anchors.layer_states.items():
            if layer_state.conv_state.grad is None or layer_state.recurrent_state.grad is None:
                raise RuntimeError(f"layer {layer_number} GDN prefix anchor gradient is missing")
            self.accumulate_layer_gradients(
                layer_number,
                layer_state.conv_state.grad,
                layer_state.recurrent_state.grad,
            )

    def consume_gradients(self) -> Mapping[int, GDNLayerState]:
        self._ensure_live()
        if self._gradients and tuple(self._gradients) != self.layer_numbers:
            raise RuntimeError(
                f"segment {self.segment_id} GDN gradient layers must be {self.layer_numbers}, "
                f"got {tuple(self._gradients)}"
            )
        gradients = self._gradients
        self._gradients = {}
        self._generation += 1
        return MappingProxyType(gradients)

    def release(self) -> None:
        if self._released:
            return
        self._layer_states.clear()
        self._gradients.clear()
        self._released = True
        self._generation += 1

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError(f"GDN prefix state for segment {self.segment_id} has been released")


