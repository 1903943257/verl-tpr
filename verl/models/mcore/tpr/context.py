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

"""Per-forward bridge between the Engine executor and Megatron attention layers.

Only ephemeral references live here. The Engine state.py owns long-lived KV,
detached Prefix caches and delayed gradients. This module never clones/detaches.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from torch import Tensor

if TYPE_CHECKING:
    from .prefix_state import GDNLayerState

KVPair = tuple[Tensor, Tensor]
RotaryPosEmb = Tensor | tuple[Tensor, Tensor]


class TPRAttentionBackend(Protocol):
    """Rank-local backend supplied by Engine Ring CP when CP > 1."""

    @property
    def global_prefix_length(self) -> int: ...

    @property
    def global_suffix_length(self) -> int: ...

    @property
    def local_suffix_length(self) -> int: ...

    @property
    def parallel_size(self) -> int: ...

    def attention(
        self, layer_number: int, query: Tensor, new_key: Tensor, new_value: Tensor,
        *, softmax_scale: float | None,
    ) -> Tensor: ...


def _validate_layer_number(layer: int) -> None:
    if type(layer) is not int or layer <= 0:
        raise ValueError(f"layer_number must be a positive integer, got {layer!r}")


def _validate_kv_pair(
    layer: int, key: Tensor, value: Tensor, *, expected_sequence_length: int, kind: str,
) -> None:
    if not isinstance(key, Tensor) or not isinstance(value, Tensor):
        raise TypeError(f"layer {layer} {kind} K/V must be torch tensors")
    if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
        raise ValueError(f"layer {layer} {kind} K/V must share shape [sequence, batch, heads, head_dim]")
    if key.shape[0] != expected_sequence_length:
        raise ValueError(
            f"layer {layer} {kind} KV sequence length must be {expected_sequence_length}, "
            f"got {key.shape[0]}"
        )
    if key.shape[1] != 1:
        raise ValueError(f"layer {layer} {kind} KV batch size must be 1, got {key.shape[1]}")
    if key.device != value.device or key.dtype != value.dtype:
        raise ValueError(f"layer {layer} {kind} K/V device and dtype must match")


def _assert_layers(actual: Mapping[int, object], expected: Iterable[int], *, kind: str) -> None:
    expected_set = set(expected)
    for layer in expected_set:
        _validate_layer_number(layer)
    if set(actual) != expected_set:
        raise RuntimeError(
            f"new {kind} layer mismatch: missing={sorted(expected_set - set(actual))}, "
            f"unexpected={sorted(set(actual) - expected_set)}"
        )


@dataclass(slots=True)
class TPRAttentionContext:
    """One Segment forward: Prefix KV in, RoPE/backend in, per-layer new KV out."""

    prefix_length: int
    suffix_length: int
    past_key_values: Mapping[int, KVPair] = field(default_factory=dict)
    suffix_rotary_pos_emb: RotaryPosEmb | None = None
    attention_backend: TPRAttentionBackend | None = None

    # Legacy GDN extension uses the same forward scope; Dense PPO never fills it.
    initial_gdn_states: Mapping[int, GDNLayerState] = field(default_factory=dict)
    capture_gdn_final_state: bool = True
    _new_key_values: dict[int, KVPair] = field(default_factory=dict, init=False, repr=False)
    _new_gdn_states: dict[int, GDNLayerState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("prefix_length", self.prefix_length, 0),
            ("suffix_length", self.suffix_length, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        if not isinstance(self.capture_gdn_final_state, bool):
            raise TypeError("capture_gdn_final_state must be a bool")
        if self.prefix_length == 0:
            if self.past_key_values:
                raise ValueError("past_key_values must be empty when prefix_length is 0")
            if self.initial_gdn_states:
                raise ValueError("initial_gdn_states must be empty when prefix_length is 0")

        backend = self.attention_backend
        if backend is not None:
            required = ("global_prefix_length", "global_suffix_length", "local_suffix_length", "parallel_size", "attention")
            missing = [name for name in required if not hasattr(backend, name)]
            if missing or not callable(getattr(backend, "attention", None)):
                raise TypeError(f"attention_backend must implement TPRAttentionBackend; missing={missing}")
            if self.past_key_values:
                raise ValueError("past_key_values must be empty when an attention_backend is active")
            if (backend.global_prefix_length, backend.global_suffix_length) != (
                self.prefix_length, self.suffix_length
            ):
                raise ValueError("attention backend Prefix/Suffix lengths must match TPRAttentionContext")
            if backend.local_suffix_length <= 0:
                raise ValueError("attention backend local suffix length must be positive")

        for layer, pair in self.past_key_values.items():
            _validate_layer_number(layer)
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"layer {layer} past KV must be a (key, value) tuple")
            _validate_kv_pair(layer, *pair, expected_sequence_length=self.prefix_length, kind="past")
        self.past_key_values = MappingProxyType(dict(self.past_key_values))

        for layer, state in self.initial_gdn_states.items():
            _validate_layer_number(layer)
            _validate_gdn_layer_state(layer, state)
        self.initial_gdn_states = MappingProxyType(dict(sorted(self.initial_gdn_states.items())))

    @property
    def local_suffix_length(self) -> int:
        return self.suffix_length if self.attention_backend is None else self.attention_backend.local_suffix_length

    @property
    def new_key_values(self) -> Mapping[int, KVPair]:
        return MappingProxyType(self._new_key_values)

    def get_past_kv(self, layer_number: int) -> KVPair | None:
        _validate_layer_number(layer_number)
        if self.attention_backend is not None:
            raise RuntimeError("sharded attention must obtain prefix state from its attention backend")
        if self.prefix_length == 0:
            return None
        try:
            return self.past_key_values[layer_number]
        except KeyError as exc:
            raise KeyError(f"missing past KV for layer {layer_number}") from exc

    def set_new_kv(self, layer_number: int, key: Tensor, value: Tensor) -> None:
        _validate_layer_number(layer_number)
        if layer_number in self._new_key_values:
            raise RuntimeError(f"new KV for layer {layer_number} was already recorded")
        _validate_kv_pair(layer_number, key, value, expected_sequence_length=self.local_suffix_length, kind="new")
        self._new_key_values[layer_number] = (key, value)

    def assert_new_kv_layers(self, expected_layer_numbers: Iterable[int]) -> None:
        _assert_layers(self._new_key_values, expected_layer_numbers, kind="KV")

    @property
    def new_gdn_states(self) -> Mapping[int, GDNLayerState]:
        return MappingProxyType(self._new_gdn_states)

    def get_initial_gdn_state(self, layer_number: int) -> GDNLayerState | None:
        _validate_layer_number(layer_number)
        if self.prefix_length == 0:
            return None
        try:
            return self.initial_gdn_states[layer_number]
        except KeyError as exc:
            raise KeyError(f"missing initial GDN state for layer {layer_number}") from exc

    def set_new_gdn_state(self, layer_number: int, state: GDNLayerState) -> None:
        _validate_layer_number(layer_number)
        if layer_number in self._new_gdn_states:
            raise RuntimeError(f"new GDN state for layer {layer_number} was already recorded")
        _validate_gdn_layer_state(layer_number, state)
        self._new_gdn_states[layer_number] = state

    def assert_new_gdn_layers(self, expected_layer_numbers: Iterable[int]) -> None:
        _assert_layers(self._new_gdn_states, expected_layer_numbers, kind="GDN")


def _validate_gdn_layer_state(layer: int, state: GDNLayerState) -> None:
    from .prefix_state import GDNLayerState
    if not isinstance(state, GDNLayerState):
        raise TypeError(f"layer {layer} GDN state must be GDNLayerState, got {type(state).__name__}")


_CURRENT: ContextVar[TPRAttentionContext | None] = ContextVar("tpr_attention_context", default=None)


def get_tpr_attention_context() -> TPRAttentionContext | None:
    return _CURRENT.get()


@contextmanager
def use_tpr_attention_context(context: TPRAttentionContext) -> Iterator[TPRAttentionContext]:
    if not isinstance(context, TPRAttentionContext):
        raise TypeError(f"context must be TPRAttentionContext, got {type(context).__name__}")
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)
