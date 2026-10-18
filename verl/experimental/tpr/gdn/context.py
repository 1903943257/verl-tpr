# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Experimental recurrent-state extension; the Dense runtime has no GDN state."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from verl.models.mcore.tpr.context import (
    TPRAttentionContext,
    _assert_layers,
    _validate_layer_number,
    get_tpr_attention_context,
    use_tpr_attention_context,
)

from .prefix_state import GDNLayerState

__all__ = ["GDNTPRAttentionContext", "get_tpr_attention_context", "use_tpr_attention_context"]


def _validate_gdn_layer_state(layer: int, state: GDNLayerState) -> None:
    if not isinstance(state, GDNLayerState):
        raise TypeError(f"layer {layer} GDN state must be GDNLayerState, got {type(state).__name__}")


@dataclass(slots=True)
class GDNTPRAttentionContext(TPRAttentionContext):
    """Share only the dynamic forward scope with Dense attention layers."""

    initial_gdn_states: Mapping[int, GDNLayerState] = field(default_factory=dict)
    capture_gdn_final_state: bool = True
    _new_gdn_states: dict[int, GDNLayerState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        TPRAttentionContext.__post_init__(self)
        if not isinstance(self.capture_gdn_final_state, bool):
            raise TypeError("capture_gdn_final_state must be a bool")
        if self.prefix_length == 0 and self.initial_gdn_states:
            raise ValueError("initial_gdn_states must be empty when prefix_length is 0")
        for layer, state in self.initial_gdn_states.items():
            _validate_layer_number(layer)
            _validate_gdn_layer_state(layer, state)
        self.initial_gdn_states = MappingProxyType(dict(sorted(self.initial_gdn_states.items())))

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
