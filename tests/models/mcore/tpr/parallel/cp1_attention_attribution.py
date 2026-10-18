"""Test-only Qwen3-4B CP1 TPRSelfAttention shape and gradient attribution.

Nonintrusive forward hooks capture bounded *absolute token coordinates* from
Full (no context), per-row Root/Leaf split and shared tree. The QKV M replay
runs the original unmodified Megatron projection on the SAME input and weight
at M=whole versus M=Root. A separate same-(Q,K,V) causal square-vs-rectangle
probe uses the real rectangular CANN op without Fixed Tile or source changes.

Snapshots only represent sampled token positions; NEVER call their relative L2
whole-layer or whole-model errors. No optimizer, recompute, FP32 GEMM or patch
to production TPR is introduced.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import torch

import verl.models.mcore.tpr.attention as attention_source
from verl.models.mcore.tpr.context import get_tpr_attention_context


STAGES = (
    "attn_in", "qkv_in", "qkv_out", "proj_in", "proj_out",
    "attn_out", "fc1_in", "fc1_out", "fc2_in", "fc2_out",
)
SELECTED_LAYERS = (1, 2, 3, 4, 8, 16, 36)


def _tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)) and obj and isinstance(obj[0], torch.Tensor):
        return obj[0]
    return None


def _difference(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        raise AssertionError(f"Comparison shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if not torch.isfinite(a.float()).all() or not torch.isfinite(b.float()).all():
        raise FloatingPointError("nonfinite attribution tensors")
    x, y = a.detach().float(), b.detach().float()
    d = x - y
    return {
        "max_abs": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "rel_l2": float(d.norm().item() / max(x.norm().item(), 1e-30)),
        "changed": int(torch.count_nonzero(d).item()),
        "count": int(d.numel()),
        "exact": bool(torch.equal(a, b)),
    }


class CP1AttentionTrace:
    def __init__(self, *, model, batch, forest, mode: str, directory: Path):
        if mode not in ("full", "split_no_share", "shared"):
            raise ValueError("Tracing only supports full, split_no_share and shared")
        self.model = model
        self.mode = mode
        self.directory = directory
        self.tokens = [t.detach().cpu().long() for t in batch["input_ids"].unbind()]
        self.root_len = (
            forest.trees[0].segment_plan.get(0).length if mode != "full"
            else int(batch["prompts"][0].numel())
        )
        if len(self.tokens) != 2:
            raise AssertionError("Expected exactly two authentic sibling trajectories")
        if mode != "full" and not (
            0 < self.root_len < min(t.numel() for t in self.tokens)
        ):
            raise AssertionError("Root must be a nonempty proper prefix")
        self.row_cursor = -1
        self.current = None
        self.attention_call_index = 0
        self.handles = []
        self.records = {}
        self.full_first_qkv_x = None
        self.root_first_qkv_x = None
        self.root_queries = {}
        self.same_qkv_attention = {}
        self.original_rectangular = None
        self.replay = {}
        self.relay_accumulation = {}
        self.original_accumulate = None
        self.call_counts = {"root_push": 0, "root_pop": 0, "leaf": 0, "full": 0}

    def _model_pre(self, _module, _args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise AssertionError("Unexpected Qwen3 model call for attribution")
        tokens = ids[0].detach().cpu().long()
        context = get_tpr_attention_context()
        if self.mode == "full":
            if context is not None:
                raise AssertionError("Full Qwen3 forward unexpectedly has TPR context")
            matches = [i for i, row in enumerate(self.tokens) if torch.equal(tokens, row)]
            if len(matches) != 1:
                raise AssertionError("Full input does not uniquely match an original row")
            row, phase, start = matches[0], "full", 0
        else:
            if context is None:
                raise AssertionError("Split/Shared model forward has no TPR context")
            if context.prefix_length == 0:
                if not torch.is_grad_enabled():
                    self.row_cursor += 1
                    phase = "push"
                    self.call_counts["root_push"] += 1
                else:
                    phase = "pop"
                    self.call_counts["root_pop"] += 1
                row = 0 if self.mode == "shared" else self.row_cursor
                if self.mode == "shared" and self.row_cursor != 0:
                    raise AssertionError("Shared tree pushed Root multiple times")
                if not 0 <= row < 2 or not torch.equal(tokens, self.tokens[row][:self.root_len]):
                    raise AssertionError("Root tokens or traversal order incorrect")
                start = 0
            else:
                start = context.prefix_length
                matches = [
                    i for i, row in enumerate(self.tokens)
                    if len(row) == start + len(tokens)
                    and torch.equal(tokens, row[start:])
                ]
                if len(matches) != 1:
                    raise AssertionError("Leaf tokens do not uniquely match an original sibling")
                row, phase = matches[0], "leaf"
                self.call_counts["leaf"] += 1
                if self.mode == "split_no_share" and row != self.row_cursor:
                    raise AssertionError("Independent split did not preserve row traversal")
        self.current = (row, phase, start)
        self.attention_call_index = 0
        self.call_counts["full"] += int(phase == "full")

    def _indices(self, width: int, start: int, *, row: int) -> list[int]:
        boundary = self.root_len
        full_len = self.tokens[row].numel()
        choices = (
            list(range(0, 16))
            + list(range(max(0, boundary - 16), boundary + 32))
            + list(range(max(0, full_len - 16), full_len))
        )
        return sorted({i for i in choices if start <= i < start + width})

    def _snapshot(self, layer: int, stage: str, tensor, *, backward: bool = False):
        t = _tensor(tensor)
        if layer not in SELECTED_LAYERS or t is None or t.ndim < 2:
            return
        if self.current is None:
            raise AssertionError("Forward hook ran without model frame")
        row, phase, start = self.current
        coords = self._indices(t.shape[0], start, row=row)
        if not coords:
            return
        local = torch.tensor([i - start for i in coords], device=t.device, dtype=torch.long)
        with torch.no_grad():
            subset = t.detach().index_select(0, local).contiguous().cpu()
        suffix = "_grad_out" if backward else ""
        key = f"r{row}|{phase}|L{layer}|{stage}{suffix}"
        if key in self.records:
            raise AssertionError(f"Duplicate forward/backward trace {key}")
        self.records[key] = {"positions": coords, "values": subset}

        if not backward and stage == "qkv_in" and layer == 1:
            if phase == "full" and row == 0:
                self.full_first_qkv_x = t.detach().cpu().contiguous()
            if phase == "push" and row == 0:
                self.root_first_qkv_x = t.detach().cpu().contiguous()

        if (not backward and torch.is_grad_enabled() and t.requires_grad
                and stage in ("qkv_out", "proj_out", "fc2_out")
                and layer in (1, 2, 8, 36)):
            # Save the exact forward stage metadata; backwards execute after
            # other forwards and cannot use self.current at callback time.
            def record_backward(gradient, layer=layer, stage=stage, row=row,
                                phase=phase, start=start, coords=coords):
                chosen = torch.tensor(
                    [p-start for p in coords], device=gradient.device,
                    dtype=torch.long,
                )
                value = gradient.detach().index_select(0, chosen).contiguous().cpu()
                grad_key = f"r{row}|{phase}|L{layer}|{stage}_grad_out"
                if grad_key in self.records:
                    raise AssertionError(f"Duplicate gradient trace {grad_key}")
                self.records[grad_key] = {"positions": coords, "values": value}
                return gradient
            t.register_hook(record_backward)

    def _pre(self, layer, stage):
        def handler(_module, args):
            self._snapshot(layer, stage, args[0] if args else None)
        return handler

    def _post(self, layer, stage, *, attention=False):
        def handler(_module, _args, output):
            self._snapshot(layer, stage, output)
            if attention and layer in SELECTED_LAYERS and self.current is not None:
                ctx = get_tpr_attention_context()
                if ctx is not None:
                    pair = ctx.new_key_values.get(layer)
                    if pair is None:
                        raise AssertionError(f"Missing new KV for layer {layer}")
                    for name, value in zip(("post_rope_k", "raw_v"), pair, strict=True):
                        self._snapshot(layer, name, value)
        return handler

    def _patched_rectangular(self, query, key, value, **kwargs):
        self.attention_call_index += 1
        layer = self.attention_call_index
        out = self.original_rectangular(query, key, value, **kwargs)
        if self.mode == "full" or layer not in (1, 2, 8, 36):
            return out
        row, phase, _ = self.current
        if phase == "push":
            self.root_queries[(row, layer)] = query.detach()
        elif phase == "leaf":
            stored_row = 0 if self.mode == "shared" else row
            prefix_q = self.root_queries.get((stored_row, layer))
            if prefix_q is None or prefix_q.shape[0] != key.shape[0] - query.shape[0]:
                raise AssertionError(f"Layer {layer} Prefix Q was not saved")
            with torch.no_grad():
                same_qkv_square = self.original_rectangular(
                    torch.cat((prefix_q, query.detach()), dim=0),
                    key.detach(), value.detach(), **kwargs,
                )[-query.shape[0]:]
            self.same_qkv_attention[f"r{row}|L{layer}"] = _difference(
                out.detach(), same_qkv_square,
            )
        return out

    def install(self):
        self.handles.append(
            self.model.register_forward_pre_hook(self._model_pre, with_kwargs=True)
        )
        for layer in self.model.decoder.layers:
            attn = layer.self_attention
            number = attn.layer_number
            if number not in SELECTED_LAYERS:
                continue
            for mod, label in (
                (attn, "attn"),
                (attn.linear_qkv, "qkv"),
                (attn.linear_proj, "proj"),
                (layer.mlp.linear_fc1, "fc1"),
                (layer.mlp.linear_fc2, "fc2"),
            ):
                self.handles.append(mod.register_forward_pre_hook(self._pre(number, f"{label}_in")))
                self.handles.append(mod.register_forward_hook(
                    self._post(number, f"{label}_out", attention=label == "attn")
                ))
        self.original_rectangular = attention_source.rectangular_causal_attention
        if self.mode != "full":
            from verl.workers.engine.megatron.tpr.state import KVPrefixState
            self.original_accumulate = KVPrefixState.accumulate_layer_gradients

            def audited_accumulate(prefix_state, layer_number, key_grad, value_grad):
                if layer_number not in (1, 2, 8, 36):
                    return self.original_accumulate(prefix_state, layer_number, key_grad, value_grad)
                row = 0 if self.mode == "shared" else self.row_cursor
                key = f"root_r{row}|L{layer_number}"
                previous = self.relay_accumulation.get(key)
                inputs = (key_grad.detach().float().cpu(), value_grad.detach().float().cpu())
                if previous is None:
                    reference = tuple(x.clone() for x in inputs)
                    count = 1
                else:
                    reference = tuple(
                        a + b for a, b in zip(previous["_fp32"], inputs, strict=True)
                    )
                    count = previous["count"] + 1
                self.original_accumulate(prefix_state, layer_number, key_grad, value_grad)
                stored = prefix_state.gradients[layer_number]
                err = {
                    name: _difference(ref.float(), actual.detach().float().cpu())
                    for name, ref, actual in zip(
                        ("dK", "dV"), reference, stored, strict=True,
                    )
                }
                self.relay_accumulation[key] = {
                    "_fp32": reference,
                    "count": count,
                    "bf16_vs_fp32_accumulate": err,
                }

            KVPrefixState.accumulate_layer_gradients = audited_accumulate
        if self.mode != "full":
            attention_source.rectangular_causal_attention = self._patched_rectangular
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.original_rectangular is not None:
            attention_source.rectangular_causal_attention = self.original_rectangular
        if self.original_accumulate is not None:
            from verl.workers.engine.megatron.tpr.state import KVPrefixState
            KVPrefixState.accumulate_layer_gradients = self.original_accumulate
        self.root_queries.clear()

    @torch.no_grad()
    def replay_root_qkv_m(self, original_full_trace: Path):
        """Same input + unchanged module/weights, physical M only differs."""
        if self.mode != "split_no_share":
            return
        full = torch.load(original_full_trace, map_location="cpu", weights_only=False)
        x_full = full.get("full_first_qkv_x")
        x_root = self.root_first_qkv_x
        if x_full is None or x_root is None or x_root.shape[0] >= x_full.shape[0]:
            raise AssertionError("Missing authentic first-layer Full/Root QKV operands")
        if not torch.equal(x_full[:x_root.shape[0]], x_root):
            self.replay = {
                "input_equal": False,
                "input_error": _difference(x_full[:x_root.shape[0]], x_root),
                "physical_m_full": int(x_full.shape[0]),
                "physical_m_root": int(x_root.shape[0]),
                "conclusion": "First-layer input mismatch: do NOT attribute QKV error to M yet",
            }
            return
        module = self.model.decoder.layers[0].self_attention.linear_qkv
        device = next(module.parameters()).device
        # Do not use FP32 or a new kernel: invoke the original Megatron module.
        x = x_full.to(device=device)
        y_full = _tensor(module(x)).detach()
        y_root = _tensor(module(x_root.to(device=device))).detach()
        prefix = y_full[:len(x_root)]
        self.replay = {
            "input_equal": True,
            "physical_m_full": int(x_full.shape[0]),
            "physical_m_root": int(x_root.shape[0]),
            "qkv_same_input_m_replay": _difference(prefix, y_root),
        }
        # Cross-check the actual recorded first-layer QKV outputs at the
        # selected absolute token coordinates: replay must match each arm.
        for arm, records, result in (
            ("full", full["records"], y_full),
            ("split_root", self.records, y_root),
        ):
            key = "r0|full|L1|qkv_out" if arm == "full" else "r0|push|L1|qkv_out"
            stage = records[key]
            index = torch.tensor(stage["positions"], device=device)
            expected = stage["values"].to(device)
            self.replay[f"{arm}_qkv_replay_vs_actual"] = _difference(
                expected, result.index_select(0, index),
            )
        del x, y_full, y_root

    def save(self, destination: Path):
        if destination.exists():
            raise FileExistsError(destination)
        if self.mode == "full" and self.call_counts["full"] != 2:
            raise AssertionError(f"Full must execute exactly twice: {self.call_counts}")
        if self.mode == "split_no_share" and self.call_counts != {
            "root_push": 2, "root_pop": 2, "leaf": 2, "full": 0,
        }:
            raise AssertionError(f"Split run did not execute Root/Leaf twice: {self.call_counts}")
        if self.mode == "shared" and self.call_counts != {
            "root_push": 1, "root_pop": 1, "leaf": 2, "full": 0,
        }:
            raise AssertionError(f"Shared run did not reuse one Root: {self.call_counts}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "mode": self.mode,
            "root_length": self.root_len,
            "row_lengths": [len(t) for t in self.tokens],
            "call_counts": self.call_counts,
            "records": self.records,
            "same_qkv_attention": self.same_qkv_attention,
            "same_input_gemm_replay": self.replay,
            "kv_relay_roundoff": {
                k: {
                    "count": v["count"],
                    "bf16_vs_fp32_accumulate": v["bf16_vs_fp32_accumulate"],
                }
                for k, v in self.relay_accumulation.items()
            },
            "full_first_qkv_x": self.full_first_qkv_x,
            "root_first_qkv_x": self.root_first_qkv_x,
            "selected_layers": SELECTED_LAYERS,
            "stage_names": STAGES,
            "no_fixed_tile": True,
        }, destination)
        return destination
