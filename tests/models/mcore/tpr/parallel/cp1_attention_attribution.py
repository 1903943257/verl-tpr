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
        self.first_layer_linear_inputs_full = {}
        self.first_layer_linear_inputs_root = {}
        self.root_queries = {}
        self.same_qkv_attention = {}
        self.same_qkv_attention_backward = {}
        self.same_qkv_backward_inputs = {}
        self.original_rectangular = None
        self.replay = {}
        self.relay_accumulation = {}
        self.original_accumulate = None
        self.original_anchor_accumulate = None
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

        if not backward and layer == 1 and stage in (
            "qkv_in", "proj_in", "fc1_in", "fc2_in"
        ) and row == 0:
            if phase == "full":
                self.first_layer_linear_inputs_full[stage] = t.detach().cpu().contiguous()
            elif phase == "push":
                self.first_layer_linear_inputs_root[stage] = t.detach().cpu().contiguous()
            if stage == "qkv_in":
                if phase == "full":
                    self.full_first_qkv_x = self.first_layer_linear_inputs_full[stage]
                elif phase == "push":
                    self.root_first_qkv_x = self.first_layer_linear_inputs_root[stage]

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

            # Capture one real Leaf's exact Q/K/V and its actual upstream dO.
            # The post-backward probe replays these identical tensors through
            # both the rectangular adapter and MCore's square core-attention
            # module, isolating attention-backend/shape effects from upstream
            # gradient differences. Limit this diagnostic to row 0.
            if row == 0:
                probe_key = f"r{row}|L{layer}"
                if probe_key in self.same_qkv_backward_inputs:
                    raise AssertionError(f"Duplicate same-QKV backward probe {probe_key}")
                attention_kwargs = {
                    name: kwargs.get(name)
                    for name in (
                        "softmax_scale", "dropout_p", "pre_tokens",
                        "inner_precise", "attention_mask",
                    )
                    if name in kwargs
                }
                if isinstance(attention_kwargs.get("attention_mask"), torch.Tensor):
                    attention_kwargs["attention_mask"] = (
                        attention_kwargs["attention_mask"].detach().cpu()
                    )
                self.same_qkv_backward_inputs[probe_key] = {
                    "query_prefix": prefix_q.detach().cpu(),
                    "query_suffix": query.detach().cpu(),
                    "key": key.detach().cpu(),
                    "value": value.detach().cpu(),
                    "rectangular_output": out.detach().cpu(),
                    "attention_kwargs": attention_kwargs,
                }

                def save_upstream_gradient(gradient, probe_key=probe_key):
                    self.same_qkv_backward_inputs[probe_key][
                        "grad_output"
                    ] = gradient.detach().cpu()
                    return gradient

                if not out.requires_grad:
                    raise AssertionError(
                        f"Leaf attention output has no grad_fn for {probe_key}"
                    )
                out.register_hook(save_upstream_gradient)
        return out

    def run_same_qkv_attention_backward(self):
        """Compare square MCore and rectangular backward on identical Q/K/V,dO."""
        if self.mode == "full":
            return
        expected = {f"r0|L{layer}" for layer in (1, 2, 8, 36)}
        missing = expected - self.same_qkv_backward_inputs.keys()
        if missing:
            raise AssertionError(
                f"Missing same-QKV backward probe inputs: {sorted(missing)}"
            )

        device = next(self.model.parameters()).device
        for probe_key in sorted(expected, key=lambda item: int(item.split("|L")[1])):
            item = self.same_qkv_backward_inputs[probe_key]
            if "grad_output" not in item:
                raise AssertionError(f"Missing real upstream gradient for {probe_key}")
            layer = int(probe_key.split("|L")[1])
            attention_module = self.model.decoder.layers[layer - 1].self_attention
            kwargs = dict(item["attention_kwargs"])
            if isinstance(kwargs.get("attention_mask"), torch.Tensor):
                kwargs["attention_mask"] = kwargs["attention_mask"].to(device)

            q_prefix = item["query_prefix"].to(device)
            q_suffix_data = item["query_suffix"].to(device)
            k_data = item["key"].to(device)
            v_data = item["value"].to(device)
            grad_output = item["grad_output"].to(device)

            with torch.enable_grad():
                # Rectangular path: only suffix queries produce outputs.
                q_rect = q_suffix_data.detach().requires_grad_(True)
                k_rect = k_data.detach().requires_grad_(True)
                v_rect = v_data.detach().requires_grad_(True)
                rect_output = self.original_rectangular(
                    q_rect, k_rect, v_rect, **kwargs,
                )
                rect_grads = torch.autograd.grad(
                    rect_output, (q_rect, k_rect, v_rect),
                    grad_outputs=grad_output,
                )

                # Square path: identical Q/K/V values; zero dO for prefix
                # outputs ensures only suffix-query contributions enter dK/dV.
                q_square = torch.cat(
                    (q_prefix, q_suffix_data), dim=0,
                ).detach().requires_grad_(True)
                k_square = k_data.detach().requires_grad_(True)
                v_square = v_data.detach().requires_grad_(True)
                square_output = _tensor(attention_module.core_attention(
                    q_square,
                    k_square,
                    v_square,
                    kwargs.get("attention_mask"),
                    attn_mask_type=attention_module.attn_mask_type,
                    attention_bias=None,
                    packed_seq_params=None,
                ))
                if square_output is None:
                    raise AssertionError(
                        f"MCore square core attention returned no tensor at {probe_key}"
                    )
                suffix_length = grad_output.shape[0]
                if tuple(square_output[-suffix_length:].shape) != tuple(grad_output.shape):
                    raise AssertionError(
                        f"Square output/dO mismatch at {probe_key}: "
                        f"square={tuple(square_output.shape)}, dO={tuple(grad_output.shape)}"
                    )
                square_grad_output = torch.zeros_like(square_output)
                square_grad_output[-suffix_length:] = grad_output
                square_grads = torch.autograd.grad(
                    square_output, (q_square, k_square, v_square),
                    grad_outputs=square_grad_output,
                )

            self.same_qkv_attention_backward[probe_key] = {
                "rectangular_replay_vs_actual_output": _difference(
                    item["rectangular_output"].to(device), rect_output.detach(),
                ),
                "square_mcore_vs_rectangular_forward_suffix": _difference(
                    rect_output.detach(), square_output[-suffix_length:].detach(),
                ),
                "dQ_suffix_rectangular_vs_square_mcore": _difference(
                    rect_grads[0], square_grads[0][-suffix_length:],
                ),
                "dK_rectangular_vs_square_mcore": _difference(
                    rect_grads[1], square_grads[1],
                ),
                "dV_rectangular_vs_square_mcore": _difference(
                    rect_grads[2], square_grads[2],
                ),
                "same_upstream_dO": True,
                "suffix_length": int(suffix_length),
                "kv_length": int(k_data.shape[0]),
                "q_dtype": str(q_suffix_data.dtype),
                "attention_module": type(attention_module.core_attention).__name__,
            }
            del q_rect, k_rect, v_rect, q_square, k_square, v_square
            del rect_output, square_output, rect_grads, square_grads

        self.same_qkv_backward_inputs.clear()

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

            # Audit every incoming anchor gradient before native consume/
            # accumulate. The first contribution can initialize the stored
            # buffer directly, bypassing accumulate_layer_gradients.
            self.original_anchor_accumulate = (
                KVPrefixState.accumulate_anchor_gradients
            )

            def audited_anchor_accumulate(
                prefix_state, anchors, *, consume=False
            ):
                row = 0 if self.mode == "shared" else self.row_cursor
                pending = []

                for layer, (key, value) in anchors.key_values.items():
                    if layer not in (1, 2, 8, 36):
                        continue
                    if key.grad is None or value.grad is None:
                        raise AssertionError(
                            f"Missing anchor gradients at layer {layer}"
                        )

                    incoming = (
                        key.grad.detach().float().cpu(),
                        value.grad.detach().float().cpu(),
                    )
                    audit_key = f"root_r{row}|L{layer}"
                    previous = self.relay_accumulation.get(audit_key)
                    if previous is None:
                        reference = tuple(x.clone() for x in incoming)
                        count = 1
                    else:
                        reference = tuple(
                            a + b
                            for a, b in zip(
                                previous["_fp32"], incoming, strict=True
                            )
                        )
                        count = previous["count"] + 1

                    self.relay_accumulation[audit_key] = {
                        "_fp32": reference,
                        "count": count,
                    }
                    pending.append((layer, audit_key, reference))

                # Preserve production accumulation behavior exactly.
                self.original_anchor_accumulate(
                    prefix_state, anchors, consume=consume
                )

                # Compare the complete FP32 sum with the actual stored buffer.
                for layer, audit_key, reference in pending:
                    stored = prefix_state.gradients[layer]
                    error = {
                        name: _difference(
                            ref, actual.detach().float().cpu()
                        )
                        for name, ref, actual in zip(
                            ("dK", "dV"), reference, stored, strict=True
                        )
                    }
                    self.relay_accumulation[audit_key][
                        "bf16_vs_fp32_accumulate"
                    ] = error

            KVPrefixState.accumulate_anchor_gradients = (
                audited_anchor_accumulate
            )
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
        if self.original_anchor_accumulate is not None:
            from verl.workers.engine.megatron.tpr.state import KVPrefixState
            KVPrefixState.accumulate_anchor_gradients = (
                self.original_anchor_accumulate
            )
        self.root_queries.clear()

    @torch.no_grad()
    def replay_root_qkv_m(self, original_full_trace: Path):
        """Same inputs and unchanged native modules, only physical M differs.

        Replays first-layer QKV, Attention projection, FC1 and FC2. For each,
        root input equality is reported separately: a shape-only local
        intervention is not proof of the actual segmented output's cause when
        upstream activations are already different.
        """
        if self.mode != "split_no_share":
            return
        full = torch.load(original_full_trace, map_location="cpu", weights_only=False)
        original_inputs = full.get("first_layer_linear_inputs_full", {})
        root_inputs = self.first_layer_linear_inputs_root
        if set(original_inputs) != {"qkv_in", "proj_in", "fc1_in", "fc2_in"}:
            raise AssertionError("Missing all four authentic first-layer Full GEMM operands")
        if set(root_inputs) != set(original_inputs):
            raise AssertionError("Missing Root projection/MLP operands")
        layer = self.model.decoder.layers[0]
        modules = {
            "qkv": layer.self_attention.linear_qkv,
            "proj": layer.self_attention.linear_proj,
            "fc1": layer.mlp.linear_fc1,
            "fc2": layer.mlp.linear_fc2,
        }
        observed = {}
        for label, module in modules.items():
            stage = f"{label}_in"
            full_input = original_inputs[stage]
            root_input = root_inputs[stage]
            if not 0 < root_input.shape[0] < full_input.shape[0]:
                raise AssertionError(f"Invalid physical M for {label}")
            same_real_input = torch.equal(
                full_input[:root_input.shape[0]], root_input,
            )
            result = {
                "input_equal": same_real_input,
                "physical_m_full": int(full_input.shape[0]),
                "physical_m_root": int(root_input.shape[0]),
                "actual_prefix_input_mismatch": _difference(
                    full_input[:root_input.shape[0]], root_input,
                ),
            }
            device = next(module.parameters()).device
            x_full = full_input.to(device=device)
            # Original Megatron modules and BF16 math. No forced M, FP32
            # Linear, or "equalizing" the native Full baseline.
            output_full = _tensor(module(x_full)).detach()
            output_cutoff = _tensor(module(
                full_input[:root_input.shape[0]].to(device=device)
            )).detach()
            result["same_full_input_m_replay"] = _difference(
                output_full[:root_input.shape[0]], output_cutoff,
            )
            full_record = full["records"][f"r0|full|L1|{label}_out"]
            idx = torch.tensor(full_record["positions"], device=device)
            result["full_replay_vs_actual"] = _difference(
                full_record["values"].to(device),
                output_full.index_select(0, idx),
            )
            root_record = self.records[f"r0|push|L1|{label}_out"]
            root_indices = torch.tensor(root_record["positions"], device=device)
            if same_real_input:
                # Only when the natural Root input is identical can the
                # shape-only cutoff replay be identified with actual Root.
                result["root_replay_vs_actual"] = _difference(
                    root_record["values"].to(device),
                    output_cutoff.index_select(0, root_indices),
                )
            else:
                result["root_replay_vs_actual"] = None
            observed[label] = result
            del x_full, output_full, output_cutoff
        self.replay = {
            "input_equal": observed["qkv"]["input_equal"],
            "physical_m_full": observed["qkv"]["physical_m_full"],
            "physical_m_root": observed["qkv"]["physical_m_root"],
            "qkv_same_input_m_replay": observed["qkv"]["same_full_input_m_replay"],
            "full_qkv_replay_vs_actual": observed["qkv"]["full_replay_vs_actual"],
            "split_root_qkv_replay_vs_actual": observed["qkv"]["root_replay_vs_actual"],
            "individual_original_linear_replays": observed,
        }

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
        if self.mode == "shared":
            expected_audit_keys = {
                f"root_r0|L{layer}" for layer in (1, 2, 8, 36)
            }
            missing = expected_audit_keys - self.relay_accumulation.keys()
            wrong_counts = {
                key: self.relay_accumulation[key]["count"]
                for key in expected_audit_keys & self.relay_accumulation.keys()
                if self.relay_accumulation[key]["count"] != 2
            }
            if missing or wrong_counts:
                raise AssertionError(
                    "Shared KV accumulation audit must observe two Leaf "
                    f"contributions per selected layer; missing={sorted(missing)}, "
                    f"wrong_counts={wrong_counts}"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "mode": self.mode,
            "root_length": self.root_len,
            "row_lengths": [len(t) for t in self.tokens],
            "call_counts": self.call_counts,
            "records": self.records,
            "same_qkv_attention": self.same_qkv_attention,
            "same_qkv_attention_backward": self.same_qkv_attention_backward,
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
            "first_layer_linear_inputs_full": self.first_layer_linear_inputs_full,
            "first_layer_linear_inputs_root": self.first_layer_linear_inputs_root,
            "selected_layers": SELECTED_LAYERS,
            "stage_names": STAGES,
            "no_fixed_tile": True,
        }, destination)
        return destination
