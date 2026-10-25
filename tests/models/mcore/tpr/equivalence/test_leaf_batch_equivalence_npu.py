"""Isolate two sibling leaves: sequential SegmentExecutor vs grouped leaf forward.

This intentionally avoids VERL/PPO capture, sequence-length attribution, rollout,
and the 4B checkpoint. It exercises the current SegmentExecutor + TPRSelfAttention
path on a tiny real Megatron GPT, including shared-prefix KV gradient relay.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel import layers as tensor_parallel_layers
from megatron.core.tensor_parallel import mappings as tensor_parallel_mappings
from megatron.core.transformer.transformer_config import TransformerConfig
from verl.models.mcore.tpr import replace_self_attention_with_tpr
from verl.utils.device import is_torch_npu_available
from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
from verl.workers.engine.megatron.tpr.tree import SegmentPlan, SegmentSpec

pytestmark = pytest.mark.skipif(
    not is_torch_npu_available(check_device=True),
    reason="Requires an Ascend NPU",
)

PREFIX_LEN = 32
LEAF_LEN = 16
VOCAB_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BF16_GRAD_REL_TOL = 5e-2
FP32_GRAD_REL_TOL = 1e-4


class _SingleProcessGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class _NpuGlobalMemoryBuffer:
    def __init__(self, device):
        self.device = device
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        del mem_alloc_context
        count = 1
        for dimension in tensor_shape:
            count *= dimension
        key = (name, dtype)
        if key not in self.buffer or self.buffer[key].numel() < count:
            self.buffer[key] = torch.empty(count, dtype=dtype, device=self.device)
        return self.buffer[key][:count].view(*tensor_shape)


class _ZeroDropoutRngTracker:
    def fork(self, *args, **kwargs):
        del args, kwargs
        return nullcontext()


def _install_single_rank_runtime(monkeypatch, device):
    parallel_state._GLOBAL_MEMORY_BUFFER = _NpuGlobalMemoryBuffer(device)
    monkeypatch.setattr(tensor_parallel, "get_cuda_rng_tracker", lambda: _ZeroDropoutRngTracker())
    original_reduce = tensor_parallel_mappings._reduce
    monkeypatch.setattr(
        tensor_parallel_mappings, "_reduce",
        lambda tensor, group: tensor if group is None else original_reduce(tensor, group),
    )
    original_gather = tensor_parallel_layers.gather_from_tensor_model_parallel_region
    monkeypatch.setattr(
        tensor_parallel_layers, "gather_from_tensor_model_parallel_region",
        lambda tensor, group=None: (
            tensor if group is None or group.size() == 1 else original_gather(tensor, group)
        ),
    )


def _make_model(device, dtype):
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        autocast_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        sequence_parallel=False,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
    )
    process_group = _SingleProcessGroup()
    pg_collection = SimpleNamespace(tp=process_group, cp=process_group, pp=process_group, embd=None)
    spec = replace_self_attention_with_tpr(
        get_gpt_decoder_block_spec(config, use_transformer_engine=False, pp_rank=0)
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=PREFIX_LEN + LEAF_LEN,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        pg_collection=pg_collection,
    ).to(device=device, dtype=dtype)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(device)
    for module in model.modules():
        if hasattr(module, "tp_group") and module.tp_group is None:
            module.tp_group = process_group
    model.train()
    return model


def _make_plan():
    # Two distinct suffixes sharing exactly one 32-token Root.
    prefix = torch.arange(PREFIX_LEN, dtype=torch.long) % VOCAB_SIZE
    leaf_a = (torch.arange(32, 32 + LEAF_LEN, dtype=torch.long) * 3 + 7) % VOCAB_SIZE
    leaf_b = (torch.arange(64, 64 + LEAF_LEN, dtype=torch.long) * 5 + 11) % VOCAB_SIZE
    return SegmentPlan(
        (
            SegmentSpec(0, None, prefix, 0, 0),
            SegmentSpec(1, 0, leaf_a, PREFIX_LEN, PREFIX_LEN),
            SegmentSpec(2, 0, leaf_b, PREFIX_LEN, PREFIX_LEN),
        ),
        root_id=0,
    )


def _segment_loss(segment, logits):
    del segment
    # A deterministic per-leaf token objective. Both execution modes receive
    # the same objective; this test is about forward/backward batching only.
    targets = torch.arange(logits.shape[1], device=logits.device, dtype=torch.long)
    targets = (targets * 7 + 3) % logits.shape[-1]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets, reduction="mean"), {}


def _grads(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float().clone()
        if not torch.isfinite(grad).all().item():
            raise AssertionError(f"Non-finite gradient in {name}")
        result[name] = grad
    if not result:
        raise AssertionError("No model gradients were produced")
    return result


def _run_one(model, plan, k, monkeypatch):
    monkeypatch.setenv("TPR_LEAF_BATCH_SIZE", str(k))
    monkeypatch.setenv("TPR_LEAF_BATCH_MAX_TOKENS", "8192")
    model.zero_grad(set_to_none=True)
    executor = SegmentExecutor(
        model,
        plan,
        expected_layer_numbers=tuple(range(1, NUM_LAYERS + 1)),
        segment_loss_fn=_segment_loss,
        segment_loss_term_counts={0: 0, 1: 1, 2: 1},
    )
    loss = executor.run()
    torch.npu.synchronize()
    return {
        "loss": float(loss.detach().float().item()),
        "grads": _grads(model),
        "forward_calls": executor.model_forward_calls,
        "grouped_calls": executor.grouped_leaf_calls,
        "grouped_segments": executor.grouped_leaf_segments,
    }


def _compare(k1, k2):
    if k1["grads"].keys() != k2["grads"].keys():
        raise AssertionError("K1/K2 gradient parameter sets differ")
    diff_sq = ref_sq = dot = 0.0
    per_parameter = []
    for name in k1["grads"]:
        a, b = k1["grads"][name], k2["grads"][name]
        d = b - a
        diff_sq += torch.sum(d.double() ** 2).item()
        ref_sq += torch.sum(a.double() ** 2).item()
        dot += torch.sum(a.double() * b.double()).item()
        per_parameter.append({
            "name": name,
            "rel_l2": float(torch.linalg.vector_norm(d) /
                            torch.linalg.vector_norm(a).clamp_min(1e-30)),
        })
    rel = (diff_sq ** 0.5) / max(ref_sq ** 0.5, 1e-30)
    cosine = dot / max(ref_sq ** 0.5 * (
        sum(torch.sum(x.double() ** 2).item() for x in k2["grads"].values()) ** 0.5
    ), 1e-30)
    return {
        "loss_abs": abs(k1["loss"] - k2["loss"]),
        "grad_rel_l2": rel,
        "grad_cosine": cosine,
        "top_parameter_rel_l2": sorted(per_parameter, key=lambda x: x["rel_l2"], reverse=True)[:8],
    }


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    ((torch.float32, FP32_GRAD_REL_TOL), (torch.bfloat16, BF16_GRAD_REL_TOL)),
    ids=("fp32", "bf16"),
)
def test_two_sibling_leaves_separate_vs_grouped(monkeypatch, dtype, tolerance):
    torch.manual_seed(20261025)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    model = _make_model(device, dtype)
    plan = _make_plan()

    k1 = _run_one(model, plan, 1, monkeypatch)
    # Same model instance and unchanged weights; only clear accumulated grads.
    k2 = _run_one(model, plan, 2, monkeypatch)
    metrics = _compare(k1, k2)
    print(
        "TPR_TWO_LEAF_BATCH_EQ "
        + str({
            "dtype": str(dtype),
            "K1": {key: k1[key] for key in ("loss", "forward_calls", "grouped_calls", "grouped_segments")},
            "K2": {key: k2[key] for key in ("loss", "forward_calls", "grouped_calls", "grouped_segments")},
            **metrics,
            "grad_rel_tolerance": tolerance,
        }),
        flush=True,
    )

    assert k1["grouped_calls"] == 0 and k1["grouped_segments"] == 0
    assert k2["grouped_calls"] == 1 and k2["grouped_segments"] == 2
    assert k1["forward_calls"] == 4 and k2["forward_calls"] == 3
    assert metrics["grad_rel_l2"] <= tolerance, (
        f"{dtype} two-leaf grouping parity failed: "
        f"grad_rel_l2={metrics['grad_rel_l2']:.6g}, tolerance={tolerance}"
    )
    assert metrics["loss_abs"] <= (1e-5 if dtype == torch.float32 else 2e-2)
    del model
    torch.npu.empty_cache()
