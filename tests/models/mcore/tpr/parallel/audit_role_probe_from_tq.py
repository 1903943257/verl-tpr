"""Audit experimental role-probe tree planning on a trusted local TQ/Actor capture.

Example:
  TPR_ROLE_TOKENIZER_PATH=/workspace/hf_models/Qwen3-4B \\
    python tests/models/mcore/tpr/parallel/audit_role_probe_from_tq.py \\
      --capture /workspace/tq_dump/my_run/1/0/tq_batch.pt

Does not run or modify training. Returns nonzero on token-path, objective-ref,
or trie-vs-LCP plan disagreement. Template-inferred semantic roles are
reported with a Qwen reference only when that tokenizer exposes ChatML IDs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[5]
TREE_FILE = ROOT / "verl/workers/engine/megatron/tpr/tree.py"
spec = importlib.util.spec_from_file_location("tpr_role_probe_audit", TREE_FILE)
tree = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tree
spec.loader.exec_module(tree)


def _rows(batch, name):
    v = batch[name]
    return list(v) if isinstance(v, (tuple, list)) else list(v.unbind())


def _validate(forest, batch):
    inputs = _rows(batch, "input_ids")
    prompts = _rows(batch, "prompts")
    masks = _rows(batch, "response_mask")
    expected = {
        (row, off) for row, mask in enumerate(masks)
        for off in torch.nonzero(mask.to(torch.bool), as_tuple=False).flatten().tolist()
    }
    found = set()
    covered_rows = set()
    for item in forest.trees:
        plan = item.segment_plan
        for row in item.member_rows:
            if row in covered_rows:
                raise AssertionError(f"row {row} appears in more than one tree")
            covered_rows.add(row)
            sequence = inputs[row].cpu().long()
            if not any(
                torch.equal(torch.cat([s.token_ids for s in plan.path_to(end.segment_id)]), sequence)
                for end in plan.segments.values() if end.position_end == len(sequence)
            ):
                raise AssertionError(f"row {row} cannot be replayed exactly")
        for ref in item.objective_refs:
            key = (ref.sample_row, ref.response_offset)
            if key in found:
                raise AssertionError(f"duplicate PPO ref {key}")
            found.add(key)
            row, off = key
            response_start = len(prompts[row])
            target_pos = response_start + off
            segment = plan.get(ref.segment_id)
            query_pos = segment.position_start + ref.query_offset
            sequence = inputs[row]
            if query_pos + 1 != target_pos:
                raise AssertionError(f"wrong target/query alignment {key}")
            if int(sequence[target_pos]) != ref.target_token_id:
                raise AssertionError(f"wrong target token {key}")
            if int(sequence[query_pos]) != int(segment.token_ids[ref.query_offset]):
                raise AssertionError(f"wrong query token {key}")
    if found != expected:
        raise AssertionError(
            f"PPO ref mismatch missing={len(expected - found)} extra={len(found - expected)}"
        )
    if covered_rows != set(range(len(inputs))):
        raise AssertionError(f"uncovered rows: {set(range(len(inputs))) - covered_rows}")
    if forest.logical_loss_tokens != len(expected):
        raise AssertionError("logical_loss_tokens disagrees with response_mask")


def _summary(forest, batch):
    inputs = _rows(batch, "input_ids")
    logical = sum(len(seq) for seq in inputs)
    unique = sum(
        seg.length for item in forest.trees for seg in item.segment_plan.segments.values()
    )
    return {
        "trees": len(forest.trees),
        "rows": len(inputs),
        "logical_tokens": logical,
        "unique_tokens": unique,
        "reuse_factor": round(logical / unique, 5) if unique else None,
        "segments": forest.segment_count,
        "logical_loss_tokens": forest.logical_loss_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--tokenizer", default=os.environ.get("TPR_ROLE_TOKENIZER_PATH"))
    parser.add_argument("--minimum", type=int, default=128)
    args = parser.parse_args()
    if not args.tokenizer:
        parser.error("set --tokenizer or TPR_ROLE_TOKENIZER_PATH")
    # Only inspect captures created by the user. weights_only=False unpickles objects.
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "tensordict" not in payload or "keys" not in payload:
        raise ValueError("capture must contain keys and tensordict")
    batch = payload["tensordict"]
    keys = [str(x) for x in payload["keys"]]
    if len(keys) != len(_rows(batch, "input_ids")):
        raise ValueError("key count disagrees with TQ row count")
    os.environ["TPR_ROLE_TOKENIZER_PATH"] = args.tokenizer
    started = time.perf_counter()
    probe = tree._role_probe_from_path(args.tokenizer)
    initialized = time.perf_counter()
    variants = {}
    for name in ("trie", "lcp_sort"):
        start = time.perf_counter()
        forest = tree.build_tree_execution_plans(
            keys, batch, tree_policy="role_probe", tree_builder=name,
            turn_min_segment_tokens=args.minimum
        )
        elapsed = (time.perf_counter() - start) * 1000
        _validate(forest, batch)
        variants[name] = (forest, {"build_ms": round(elapsed, 2), **_summary(forest, batch)})
    if tree.digest_forest_plan(variants["trie"][0]) != tree.digest_forest_plan(variants["lcp_sort"][0]):
        raise AssertionError("trie/lcp_sort plan digest mismatch")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    reference_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    has_chatml_reference = (
        isinstance(reference_id, int) and reference_id >= 0
        and tokenizer.convert_ids_to_tokens(reference_id) == "<|im_start|>"
    )
    match_count = reference_count = recovered_count = 0
    role_label_checked = role_label_correct = 0
    role_counts = {}
    for tokens in _rows(batch, "input_ids"):
        cpu = tokens.detach().cpu().long()
        anchors = tree._role_anchors(cpu, probe)
        recovered_count += len(anchors)
        for _at, role in anchors:
            role_counts[role] = role_counts.get(role, 0) + 1
        if has_chatml_reference:
            real = set(torch.nonzero(cpu == reference_id, as_tuple=False).flatten().tolist())
            estimated = dict(anchors)
            match_count += len(real & set(estimated))
            reference_count += len(real)
            for position in real & set(estimated):
                rendered = tokenizer.decode(
                    cpu[position + 1:min(len(cpu), position + 24)].tolist(),
                    skip_special_tokens=False
                )
                label = rendered.split("\n", 1)[0].strip()
                if label in {"system", "developer", "user", "assistant", "tool"}:
                    role_label_checked += 1
                    role_label_correct += int(estimated[position] == label)

    output = {
        "role_template_init_ms": round((initialized - started) * 1000, 2),
        "role_header_patterns": len(probe.headers),
        "recovered_role_headers": recovered_count,
        "recovered_role_counts": role_counts,
        "qwen_chatml_reference": {
            "available": has_chatml_reference,
            "reference_message_starts": reference_count if has_chatml_reference else None,
            "matched": match_count if has_chatml_reference else None,
            "recall": round(match_count / reference_count, 5) if reference_count else None,
            "precision": round(match_count / recovered_count, 5) if recovered_count and has_chatml_reference else None,
            "role_labels_compared": role_label_checked if has_chatml_reference else None,
            "role_label_accuracy": round(role_label_correct / role_label_checked, 5) if role_label_checked else None,
            "note": "Only a Qwen tokenizer-specific offline reference; in-band markers can be ambiguous.",
        },
        "trie": variants["trie"][1],
        "lcp_sort": variants["lcp_sort"][1],
        "plan_digest_equal": True,
        "token_path_and_ppo_refs_checked": True,
    }
    try:
        baseline = tree.build_tree_execution_plans(
            keys, batch, tree_policy="qwen_turn", tree_builder="trie",
            turn_min_segment_tokens=args.minimum
        )
        _validate(baseline, batch)
        output["legacy_qwen_turn"] = _summary(baseline, batch)
    except Exception as e:
        output["legacy_qwen_turn_error"] = str(e)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
