"""Build and audit a Token-LCP Forest from a local TQ capture.

This is a Token-Level-only audit: it does not load a tokenizer, inspect a
Chat Template, probe roles, or construct a legacy qwen_turn baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[5]
TREE_FILE = ROOT / "verl/workers/engine/megatron/tpr/tree.py"
spec = importlib.util.spec_from_file_location("tpr_token_lcp_audit", TREE_FILE)
tree = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tree
spec.loader.exec_module(tree)


def _rows(batch, name):
    value = batch[name]
    return list(value) if isinstance(value, (tuple, list)) else list(value.unbind())


def _common_prefix_length(sequences):
    if not sequences:
        return 0
    limit = min(int(sequence.numel()) for sequence in sequences)
    if limit == 0:
        return 0
    reference = sequences[0][:limit].detach().cpu().long()
    common = limit
    for sequence in sequences[1:]:
        candidate = sequence[:limit].detach().cpu().long()
        mismatch = torch.nonzero(reference != candidate, as_tuple=False)
        if mismatch.numel():
            common = min(common, int(mismatch[0, 0]))
    return common


def _validate(forest, batch):
    inputs = _rows(batch, "input_ids")
    prompts = _rows(batch, "prompts")
    masks = _rows(batch, "response_mask")
    expected = {
        (row, offset)
        for row, mask in enumerate(masks)
        for offset in torch.nonzero(
            mask.detach().cpu().to(torch.bool), as_tuple=False
        ).flatten().tolist()
    }
    found = set()
    covered_rows = set()
    row_replay = {}

    for item in forest.trees:
        plan = item.segment_plan
        for row in item.member_rows:
            if row in covered_rows:
                raise AssertionError(f"row {row} appears in more than one physical tree")
            covered_rows.add(row)
            sequence = inputs[row].detach().cpu().long()
            replayed = False
            for end in plan.segments.values():
                if end.position_end != sequence.numel():
                    continue
                path = plan.path_to(end.segment_id)
                replay = torch.cat([segment.token_ids for segment in path])
                if torch.equal(replay, sequence):
                    replayed = True
                    break
            if not replayed:
                raise AssertionError(f"row {row} cannot be reconstructed from its tree path")
            row_replay[row] = True

        for ref in item.objective_refs:
            key = (ref.sample_row, ref.response_offset)
            if key in found:
                raise AssertionError(f"duplicate PPO ObjectiveRef {key}")
            found.add(key)
            row, offset = key
            target_pos = int(prompts[row].numel()) + offset
            segment = plan.get(ref.segment_id)
            query_pos = segment.position_start + ref.query_offset
            sequence = inputs[row].detach().cpu().long()
            if query_pos + 1 != target_pos:
                raise AssertionError(f"query/target position mismatch for {key}")
            if int(sequence[target_pos]) != ref.target_token_id:
                raise AssertionError(f"target Token ID mismatch for {key}")
            if int(sequence[query_pos]) != int(segment.token_ids[ref.query_offset]):
                raise AssertionError(f"query Token ID mismatch for {key}")

    if found != expected:
        raise AssertionError(
            f"PPO ObjectiveRef mismatch: missing={len(expected - found)}, "
            f"extra={len(found - expected)}"
        )
    if covered_rows != set(range(len(inputs))):
        raise AssertionError(
            f"uncovered rows: {sorted(set(range(len(inputs))) - covered_rows)}"
        )
    if forest.logical_loss_tokens != len(expected):
        raise AssertionError("Forest logical_loss_tokens disagrees with response_mask")
    return {
        "all_rows_reconstructed": len(row_replay) == len(inputs),
        "reconstructed_rows": len(row_replay),
        "objective_refs_exact": len(found) == len(expected),
        "objective_refs": len(found),
        "logical_loss_tokens": forest.logical_loss_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--minimum", type=int, default=1024)
    parser.add_argument("--tree-builder", choices=("lcp_sort", "trie"), default="lcp_sort")
    args = parser.parse_args()
    if args.minimum < 1:
        parser.error("--minimum must be positive")

    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "tensordict" not in payload or "keys" not in payload:
        raise ValueError("capture must contain keys and tensordict")
    batch = payload["tensordict"]
    keys = [str(value) for value in payload["keys"]]
    inputs = _rows(batch, "input_ids")
    if len(keys) != len(inputs):
        raise ValueError("key count disagrees with TQ row count")

    started = time.perf_counter()
    forest = tree.build_tree_execution_plans(
        keys,
        batch,
        tree_policy="token_lcp",
        tree_builder=args.tree_builder,
        min_shared_prefix_tokens=args.minimum,
    )
    build_ms = (time.perf_counter() - started) * 1000

    validation = _validate(forest, batch)
    logical_tokens = sum(int(sequence.numel()) for sequence in inputs)
    unique_tokens = sum(
        int(segment.length)
        for item in forest.trees
        for segment in item.segment_plan.segments.values()
    )
    per_tree = []
    for item in forest.trees:
        rows = list(item.member_rows)
        sequences = [inputs[row].detach().cpu().long() for row in rows]
        logical = sum(int(sequence.numel()) for sequence in sequences)
        unique = sum(int(segment.length) for segment in item.segment_plan.segments.values())
        common_prefix = _common_prefix_length(sequences) if len(rows) > 1 else 0
        per_tree.append({
            "tree_index": item.tree_index,
            "member_rows": rows,
            "member_uids": list(item.member_uids),
            "row_count": len(rows),
            "logical_tokens": logical,
            "unique_tokens": unique,
            "tokens_saved": logical - unique,
            "common_prefix_tokens": common_prefix,
            "segments": len(item.segment_plan.segments),
            "objective_refs": len(item.objective_refs),
        })

    output = {
        "policy": "token_lcp",
        "tree_builder": args.tree_builder,
        "minimum_shared_prefix_tokens": args.minimum,
        "capture": str(args.capture),
        "build_ms": round(build_ms, 2),
        "rows": len(inputs),
        "trees": len(forest.trees),
        "cross_uid_trees": sum(len(item.member_uids) > 1 for item in forest.trees),
        "logical_tokens": logical_tokens,
        "unique_tokens": unique_tokens,
        "tokens_saved": logical_tokens - unique_tokens,
        "reuse_factor": round(logical_tokens / unique_tokens, 6) if unique_tokens else None,
        "segments": forest.segment_count,
        "validation": validation,
        "per_tree": per_tree,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
