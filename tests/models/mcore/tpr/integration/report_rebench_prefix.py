#!/usr/bin/env python3
"""Summarize EXACT shared prefix and suffix lengths from real actor PPO captures.

Read only trusted, locally produced torch.save snapshots. Actor capture files
may contain only a minibatch of a rollout group, so aggregate ALL captures
by (global_step, uid) and deduplicate Sessions before measuring N siblings.
The exact LCP ratio is a DATASET selection signal, NOT the exact realized
Qwen-turn tree reuse/TPR performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch

# Shared by the existing 20-step real actor capture replay code.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parallel"))
from offline_session_selection import session_rows

NAME = re.compile(r"actor_update_step(?P<step>\d+)_rank0_capture(?P<ordinal>\d+)_batch(?P<batch>\d+)\.pt$")


def common_prefix_length(sequences):
    if len(sequences) < 2:
        return 0
    upto = min(len(row) for row in sequences)
    first = sequences[0]
    for pos in range(upto):
        if any(row[pos] != first[pos] for row in sequences[1:]):
            return pos
    return upto


def summarize_group(step, uid, sessions, expected_n):
    rows = list(sessions.values())
    seqs = [row["input_ids"] for row in rows]
    p = common_prefix_length(seqs)
    suffix = [len(row) - p for row in seqs]
    logical = sum(map(len, seqs))
    unique_root = p + sum(suffix)
    mean_s = sum(suffix) / len(suffix)
    response = [row["response_length"] for row in rows]
    ppo = [row["ppo_tokens"] for row in rows]
    return {
        "step": step, "uid": uid, "n": len(rows),
        "expected_n": expected_n, "complete_n": len(rows) == expected_n,
        "logical_tokens": logical,
        "shared_prefix_tokens": p,
        "mean_suffix_tokens": round(mean_s, 2),
        "min_suffix_tokens": min(suffix),
        "max_suffix_tokens": max(suffix),
        "measured_p_over_mean_s": round(p / mean_s, 4) if mean_s else None,
        "mean_response_tokens": round(sum(response) / len(response), 2),
        "mean_ppo_tokens": round(sum(ppo) / len(ppo), 2),
        "root_only_unique_tokens": unique_root,
        "root_only_dedup_fraction": round(1.0 - unique_root / logical, 6),
        "sessions_with_trajectory_ordinal": sum(row["has_ordinal"] for row in rows),
        "reward_variance": None,
    }


def analyze(directory, expected_n):
    captures = []
    for path in directory.glob("actor_update_step*_rank0_capture*_batch*.pt"):
        match = NAME.fullmatch(path.name)
        if match:
            captures.append((int(match["step"]), int(match["ordinal"]),
                             int(match["batch"]), path))
    captures.sort()
    if not captures:
        raise FileNotFoundError(f"No actor-update captures in {directory}")

    # (global_step, uid) -> (session_id -> representative first trajectory)
    groups = {}
    for step, capture_index, batch_idx, path in captures:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if raw.get("capture_stage") != "actor_update_mini_batch":
            raise ValueError(f"{path}: not an actual PPO actor-update minibatch")
        if int(raw.get("capture_global_step", -1)) != step:
            raise ValueError(f"{path}: global_step mismatch")
        td = raw["tensordict"]
        identities, ordinals, _ = session_rows(td)
        for row_idx, (uid, session) in enumerate(identities):
            if uid.startswith("pad"):
                continue
            tokens = td["input_ids"][row_idx].detach().cpu().tolist()
            prompt = td["prompts"][row_idx]
            response = td["responses"][row_idx]
            if len(tokens) != prompt.numel() + response.numel():
                raise ValueError(f"{path}, row {row_idx}: token alignment mismatch")
            rm = td["response_mask"][row_idx]
            if rm.numel() != response.numel():
                raise ValueError(f"{path}, row {row_idx}: response_mask mismatch")
            score = (0, ordinals[row_idx], capture_index, batch_idx, row_idx) if ordinals else (
                1, capture_index, batch_idx, row_idx, 0
            )
            sessions = groups.setdefault((step, uid), {})
            prior = sessions.get(session)
            if prior is None or score < prior["selection_order"]:
                sessions[session] = {
                    "selection_order": score,
                    "has_ordinal": ordinals is not None,
                    "input_ids": tokens,
                    "response_length": int(response.numel()),
                    "ppo_tokens": int(rm.bool().sum().item()),
                }
    summary = [
        summarize_group(step, uid, rows, expected_n)
        for (step, uid), rows in sorted(groups.items())
    ]
    if not summary:
        raise ValueError("No non-padding task groups found")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--expected-n", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = ap.parse_args()
    if args.expected_n < 2:
        ap.error("--expected-n must be >= 2")
    rows = analyze(args.captures, args.expected_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r["complete_n"]]
    print(json.dumps({
        "groups": len(rows),
        "complete_n_groups": len(valid),
        "incomplete_groups": len(rows) - len(valid),
        "results_csv": str(args.out),
        "best_complete_groups_by_ratio": sorted(
            [{"step": r["step"], "uid": r["uid"],
              "p_over_s": r["measured_p_over_mean_s"],
              "root_only_dedup_fraction": r["root_only_dedup_fraction"]}
             for r in valid],
            key=lambda x: (x["p_over_s"] if x["p_over_s"] is not None else -1),
            reverse=True,
        )[:10],
        "warning": ("This is exact token LCP and root-only dedup, not Qwen-turn "
                    "realized tree reuse; reward diversity needs independent audit."),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
