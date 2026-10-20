"""Read-only inventory of genuine eight-row Actor PPO capture lengths.

Find N8 1:1 workloads and fixed-18K alternatives. This is a LENGTH audit:
a shortlisted source must still pass prepare_tpr_cp1_n8_perf_capture's
token-identity, supervision, and qwen_turn gates before a benchmark.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .select_tpr_n8_perf_source import DEFAULT_ROOTS


def _round_down(value: int, alignment: int = 16) -> int:
    return value // alignment * alignment


def inventory(roots: list[Path]) -> dict:
    paths = sorted({
        p for root in roots if root.is_dir()
        for p in root.rglob("actor_update*batch*.pt")
    })
    records = []
    errors = Counter()
    extremes = {"longest_response": None, "shortest_response": None}
    for path in paths:
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if raw.get("capture_stage") != "actor_update_mini_batch":
                errors["wrong_capture_stage"] += 1
                continue
            data = raw["tensordict"]
            if len(raw["keys"]) != 8 or tuple(data.batch_size) != (8,):
                errors["not_eight_rows"] += 1
                continue
            prompts = [int(data["prompts"][i].numel()) for i in range(8)]
            responses = [int(data["responses"][i].numel()) for i in range(8)]
            for i, length in enumerate(responses):
                row = {"source": str(path), "row": i, "response_tokens": length,
                       "prompt_tokens": prompts[i]}
                for name, better in (("longest_response", lambda x, y: x > y),
                                     ("shortest_response", lambda x, y: x < y)):
                    old = extremes[name]
                    if old is None or better(length, old["response_tokens"]):
                        extremes[name] = row
            minp, minr = min(prompts), min(responses)
            # Longest balanced P=S, retaining all 8 original rows.
            n8_balanced = _round_down(min(minp, minr))
            # Closest-to-1:1 18,432-token case while retaining all eight rows.
            s_fixed = _round_down(min(9216, minr))
            p_fixed = 18432 - s_fixed
            possible_fixed = s_fixed > 0 and p_fixed <= minp
            # At least six source rows eligible for BOTH CP2 18K ratios.
            rows_n6 = [i for i in range(8)
                       if prompts[i] >= 16384 and responses[i] >= 9216]
            records.append({
                "source": str(path),
                "prompt_min": minp, "prompt_max": max(prompts),
                "response_min": minr, "response_max": max(responses),
                "response_min_row": responses.index(minr),
                "response_lengths": responses,
                "prompt_lengths": prompts,
                "n8_balanced_p_s": n8_balanced,
                "n8_fixed_18k_p": p_fixed if possible_fixed else None,
                "n8_fixed_18k_s": s_fixed if possible_fixed else None,
                "n6_common_18k_rows": rows_n6 if len(rows_n6) >= 6 else [],
                "n6_common_18k_eligible_by_length": len(rows_n6) >= 6,
            })
        except Exception as exc:
            errors[f"load:{type(exc).__name__}"] += 1
    return {"files_found": len(paths), "files_valid_n8": len(records),
            "errors": dict(errors), "extremes": extremes, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", type=Path,
                        help="Optional complete, per-capture inventory JSON")
    args = parser.parse_args()
    report = inventory(args.roots)
    rows = report["records"]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("TPR_CAPTURE_INVENTORY",
          json.dumps({k: report[k] for k in ("files_found", "files_valid_n8",
                                             "errors", "extremes")}, ensure_ascii=False))
    by_balanced = sorted(rows, key=lambda x: x["n8_balanced_p_s"], reverse=True)
    by_fixed = sorted(
        (r for r in rows if r["n8_fixed_18k_s"] is not None),
        key=lambda x: x["n8_fixed_18k_s"], reverse=True,
    )
    by_n6 = [r for r in rows if r["n6_common_18k_eligible_by_length"]]
    print("TPR_CAPTURE_BEST_N8_1TO1_LENGTHS", json.dumps(
        [{k: r[k] for k in ("source", "n8_balanced_p_s", "prompt_min",
                           "response_min", "response_min_row")}
         for r in by_balanced[:args.top]], ensure_ascii=False))
    print("TPR_CAPTURE_BEST_N8_FIXED18K_LENGTHS", json.dumps(
        [{k: r[k] for k in ("source", "n8_fixed_18k_p", "n8_fixed_18k_s",
                           "response_min", "response_min_row")}
         for r in by_fixed[:args.top]], ensure_ascii=False))
    print("TPR_CAPTURE_N6_BOTH_18K_LENGTHS", json.dumps(
        [{k: r[k] for k in ("source", "n6_common_18k_rows", "response_min")}
         for r in by_n6[:args.top]], ensure_ascii=False))
    print("TPR_CAPTURE_INVENTORY_COUNTS", json.dumps({
        "n8_length_1to1_at_least_6016": sum(r["n8_balanced_p_s"] >= 6016 for r in rows),
        "n8_length_1to1_at_least_9216": sum(r["n8_balanced_p_s"] >= 9216 for r in rows),
        "n8_fixed18k_length_candidates": len(by_fixed),
        "n6_both_18k_length_candidates": len(by_n6),
        "note": "Length-only preselection. Confirm shared prefix, distinct responses, "
                "PPO supervision and tree root via the strict capture preparer.",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
