"""Find a real eight-row Actor PPO capture eligible for an exact P/S shape.

Read-only CPU probe; never pads or synthesizes prompts, responses or PPO labels.
Output is a single source filename on stdout, diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

DEFAULT_ROOTS = (
    Path("/workspace/tpr_actor_capture_4b_n8_run5"),
    Path("/workspace/tpr_actor_capture_4b_n8_run3"),
    Path("/workspace/tpr_actor_capture_4b_n8_run2"),
)


def is_eligible(path: Path, p: int, s: int) -> tuple[bool, str]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if raw.get("capture_stage") != "actor_update_mini_batch":
            return False, "wrong_stage"
        batch = raw["tensordict"]
        if len(raw["keys"]) != 8 or tuple(batch.batch_size) != (8,):
            return False, "not_eight_rows"
        fields = ("input_ids", "prompts", "responses", "response_mask",
                  "loss_mask", "old_log_probs", "advantages", "temperature")
        if any(k not in batch for k in fields):
            return False, "missing_ppo_fields"
        prompts = [batch["prompts"][i] for i in range(8)]
        responses = [batch["responses"][i] for i in range(8)]
        if any(v.numel() < p for v in prompts):
            return False, "prompt_shortfall"
        if any(v.numel() < s for v in responses):
            return False, "response_shortfall"
        shared = prompts[0][-p:]
        if any(not torch.equal(v[-p:], shared) for v in prompts[1:]):
            return False, "nonidentical_prompt"
        distinct = {v[:s].contiguous().numpy().tobytes() for v in responses}
        if len(distinct) != 8:
            return False, "non_distinct_responses"
        for i in range(8):
            if not torch.equal(batch["input_ids"][i],
                               torch.cat((prompts[i], responses[i]))):
                return False, "input_mismatch"
            for key in ("response_mask", "loss_mask", "old_log_probs", "advantages"):
                if batch[key][i].numel() != responses[i].numel():
                    return False, "ppo_length_mismatch"
            if not batch["response_mask"][i][:s].bool().any().item():
                return False, "unsupervised_response"
        return True, "eligible"
    except Exception as exc:
        return False, f"read_or_validation_error:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--suffix", type=int, required=True)
    parser.add_argument("--roots", type=Path, nargs="*", default=DEFAULT_ROOTS)
    args = parser.parse_args()
    candidates = sorted({
        path for root in args.roots if root.is_dir()
        for path in root.rglob("actor_update*batch*.pt")
    })
    failures = Counter()
    for path in candidates:
        ok, reason = is_eligible(path, args.prefix, args.suffix)
        if ok:
            print(f"TPR_N8_SOURCE_FOUND P={args.prefix} S={args.suffix} "
                  f"examined={sum(failures.values())+1} path={path}", file=sys.stderr)
            print(path)
            return
        failures[reason] += 1
    print(f"TPR_N8_NO_ELIGIBLE_SOURCE P={args.prefix} S={args.suffix} "
          f"examined={len(candidates)} reasons={dict(failures)} "
          f"roots={[str(r) for r in args.roots]}. "
          "Supply TPR_CP4_N8_SOURCE_18K_18K or TPR_CP4_N8_SOURCE_32K_4K "
          "pointing to an eligible original actor_update_mini_batch capture.",
          file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
