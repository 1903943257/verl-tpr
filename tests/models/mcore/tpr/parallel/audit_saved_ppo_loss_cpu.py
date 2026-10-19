"""Independent CPU-only PPO loss audit using ORIGINAL Actor Capture labels.

Reports a full-precision scalar from the captured, identity-aligned Logprobs,
old_log_probs and advantages. It never edits old 8-rank snapshots, and it is
NOT a replay of the BF16 forward/backward scalar; differences may remain.
Use it to diagnose the FP32-vs-BF16 reporting all_reduce bug.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from .compare_real_actor_tp2cp4_offline_ppo import _logprobs


def _ppo_from_token_vectors(new: torch.Tensor, old: torch.Tensor,
                            advantage: torch.Tensor,
                            is_weight: torch.Tensor | None = None) -> float:
    """Independent FP32 reconstruction of VERL vanilla clip=0.2 token mean."""
    new, old, advantage = (x.to(torch.float32).flatten()
                           for x in (new, old, advantage))
    if not new.numel() or new.shape != old.shape or new.shape != advantage.shape:
        raise ValueError("PPO token vectors must be equal, nonempty")
    ratio = torch.exp((new - old).clamp(-20.0, 20.0))
    pg1 = -advantage * ratio
    pg2 = -advantage * ratio.clamp(0.8, 1.2)
    pg = torch.maximum(pg1, pg2)
    # Match compute_policy_loss_vanilla's negative-advantage dual-clip path.
    pg = torch.where(advantage < 0, torch.minimum(pg, -advantage * 3.0), pg)
    if is_weight is not None:
        if is_weight.numel() != new.numel():
            raise ValueError("Missing rollout importance weights")
        pg = pg * is_weight.float().flatten()
    return float(pg.sum().item() / new.numel())


def audit(root: Path, steps: tuple[int, ...]) -> dict:
    result = {"kind": "INDEPENDENT_CPU_FP32_PPO_SCALAR_RECONSTRUCTION",
              "caution": (
                  "Uses saved float32-represented Logprobs and unchanged Actor "
                  "labels; not bit-identical to the original BF16 loss graph. "
                  "Does not repair, certify, or rewrite original PPO gradients "
                  "or optimizer updates."
              ), "steps": []}
    for step in steps:
        modes = {}
        for mode in ("full", "tpr"):
            saved = [
                torch.load(root / f"{mode}_step{step}_rank{rank}.pt",
                           map_location="cpu", weights_only=False)
                for rank in range(8)
            ]
            source = Path(saved[0]["source_capture"])
            original = torch.load(source, map_location="cpu", weights_only=False)
            batch = original["tensordict"]
            rows = [int(x) for x in saved[0]["source_rows"]]
            lp = _logprobs(saved)
            # Gather ORIGINAL PPO data only from the exact supervised
            # response offsets recorded by the 8-rank execution.
            old, adv, new, weights = [], [], [], []
            has_weights = "rollout_is_weights" in batch
            for row_index, source_row in enumerate(rows):
                masks = batch["response_mask"].unbind(0)[source_row].bool()
                old_row = batch["old_log_probs"].unbind(0)[source_row]
                adv_row = batch["advantages"].unbind(0)[source_row]
                weight_row = (
                    batch["rollout_is_weights"].unbind(0)[source_row]
                    if has_weights else None
                )
                for offset in torch.nonzero(masks, as_tuple=False).flatten().tolist():
                    key = (row_index, offset)
                    if key not in lp:
                        raise AssertionError(f"{mode} missing captured logprob {key}")
                    new.append(float(lp[key]))
                    old.append(float(old_row[offset]))
                    adv.append(float(adv_row[offset]))
                    if has_weights:
                        weights.append(float(weight_row[offset]))
            if len(new) != len(lp) or len(new) != saved[0]["tokens"]:
                raise AssertionError("Saved Logprobs do not exactly cover original PPO tokens")
            canonical = _ppo_from_token_vectors(
                torch.tensor(new), torch.tensor(old), torch.tensor(adv),
                torch.tensor(weights) if has_weights else None,
            )
            original_losses = {
                str(x["cp_rank"]): float(x["loss"]) for x in saved
                if x["tp_rank"] == 0
            }
            modes[mode] = {
                "reconstructed_fp32_loss": canonical,
                "saved_by_cp": original_losses,
                "saved_minus_fp32_by_cp": {
                    k: v-canonical for k,v in original_losses.items()
                },
                "supervised_tokens": len(new),
            }
            print(
                f"TPR_SAVED_PPO_LOSS_AUDIT step={step} mode={mode} "
                f"fp32_recomputed={canonical:.9g} "
                f"old_saved_by_cp={original_losses} "
                f"saved_minus_recomputed={modes[mode]['saved_minus_fp32_by_cp']}",
                flush=True,
            )
        result["steps"].append({"step": step, "modes": modes})
    target = root / "saved_logprob_based_ppo_loss_audit.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"TPR_SAVED_PPO_LOSS_AUDIT_REPORT={target}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", default="1,2,5")
    args = parser.parse_args()
    audit(args.root, tuple(int(s) for s in args.steps.split(",") if s))


if __name__ == "__main__":
    main()
