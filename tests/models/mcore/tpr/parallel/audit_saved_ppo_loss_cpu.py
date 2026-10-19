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
from .native_full_megatron_schedule import native_cp_query_indices


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


def _ppo_token_gradient_diagnostics(
    full_new: torch.Tensor, tpr_new: torch.Tensor,
    old: torch.Tensor, adv: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    """Compare independent FP32 PPO upstream gradients on *identical* labels.

    Diagnostic derivative of compute_policy_loss_vanilla token-mean loss;
    this is NOT the gradient through BF16 logits or Megatron and does not
    describe parameter-grad differences caused by attention/backward.
    """
    a, b, old, adv = [
        x.to(torch.float32).flatten() for x in
        (full_new, tpr_new, old, adv)
    ]
    if not a.numel() or any(x.shape != a.shape for x in (b, old, adv)):
        raise ValueError("PPO token diagnostic shapes must match")
    if weights is None:
        weight = torch.ones_like(a)
    else:
        weight = weights.to(torch.float32).flatten()
        if weight.shape != a.shape:
            raise ValueError("PPO importance-weight shape mismatch")

    def gradient_and_effective_clip(new):
        log_ratio = new - old
        ratio = log_ratio.clamp(-20.0, 20.0).exp()
        # VERL's dual-clip option imposes an additional A<0 ratio>3
        # saturation, beyond the ordinary PPO lower-clip boundary.
        ineffective = (
            ((adv > 0) & (ratio > 1.2)) |
            ((adv < 0) & ((ratio < 0.8) | (ratio > 3.0))) |
            (log_ratio < -20.0) |
            (log_ratio > 20.0)
        )
        active_gradient = torch.where(
            ineffective, torch.zeros_like(ratio),
            -adv * ratio * weight / a.numel(),
        )
        return active_gradient, ineffective, ratio

    g_full, clip_full, ratio_full = gradient_and_effective_clip(a)
    g_tpr, clip_tpr, ratio_tpr = gradient_and_effective_clip(b)
    full_norm = torch.linalg.vector_norm(g_full).item()
    tpr_norm = torch.linalg.vector_norm(g_tpr).item()
    diff_norm = torch.linalg.vector_norm(g_tpr - g_full).item()
    dot = torch.dot(g_full, g_tpr).item()
    abs_lp = (b-a).abs()
    return {
        "tokens": int(a.numel()),
        "logprob_mean_abs": float(abs_lp.mean()),
        "logprob_max_abs": float(abs_lp.max()),
        "logprob_rel_l2": float(
            torch.linalg.vector_norm(b-a).item() /
            max(torch.linalg.vector_norm(a).item(), 1e-30)
        ),
        "ratio_mean_abs": float((ratio_tpr-ratio_full).abs().mean()),
        "effective_clip_full": int(clip_full.sum()),
        "effective_clip_tpr": int(clip_tpr.sum()),
        "effective_clip_flip": int((clip_full ^ clip_tpr).sum()),
        "effective_clip_flip_fraction": float((clip_full ^ clip_tpr).float().mean()),
        "upstream_ppo_gradient_rel_l2": diff_norm / max(full_norm, 1e-30),
        "upstream_ppo_gradient_cosine": (
            dot / max(full_norm*tpr_norm, 1e-30)
        ),
        "upstream_ppo_gradient_full_norm": full_norm,
        "upstream_ppo_gradient_tpr_norm": tpr_norm,
    }


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
        token_evidence = {}
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
            cp_token_ownership = {str(cp): 0 for cp in range(4)}
            has_weights = "rollout_is_weights" in batch
            for row_index, source_row in enumerate(rows):
                masks = batch["response_mask"].unbind(0)[source_row].bool()
                tokens = batch["input_ids"].unbind(0)[source_row]
                prompt_len = len(batch["prompts"].unbind(0)[source_row])
                logit_indices = {
                    prompt_len + offset - 1
                    for offset in torch.nonzero(masks, as_tuple=False).flatten().tolist()
                }
                for cp in range(4):
                    physically_owned = set(
                        native_cp_query_indices(len(tokens), cp)
                    )
                    cp_token_ownership[str(cp)] += len(
                        physically_owned & logit_indices
                    )
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
            if sum(cp_token_ownership.values()) != len(new):
                raise AssertionError(
                    f"Native CP ownership does not cover tokens: {cp_token_ownership}"
                )
            token_evidence[mode] = {
                "new": torch.tensor(new),
                "old": torch.tensor(old),
                "adv": torch.tensor(adv),
                "weights": torch.tensor(weights) if has_weights else None,
            }
            canonical = _ppo_from_token_vectors(
                token_evidence[mode]["new"],
                token_evidence[mode]["old"],
                token_evidence[mode]["adv"],
                token_evidence[mode]["weights"],
            )
            original_losses = {
                str(x["cp_rank"]): float(x["loss"]) for x in saved
                if x["tp_rank"] == 0
            }
            modes[mode] = {
                "reconstructed_fp32_loss": canonical,
                "native_cp_original_supervised_token_counts": cp_token_ownership,
                "saved_by_cp": original_losses,
                "saved_minus_fp32_by_cp": {
                    k: v-canonical for k,v in original_losses.items()
                },
                "supervised_tokens": len(new),
            }
            print(
                f"TPR_SAVED_PPO_LOSS_AUDIT step={step} mode={mode} "
                f"fp32_recomputed={canonical:.9g} "
                f"native_cp_owned_ppo_tokens={cp_token_ownership} "
                f"old_saved_by_cp={original_losses} "
                f"saved_minus_recomputed={modes[mode]['saved_minus_fp32_by_cp']}",
                flush=True,
            )
        full, tpr = token_evidence["full"], token_evidence["tpr"]
        for field in ("old", "adv", "weights"):
            x, y = full[field], tpr[field]
            if (x is None) != (y is None) or (
                x is not None and not torch.equal(x, y)
            ):
                raise AssertionError(
                    f"Step {step}: original PPO {field} differs across modes"
                )
        cross = _ppo_token_gradient_diagnostics(
            full["new"], tpr["new"], full["old"], full["adv"],
            full["weights"],
        )
        cross["loss_tpr_minus_full"] = (
            modes["tpr"]["reconstructed_fp32_loss"] -
            modes["full"]["reconstructed_fp32_loss"]
        )
        result["steps"].append({
            "step": step, "modes": modes,
            "full_vs_tpr_ppo_upstream_diagnostics": cross,
        })
        print(
            f"TPR_SAVED_PPO_UPSTREAM_AUDIT step={step} "
            f"lp_mae={cross['logprob_mean_abs']:.9g} "
            f"clip_flip={cross['effective_clip_flip']}/{cross['tokens']} "
            f"dloss_dlp_rel_l2={cross['upstream_ppo_gradient_rel_l2']:.9g} "
            f"dloss_dlp_cos={cross['upstream_ppo_gradient_cosine']:.9g} "
            f"audited_ppo_loss_gap={cross['loss_tpr_minus_full']:.9g}",
            flush=True,
        )
    target = root / "saved_ppo_loss_and_upstream_audit.json"
    if target.exists():
        raise FileExistsError(
            f"Original saved PPO upstream audit exists; refusing overwrite: {target}"
        )
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
