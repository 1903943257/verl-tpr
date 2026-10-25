"""Prepare a short, SAME-DATA Full/TPR training-side replay without rollout.

Inputs:
  - an existing original or derived eight-row Actor capture .pt, OR
  - a directory of rank0 original Actor captures (one original minibatch
    with eight independent Session rows per online step).

The derived captures and manifest are written exclusively under a NEW
experiment output directory. This is controlled PPO numerical replay:
10K windows preserve original PPO fields on retained tokens but can change
the causal conditioning, so they must never be called the original online PPO.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .prepare_real_actor_20step_manifest import CAPTURE_NAME, REQUIRED
from .prepare_tpr_10k_parity_capture import derive_10k
from .prepare_tpr_30k_one_step import derive_capture

_ALLOWED_CAPS = (0, 10240, 30720)


def _rank0_sources(root: Path, steps: int) -> list[Path]:
    if root.is_file():
        if root.suffix != ".pt" or steps != 1:
            raise ValueError("An existing .pt capture is only a single Replay step")
        return [root.resolve()]
    if not root.is_dir():
        raise FileNotFoundError(root)
    found = {}
    for path in sorted(root.glob("actor_update_step*_rank0_capture*_batch*.pt")):
        m = CAPTURE_NAME.fullmatch(path.name)
        if m is None:
            continue
        found.setdefault(int(m.group("step")), []).append(path.resolve())
    selected = []
    for step in range(1, steps + 1):
        matches = found.get(step, ())
        if len(matches) != 1:
            raise ValueError(
                f"online step {step} has {len(matches)} rank0 Actor captures, "
                "expected exactly one; for multi-minibatch steps use the "
                "existing session-first merged manifest, not a lossy shortcut"
            )
        selected.append(matches[0])
    return selected


def _validate_8session_capture(path: Path) -> tuple[dict, dict]:
    from .offline_session_selection import session_rows

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch":
        raise ValueError(f"Not a real Actor minibatch capture: {path}")
    batch = raw["tensordict"]
    keys = tuple(raw["keys"])
    if len(keys) != 8 or len(batch["input_ids"]) != 8:
        raise ValueError(f"Require precisely eight captured Actor trajectories: {path}")
    if len(set(keys)) != 8 or any(not isinstance(x, str) for x in keys):
        raise ValueError("Captured trajectory keys must be eight distinct strings")
    if any(field not in batch for field in REQUIRED):
        raise ValueError(f"Original PPO fields missing in {path}")
    for row in range(8):
        tokens = batch["input_ids"][row]
        prompt = batch["prompts"][row]
        response = batch["responses"][row]
        if not torch.equal(tokens, torch.cat((prompt, response))):
            raise AssertionError(f"input_ids != prompts + responses at row {row}")
        for name in ("response_mask", "loss_mask", "old_log_probs", "advantages"):
            if len(batch[name][row]) != len(response):
                raise AssertionError(f"{name} misaligned at row {row}")
        if not torch.equal(
            batch["response_mask"][row].bool(), batch["loss_mask"][row].bool()
        ):
            raise ValueError(
                "Native Full and TPR cannot use different supervised PPO masks"
            )
        if not bool(batch["response_mask"][row].bool().any()):
            raise ValueError(f"At least one PPO token required in each row, got row {row}")

    # Already-derived historical captures may intentionally omit the
    # session_id field. Do not invent identities, but require unique keys.
    if "uid" in batch and "session_id" in batch:
        identities, _, _ = session_rows(batch)
        if len(set(identities)) != 8:
            raise ValueError("Original 8 rows are not eight distinct Sessions")

    sizes = [int(x.numel()) for x in batch["input_ids"].unbind(0)]
    tokens = int(sum(x.bool().sum().item() for x in batch["response_mask"].unbind(0)))
    return raw, {
        "sequence_lengths": sizes,
        "ppo_tokens": tokens,
        "logical_full_tokens": sum(sizes),
        "max_sequence_length": max(sizes),
        "min_sequence_length": min(sizes),
    }


def prepare_short_manifest(source: Path, output_root: Path, *,
                           steps: int = 1, cap: int = 10240,
                           require_shared_prefix: bool = True) -> Path:
    if not 1 <= steps <= 20:
        raise ValueError("steps must be 1..20")
    if cap not in _ALLOWED_CAPS:
        raise ValueError(f"short replay supports only existing validated caps {_ALLOWED_CAPS}")
    output_root = output_root.resolve()
    manifest_path = output_root / "capture_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite capture Manifest: {manifest_path}")
    sources = _rank0_sources(source.resolve(), steps)
    items, audits = [], []
    for step, original in enumerate(sources, start=1):
        source_raw, _ = _validate_8session_capture(original)
        selected = original
        shortening = None
        if cap and any(
            int(ids.numel()) > cap
            for ids in source_raw["tensordict"]["input_ids"].unbind(0)
        ):
            selected = output_root / "short_captures" / f"step{step}_cap{cap}.pt"
            if selected.exists():
                raise FileExistsError(f"Refusing to overwrite derived PPO capture {selected}")
            if cap == 10240:
                shortening = derive_10k(original, selected)
            else:
                shortening = derive_capture(original, selected, cap=cap)
        payload, stats = _validate_8session_capture(selected)
        if cap and stats["max_sequence_length"] > cap:
            raise AssertionError("Capture was not reduced to the requested sequence cap")
        if source_raw.get("capture_global_step") is not None:
            source_step = int(source_raw["capture_global_step"])
        else:
            source_step = step

        # Compute ACTUAL qwen_turn topology on shortened bytes, not a
        # fabricated estimated prefix ratio. Validate numerical experiment
        # really enters tree segments rather than only eight unshared roots.
        from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans
        forest = build_tree_execution_plans(
            tuple(payload["keys"]), payload["tensordict"],
            tree_policy="qwen_turn", turn_min_segment_tokens=128,
            require_loss_mask_alignment=True,
        )
        unique_tokens = sum(
            segment.length for tree in forest.trees
            for segment in tree.segment_plan.segments.values()
        )
        ratio = stats["logical_full_tokens"] / unique_tokens
        actual_segmentation = forest.segment_count > len(forest.trees)
        if require_shared_prefix and (ratio <= 1.0 or not actual_segmentation):
            raise ValueError(
                f"step={step} has no genuine shared prefix tree: "
                f"trees={len(forest.trees)} segments={forest.segment_count} "
                f"reuse={ratio:.6f}; this capture tests DDP/optimizer but not "
                "split/shared prefix correctness. Choose another capture "
                "or set TPR_REPLAY_REQUIRE_SHARED_PREFIX=0 explicitly."
            )
        item = {
            "step": step,
            "source_global_step": source_step,
            "source_capture_index": source_raw.get("capture_index"),
            "capture": str(selected),
            "rows": list(range(8)),
            "session_count": 8,
            "original_capture_count": 1,
            "original_rows": 8,
            "ppo_tokens": stats["ppo_tokens"],
            "logical_full_tokens": stats["logical_full_tokens"],
            "max_sequence_length": stats["max_sequence_length"],
            "min_sequence_length": stats["min_sequence_length"],
            "actual_tree_count": len(forest.trees),
            "actual_segment_count": forest.segment_count,
            "actual_prefix_reuse_ratio": ratio,
            "source_capture": str(original),
            "effective_sequence_cap": cap or None,
            "conditioning_changed": bool(shortening and (
                shortening.get("changes_original_conditioning")
                or shortening.get("conditioning_changed")
            )),
        }
        items.append(item)
        audits.append({
            "step": step,
            "original_capture": str(original),
            "derived_capture": str(selected),
            "derived_audit": shortening,
            "actual_tree_count": len(forest.trees),
            "actual_segment_count": forest.segment_count,
            "actual_prefix_reuse_ratio": ratio,
            "token_count": stats["ppo_tokens"],
            "max_sequence_length": stats["max_sequence_length"],
        })
        print(
            f"REAL_ENGINE_SHORT_CAPTURE step={step} "
            f"max_len={stats['max_sequence_length']} "
            f"ppo_tokens={stats['ppo_tokens']} "
            f"trees={len(forest.trees)} segments={forest.segment_count} "
            f"reuse={ratio:.6f} derived={selected != original}",
            flush=True,
        )
    result = {
        "protocol": "real_actor_session_first_8merged_v3",
        "online_steps": steps,
        "steps": steps,
        "expected_sessions": 8,
        "max_seq_len": max(item["max_sequence_length"] for item in items),
        "historical_ppo_labels_modified": False,
        "controlled_short_replay": True,
        "purpose": "same_data_full_tpr_segment_and_megatron_ddp_parity_only",
        "items": items,
        "session_audits": audits,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2))
    (output_root / "capture_derivation_audit.json").write_text(
        json.dumps(audits, indent=2)
    )
    print(f"REAL_ENGINE_SHORT_MANIFEST_READY {manifest_path}", flush=True)
    return manifest_path
