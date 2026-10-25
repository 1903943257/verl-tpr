# SWE-reBench short-fix candidates (TPR/Claude Code)

This is **stage 1 only**: a CPU-side screening tool for UniAgent SWE-reBench
tasks. It does not modify VERL/TPR runtime or UniAgent rollout, and does **not**
claim a 2K/4K response or any prefix-reuse speedup.

Script: `filter_swe_rebench_shortfix.py`
Test: `test_filter_swe_rebench_shortfix_cpu.py`

## Input and output

The filter supports both metadata layouts:

- `extra_info.tools_kwargs.task.metadata` (current UniAgent format)
- `extra_info.tools_kwargs.reward.metadata` (legacy local 50-row Parquet)

It writes two Parquet files intentionally:

- `shortfix_candidates.parquet`: filtered rows preserving the input schema.
- `shortfix_candidates_task.parquet`: normalized to the current UniAgent
  framework schema, with `extra_info.tools_kwargs.task`, for
  `examples/inference/parallel_infer_verl.py`.

The legacy-to-current conversion maps `reward.name` to `task.name` and
`reward.metadata` to `task.metadata`, and keeps the original prompt and
agent_name.

**Image migration matters.** The legacy rows use `swe-local:3.12` and a
`post_setup_cmd` that clones a repository into `/testbed`. The current
OpenYuanrong recipe instead maps per-instance image names
`swerebench/sweb.eval.x86_64.<normalized-instance-id>` to the internal
SWE-reBench image registry. The converter therefore derives the canonical
per-instance image name from `instance_id`; it does not copy the local image
or the old setup shell command into the current task config. The image must
already contain `/testbed` checked out at `base_commit`. Verify this with a
single-task smoke test before running a batch. If the image is unavailable or
the checkout is wrong, stop and fix the image mapping rather than silently
restoring the old setup command.

The gold patch is used **offline only** for task selection; it is not inserted
into the model-facing prompt.

## Run stage 1

From the `verl-tpr` repository root:

    python -m pytest -q tests/models/mcore/tpr/parallel/test_filter_swe_rebench_shortfix_cpu.py

For the local 50-row file:

    python tests/models/mcore/tpr/parallel/filter_swe_rebench_shortfix.py \
        /workspace/data/swe_agent/swe_rebench_filtered_local.parquet \
        --out /workspace/tpr_experiments/shortfix \
        --limit 50

Outputs:

- `shortfix_counts.json`: input, selected, strict/relaxed/reject counts;
- `shortfix_ranked.csv`: selected instance IDs and static audit metrics;
- `shortfix_candidates.parquet`: original row schema;
- `shortfix_candidates_task.parquet`: current framework Task Config schema.

Use **`shortfix_candidates_task.parquet`** as `--data-path` for the current
`examples/inference/parallel_infer_verl.py` entrypoint. Do not pass the legacy
`shortfix_candidates.parquet` directly: the framework runner requires
`tools_kwargs['task']`, while legacy rows only contain `env` and `reward`.

Strict tier: 1 modified source file (1 total), at most 30 changed lines,
at most 3 diff hunks, 1–3 FAIL_TO_PASS tests and at most 3K issue characters.
Relaxed tier: at most 2 modified source files, at most 60 changed lines,
at most 4 diff hunks, 1–5 FAIL_TO_PASS tests and at most 4.5K issue characters.

Gold-patch scope is only a **proxy** for short Claude Code completions.

## Stage 2: actual rollout audit (required before performance claims)

Run unchanged Claude Code + UniAgent inference, initially `n=1` or `n=2`
on a small candidate batch. Use the actual existing inference launcher and
configuration, pointing `--data-path` at `shortfix_candidates_task.parquet`.

For a validated candidate, repeat with `--n 8`; do not infer any 8-way
sharing from gold patch size.

Measure from real model-facing token IDs / final training captures:

1. Exact longest common prefix in the eight actual tokenized trajectories.
2. Per-trajectory accumulated generated suffix tokens, not only the final
   assistant message, excluding padding.
3. Finished status, verified reward, and nonzero within-group reward variance.
4. Actual forest unique-token ratio and loss masks (not nominal P:S).

First aim for at most 4K successful suffixes; 2K is a stretch goal. Prefer at
least 70% measured token deduplication for high-reuse comparisons, but report
all screened tasks and selection counts. Do not shorten otherwise successful
trajectories after rollout or repeat prompt text to force high reuse.
Preloading source text changes the workload and must be described separately
rather than passed off as unmodified SWE-reBench.

Keep the earlier Django long-trajectory case as the low-reuse baseline.
