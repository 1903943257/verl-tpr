# SWE-reBench short-fix candidates (TPR/Claude Code)

This is **stage 1 only**: a CPU-side screening tool for UniAgent
SWE-reBench tasks. It does not modify VERL/TPR runtime or UniAgent rollout,
and does **not** claim a 2K/4K response or any prefix-reuse speedup.

Script: filter_swe_rebench_shortfix.py
Test: test_filter_swe_rebench_shortfix_cpu.py

## Input contract

The filter supports both metadata layouts:

- \`extra_info.tools_kwargs.task.metadata\` (newer UniAgent preprocessor output)
- \`extra_info.tools_kwargs.reward.metadata\` (the local 50-row Parquet layout)

It reads the gold patch and task metadata **offline only**. The output Parquet
preserves original rows, including the existing prompt, agent name, environment,
reward configuration and metadata. The gold patch is not inserted into any
model-facing prompt.

## Run stage 1

From the \`verl-tpr\` repository root:

    python -m pytest -q tests/models/mcore/tpr/parallel/test_filter_swe_rebench_shortfix_cpu.py

For the local 50-row file already present in this environment:

    python tests/models/mcore/tpr/parallel/filter_swe_rebench_shortfix.py \
        /workspace/data/swe_agent/swe_rebench_filtered_local.parquet \
        --out /workspace/tpr_experiments/shortfix \
        --limit 50

Alternatively, pass the output from the UniAgent preprocessor:

    python tests/models/mcore/tpr/parallel/filter_swe_rebench_shortfix.py \
        /workspace/tpr_experiments/datasets/swe_rebench_filtered.parquet \
        --out /workspace/tpr_experiments/shortfix \
        --limit 50

Outputs:
- \`shortfix_counts.json\`: input, selected, strict/relaxed/reject counts;
- \`shortfix_ranked.csv\`: selected instance IDs and static audit metrics;
- \`shortfix_candidates.parquet\`: original UniAgent rows, reordered/filtered
  and retaining the original schema.

Strict tier: 1 modified source file (1 total), <=30 changed lines,
<=3 diff hunks, 1-3 FAIL_TO_PASS tests and <=3K issue characters.
Relaxed tier: <=2 modified source files (2 total), <=60 changed lines,
<=4 diff hunks, 1-5 FAIL_TO_PASS tests and <=4.5K issue characters.

Gold-patch scope is only a **proxy** for short Claude Code completions.

## Stage 2: actual rollout audit (required before performance claims)

Run unchanged Claude Code + UniAgent inference, initially \`n=1\` or \`n=2\`
on a small candidate batch. Use the actual existing inference launcher and
configuration, and point \`--data-path\` at \`shortfix_candidates.parquet\`.

For a validated candidate, repeat with \`--n 8\`; do not infer any 8-way
sharing from gold patch size.

Measure from real model-facing token IDs / final training captures:
1. exact longest common prefix in the eight actual tokenized trajectories;
2. per-trajectory accumulated generated suffix tokens, not only the final
   assistant message, and excluding padding;
3. finished status, verified reward, and nonzero within-group reward variance;
4. actual forest unique-token ratio and loss masks (not nominal P:S).

First aim for <=4K successful suffixes; <=2K is a stretch goal. Prefer
>=70% measured token deduplication for high-reuse comparisons, but report
all screened tasks and selection counts. Do not shorten otherwise successful
trajectories after rollout or repeat prompt text to force high reuse.
Preloading source text changes the workload and must be described separately
rather than passed off as unmodified SWE-reBench.

Keep the earlier Django long-trajectory case as the low-reuse baseline.
