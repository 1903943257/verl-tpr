# Repeat the ten-update Base CP1 control

Pull this branch on the host, then run in the same training image, with the
same checkpoint and physical NPU used by the original Base CP1:

```bash
export TPR_TRAIN_MODEL_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_TRAIN10_REPEAT_SOURCE=/workspace/tpr_experiments/p1024s1024_train10_1
export TPR_TRAIN10_OUTPUT=/workspace/tpr_experiments/p1024s1024_base_cp1_repeat_1
export TPR_TRAIN10_REPEAT_DEVICE=0
bash tests/models/mcore/tpr/parallel/run_training_base_cp1_repeat.sh
cat "$TPR_TRAIN10_OUTPUT/summary.md"
```

This runs **only one new Base CP1 job, ten native updates**. It loads the
original HF weights and a fresh optimizer, not the original final checkpoint.
The saved seed, LR, bridge, fusion and precision flags are restored from the
original result. Exact initial weights, input digest, package versions and
native runtime precision/topology must match. The original fixture is read
without rewriting its tokens, old logprobs or advantages. The eleven native
fixture-preparation forwards are replayed to match the first Base CP1 job's
call sequence; their outputs never replace the frozen labels.

The experiment measures run-to-run variation at an unchanged configuration.
`HCCL_DETERMINISTIC=true` is retained as in the original launcher; the repeat
does not add a new deterministic-algorithm setting that would change the
baseline configuration. Reuse the same hardware and software environment.

The historical experiment is read-only. The new output must be outside it
and must not already exist. The other three routes do not run again.

## What is compared

* Initial held-out probe output, plus probe logprob gaps at updates 1/5/10.
* The saved, fixed FP32 master-update coordinates from every parameter at
  updates 1/5/10: relative L2, cosine and max absolute difference.
* All ten steps' losses, native gradient norms, learning rates and PPO metrics.
* If all original Base CP1 full update blocks still exist, the final complete
  cumulative update is also compared using the same bounded CPU observer.

The normal four-route experiment deletes full references after successful
aggregation. When those blocks are gone, this repeat **cannot reconstruct a
historical full update vector** from its norm. It reports sampled update
errors explicitly and skips the full-model final scan. No new full reference
is saved. Existing small sample/probe files and manifests are enough.

If the original four-route `comparison.json` exists, sampled repeat error is
divided by the **sampled** native Base CP1/CP2 error at the same step and same
coordinates. It is never divided by the previous all-parameter error. Ratios
with a zero native denominator are undefined. No full-update claim is made
when only samples are available.

`summary.md` is short; `comparison.json` includes per-step metric gaps. Logs
and the native route/Adam-state audit are in `base_cp1_repeat/`. Peak NPU and
host memory are similar to the previous Base CP1 job; output disk is small
because full update blocks are never saved by the repeat. Failed repeat
folders are preserved; use another new output directory for another attempt.
