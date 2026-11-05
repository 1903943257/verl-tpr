# TPR numerical acceptance — phase 1

Scope: VERL/Megatron/MindSpeed native training with Qwen3-1.7B, BF16 model,
FP32 `main_grad` and optimizer masters, fused gradient accumulation,
TP=1, CP=1/2, DP=1. Fixed ten PPO minibatches, two P1024/S1024 rows
sharing a 1024-token prefix, AdamW at LR=1e-6. No online rollout,
recomputation or sequence parallelism in this experiment.

## First-backward full-gradient comparisons

All values below are relative L2 of complete FP32 `main_grad` vectors
at the native backward/finalize boundary before clipping and Adam.

| Comparison | Full gradient rel-L2 |
| --- | ---: |
| Base CP1 vs an independently repeated Base CP1 | 0.91225% |
| Base CP1 vs TPR CP1 | 4.67840% |
| Base CP2 vs TPR CP2 (Merge off) | 8.88371% |
| Base CP1 vs Base CP2 | 14.36803% |
| Base CP1 vs TPR CP2 (Merge off) | 16.21392% |

The Base CP1 repeat covers 1,720,574,976 gradient elements, has cosine
0.9999584055, loss gap 0, training logprob max gap 0 and initial probe
max gap 0. Native gradient norms: 20.5345131 vs 20.5373418. This is
one independent pair, **not** a measured noise distribution or a
production tolerance.

## Six-route, ten-update control

The numbers below compare **181,248 sampled coordinates of the cumulative
FP32 master-weight update**, `W_step10 - W_initial`, against the first
Base CP1 run. They are not full-gradient or full-update errors.
Probe MAE is measured at step 10 under native whole-sequence inference.

| Route | Step-10 sampled update rel-L2 | Cosine | Probe logprob MAE |
| --- | ---: | ---: | ---: |
| Base CP1 | 0% | 1.000000 | 0 |
| Base CP1 repeat | 10.97426% | 0.994006 | 0.019245 |
| Base CP2 | 12.96001% | 0.991640 | 0.019951 |
| TPR CP1 | 11.87961% | 0.992949 | 0.018764 |
| TPR CP2, Merge off | 15.70082% | 0.987845 | 0.021593 |
| TPR CP2, Merge on | 12.01240% | 0.992801 | 0.020309 |

The sampled first-update Base CP1 repeat discrepancy is **4.34842%**;
it is an Adam parameter-update statistic, **not a gradient statistic**.
Native Base CP1 reruns are not bitwise identical; the observed 0.91225%
first-backward gradient discrepancy has not been attributed to a specific
operator. Do not subtract it as noise or claim it is an official tolerance.

TPR CP2 Merge off/on also has a sampled step-10 update rel-L2 of
14.07788%; the variants must not be described as numerically identical.
The P1024/S1024 CP2 Merge-on run is physically aligned, so it does not
independently validate ragged padding. CP4 ragged-padding Merge validation
is a separate test already carried on `main`.

Conclusion: acceptable stage-1 numerical alignment for the validated
configurations, not bitwise equivalence, not on-policy RL convergence,
and not a universal error bound.

Historical diagnostic harness/results remain available at the branch
ancestor `53eae53bdba1ae48dbc3d03054ed2868cbeff778` and in the
original experiment outputs. The diagnostic runtime hooks and optional
FP32 delayed-dKV experiment are intentionally excluded from production.
