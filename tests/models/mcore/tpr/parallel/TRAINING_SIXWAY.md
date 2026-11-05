# Six-route comparison: reuse four completed jobs, add only two

The six routes are Base CP1, Base CP2, TPR CP1, TPR CP2, **Base CP1 repeat**
and **TPR CP2 QKV merge**. Pull this branch on the host; run inside the same
training image with the same checkpoint and physical NPU IDs:

```bash
export TPR_TRAIN_MODEL_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_TRAIN_DEVICES=0,1
export TPR_TRAIN10_SOURCE=/workspace/tpr_experiments/p1024s1024_train10_1
export TPR_TRAIN10_OUTPUT=/workspace/tpr_experiments/p1024s1024_train10_sixway_1
bash tests/models/mcore/tpr/parallel/run_training_multistep_sixway.sh
cat "$TPR_TRAIN10_OUTPUT/summary.md"
```

The original four jobs are **not rerun or modified**. The output must be new,
outside the original experiment. The script runs two independent jobs in
sequence: a one-NPU Base CP1 repeat on the first selected device, and a two-NPU
TPR CP2 QKV merge run. Each starts from the original HF weights and fresh
native optimizer state and performs ten real native updates.

Both reuse the historical `actor_fixture.pt`, preserving all ten minibatches,
old logprobs, advantages and the held-out probe. Seed, LR, bridge, fused wgrad
and precision settings are restored from the historical result. Exact initial
weights, package versions and native precision/topology must match. Base CP1
replays its eleven original preparation forwards without replacing labels;
the TPR CP2 supplement retains its original preparation sequence.

The original routes and Base repeat keep merge off. Only `tpr_cp2_qkvmerge`
sets **TPR_RING_COALESCE_PREFIX_FULL/QUERY/PADDED=1/1/1**. This is the retained
Ring Prefix Query/KV coalescing optimization, not a replacement linear or
attention implementation. The first update observes the existing metadata
trace and must see joint Query/KV merge in **forward and backward on both CP
ranks**. Missing execution fails the experiment even if flags were requested.
The trace hook is removed afterward; updates 2–10 run without that hook or
the Python call profiler. Audits retain the actual call counts and Q/KV shapes.

P1024/S1024 is aligned to CP2. The padded-merge capability is enabled, but
this input does **not** validate ragged/padded sequence boundaries.

## Read the results

The original four-route experiment normally deletes its full update vectors.
The six-route table therefore uses the **same saved master-update coordinates
from every named parameter** for all six routes. These are sampled metrics,
not all-element metrics. The original four all-parameter measurements remain
in the original experiment's `comparison.json`; do not mix the two scopes.

The short combined `summary.md` shows:

* Six final sampled-update comparisons against the original Base CP1, and
  ratios to native Base CP1/CP2 using the same sampled coordinates.
* Base CP1 versus its independent repeat at steps 1/5/10, measuring its own
  run-to-run variation.
* TPR CP2 merge OFF versus ON at steps 1/5/10, isolating the merge effect.

All routes evaluate through native whole-sequence inference at their own CP
size. The merge OFF/ON pair therefore uses the same CP2 evaluator; cross-CP
probe gaps still include evaluation-layout differences.

Supplement reports, step metric gaps and native execution audits are under
`base_cp1_repeat_run/` and `qkvmerge_run/`. No new full update blocks are saved.
If historical Base CP1 full blocks happen to remain, its repeat also compares
the complete update; the six-route table still consistently uses samples.

To recover after only the first supplemental job completed, keep that output
and run the second job by itself:

```bash
TPR_TRAIN10_REPEAT_SOURCE="$TPR_TRAIN10_SOURCE" \
TPR_TRAIN10_OUTPUT="$TPR_TRAIN10_OUTPUT/qkvmerge_run" \
bash tests/models/mcore/tpr/parallel/run_training_cp2_qkvmerge.sh
TPR_TRAIN10_REPEAT_SOURCE="$TPR_TRAIN10_SOURCE" \
python -m tests.models.mcore.tpr.parallel.training_multistep_sixway
```

An incomplete job directory is preserved and cannot be overwritten. If the
second job started and failed, preserve/rename its directory before retrying.
