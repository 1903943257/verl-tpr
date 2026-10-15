# TPR development baseline

This branch starts from the VERL commit used by the Uni-Agent-CC server:
`3efe38c759c14622fd1b2c9e3679f2d02f86bdac` (2026-09-18).

The initial TPR model code and tests were copied from Bridge commit
`c839c12eaefdc0a6795980931c3a0241caea4dd5` (2026-10-14). The three
VERL integration edits follow Bridge's `apply_uniagent_cc_tpr.py` preflight:

- `verl/workers/config/engine.py`: opt-in TPR engine settings.
- `verl/workers/engine/megatron/transformer_impl.py`: TPR model spec installation
  and training dispatch after global token metadata, before microbatch packing.
- `verl/workers/engine_workers.py`: opt-in actor minibatch capture.

The TPR test and experiment-document changes through Bridge commit
`9079167a274ae43abbf4478cc615160a276517f7` (2026-10-15) were synchronized
after the initial import. This update did not change TPR runtime code or the
VERL integration hooks.

The 10K TP/CP/DP parity probes and their explicitly gated offline DP2 adapter
change were synchronized through Bridge commit
`d1ecfa17152185e73143a1e67ff058c44ccb9998` (2026-10-15). This does not
enable production VERL DP2 training.

The three legacy installers in `patches/` are retained for imported regression
tests and provenance. Do not apply them to this already-integrated branch.

The server checkout has additional uncommitted changes, including
`verl/utils/device.py`, and Uni-Agent-side framework/gateway changes. Their
exact diffs were not available when this baseline was created. This branch
therefore records the reproducible Bridge integration, not a byte-for-byte
snapshot of every server-local edit. Review those diffs separately before
retiring the old server checkout.

Develop TPR runtime code and tests in this VERL branch. Once a writable VERL
remote is configured, publish this branch from a machine that can commit and
push; the server can fetch it. Keep the original Uni-Agent VERL submodule and
server worktree intact until their uncommitted changes have been inventoried
and the new branch has passed the required NPU integration checks.
