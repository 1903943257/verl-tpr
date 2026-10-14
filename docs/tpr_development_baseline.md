# TPR development baseline

This branch starts from the VERL commit used by the Uni-Agent-CC server:
`3efe38c759c14622fd1b2c9e3679f2d02f86bdac` (2026-09-18).

The TPR model code and tests were copied from Bridge commit
`c839c12eaefdc0a6795980931c3a0241caea4dd5` (2026-10-14). The three
VERL integration edits follow Bridge's `apply_uniagent_cc_tpr.py` preflight:

- `verl/workers/config/engine.py`: opt-in TPR engine settings.
- `verl/workers/engine/megatron/transformer_impl.py`: TPR model spec installation
  and training dispatch after global token metadata, before microbatch packing.
- `verl/workers/engine_workers.py`: opt-in actor minibatch capture.

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
