# Token-Level Prefix Sharing

Last updated: 11/02/2026

## Goal

Make physical Prefix Tree construction independent of model-specific message serialization. The builder consumes only the exact input_ids supplied to one Actor PPO minibatch. It does not load a tokenizer, inspect a Chat Template, recognize Role headers, or depend on Qwen special-token IDs.

Logical GRPO UIDs remain attached to the resulting physical tree as TreeExecutionPlan.member_uids. UID is not a physical sharing boundary in this mode.

## Scope and isolation

- One call to the Actor minibatch builder is one sharing domain.
- Rows from different UIDs in that minibatch may share an exact Token prefix.
- The call receives only the local DP replica's minibatch; no cross-DP sharing is possible.
- A later minibatch is processed by a separate call; no cross-minibatch sharing is possible.
- Each PPO ObjectiveRef retains its original sample_row, response offset, and target token. Reward, Advantage, and UID ownership are not merged.

## Exact token-prefix clustering

1. Sort rows lexicographically by their exact int64 Token-ID byte sequence.
2. Compute the exact adjacent longest-common-prefix length in Token IDs.
3. Split sorted rows whenever adjacent LCP is shorter than min_shared_prefix_tokens (default 1024).
4. Within each cluster, use all adjacent LCP depths and prompt boundaries as batch-global cut points. This keeps each row represented by a small number of exact byte-signature units, without allocating one Python Trie node per Token.
5. Build the compressed Trie or LCP-sorted topology from those aligned units.
6. Keep one ObjectiveRef per original supervised response token.

A cluster therefore shares a prefix only when every member has an exact common prefix at least as long as the configured threshold. Rows below the threshold are kept in separate physical trees.

## Configuration

Production defaults:

```bash
export TPR_TREE_POLICY=token_lcp
export TPR_MIN_SHARED_PREFIX_TOKENS=1024
export TPR_TREE_BUILDER=lcp_sort
```

The Actor adapter now selects Token-LCP and the LCP-sorted builder by default.
The legacy `qwen_turn` policy and diagnostic `role_probe` remain available
only as explicit temporary compatibility/rollback paths; they are not the
production construction strategy.

## Correctness requirements

Token-Level boundaries may occur inside a serialized message. This is intentional: causal Transformer computation is token-sequence based, not message-boundary based. The executor must nevertheless preserve exact token order, position IDs, attention/KV state, query ownership, gradient accumulation, and independent PPO ObjectiveRefs.

CPU tests cover cross-UID sharing in one minibatch; no sharing below the minimum prefix threshold; exact-threshold and duplicate-row behavior; arbitrary divergence positions inside the prompt without Qwen marker IDs; Trie/LCP plan agreement; direct Session Tree consumption and equivalence to the flattened-row oracle.

CPU topology tests do not replace Full-vs-TPR forward/backward parity or NPU validation. Token-LCP is now the selected production construction policy, but this change does not by itself certify full training correctness or performance; retain the existing NPU correctness and parity gates before declaring the rollout complete.

## Phase 2: explicit message metadata

The follow-up is to have UniAgent persist original message metadata alongside the serialized token stream, including role and exact token spans (and optionally channel/recipient). That metadata will support semantic auditing and message-aware scheduling. It is not needed for exact Token-Level prefix sharing.