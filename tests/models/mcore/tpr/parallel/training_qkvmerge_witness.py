# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Read-only proof that TPR Ring executes joint Query/KV merge."""

import importlib
from collections import Counter


class QKVMergeWitness:
    def __init__(self):
        self.counts = Counter()
        self.examples = {}

    def __enter__(self):
        self.module = importlib.import_module("verl.workers.engine.megatron.tpr.ring_attention")
        self.previous = self.module._trace_ring_block
        if self.previous is not None:
            raise RuntimeError("QKV merge audit requires an unused metadata trace hook")
        self.module._trace_ring_block = self.observe
        return self

    def observe(self, **event):
        if event.get("query_coalesced") and event.get("coalesced") and event.get("fa_called"):
            phase = event["phase"]
            self.counts[phase] += 1
            if phase not in self.examples:
                self.examples[phase] = {
                    key: event[key] for key in ("rank", "query_length", "kv_length", "query_chunk", "kv_chunks")
                }

    def validate(self):
        if not self.counts["forward"] or not self.counts["backward"]:
            raise AssertionError("Requested QKV merge did not execute in both Ring forward and backward")

    def snapshot(self):
        return dict(joint_query_kv_calls=dict(self.counts), examples=self.examples)

    def __exit__(self, *_exc):
        self.module._trace_ring_block = self.previous
