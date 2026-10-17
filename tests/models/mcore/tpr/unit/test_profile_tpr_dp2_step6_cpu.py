# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stdlib CLI/provenance contracts; these tests do not execute NPU replay."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "parallel/profile_tpr_dp2_step6.py"
SPEC = importlib.util.spec_from_file_location("_step6_profile_cli_contract", SCRIPT)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class ProfileCLIContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.capture = self.root / "capture.pt"
        self.capture.touch()
        self.reference = {
            "source_step": 6, "mode": "tpr", "effective_sequence_cap": 30720,
            "topology": {"tp": 2, "cp": 2, "dp": 2, "world": 8},
            "ppo_logprob_chunk_tokens": 512, "gradient_sync_reduce_dtype": "bf16",
            "updated_weights": False, "recompute": False,
            "input_sha256": "a" * 64, "dp_partitions": [[0, 2, 4], [1, 3, 5, 6, 7]],
            "global_ppo_tokens": 100, "source_capture": str(self.capture),
            "checkpoint": str(self.root), "capture_logprobs": True,
        }
        self.reference_path = self.root / "step6.json"
        self.reference_path.write_text(json.dumps(self.reference), encoding="utf-8")

    def args(self):
        return ["--repo", str(self.root), "--reference-json", str(self.reference_path),
                "--revision", "a" * 40, "--output-root", str(self.root / "result"), "--label", "main"]

    def test_cli_help_without_torch_or_npu(self):
        process = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"],
                                 capture_output=True, text=True, check=True)
        self.assertIn("--reference-json", process.stdout)
        self.assertNotIn("Traceback", process.stderr)

    def test_defaults_and_explicit_timing_vs_profile(self):
        ordinary = driver.parse_args(self.args())
        traced = driver.parse_args(self.args() + ["--profile"])
        self.assertEqual((ordinary.warmups, ordinary.repeats, ordinary.profile), (3, 10, False))
        self.assertEqual((traced.warmups, traced.repeats, traced.profile), (3, 1, True))
        self.assertEqual(driver.parse_args(self.args() + ["--repeats", "2"]).repeats, 2)

    def test_full_revision_required(self):
        arguments = self.args()
        arguments[arguments.index("--revision") + 1] = "2dc82406"
        with self.assertRaises(SystemExit):
            driver.parse_args(arguments)

    def test_accepted_json_needs_no_40k_manifest(self):
        actual = driver.load_reference(self.reference_path)
        self.assertEqual(actual["source_step"], 6)
        self.assertEqual(actual["effective_sequence_cap"], 30720)

    def test_reference_rejects_changed_acceptance_invariants(self):
        for key, changed in (("source_step", 5), ("gradient_sync_reduce_dtype", "fp32"),
                             ("effective_sequence_cap", 40960), ("updated_weights", True),
                             ("dp_partitions", [[0], [0]]), ("input_sha256", "invalid")):
            with self.subTest(key=key):
                data = dict(self.reference, **{key: changed})
                self.reference_path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    driver.load_reference(self.reference_path)

    def test_controls_reject_conflicts_without_overwriting(self):
        for key, value in (("TPR_RING_COALESCE_PREFIX_FULL", "1"),
                           ("TPR_OFFLINE_GRAD_SYNC_DTYPE", "fp32"),
                           ("TPR_EXEC_PROFILE", "1"), ("TPR_FIXED_TILE_TOKENS", "512"),
                           ("TPR_MEM_TRACE", "1"), ("TPR_BENCH_CAPTURE_LOGPROBS", "1")):
            with self.subTest(key=key):
                environment = {key: value}
                with self.assertRaises(ValueError):
                    driver.configure_environment(self.reference, environment)
                self.assertEqual(environment, {key: value})
        environment = {}
        actual = driver.configure_environment(self.reference, environment)
        self.assertEqual(actual["TPR_EXEC_PROFILE"], "0")
        self.assertEqual(actual["TPR_BENCH_CAPTURE_LOGPROBS"], "0")
        self.assertEqual(actual["TPR_PPO_LOGPROB_CHUNK_TOKENS"], "512")
        self.assertEqual(environment, actual)

    def test_git_unavailable_is_explicitly_unverified(self):
        with patch.object(driver.subprocess, "run", side_effect=FileNotFoundError("git")):
            report = driver.revision_metadata(self.root, "a" * 40)
        self.assertEqual(report["verification"], "host_declared_not_verified")

    def test_git_rejects_wrong_head_and_dirty_production(self):
        for responses in (("b" * 40,), ("a" * 40, str(self.root), " M verl/file.py")):
            with self.subTest(responses=responses), patch.object(
                    driver.subprocess, "run", side_effect=[SimpleNamespace(stdout=s) for s in responses]):
                with self.assertRaises(ValueError):
                    driver.revision_metadata(self.root, "a" * 40)

    def test_module_provenance_rejects_wrong_checkout(self):
        module = SimpleNamespace(__name__="verl.fake", __file__=str(self.capture))
        self.assertIn("verl.fake", driver.verify_module_paths(self.root, [module]))
        with self.assertRaises(RuntimeError):
            driver.verify_module_paths(self.root / "other", [module])

    def test_percentile_is_defined_for_one_repeat(self):
        self.assertEqual(driver.percentile90([3]), 3)
        self.assertAlmostEqual(driver.percentile90([1, 2, 3]), 2.8)

    def test_runtime_scrubs_and_restores_argv_even_on_failure(self):
        with patch.object(driver.sys, "argv", ["driver", "--profile", "--repo", "candidate"]):
            original = driver.sys.argv[:]
            def factory():
                self.assertEqual(driver.sys.argv, ["driver"])
                raise RuntimeError("bootstrap failed")
            with self.assertRaisesRegex(RuntimeError, "bootstrap failed"):
                driver.runtime_without_cli(factory)
            self.assertEqual(driver.sys.argv, original)

    def test_existing_rank_output_fails_before_heavy_imports(self):
        (self.root / "result/rank0").mkdir(parents=True)
        with patch.dict(driver.os.environ, {"RANK": "0", "WORLD_SIZE": "8"}, clear=True), patch.object(
                driver, "revision_metadata", return_value={}), patch.object(
                driver.importlib, "import_module", side_effect=AssertionError("NPU import reached")):
            with self.assertRaises(FileExistsError):
                driver.main(self.args())

    def test_environment_records_controls_without_leaking_secrets(self):
        metadata = driver.redacted_environment({
            "TPR_PPO_LOGPROB_CHUNK_TOKENS": "512", "HCCL_DETERMINISTIC": "true",
            "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
            "TPR_HF_TOKEN": "sensitive", "HCCL_PASSWORD": "sensitive",
            "ASCEND_API_KEY": "sensitive", "HCCL_AUTHENTICATION_KEY": "sensitive",
            "TPR_CUSTOM_KEY": "sensitive",
            "UNRELATED": "not-included",
        })
        self.assertEqual(metadata["TPR_PPO_LOGPROB_CHUNK_TOKENS"], "512")
        self.assertEqual(metadata["HCCL_DETERMINISTIC"], "true")
        self.assertNotIn("sensitive", metadata.values())
        self.assertNotIn("UNRELATED", metadata)

    def test_visible_device_mapping_preserves_explicit_eight_cards(self):
        for visible in ("8,9,10,11,12,13,14,15", "0,1,2,3,4,5,6,7"):
            environment = {"ASCEND_RT_VISIBLE_DEVICES": visible}
            controls = driver.configure_environment(self.reference, environment)
            self.assertEqual(controls["ASCEND_RT_VISIBLE_DEVICES"], visible)
        for visible in ("0,1", "0,1,2,3,4,5,6,6", "-1,1,2,3,4,5,6,7", "0,1,2,3,4,5,6,x"):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                driver.configure_environment(self.reference, {"ASCEND_RT_VISIBLE_DEVICES": visible})

    def test_ppo_snapshot_includes_class_defaults_and_is_independent(self):
        class PPO:
            policy_loss = {"loss_mode": "vanilla"}
            entropy_coeff = 0.0
            _private = "omit"

            def __init__(self):
                self.global_batch_info = {}

            def get(self):
                return None

        original = PPO()
        snapshot = driver.ppo_configuration(original)
        self.assertEqual(snapshot, {"policy_loss": {"loss_mode": "vanilla"},
                                    "entropy_coeff": 0.0, "global_batch_info": {}})
        original.global_batch_info["dp_size"] = 2
        original.policy_loss["loss_mode"] = "changed"
        self.assertEqual(snapshot["global_batch_info"], {})
        self.assertEqual(snapshot["policy_loss"], {"loss_mode": "vanilla"})

    def test_group_metadata_only_queries_existing_membership(self):
        runtime = SimpleNamespace(tp_group="tp", cp_group="cp", dp_group="dp",
                                  dp_cp_group="dp_cp", pp_group="pp")
        groups = {"tp": [0, 1], "cp": [0, 2], "dp": [0, 4], "dp_cp": [0, 2, 4, 6], "pp": [0]}
        distributed = SimpleNamespace(get_process_group_ranks=lambda group: groups[group])
        actual = driver.group_membership(runtime, distributed)
        self.assertEqual(actual, groups)
        self.assertIsNot(actual["tp"], groups["tp"])

    def test_capture_sha_hashes_all_file_bytes(self):
        payload = b"capture including temperature and metadata" * 40000
        self.capture.write_bytes(payload)
        self.assertEqual(driver.sha256(self.capture), hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
