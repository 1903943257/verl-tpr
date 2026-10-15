"""Dependency-free tests for current Phase-B calibrated gradient-noise gate."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "profiling/_offload_b_gate.py"
spec = importlib.util.spec_from_file_location("offload_b_gate", PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def metrics(l2=0.006, *, finite=True, mismatched=10, elements=8192):
    return dict(relative_l2=l2, finite=finite,
                mismatched=mismatched, elements=elements)


class GateTest(unittest.TestCase):
    def test_supplied_repeat_noise_is_accepted(self):
        for baseline, on, repeated in ((0.0060, 0.0065, 0.0059),
                                      (0.0123, 0.0117, 0.0119),
                                      (0.0108, 0.0116, 0.0105)):
            for value in (on, repeated):
                self.assertTrue(gate.gradient_gate(metrics(value), metrics(baseline))[0])

    def test_relative_l2_can_fail_calibrated_noise_budget(self):
        baseline = metrics(0.006)
        passed, limits, severity = gate.gradient_gate(metrics(0.018), baseline)
        self.assertFalse(passed)
        self.assertGreater(severity, 1.0)
        self.assertEqual(limits["relative_l2"], 0.009)
        self.assertFalse(gate.gradient_gate(metrics(0.004), baseline)[1].get("max_abs"))

    def test_zero_noise_does_not_produce_zero_budget(self):
        baseline = metrics(0)
        self.assertTrue(gate.gradient_gate(metrics(0.002), baseline)[0])
        self.assertFalse(gate.gradient_gate(metrics(0.003), baseline)[0])
        self.assertTrue(gate.gradient_gate(metrics(0.015, elements=1024), baseline)[0])

    def test_nonfinite_reference_or_actual_fails(self):
        for invalid in (metrics(float("nan")), metrics(float("inf")), metrics(finite=False)):
            self.assertFalse(gate.gradient_gate(invalid, None)[0])
            self.assertFalse(gate.gradient_gate(invalid, metrics())[0])
            self.assertFalse(gate.gradient_gate(metrics(), invalid)[0])

    def test_calibration_requires_baseline_except_elementwise_pass(self):
        self.assertFalse(gate.gradient_gate(metrics(0.0123), None)[0])
        self.assertTrue(gate.gradient_gate(metrics(0.0123, mismatched=0), None)[0])
        earlier = gate.merge_baseline(metrics(0.006), metrics(0.009))
        self.assertEqual(earlier["relative_l2"], 0.009)
        self.assertEqual(gate.merge_baseline(earlier, metrics(0.005)), earlier)


if __name__ == "__main__":
    unittest.main()
