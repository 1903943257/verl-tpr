# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Historical repeat comparisons survive full-reference cleanup without changing inputs."""

import pytest
import torch

from tests.models.mcore.tpr.parallel import training_multistep_repeat as repeat
from tests.models.mcore.tpr.parallel import training_multistep_runtime as runtime
from tests.models.mcore.tpr.parallel.training_multistep_io import InitialWeights, owned_master_ranges
from tests.models.mcore.tpr.parallel.training_multistep_npu import controls_from_environment
from tests.models.mcore.tpr.parallel.training_precision_io import write_json
from tests.models.mcore.tpr.unit.test_training_multistep_cpu import Model, Owner, advance, write_route


def source_run(root, *, keep_full):
    root.mkdir()
    controls = controls_from_environment()
    item = runtime.input_item(151936, controls["seed"], 0)
    item["old_log_probs"] = torch.full((2, runtime.SUFFIX), -7.0)
    fixture = dict(controls=controls, initial_weights_sha256="weights", batches=[item] * runtime.STEPS, probe=item)
    fixture["sha256"] = runtime.fixture_digest(fixture)
    torch.save(fixture, str(root / "actor_fixture.pt"))
    write_route(root, "base_cp1", 1, controls)
    folder = root / "base_cp1"
    result = repeat.read(folder / "result.json")
    result["input_sha256"] = fixture["sha256"]
    for step in result["steps"]:
        step["native_metrics"] = {"actor/pg_clipfrac": 0.1}
    write_json(folder / "result.json", result)
    write_json(
        folder / "runtime_audit_rank0.json",
        dict(status="ten_updates_validated", optimizer=dict(completed_updates=10), precision={}),
    )
    write_json(
        root / "comparison.json",
        dict(
            controls=controls,
            model_elements=16,
            pairs={
                "base_cp1_vs_base_cp2": {
                    "trend": [dict(step=step, sampled_update_relative_l2=0.2) for step in runtime.CHECKPOINTS[1:]]
                }
            },
        ),
    )
    if not keep_full:
        index = repeat.read(folder / "updates/index.json")
        for blocks in index["parts"]:
            for block in blocks:
                (folder / "updates" / block["file"]).unlink()
    return fixture, result


def repeated_run(root, context, factor):
    mode = context["mode"]
    cp, _ = runtime.mode_topology(mode)
    folder = root / mode
    folder.mkdir(parents=True)
    for rank in range(cp):
        model = Model()
        spans = [(0, 11), (0, 5)] if cp == 1 else ([(0, 6), None] if rank == 0 else [(6, 11), (0, 5)])
        owner = Owner(model, spans)
        initial = InitialWeights(model, owner)
        samples = {}
        for step in runtime.CHECKPOINTS[1:]:
            for _index, name, lo, hi, master in owned_master_ranges(model, owner):
                master.copy_(initial.values[name][lo:hi].float())
            advance(model, owner, factor * step)
            samples[step] = initial.samples(model, owner)
        torch.save(samples, str(folder / f"samples_rank{rank}.pt"))
        report = repeat.read(context["source"] / context["reference_mode"] / f"rank{rank}.json")
        report.update(mode=mode, cp=cp, input_sha256=context["baseline"]["input_sha256"])
        report["layout"] = initial.layout
        for step in report["steps"]:
            step["native_metrics"] = {"actor/pg_clipfrac": 0.1}
        report["full_update"] = (
            initial.full_update(
                model,
                owner,
                folder / "updates",
                rank,
                {"base_cp1": context["source"] / "base_cp1/updates"},
                save_reference=False,
            )
            if context["full_update_available"]
            else None
        )
        write_json(folder / f"rank{rank}.json", report)
        if mode == runtime.TPR_CP2_QKVMERGE:
            write_json(
                folder / f"runtime_audit_rank{rank}.json",
                dict(
                    status="ten_updates_validated",
                    optimizer=dict(completed_updates=10),
                    qkv_merge_execution={"joint_query_kv_calls": {"forward": 4, "backward": 4}},
                    ring_merge_settings={
                        f"TPR_RING_COALESCE_PREFIX_{name}": "1" for name in ("FULL", "QUERY", "PADDED")
                    },
                ),
            )
    probes = {
        step: value + (factor - 1) * step * 0.01
        for step, value in repeat.load(context["source"] / context["reference_mode"] / "probes.pt").items()
    }
    torch.save(probes, str(folder / "probes.pt"))


@pytest.mark.parametrize("keep_full", [False, True])
@pytest.mark.parametrize("factor", [1, 1.03])
def test_repeat_measures_unchanged_or_perturbed_state_without_fabricating_full_metrics(tmp_path, keep_full, factor):
    source, output = tmp_path / "source", tmp_path / "repeat"
    source_run(source, keep_full=keep_full)
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    context = repeat.check_source(source, output=output, packages={})
    assert context["full_update_available"] == keep_full
    repeated_run(output, context, factor)
    result = repeat.complete_repeat(output, context)
    expected = factor - 1
    assert result["trend"][-1]["sampled_update"]["update_relative_l2"] == pytest.approx(expected, abs=0.0003)
    last = result["trend"][-1]
    assert last["repeat_to_native_cp_sampled_ratio"] == pytest.approx(
        last["sampled_update"]["update_relative_l2"] / 0.2
    )
    assert result["initial_probe"]["logprob_mean_abs"] == 0
    if keep_full:
        assert result["full_update_comparison"]["update_relative_l2"] == pytest.approx(expected, abs=0.0003)
    else:
        assert result["full_update_comparison"] is None
        assert "not every parameter element" in (output / "summary.md").read_text()
    assert not (output / runtime.BASE_CP1_REPEAT / "updates").exists()
    after = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert after == before


def test_repeat_preflight_rejects_wrong_packages_changed_fixture_and_overlapping_output(tmp_path):
    source = tmp_path / "source"
    fixture, _ = source_run(source, keep_full=False)
    with pytest.raises(AssertionError, match="package versions"):
        repeat.check_source(source, packages={"torch": "different"})
    with pytest.raises(ValueError, match="separate"):
        repeat.check_source(source, output=source / "repeat")
    fixture["probe"]["old_log_probs"][0, 0] += 0.5
    torch.save(fixture, str(source / "actor_fixture.pt"))
    with pytest.raises(AssertionError, match="unchanged actor fixture"):
        repeat.check_source(source)


def test_saved_controls_override_shell_defaults_exactly(monkeypatch):
    original = controls_from_environment()
    for variable in repeat.ENVIRONMENT_CONTROLS.values():
        monkeypatch.setenv(variable, "0")
    repeat.restore_controls(original)
    assert controls_from_environment() == original


def test_repeat_replays_original_preparation_forwards_without_replacing_labels(tmp_path, monkeypatch):
    source = tmp_path / "source"
    fixture, _ = source_run(source, keep_full=False)
    original_bytes = (source / "actor_fixture.pt").read_bytes()
    calls = []

    class Engine:
        def eval_mode(self):
            from contextlib import nullcontext

            return nullcontext()

        def forward_backward_batch(self, batch, loss, *, forward_only):
            assert forward_only and loss is None
            assert torch.count_nonzero(batch[1]) == 0
            calls.append(batch[0])
            return {"replacement_labels": torch.full((2, runtime.SUFFIX), -99.0)}

    monkeypatch.setattr(runtime, "actor_batch", lambda tokens, old, advantages: (tokens, old, advantages))
    monkeypatch.setattr(runtime, "validate_forest", lambda _: None)
    monkeypatch.setattr(runtime, "_native_response_logprobs", lambda output, _: output["replacement_labels"])
    loaded = runtime.prepare_fixture(
        Engine(),
        {**fixture["controls"], "mode": runtime.BASE_CP1_REPEAT},
        fixture["initial_weights_sha256"],
        source / "actor_fixture.pt",
        0,
        lambda value: value,
        lambda: None,
    )
    assert len(calls) == 11
    assert torch.equal(loaded["probe"]["old_log_probs"], fixture["probe"]["old_log_probs"])
    assert (source / "actor_fixture.pt").read_bytes() == original_bytes
