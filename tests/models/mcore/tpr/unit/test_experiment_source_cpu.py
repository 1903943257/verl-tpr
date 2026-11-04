# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unavailable host worktree metadata must not fail container experiments."""

import subprocess

import pytest

from tests.models.mcore.tpr.parallel.experiment_source import source_metadata


def test_host_revision_override_does_not_invoke_git(monkeypatch):
    monkeypatch.setenv("TPR_SOURCE_HEAD", "  d497032b  ")

    def unexpected(*args, **kwargs):
        raise AssertionError("Do not invoke Git when the host supplies a revision")

    monkeypatch.setattr(subprocess, "run", unexpected)
    metadata = source_metadata()
    assert metadata["source_head"] == "d497032b"
    assert metadata["source_head_origin"] == "environment"
    assert metadata["source_head_error"] is None


def test_git_revision_is_read_from_source_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("TPR_SOURCE_HEAD", raising=False)
    revision = "a" * 40

    def git(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == 3 and kwargs["capture_output"]
        return subprocess.CompletedProcess(command, 0, stdout=revision + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", git)
    metadata = source_metadata(tmp_path)
    assert metadata["source_head"] == revision
    assert metadata["source_head_origin"] == "git"


@pytest.mark.parametrize("failure", ["broken_worktree", "missing_git", "timeout", "empty_head"])
def test_unavailable_source_is_metadata_not_an_exception(tmp_path, monkeypatch, capsys, failure):
    monkeypatch.delenv("TPR_SOURCE_HEAD", raising=False)

    def unavailable(command, **kwargs):
        if failure == "missing_git":
            raise FileNotFoundError("git")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(
            command,
            128 if failure == "broken_worktree" else 0,
            stdout="",
            stderr="fatal: not a git repository: /host/.git/worktrees/source" if failure == "broken_worktree" else "",
        )

    monkeypatch.setattr(subprocess, "run", unavailable)
    metadata = source_metadata(tmp_path)
    assert metadata["source_head"] is None
    assert metadata["source_head_origin"] == "unavailable"
    assert metadata["source_head_error"]
    assert not capsys.readouterr().err
