import subprocess

import pytest
from agentwatch_sandbox.coding_task import coding_task, run_tests, summarize


def test_project_handles_empty_and_signed_values():
    assert summarize([])["mean"] is None
    assert summarize([-3, 1])["mean"] == -1


def test_test_tool_runs_in_subprocess_with_matching_output():
    result = coding_task(1000)
    assert result["checksum"] == run_tests(1000)["checksum"]
    assert result["finished_ns"] > result["started_ns"]
    assert result["cpu_seconds"] > 0


def test_failed_tool_is_not_returned_as_success(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="test failed")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        coding_task(1000)
