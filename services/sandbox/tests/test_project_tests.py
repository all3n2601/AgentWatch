import subprocess
import threading
import time

import pytest
from agentwatch_sandbox.project_tests import run_project_tests


def write_project(path, test_body):
    path.mkdir()
    (path / "test_fixture.py").write_text(
        "import unittest\n\n\nclass FixtureTests(unittest.TestCase):\n" + test_body
    )
    return path


def test_passing_suite_reports_counts_and_cpu(tmp_path):
    project = write_project(
        tmp_path / "project",
        "    def test_one(self):\n        self.assertEqual(sum(range(10_000)), 49_995_000)\n"
        "    def test_two(self):\n        self.assertTrue(True)\n",
    )
    result = run_project_tests(project, timeout_seconds=30)
    assert result.outcome == "passed"
    assert result.returncode == 0
    assert result.tests_run == 2
    assert result.cpu_seconds > 0
    assert result.wall_seconds > 0
    assert not list(project.rglob("__pycache__")), "fixture copies must stay byte-clean"


def test_failing_suite_is_a_failed_outcome_with_redacted_output(tmp_path):
    project = write_project(
        tmp_path / "project", "    def test_fails(self):\n        self.assertEqual(1, 2)\n"
    )
    result = run_project_tests(project, timeout_seconds=30)
    assert result.outcome == "failed"
    assert result.returncode == 1
    assert result.tests_run == 1
    assert "AssertionError" in result.output_tail
    assert str(project) not in result.output_tail
    assert "<project>" in result.output_tail


def test_environment_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTWATCH_SECRET_FOR_TEST", "do-not-leak")
    project = write_project(
        tmp_path / "project",
        "    def test_env(self):\n        import os\n"
        "        self.assertNotIn('AGENTWATCH_SECRET_FOR_TEST', os.environ)\n",
    )
    assert run_project_tests(project, timeout_seconds=30).outcome == "passed"


def process_gone(pid: int, timeout: float = 5) -> bool:
    """Wait until `pid` has exited (or is an unreaped zombie), checking its state, not a timer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return True
    return False


def test_timeout_kills_the_whole_process_group(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        f"import os, pathlib, time\npathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    project = write_project(
        tmp_path / "project",
        "    def test_hangs(self):\n"
        "        import pathlib, subprocess, sys, time\n"
        f"        subprocess.Popen([sys.executable, {str(grandchild)!r}])\n"
        f"        while not pathlib.Path({str(pid_file)!r}).exists():\n"
        "            time.sleep(0.01)\n"
        "        time.sleep(60)\n",
    )
    started = time.monotonic()
    result = run_project_tests(project, timeout_seconds=1)
    assert result.outcome == "timed_out"
    assert result.returncode is None
    assert time.monotonic() - started < 10
    assert process_gone(int(pid_file.read_text())), "a grandchild survived process-group cleanup"


def test_cancellation_stops_a_running_suite(tmp_path):
    project = write_project(
        tmp_path / "project", "    def test_hangs(self):\n        import time; time.sleep(60)\n"
    )
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    result = run_project_tests(project, timeout_seconds=30, cancel=cancel)
    assert result.outcome == "cancelled"
    assert result.returncode is None
    assert time.monotonic() - started < 10


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeouts_are_rejected(tmp_path, timeout):
    with pytest.raises(ValueError):
        run_project_tests(tmp_path, timeout_seconds=timeout)


def test_missing_project_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_project_tests(tmp_path / "absent", timeout_seconds=1)
