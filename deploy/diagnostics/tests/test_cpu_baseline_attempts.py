"""Public-behavior checks for cpu-baseline-attempts.sh argument validation.

The script is copied into a temporary repository layout so it can only write evidence there,
and `seq`/`uv` are replaced by recorders that schedule and run nothing. No experiment runs.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "cpu-baseline-attempts.sh"
BASH = shutil.which("bash")
TIMEOUT_SECONDS = 30

pytestmark = pytest.mark.skipif(BASH is None, reason="bash is required to run the script")


def run_script(tmp_path: Path, *arguments: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    repo = tmp_path / "repo"
    script = repo / "deploy" / "diagnostics" / SCRIPT.name
    script.parent.mkdir(parents=True)
    shutil.copy(SCRIPT, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    for tool in ("seq", "uv"):
        recorder = bin_dir / tool
        recorder.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{calls}"\n')
        recorder.chmod(0o755)
    environment = os.environ | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    completed = subprocess.run(
        [BASH, str(script), *arguments],
        capture_output=True,
        text=True,
        env=environment,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    return completed, repo, calls


@pytest.mark.parametrize(
    "attempts",
    [
        "999999999999999999999999999999",
        "51",
        "100",
        "0",
        "00",
        "01",
        "050",
        "-1",
        "+5",
        "1.5",
        "abc",
        " 5",
        "5 ",
    ],
)
def test_invalid_attempt_counts_exit_before_scheduling_or_evidence(tmp_path, attempts):
    completed, repo, calls = run_script(tmp_path, attempts)
    assert completed.returncode == 2
    assert "ATTEMPTS must be a whole number from 1 to 50" in completed.stderr
    assert not (repo / ".data").exists()
    assert not calls.exists()


@pytest.mark.parametrize(("argument", "scheduled"), [("1", "1"), ("50", "50"), ("", "5")])
def test_valid_attempt_counts_are_scheduled_unchanged(tmp_path, argument, scheduled):
    completed, repo, calls = run_script(tmp_path, argument)
    assert completed.returncode == 0, completed.stderr
    assert calls.read_text().splitlines() == [f"seq 1 {scheduled}"]
    assert f"passed 0 of {scheduled} declared attempts" in completed.stdout
    assert len(list((repo / ".data").glob("a01-cpu-baseline-*"))) == 1
