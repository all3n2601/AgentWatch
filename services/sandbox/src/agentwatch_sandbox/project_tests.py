"""Run a trusted fixture project's unittest suite in a separate, cancellable process.

The project and every patch applied to it come from reviewed task fixtures; model output only
selects among them. This is a process boundary with a timeout, a scrubbed environment and
process-group cleanup. It is not an isolation sandbox: the tests run with the caller's
filesystem, network and privileges, so it must never run model-written code.
"""

import math
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

MAX_OUTPUT_CHARS = 4000
DEFAULT_POLL_SECONDS = 0.05
_TESTS_RAN = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)

# `resource_limit` is produced only by the isolated (container) runner.
Outcome = Literal["passed", "failed", "timed_out", "cancelled", "resource_limit"]


@dataclass(frozen=True)
class ProjectTestResult:
    outcome: Outcome
    returncode: int | None
    tests_run: int | None
    output_tail: str
    # Host-measured elapsed time; for a container it includes container start-up.
    wall_seconds: float
    # CPU time of the test process and the descendants it waited for. None when unmeasured,
    # e.g. a container killed on timeout before its in-sandbox report: never a fabricated zero.
    cpu_seconds: float | None
    isolation: Literal["process", "container"] = "process"
    # Which limit ended the suite, for `resource_limit` outcomes ("memory" or "output").
    limit_exceeded: str | None = None
    # The suite's own wall time measured inside the container, excluding start-up.
    in_sandbox_wall_seconds: float | None = None
    # Resources released and verified gone after the run (container runner only).
    cleanup: dict[str, bool] = field(default_factory=dict)


def tests_run_from_output(text: str) -> int | None:
    """The `Ran N tests` count unittest prints, or None when it printed none."""
    counts = _TESTS_RAN.findall(text)
    return int(counts[-1]) if counts else None


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _kill_process_group(process: subprocess.Popen) -> None:
    # start_new_session makes the child its own process-group leader, so this also reaches
    # anything the tests spawned. A group that already exited is not an error.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _redact(text: str, project: Path) -> str:
    for form in {str(project), str(project.resolve())}:
        text = text.replace(form, "<project>")
    return text


def run_project_tests(
    project: Path,
    *,
    timeout_seconds: float,
    cancel: threading.Event | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> ProjectTestResult:
    """Discover and run `test_*.py` in `project`; kill the process group on timeout/cancel.

    Test failures are a normal `failed` outcome. Only an inability to start the interpreter
    raises, because that is a runner fault rather than evidence about the project.
    """
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite")
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("timeout_seconds and poll_seconds must be positive")
    if not project.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {project}")
    # -I ignores user site-packages and PYTHON* variables; -B keeps fixture copies clean.
    command = [sys.executable, "-I", "-B", "-m", "unittest", "discover"]
    command += ["-s", ".", "-t", ".", "-p", "test_*.py"]
    environment = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C.UTF-8"}
    outcome: Outcome | None = None
    with tempfile.TemporaryFile() as output:
        cpu_before = _children_cpu_seconds()
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=project,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = started + timeout_seconds
            while True:
                try:
                    # Never wait past the deadline: a suite that exits after it has timed out,
                    # however close to the deadline it finishes.
                    process.wait(timeout=max(min(poll_seconds, deadline - time.monotonic()), 0))
                    break
                except subprocess.TimeoutExpired:
                    if cancel is not None and cancel.is_set():
                        outcome = "cancelled"
                    elif time.monotonic() >= deadline:
                        outcome = "timed_out"
                    else:
                        continue
                    _kill_process_group(process)
                    process.wait()
                    break
        finally:
            # Also reaps background processes a completed test run left behind.
            _kill_process_group(process)
            if process.returncode is None:
                process.wait()
        wall_seconds = time.monotonic() - started
        cpu_seconds = max(_children_cpu_seconds() - cpu_before, 0.0)
        output.seek(0)
        text = output.read().decode("utf-8", errors="replace")
    if outcome is None:
        outcome = "passed" if process.returncode == 0 else "failed"
    return ProjectTestResult(
        outcome=outcome,
        returncode=process.returncode if outcome in {"passed", "failed"} else None,
        tests_run=tests_run_from_output(text),
        output_tail=_redact(text, project)[-MAX_OUTPUT_CHARS:],
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
    )
