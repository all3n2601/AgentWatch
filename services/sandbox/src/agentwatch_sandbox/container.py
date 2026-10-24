"""Run a project's unittest suite in a hardened, disposable Docker container.

`run_project_tests_isolated` is a drop-in for `project_tests.run_project_tests` that returns the
same `ProjectTestResult`. The isolation design, its boundaries and their verification are in
docs/adr/0004-container-isolation-for-tool-execution.md. In short:

- filesystem: a private, read-only staging copy at /workspace, a read-only root, a small noexec
  /tmp and no /dev/shm;
- network: none; privilege: uid 65534, no capabilities, no-new-privileges;
- resources: memory (no swap), CPU, process, open-file and captured-output limits;
- time: a host deadline that kills the container, plus a backstop timer in a wrapper that runs
  as PID 1 (the suite cannot stop or signal it) and is non-dumpable (the suite cannot read it);
- supply chain: a digest-pinned image, never pulled implicitly.

Every outcome releases its resources: the container and the staging copy are removed and the
removal is verified. A failed removal raises SandboxCleanupError instead of returning.
"""

import dataclasses
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from agentwatch_sandbox.project_tests import (
    DEFAULT_POLL_SECONDS,
    MAX_OUTPUT_CHARS,
    Outcome,
    ProjectTestResult,
    tests_run_from_output,
)

# python:3.13-slim, pinned by its multi-architecture index digest (linux/amd64 and arm64).
SANDBOX_IMAGE = (
    "python:3.13-slim@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962"
)
SANDBOX_LABEL = "agentwatch.sandbox"
CREATED_LABEL = "agentwatch.sandbox.created"
DEADLINE_LABEL = "agentwatch.sandbox.deadline"
WORKSPACE = "/workspace"
SANDBOX_USER = "65534:65534"
DOCKER_COMMAND_TIMEOUT_SECONDS = 30
# How long past its backstop a container may still exist before a sweep removes it: covers the
# host's kill and release steps, so a sweep never removes a container a runner still owns.
SWEEP_MARGIN_SECONDS = 4 * DOCKER_COMMAND_TIMEOUT_SECONDS
BACKSTOP_EXIT_CODE = 124
# Output is a resource too: the suite is stopped once it writes more than this. The host only
# ever stores a bounded tail of each stream, however much arrives before the stop lands.
MAX_CAPTURED_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_REPORT_BYTES = 64 * 1024
CAPTURE_READ_BYTES = 64 * 1024
_PINNED_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$")
# `--mount` is a comma-separated key=value list, so these characters would change its meaning.
_UNSAFE_MOUNT_PATH = re.compile(r"[,=\s\"]")

# Runs as PID 1 inside the container: `python -I -B -c WRAPPER <backstop seconds>`.
# - PID 1 of a PID namespace ignores signals from inside the namespace that it has no handler
#   for, so the suite cannot SIGSTOP or SIGKILL it; its exit ends every other process.
#   CPython handles SIGINT (KeyboardInterrupt), so the wrapper replaces that handler with one
#   that does nothing. Handlers reset on exec, so the suite keeps its normal SIGINT behavior.
# - It makes itself non-dumpable, so the suite (same uid) cannot read its memory or write to
#   its file descriptors through /proc.
# - It reads a random nonce from stdin and reports measurements on stdout behind it; the suite
#   runs with stdin closed and its output on stderr.
# - The backstop ends the container if the host runner is gone. A suite can send SIGALRM to
#   PID 1, so an alarm that arrives before the backstop is ignored.
WRAPPER = r"""
import ctypes, json, os, resource, signal, subprocess, sys, time
if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE, 0
    sys.stderr.write("sandbox wrapper: could not become non-dumpable\n")
    sys.exit(70)
signal.signal(signal.SIGINT, lambda signum, frame: None)
nonce = sys.stdin.readline().strip()
backstop = float(sys.argv[1])
started = time.monotonic()
def expire(signum, frame):
    if time.monotonic() - started < backstop - 0.5:
        return
    sys.stdout.write(nonce + json.dumps({"backstop_expired": True}) + "\n")
    sys.stdout.flush()
    os._exit(124)
signal.signal(signal.SIGALRM, expire)
signal.setitimer(signal.ITIMER_REAL, backstop)
child = subprocess.Popen(
    [sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", ".", "-t", ".",
     "-p", "test_*.py"],
    stdin=subprocess.DEVNULL, stdout=sys.stderr.fileno(),
)
returncode = child.wait()
signal.setitimer(signal.ITIMER_REAL, 0)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
sys.stdout.write(nonce + json.dumps({
    "returncode": returncode,
    "cpu_seconds": usage.ru_utime + usage.ru_stime,
    "wall_seconds": time.monotonic() - started,
}) + "\n")
sys.stdout.flush()
sys.exit(returncode if 0 <= returncode < 124 else 1)
"""


class SandboxError(Exception):
    """Base class for sandbox failures that are not evidence about the project's tests."""


class SandboxRequestError(SandboxError, ValueError):
    """The request was rejected; any staging copy it reached has been removed."""


class SandboxUnavailable(SandboxError):
    """Docker, its daemon or the pinned image is not usable; nothing ran."""


class SandboxExecutionError(SandboxError):
    """The sandbox ran but produced no trustworthy result (e.g. its wrapper did not report)."""


class SandboxCleanupError(SandboxError):
    """A container or staging directory could not be removed. `result` holds the run's result,
    when one exists, so the evidence is not lost."""

    def __init__(self, message: str, result: ProjectTestResult | None = None):
        super().__init__(message)
        self.result = result


# What releasing can raise (the Docker CLI, the filesystem); anything else is a bug and propagates.
_RELEASE_ERRORS = (SandboxError, OSError, subprocess.SubprocessError)


@dataclass(frozen=True)
class SandboxLimits:
    """Per-run resource limits. Bounds keep a misconfiguration from disabling a limit."""

    memory_mb: int = 256
    cpus: float = 1.0
    pids: int = 64
    tmpfs_mb: int = 16
    open_files: int = 256
    # How long after the host deadline the in-container backstop ends the container.
    backstop_grace_seconds: float = 5.0

    def __post_init__(self):
        integers = {
            "memory_mb": (self.memory_mb, 32, 4096),
            "pids": (self.pids, 8, 1024),
            "tmpfs_mb": (self.tmpfs_mb, 1, 512),
            "open_files": (self.open_files, 32, 4096),
        }
        for name, (value, low, high) in integers.items():
            if type(value) is not int or not low <= value <= high:
                raise SandboxRequestError(f"{name} must be an integer in [{low}, {high}]")
        for name, value, low, high in (
            ("cpus", self.cpus, 0.1, 8.0),
            ("backstop_grace_seconds", self.backstop_grace_seconds, 1.0, 600.0),
        ):
            if not _is_number(value) or not low <= value <= high:
                raise SandboxRequestError(f"{name} must be a number in [{low}, {high}]")


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def docker_create_command(
    *,
    docker: str,
    name: str,
    staging: Path,
    image: str,
    limits: SandboxLimits,
    backstop_seconds: float,
    created_epoch: int,
) -> list[str]:
    """The exact `docker create` invocation; every isolation control is visible here."""
    deadline_epoch = created_epoch + math.ceil(backstop_seconds) + SWEEP_MARGIN_SECONDS
    return [
        docker, "create",
        "--name", name,
        "--label", f"{SANDBOX_LABEL}=1",
        "--label", f"{CREATED_LABEL}={created_epoch}",
        "--label", f"{DEADLINE_LABEL}={deadline_epoch}",
        "--pull", "never",
        "--interactive",
        "--log-driver", "none",
        "--network", "none",
        "--ipc", "none",
        "--read-only",
        "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={limits.tmpfs_mb}m",
        "--user", SANDBOX_USER,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--memory", f"{limits.memory_mb}m",
        "--memory-swap", f"{limits.memory_mb}m",
        "--cpus", f"{limits.cpus:g}",
        "--pids-limit", str(limits.pids),
        "--ulimit", f"nofile={limits.open_files}:{limits.open_files}",
        "--ulimit", "core=0",
        "--mount", f"type=bind,source={staging},target={WORKSPACE},readonly",
        "--workdir", WORKSPACE,
        "--env", "LC_ALL=C.UTF-8",
        "--hostname", "sandbox",
        image,
        "python", "-I", "-B", "-c", WRAPPER, f"{backstop_seconds:g}",
    ]  # fmt: skip


def _docker(docker: str, *args: str) -> subprocess.CompletedProcess:
    """A bounded Docker CLI call. A missing CLI or an unresponsive daemon is unavailability."""
    try:
        return subprocess.run(
            [docker, *args],
            capture_output=True,
            text=True,
            timeout=DOCKER_COMMAND_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as error:
        raise SandboxUnavailable(f"Docker CLI not found: {docker}") from error
    except subprocess.TimeoutExpired as error:
        raise SandboxUnavailable(
            f"`docker {args[0]}` did not finish within {DOCKER_COMMAND_TIMEOUT_SECONDS} s"
        ) from error


def _reject_unsafe_tree(root: Path) -> None:
    for path in root.rglob("*"):
        # A symlink could expose a host file once copied; special files have no place in a
        # test project. Both are rejected rather than silently skipped.
        if path.is_symlink():
            raise SandboxRequestError(f"Project contains a symlink: {path.relative_to(root)}")
        if not (path.is_file() or path.is_dir()):
            raise SandboxRequestError(f"Project contains a special file: {path.relative_to(root)}")


def _validate_request(
    project: Path,
    timeout_seconds: float,
    poll_seconds: float,
    image: str,
    backstop_seconds: float | None,
) -> None:
    if not _is_number(timeout_seconds):
        raise SandboxRequestError("timeout_seconds must be a finite number")
    if timeout_seconds <= 0 or not _is_number(poll_seconds) or poll_seconds <= 0:
        raise SandboxRequestError("timeout_seconds and poll_seconds must be finite and positive")
    if backstop_seconds is not None and (not _is_number(backstop_seconds) or backstop_seconds < 1):
        raise SandboxRequestError("backstop_seconds must be a finite number of at least 1")
    if not _PINNED_IMAGE.fullmatch(image):
        raise SandboxRequestError("image must be pinned by digest (name@sha256:<64 hex>)")
    if not project.is_dir() or project.is_symlink():
        raise SandboxRequestError(f"Project directory does not exist: {project}")
    _reject_unsafe_tree(project)


def _stage(project: Path) -> Path:
    """Copy the project into a private directory; the container mounts its `workspace`.

    The copy keeps symlinks as links and is re-checked, so a file swapped for a symlink after
    validation is rejected instead of copied from the host. The parent stays 0700, so other
    local users cannot reach the copy; the container only needs the mounted directory.
    """
    staging = Path(tempfile.mkdtemp(prefix="agentwatch-sandbox-"))
    try:
        if _UNSAFE_MOUNT_PATH.search(str(staging)):
            raise SandboxUnavailable(
                f"The temporary directory path cannot be mounted safely: {staging.parent}"
            )
        workspace = staging / "workspace"
        shutil.copytree(
            project, workspace, symlinks=True, ignore=shutil.ignore_patterns("__pycache__")
        )
        _reject_unsafe_tree(workspace)
        for path in [workspace, *workspace.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
        return staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _container_exists(docker: str, name: str) -> bool:
    listed = _docker(docker, "ps", "--all", "--quiet", "--filter", f"name=^{name}$")
    if listed.returncode != 0:
        raise SandboxUnavailable(f"`docker ps` failed: {listed.stderr.strip()[:300]}")
    return bool(listed.stdout.strip())


def _inspect_state(docker: str, name: str) -> dict | None:
    inspected = _docker(docker, "inspect", "--format", "{{json .State}}", name)
    if inspected.returncode != 0:
        return None
    try:
        state = json.loads(inspected.stdout)
    except json.JSONDecodeError:
        state = None
    if not isinstance(state, dict):
        raise SandboxExecutionError(
            f"`docker inspect` returned an unreadable state: {inspected.stdout.strip()[:300]}"
        )
    return state


class _Capture:
    """Drain one output stream of the attached CLI on a thread, keeping a bounded tail.

    Everything is read, so the CLI never blocks on a full pipe, and counted, so a flood can be
    detected; at most `keep_bytes` are stored. The thread ends when the CLI closes the stream.
    """

    def __init__(self, stream: BinaryIO, keep_bytes: int):
        self._keep_bytes = keep_bytes
        self._tail = bytearray()
        self._lock = threading.Lock()
        self.total_bytes = 0
        self._thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self._thread.start()

    def _drain(self, stream: BinaryIO) -> None:
        with stream:
            while chunk := os.read(stream.fileno(), CAPTURE_READ_BYTES):
                with self._lock:
                    self.total_bytes += len(chunk)
                    self._tail += chunk
                    del self._tail[: max(len(self._tail) - self._keep_bytes, 0)]

    def finish(self, timeout_seconds: float) -> bool:
        """Wait for the stream to close; False if it is still open after the timeout."""
        self._thread.join(timeout_seconds)
        return not self._thread.is_alive()

    def tail(self, limit: int) -> str:
        with self._lock:
            return bytes(self._tail[-limit:]).decode("utf-8", errors="replace")


def _report(text: str, nonce: str) -> dict | None:
    for line in reversed(text.splitlines()):
        if line.startswith(nonce):
            try:
                report = json.loads(line[len(nonce) :])
            except json.JSONDecodeError:
                return None
            return report if isinstance(report, dict) else None
    return None


def _stop(docker: str, name: str, process: subprocess.Popen) -> None:
    """Kill the container and wait for the attached CLI to exit, retrying the kill.

    The container may still be starting, or have exited on its own, so one failed kill is not
    an error; only a container that cannot be stopped within the bound is.
    """
    deadline = time.monotonic() + DOCKER_COMMAND_TIMEOUT_SECONDS
    while True:
        _docker(docker, "kill", name)
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                raise SandboxUnavailable(
                    f"Docker did not stop the sandbox within {DOCKER_COMMAND_TIMEOUT_SECONDS} s"
                ) from None


def run_project_tests_isolated(
    project: Path,
    *,
    timeout_seconds: float,
    cancel: threading.Event | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    limits: SandboxLimits | None = None,
    image: str = SANDBOX_IMAGE,
    docker: str = "docker",
    backstop_seconds: float | None = None,
) -> ProjectTestResult:
    """Run the suite in a disposable container; raise SandboxError for non-test failures.

    Outcomes match `run_project_tests`, plus `resource_limit` when the memory or captured-output
    limit ended the suite. `backstop_seconds` overrides the in-container backstop and exists for
    verification; by default it is the timeout plus the limits' grace period.
    """
    limits = limits or SandboxLimits()
    _validate_request(project, timeout_seconds, poll_seconds, image, backstop_seconds)
    if shutil.which(docker) is None:
        raise SandboxUnavailable(f"Docker CLI not found: {docker}")
    if cancel is not None and cancel.is_set():
        # Cancelled before anything was created: nothing ran, nothing to release.
        return ProjectTestResult(
            outcome="cancelled",
            returncode=None,
            tests_run=None,
            output_tail="",
            wall_seconds=0.0,
            cpu_seconds=None,
            isolation="container",
            cleanup={"container_removed": True, "staging_removed": True},
        )
    # Containers whose runner died are past their deadline label; remove them before adding one.
    remove_orphaned_sandboxes(docker=docker)
    backstop = (
        backstop_seconds
        if backstop_seconds is not None
        else (timeout_seconds + limits.backstop_grace_seconds)
    )
    name = f"agentwatch-sandbox-{uuid4().hex}"
    nonce = secrets.token_hex(16)
    staging: Path | None = None
    created = False
    process: subprocess.Popen | None = None
    stdout: _Capture | None = None
    stderr: _Capture | None = None
    try:
        staging = _stage(project)
        create = _docker(
            docker,
            *docker_create_command(
                docker=docker,
                name=name,
                staging=staging / "workspace",
                image=image,
                limits=limits,
                backstop_seconds=backstop,
                created_epoch=int(time.time()),
            )[1:],
        )
        if create.returncode != 0:
            raise SandboxUnavailable(
                f"The sandbox container could not be created: {create.stderr.strip()[-500:]}"
            )
        created = True
        started = time.monotonic()
        # The container exists before it starts, so a kill can always reach it.
        process = subprocess.Popen(
            [docker, "start", "--attach", "--interactive", name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = _Capture(process.stdout, MAX_REPORT_BYTES)
        stderr = _Capture(process.stderr, MAX_OUTPUT_CHARS)
        process.stdin.write((nonce + "\n").encode())
        process.stdin.close()
        outcome, limit = _supervise(
            docker, name, process, started + timeout_seconds, poll_seconds, cancel,
            stdout, stderr,
        )  # fmt: skip
        wall_seconds = time.monotonic() - started
        if not (
            stdout.finish(DOCKER_COMMAND_TIMEOUT_SECONDS)
            and stderr.finish(DOCKER_COMMAND_TIMEOUT_SECONDS)
        ):
            raise SandboxExecutionError("The sandbox output did not close after the CLI exited")
        state = _inspect_state(docker, name)
        if state is None:
            raise SandboxExecutionError("The sandbox container disappeared during the run")
        if str(state.get("StartedAt", "")).startswith("0001"):
            # Docker reports StartedAt 0001-01-01 for a container that never started.
            raise SandboxUnavailable(
                f"The sandbox container could not start: {stderr.tail(2000).strip()}"
            )
        output = stderr.tail(MAX_OUTPUT_CHARS)
        result = _classify(
            outcome, limit, state, _report(stdout.tail(MAX_REPORT_BYTES), nonce),
            output, wall_seconds, backstop,
        )  # fmt: skip
    except BaseException as error:
        # Keep the original error; a failed or failing release must not replace or hide it.
        try:
            cleanup = _release(docker, name, staging, process, created, (stdout, stderr))
        except _RELEASE_ERRORS as release_error:
            error.add_note(f"Releasing the sandbox also failed: {release_error!r}")
        else:
            if not all(cleanup.values()):
                error.add_note(f"Sandbox resources were not verified released: {cleanup}")
        raise
    try:
        cleanup = _release(docker, name, staging, process, created, (stdout, stderr))
    except _RELEASE_ERRORS as release_error:
        message = f"Releasing the sandbox failed: {release_error!r}"
        raise SandboxCleanupError(message, result) from release_error
    if not all(cleanup.values()):
        raise SandboxCleanupError(f"Sandbox resources were not released: {cleanup}", result)
    return dataclasses.replace(result, cleanup=cleanup)


def _supervise(
    docker: str,
    name: str,
    process: subprocess.Popen,
    deadline: float,
    poll_seconds: float,
    cancel: threading.Event | None,
    stdout: _Capture,
    stderr: _Capture,
) -> tuple[Outcome | None, str | None]:
    """Wait for the suite; stop it on cancellation, the deadline or an output flood."""
    while True:
        try:
            process.wait(timeout=max(min(poll_seconds, deadline - time.monotonic()), 0))
            return None, None
        except subprocess.TimeoutExpired:
            if cancel is not None and cancel.is_set():
                decided = ("cancelled", None)
            elif time.monotonic() >= deadline:
                decided = ("timed_out", None)
            elif (
                stderr.total_bytes > MAX_CAPTURED_OUTPUT_BYTES
                or stdout.total_bytes > MAX_REPORT_BYTES
            ):
                decided = ("resource_limit", "output")
            else:
                continue
            _stop(docker, name, process)
            return decided


def _classify(
    outcome: Outcome | None,
    limit: str | None,
    state: dict,
    report: dict | None,
    output: str,
    wall_seconds: float,
    backstop_seconds: float,
) -> ProjectTestResult:
    """Decide the outcome. Host decisions (deadline, cancellation, output) come first, then the
    runtime's OOM flag, then the wrapper's report, which the suite cannot reach. The suite's own
    exit status is still the suite's: code under test can always end its process with 0."""
    if outcome is None:
        exit_code = state.get("ExitCode")
        if state.get("OOMKilled"):
            outcome, limit = "resource_limit", "memory"
        elif (
            report is not None
            and report.get("backstop_expired")
            and exit_code == BACKSTOP_EXIT_CODE
            and wall_seconds >= backstop_seconds - 1
        ):
            outcome = "timed_out"
        elif report is not None and isinstance(report.get("returncode"), int):
            # A suite killed by a signal (e.g. SIGKILL from itself) is a failing suite.
            passed = report["returncode"] == 0 and exit_code == 0
            outcome = "passed" if passed else "failed"
        else:
            raise SandboxExecutionError(
                f"The sandbox exited {exit_code} without its wrapper report: {output[-500:]}"
            )
    measured = report is not None and outcome in {"passed", "failed"} and "cpu_seconds" in report
    return ProjectTestResult(
        outcome=outcome,
        returncode=report.get("returncode") if outcome in {"passed", "failed"} else None,
        tests_run=tests_run_from_output(output),
        output_tail=output[-MAX_OUTPUT_CHARS:],
        wall_seconds=wall_seconds,
        cpu_seconds=float(report["cpu_seconds"]) if measured else None,
        isolation="container",
        limit_exceeded=limit,
        in_sandbox_wall_seconds=float(report["wall_seconds"]) if measured else None,
    )


def _release(
    docker: str,
    name: str,
    staging: Path | None,
    process: subprocess.Popen | None,
    created: bool,
    captures: tuple[_Capture | None, ...],
) -> dict[str, bool]:
    """Remove the container and the staging copy, and report whether each is verified gone.

    A container whose creation never returned may still be created by the daemon, so removal
    is attempted whether or not `created` was confirmed, and retried briefly. The attached CLI
    is killed and its output threads joined; neither outlives the run.
    """
    if process is not None and process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=DOCKER_COMMAND_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass  # Removing the container below still ends it; it is not left running.
    container_removed = False
    try:
        for attempt in range(3):
            _docker(docker, "rm", "--force", "--volumes", name)
            if not _container_exists(docker, name):
                container_removed = True
                break
            time.sleep(0.5 * (attempt + 1))
    except SandboxUnavailable:
        # The daemon cannot be asked; a container that was never created is not a leak.
        container_removed = not created
    staging_removed = True
    if staging is not None:
        shutil.rmtree(staging, ignore_errors=True)
        staging_removed = not staging.exists()
    for capture in captures:
        if capture is not None:
            capture.finish(DOCKER_COMMAND_TIMEOUT_SECONDS)
    return {"container_removed": container_removed, "staging_removed": staging_removed}


def remove_orphaned_sandboxes(*, docker: str = "docker") -> list[str]:
    """Remove labelled sandbox containers that are past their deadline.

    A container's deadline label is its backstop plus a margin for the host's own kill and
    release, so a live run's container is never removed. Containers without a readable
    deadline are treated as expired. Returns the removed container names.
    """
    listed = _docker(
        docker,
        "ps", "--all", "--filter", f"label={SANDBOX_LABEL}=1",
        "--format", f'{{{{.Names}}}} {{{{.Label "{DEADLINE_LABEL}"}}}}',
    )  # fmt: skip
    if listed.returncode != 0:
        raise SandboxUnavailable(f"`docker ps` failed: {listed.stderr.strip()[:300]}")
    removed = []
    now = time.time()
    for line in listed.stdout.splitlines():
        name, _, deadline = line.partition(" ")
        try:
            expired = now >= int(deadline)
        except ValueError:
            expired = True
        if expired:
            _docker(docker, "rm", "--force", "--volumes", name)
            if not _container_exists(docker, name):
                removed.append(name)
    return removed


def docker_available(docker: str = "docker", image: str = SANDBOX_IMAGE) -> tuple[bool, str]:
    """Whether isolated execution can run here, with the reason when it cannot."""
    if shutil.which(docker) is None:
        return False, "Docker CLI not found"
    info = _docker(docker, "info", "--format", "{{.ServerVersion}}")
    if info.returncode != 0:
        return False, f"Docker daemon unavailable: {info.stderr.strip()[:300]}"
    if _docker(docker, "image", "inspect", image).returncode != 0:
        return False, f"Sandbox image not present; run `make sandbox-image` ({image})"
    return True, info.stdout.strip()


__all__ = [
    "MAX_CAPTURED_OUTPUT_BYTES",
    "SANDBOX_IMAGE",
    "SandboxCleanupError",
    "SandboxError",
    "SandboxExecutionError",
    "SandboxLimits",
    "SandboxRequestError",
    "SandboxUnavailable",
    "docker_available",
    "docker_create_command",
    "remove_orphaned_sandboxes",
    "run_project_tests_isolated",
]
