"""Verification of the container isolation boundaries (ADR 0004) and of resource release.

Probe suites run *inside* the sandbox and assert that a boundary holds. Tests that need Docker
run when AGENTWATCH_TEST_DOCKER=1; with it set, an unusable Docker fails instead of skipping,
because skipped isolation tests are incomplete verification.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
from agentwatch_sandbox import container
from agentwatch_sandbox.container import (
    SANDBOX_IMAGE,
    SandboxCleanupError,
    SandboxLimits,
    SandboxRequestError,
    SandboxUnavailable,
    docker_available,
    docker_create_command,
    remove_orphaned_sandboxes,
    run_project_tests_isolated,
)

DOCKER_TESTS = os.environ.get("AGENTWATCH_TEST_DOCKER") == "1"


@pytest.fixture
def nothing_staged(monkeypatch):
    """Fail if a rejected request gets as far as copying the project."""

    def refuse(_project):
        raise AssertionError("a rejected request must not create a staging copy")

    monkeypatch.setattr(container, "_stage", refuse)


@pytest.fixture
def docker():
    if not DOCKER_TESTS:
        pytest.skip("Set AGENTWATCH_TEST_DOCKER=1 to verify container isolation")
    usable, reason = docker_available()
    assert usable, f"AGENTWATCH_TEST_DOCKER=1 but the sandbox cannot run: {reason}"


@pytest.fixture
def no_leaks(docker, monkeypatch):
    """Every Docker test must remove exactly the containers and staging copies it created.

    The fixture records them as they are created, so other sandboxes running concurrently
    (another test session, a real agent run) cannot make it pass or fail.
    """
    names, stagings = [], []
    real_command, real_stage = container.docker_create_command, container._stage

    def recording_command(**kwargs):
        names.append(kwargs["name"])
        return real_command(**kwargs)

    def recording_stage(project):
        staging = real_stage(project)
        stagings.append(staging)
        return staging

    monkeypatch.setattr(container, "docker_create_command", recording_command)
    monkeypatch.setattr(container, "_stage", recording_stage)
    yield names
    left = [name for name in names if container._container_exists("docker", name)]
    assert left == [], f"sandbox containers left behind: {left}"
    assert [path for path in stagings if path.exists()] == [], "staging copies left behind"


def project(tmp_path: Path, test_body: str, name: str = "project") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "test_probe.py").write_text(
        "import os, socket, subprocess, sys, time, unittest\n\n\n"
        "class Probe(unittest.TestCase):\n" + test_body
    )
    return root


def run(root: Path, **kwargs):
    return run_project_tests_isolated(root, timeout_seconds=kwargs.pop("timeout", 60), **kwargs)


# Requests rejected before any resource exists -------------------------------------------


@pytest.mark.parametrize(
    "limits",
    [
        {"memory_mb": 8},
        {"memory_mb": 1.5},
        {"cpus": 0},
        {"cpus": float("inf")},
        {"pids": 4},
        {"tmpfs_mb": 0},
        {"open_files": 10_000},
        {"backstop_grace_seconds": True},
    ],
)
def test_limits_outside_their_bounds_are_rejected(limits):
    with pytest.raises(SandboxRequestError):
        SandboxLimits(**limits)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image": "python:3.13-slim"}, "pinned by digest"),
        ({"image": "python@sha256:abc"}, "pinned by digest"),
        ({"timeout": 0}, "finite and positive"),
        ({"timeout": float("nan")}, "must be a finite number"),
        ({"timeout": True}, "must be a finite number"),
        ({"backstop_seconds": -1}, "backstop_seconds"),
        ({"backstop_seconds": 0}, "backstop_seconds"),
    ],
)
def test_invalid_requests_are_rejected_without_creating_anything(
    tmp_path, kwargs, message, nothing_staged
):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    with pytest.raises(SandboxRequestError, match=message):
        run(root, docker="/nonexistent/docker", **kwargs)


def test_symlinks_and_missing_projects_are_rejected(tmp_path, nothing_staged):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    (root / "passwd").symlink_to("/etc/passwd")
    with pytest.raises(SandboxRequestError, match="symlink"):
        run(root, docker="/nonexistent/docker")
    with pytest.raises(SandboxRequestError, match="does not exist"):
        run(tmp_path / "absent", docker="/nonexistent/docker")


def test_special_files_are_rejected(tmp_path, nothing_staged):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    os.mkfifo(root / "pipe")
    with pytest.raises(SandboxRequestError, match="special file"):
        run(root, docker="/nonexistent/docker")


def test_missing_docker_is_unavailability_and_creates_nothing(tmp_path, nothing_staged):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    with pytest.raises(SandboxUnavailable, match="Docker CLI not found"):
        run(root, docker="/nonexistent/docker")


def test_docker_command_applies_every_isolation_control(tmp_path):
    command = docker_create_command(
        docker="docker",
        name="agentwatch-sandbox-x",
        staging=tmp_path,
        image=SANDBOX_IMAGE,
        limits=SandboxLimits(),
        backstop_seconds=65,
        created_epoch=1000,
    )
    joined = " ".join(command)
    for control in (
        "--network none",
        "--ipc none",
        "--read-only",
        "--pull never",
        "--log-driver none",
        "--user 65534:65534",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--memory 256m",
        "--memory-swap 256m",
        "--cpus 1",
        "--pids-limit 64",
        "--ulimit nofile=256:256",
        "--ulimit core=0",
        "/tmp:rw,noexec,nosuid,nodev,size=16m",
        f"type=bind,source={tmp_path},target=/workspace,readonly",
        "--label agentwatch.sandbox=1",
        "--label agentwatch.sandbox.deadline=1185",  # 1000 + 65 backstop + 120 margin
    ):
        assert control in joined, control
    # The wrapper must be PID 1 (no --init), or the suite could stop or kill it.
    for forbidden in ("--init", "--privileged", "--volume", "--cap-add"):
        assert forbidden not in command
    assert command[command.index(SANDBOX_IMAGE) + 1 :][:4] == ["python", "-I", "-B", "-c"]


# Results and resource release on every outcome ------------------------------------------


def test_passing_and_failing_suites_report_measured_results(tmp_path, no_leaks):
    passing = run(project(tmp_path, "    def test_ok(self):\n        self.assertTrue(True)\n"))
    assert (passing.outcome, passing.returncode, passing.tests_run) == ("passed", 0, 1)
    assert passing.isolation == "container"
    assert passing.cpu_seconds is not None and passing.cpu_seconds >= 0
    assert 0 < passing.in_sandbox_wall_seconds <= passing.wall_seconds
    assert passing.cleanup == {"container_removed": True, "staging_removed": True}
    failing = run(project(tmp_path, "    def test_no(self):\n        self.fail('x')\n", "b"))
    assert (failing.outcome, failing.returncode, failing.tests_run) == ("failed", 1, 1)
    assert "AssertionError" in failing.output_tail


def test_a_suite_that_cannot_import_is_a_failed_outcome(tmp_path, no_leaks):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    (root / "test_broken.py").write_text("import does_not_exist\n")
    result = run(root)
    assert result.outcome == "failed"
    assert "does_not_exist" in result.output_tail


def test_timeout_kills_the_container(tmp_path, no_leaks):
    root = project(tmp_path, "    def test_hang(self):\n        time.sleep(120)\n")
    started = time.monotonic()
    # A 60 s backstop grace means only the host's kill can end this run within 10 s.
    result = run(root, timeout=1, limits=SandboxLimits(backstop_grace_seconds=60))
    assert (result.outcome, result.returncode, result.cpu_seconds) == ("timed_out", None, None)
    assert time.monotonic() - started < 10


def test_backstop_ends_the_container_without_the_host(tmp_path, no_leaks):
    """If the host runner died, the in-container timer still ends the suite."""
    root = project(tmp_path, "    def test_hang(self):\n        time.sleep(120)\n")
    started = time.monotonic()
    result = run(root, timeout=120, backstop_seconds=1)
    assert result.outcome == "timed_out"
    assert time.monotonic() - started < 30, "the host deadline, not the backstop, ended it"


def test_cancellation_kills_the_container(tmp_path, no_leaks):
    root = project(tmp_path, "    def test_hang(self):\n        time.sleep(120)\n")
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    started = time.monotonic()
    result = run(root, timeout=120, cancel=cancel)
    assert result.outcome == "cancelled"
    assert time.monotonic() - started < 30


def test_image_that_is_not_present_is_unavailability_and_leaks_nothing(tmp_path, no_leaks):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    absent = "python:3.13-slim@sha256:" + "1" * 64
    with pytest.raises(SandboxUnavailable, match="could not be created"):
        run(root, image=absent)


def test_failed_release_is_raised_with_the_result(tmp_path, no_leaks, monkeypatch):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    real = container._container_exists
    monkeypatch.setattr(container, "_container_exists", lambda docker, name: True)
    with pytest.raises(SandboxCleanupError) as error:
        run(root)
    assert error.value.result is not None and error.value.result.outcome == "passed"
    # The container really was removed (only the check was faked), so nothing else to do.
    monkeypatch.setattr(container, "_container_exists", real)


def test_concurrent_runs_are_isolated_from_each_other(tmp_path, no_leaks):
    roots = [
        project(tmp_path, "    def test_ok(self):\n        time.sleep(1)\n", name=f"p{i}")
        for i in range(3)
    ]
    results = []
    threads = [threading.Thread(target=lambda r=r: results.append(run(r))) for r in roots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [r.outcome for r in results] == ["passed"] * 3


def test_only_expired_sandboxes_are_swept(no_leaks):
    def create(name, deadline):
        subprocess.run(
            [
                "docker", "create", "--label", "agentwatch.sandbox=1",
                "--label", f"agentwatch.sandbox.deadline={deadline}", "--name", name,
                SANDBOX_IMAGE, "true",
            ],
            capture_output=True, timeout=30, check=True,
        )  # fmt: skip

    expired, live = "agentwatch-sandbox-expired", "agentwatch-sandbox-live"
    create(expired, 0)
    create(live, int(time.time()) + 3600)
    try:
        removed = remove_orphaned_sandboxes()
        assert expired in removed and live not in removed
        assert container._container_exists("docker", live)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", expired, live], capture_output=True, timeout=30, check=False
        )


# Isolation boundaries, probed from inside ------------------------------------------------


def test_host_filesystem_is_invisible(tmp_path, no_leaks):
    secret = tmp_path / "host-secret.txt"
    secret.write_text("do not leak")
    root = project(
        tmp_path,
        "    def test_host_paths(self):\n"
        f"        self.assertFalse(os.path.exists({str(secret)!r}))\n"
        f"        self.assertFalse(os.path.exists({str(Path.home())!r}))\n"
        "        self.assertEqual(sorted(os.listdir('.')), ['test_probe.py'])\n"
        "        self.assertFalse(os.path.exists('/dev/shm'))\n",
    )
    assert run(root).outcome == "passed"


def test_workspace_and_root_are_read_only_and_tmp_is_bounded_noexec(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_read_only(self):\n"
        "        import errno\n"
        "        for path in ('/workspace/new.txt', '/new.txt', '/usr/lib/new.txt'):\n"
        "            with self.assertRaises(OSError):\n"
        "                open(path, 'w').close()\n"
        # /var/tmp is world-writable in the image, so only a read-only root stops this:
        # EROFS, not a permission error, proves the root filesystem itself is read-only.
        "        with self.assertRaises(OSError) as raised:\n"
        "            open('/var/tmp/new.txt', 'w').close()\n"
        "        self.assertEqual(raised.exception.errno, errno.EROFS)\n"
        "        with open('/proc/self/mounts') as handle:\n"
        "            root = next(line.split() for line in handle if line.split()[1] == '/')\n"
        "        self.assertIn('ro', root[3].split(','))\n"
        "    def test_tmp_is_writable_but_bounded(self):\n"
        "        with open('/tmp/small', 'wb') as f:\n"
        "            f.write(b'x' * 1024)\n"
        "        with self.assertRaises(OSError):\n"
        "            with open('/tmp/large', 'wb') as f:\n"
        "                f.write(b'x' * 64 * 1024 * 1024)\n"
        "        os.remove('/tmp/large')\n"
        "    def test_tmp_cannot_execute(self):\n"
        "        with open('/tmp/run.sh', 'w') as f:\n"
        "            f.write('#!/bin/sh\\necho escaped\\n')\n"
        "        os.chmod('/tmp/run.sh', 0o755)\n"
        "        with self.assertRaises(PermissionError):\n"
        "            subprocess.run(['/tmp/run.sh'], check=True)\n",
    )
    result = run(root)
    assert result.outcome == "passed", result.output_tail


def test_network_is_unavailable(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_no_network(self):\n"
        # A new network namespace lists the kernel's default tunnel devices; what matters is
        # that only loopback is up, there is no route, and connections fail.
        "        interfaces = [n for n in os.listdir('/sys/class/net')\n"
        "                      if os.path.isdir(f'/sys/class/net/{n}')]\n"
        "        self.assertIn('lo', interfaces)\n"
        "        for name in interfaces:\n"
        "            with open(f'/sys/class/net/{name}/flags') as handle:\n"
        "                flags = int(handle.read(), 16)\n"
        "            self.assertEqual(bool(flags & 0x1), name == 'lo', name)\n"
        "        with open('/proc/net/route') as handle:\n"
        "            routes = handle.read().splitlines()[1:]\n"
        "        self.assertEqual(routes, [])\n"
        "        with self.assertRaises(OSError):\n"
        "            socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "        with self.assertRaises(OSError):\n"
        "            socket.getaddrinfo('example.com', 443)\n",
    )
    result = run(root)
    assert result.outcome == "passed", result.output_tail


def test_runs_unprivileged_without_capabilities(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_privileges(self):\n"
        "        self.assertEqual((os.getuid(), os.getgid()), (65534, 65534))\n"
        "        status = dict(line.split(':', 1) for line in open('/proc/self/status'))\n"
        "        self.assertEqual(int(status['CapEff'].strip(), 16), 0)\n"
        "        self.assertEqual(int(status['CapPrm'].strip(), 16), 0)\n"
        # Non-root already has no effective capabilities; the bounding set proves
        # --cap-drop ALL, so nothing (e.g. a setuid binary) can regain one.
        "        self.assertEqual(int(status['CapBnd'].strip(), 16), 0)\n"
        "        self.assertEqual(status['NoNewPrivs'].strip(), '1')\n",
    )
    result = run(root)
    assert result.outcome == "passed", result.output_tail


def test_host_environment_is_not_inherited(tmp_path, no_leaks, monkeypatch):
    monkeypatch.setenv("AGENTWATCH_SECRET_FOR_TEST", "do-not-leak")
    root = project(
        tmp_path,
        "    def test_env(self):\n"
        "        self.assertNotIn('AGENTWATCH_SECRET_FOR_TEST', os.environ)\n",
    )
    assert run(root).outcome == "passed"


def test_memory_limit_ends_the_suite_as_a_resource_limit(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_allocate(self):\n"
        "        blocks = [bytearray(32 * 1024 * 1024) for _ in range(32)]\n"
        "        self.assertTrue(blocks)\n",
    )
    result = run(root, limits=SandboxLimits(memory_mb=128))
    assert result.outcome == "resource_limit", result.output_tail
    assert result.limit_exceeded == "memory"
    assert (result.returncode, result.cpu_seconds) == (None, None)


def test_process_cpu_and_file_limits_apply(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_process_limit(self):\n"
        "        children = []\n"
        "        with self.assertRaises(OSError):\n"
        "            for _ in range(200):\n"
        "                children.append(subprocess.Popen(['sleep', '30']))\n"
        "        for child in children:\n"
        "            child.kill()\n"
        "    def test_cpu_quota(self):\n"
        "        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()\n"
        "        self.assertEqual(int(quota) / int(period), 1.0)\n"
        "    def test_open_file_limit(self):\n"
        "        import resource\n"
        "        self.assertEqual(resource.getrlimit(resource.RLIMIT_NOFILE), (256, 256))\n",
    )
    result = run(root)
    assert result.outcome == "passed", result.output_tail


def test_a_suite_cannot_forge_a_passing_result(tmp_path, no_leaks):
    """The suite writes a fake success report to its parent's stdout, then fails."""
    root = project(
        tmp_path,
        "    def test_forge(self):\n"
        "        import json\n"
        "        fake = json.dumps({'returncode': 0, 'cpu_seconds': 0, 'wall_seconds': 0})\n"
        "        for target in (f'/proc/{os.getppid()}/fd/1', '/proc/1/fd/1'):\n"
        "            try:\n"
        "                with open(target, 'w') as handle:\n"
        "                    handle.write('0' * 32 + fake + '\\n' + fake + '\\n')\n"
        "            except OSError:\n"
        "                pass\n"
        "        self.fail('the real result is a failure')\n",
    )
    result = run(root)
    assert (result.outcome, result.returncode) == ("failed", 1)


# Regressions from the A03 security and correctness reviews -------------------------------


def test_cancellation_before_start_creates_nothing(tmp_path, no_leaks):
    cancel = threading.Event()
    cancel.set()
    started = time.monotonic()
    result = run(project(tmp_path, "    def test_ok(self):\n        pass\n"), cancel=cancel)
    assert (result.outcome, result.cpu_seconds) == ("cancelled", None)
    assert time.monotonic() - started < 5
    assert no_leaks == [], "no container may be created for an already-cancelled request"


def test_a_deadline_during_start_up_still_kills_the_container(tmp_path, no_leaks):
    root = project(tmp_path, "    def test_hang(self):\n        time.sleep(120)\n")
    started = time.monotonic()
    result = run(root, timeout=0.05, limits=SandboxLimits(backstop_grace_seconds=60))
    assert result.outcome == "timed_out"
    assert time.monotonic() - started < 15, "the kill missed a container that was starting"


def test_the_suite_cannot_read_or_write_the_wrapper(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_wrapper_is_unreachable(self):\n"
        # Attack the suite's parent, wherever it is; it must be the PID 1 wrapper.
        "        wrapper = os.getppid()\n"
        "        self.assertEqual(wrapper, 1)\n"
        "        for path, mode in ((f'/proc/{wrapper}/mem', 'rb'),\n"
        "                           (f'/proc/{wrapper}/environ', 'rb'),\n"
        "                           (f'/proc/{wrapper}/fd/1', 'w')):\n"
        "            with self.assertRaises(PermissionError, msg=path):\n"
        "                open(path, mode).close()\n",
    )
    result = run(root)
    assert result.outcome == "passed", result.output_tail


def test_stopping_pid_1_does_not_defeat_the_backstop(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_stop_the_wrapper(self):\n"
        "        import signal\n"
        "        os.kill(os.getppid(), signal.SIGSTOP)\n"
        "        time.sleep(120)\n",
    )
    started = time.monotonic()
    result = run(root, timeout=120, backstop_seconds=2)
    assert result.outcome == "timed_out"
    assert time.monotonic() - started < 15, "SIGSTOP on PID 1 suspended the backstop"


def test_an_early_alarm_cannot_fake_a_timeout(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_fake_timeout(self):\n"
        "        import signal\n"
        "        os.kill(os.getppid(), signal.SIGALRM)\n"
        "        time.sleep(0.5)\n"
        "        self.fail('the real result is a failure')\n",
    )
    result = run(root, timeout=60)
    assert (result.outcome, result.returncode) == ("failed", 1)


def test_a_suite_that_kills_itself_is_a_failing_suite(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_self_kill(self):\n"
        "        import signal\n"
        "        os.kill(os.getpid(), signal.SIGKILL)\n",
    )
    result = run(root)
    assert (result.outcome, result.returncode, result.limit_exceeded) == ("failed", -9, None)


def test_an_output_flood_is_a_resource_limit_and_stays_bounded(tmp_path, no_leaks):
    root = project(
        tmp_path,
        "    def test_flood(self):\n"
        "        for _ in range(256):\n"
        "            os.write(2, b'A' * (1 << 20))\n"
        "        time.sleep(60)\n",
    )
    started = time.monotonic()
    result = run(root, limits=SandboxLimits(backstop_grace_seconds=60))
    assert (result.outcome, result.limit_exceeded) == ("resource_limit", "output")
    assert len(result.output_tail) <= container.MAX_OUTPUT_CHARS
    assert time.monotonic() - started < 20


def test_a_symlink_swapped_in_after_validation_is_rejected(tmp_path, no_leaks, monkeypatch):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    (root / "data.txt").write_text("fixture")
    secret = tmp_path / "host-secret.txt"
    secret.write_text("do not copy")
    real_validate = container._validate_request

    def validate_then_swap(*args):
        real_validate(*args)
        (root / "data.txt").unlink()
        (root / "data.txt").symlink_to(secret)  # a concurrent writer, after validation

    monkeypatch.setattr(container, "_validate_request", validate_then_swap)
    with pytest.raises(SandboxRequestError, match="symlink"):
        run(root)


def test_staging_copy_is_private_to_the_host_user(tmp_path):
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    staging = container._stage(root)
    try:
        assert staging.stat().st_mode & 0o777 == 0o700
        assert (staging / "workspace").stat().st_mode & 0o777 == 0o755
        assert (staging / "workspace/test_probe.py").stat().st_mode & 0o777 == 0o644
    finally:
        import shutil

        shutil.rmtree(staging)


def test_a_temporary_directory_that_would_break_the_mount_is_refused(
    tmp_path, no_leaks, monkeypatch
):
    import tempfile as tempfile_module

    unsafe = tmp_path / "a,readonly=false"
    unsafe.mkdir()
    monkeypatch.setattr(tempfile_module, "tempdir", str(unsafe))
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    with pytest.raises(SandboxUnavailable, match="cannot be mounted safely"):
        run(root)
    assert list(unsafe.iterdir()) == []


def test_an_unreachable_daemon_reports_no_false_leak(tmp_path):
    fake = tmp_path / "docker"
    fake.write_text("#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
    fake.chmod(0o755)
    root = project(tmp_path, "    def test_ok(self):\n        pass\n")
    with pytest.raises(SandboxUnavailable, match="Cannot connect") as error:
        run(root, docker=str(fake))
    assert not getattr(error.value, "__notes__", None), "nothing was created, nothing leaked"
