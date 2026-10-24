# Sandbox (A03)

`agentwatch_sandbox` runs a project's `unittest` suite and reports a `ProjectTestResult`. It
has two runners with the same signature and result type:

| Runner | Boundary | Use |
| --- | --- | --- |
| `project_tests.run_project_tests` | Host subprocess with a timeout, a scrubbed environment and process-group cleanup | Reviewed fixtures only (today's agent runner) |
| `container.run_project_tests_isolated` | A hardened, disposable Docker container ([ADR 0004](../../docs/adr/0004-container-isolation-for-tool-execution.md)) | Isolated execution; Benoy is integrating it into the agent runner |

## Setup

```sh
make sandbox-image    # pull the digest-pinned python:3.13-slim image (never pulled implicitly)
make sandbox-verify   # prove every isolation boundary and cleanup path on this machine
make sandbox-sweep    # remove sandbox containers past their deadline (e.g. from a SIGKILLed runner)
```

Docker must be running. The isolated runner uses the Docker CLI and adds no Python
dependency. It is verified locally on Docker Desktop for macOS (arm64). The CI `python` job
runs the same tests on Docker Engine on Linux (amd64); that run is pending this change's PR.
Docker Desktop with WSL2 is expected to work but is not verified.

## Using the isolated runner

```python
from agentwatch_sandbox.container import SandboxLimits, run_project_tests_isolated

result = run_project_tests_isolated(
    project_dir,  # copied; the original is never mounted or changed
    timeout_seconds=60,
    cancel=cancel_event,  # optional threading.Event
    limits=SandboxLimits(memory_mb=256, cpus=1.0, pids=64, tmpfs_mb=16, open_files=256),
)
```

Outcomes are `passed`, `failed`, `timed_out` and `cancelled`, as for the subprocess runner.
There is also `resource_limit`, where `limit_exceeded` is `memory` (the runtime's OOM flag) or
`output` (the suite wrote more than 4 MiB, so the host stopped it). A `resource_limit` outcome
is a limit that was hit, not a failing test: the agent's `run_tests` tool reports it as a tool
failure. A suite that kills itself with a signal, or hits the process or file limit and fails,
is a `failed` suite.

Errors that are not evidence about the project raise a `SandboxError` subclass:

| Exception | Meaning | Resources |
| --- | --- | --- |
| `SandboxRequestError` | Rejected before anything was created: an invalid timeout or limit, an unpinned image, a missing project, or a symlink or special file in the project | None created |
| `SandboxUnavailable` | No Docker CLI, the daemon is down or unresponsive, the image is absent, or the container could not start | Released |
| `SandboxExecutionError` | The container ran but its wrapper produced no report | Released |
| `SandboxCleanupError` | The container or staging copy could not be removed; `.result` keeps the run's result | Reported as not released |

## Boundaries

Each control is verified by a probe suite that runs inside the container
(`tests/test_container.py`). Each of 18 controls was also checked by removing it alone and
confirming its probe fails.

- **Filesystem:**
  - The project is copied to a private (`0700`) staging directory, mounted read-only at
    `/workspace`. Symlinks and special files are rejected before and after the copy.
  - The root filesystem is read-only.
  - `/tmp` is a 16 MiB `noexec,nosuid,nodev` tmpfs. `noexec` stops running binaries from it;
    an interpreter can still read a script there.
  - `--ipc none`: there is no `/dev/shm`.
  - No other host path is mounted. `/proc/self/mountinfo` still names the host staging path.
- **Network:** `--network none`. Only loopback is up, there is no route, and DNS and outbound
  connections fail.
- **Privilege:** runs as uid/gid 65534, with an empty capability bounding set and
  `NoNewPrivs`. Docker's default seccomp profile stays on.
- **Resources:**
  - Memory equals memory plus swap (no swap).
  - CPU quota, process limit and open-file limit.
  - No core dumps.
- **Time:**
  - The host deadline or cancellation kills the container. The container is created before
    it starts, so a deadline during start-up still reaches it.
  - An in-container backstop (timeout plus 5 s) ends the suite if the host stops supervising.
    The wrapper that runs it is PID 1, so the suite cannot stop or kill it, and an early
    `SIGALRM` is ignored.
  - Output is capped (4 MiB for the suite, 64 KiB for the report), and only the tails are read.
- **Leftovers:** each container carries a deadline label (backstop plus 120 s). Every isolated
  run, and `make sandbox-sweep`, removes only labelled containers past that deadline, so live
  runs are never touched.
- **Supply chain:** a digest-pinned image with `--pull never`.

## Measurements

| Field | Meaning |
| --- | --- |
| `wall_seconds` | Host-measured, including container start-up (about 0.3–1 s) |
| `in_sandbox_wall_seconds` | The suite's own wall time, measured inside the container |
| `cpu_seconds` | CPU time of the suite's process and children, measured inside the container; `None` when the suite did not finish (timeout, cancellation, resource limit), never zero |
| `cleanup` | `{"container_removed": true, "staging_removed": true}`, checked after every run |

A small wrapper runs as PID 1 inside the container. It reads a random nonce from stdin, starts
the suite with its output sent to stderr, and writes its report to stdout behind the nonce. The
wrapper makes itself non-dumpable, so the suite cannot read its memory, environment or file
descriptors and cannot forge that report.

The outcome is decided in this order:

1. The host's own decisions: cancellation, deadline or the output cap.
2. The runtime's OOM flag.
3. The backstop. This needs the report's flag, exit code 124, and host wall time of at least
   the backstop.
4. The wrapper's report of the suite's return code. `passed` needs return code 0 and
   container exit code 0.

A run with no report raises `SandboxExecutionError`. The suite's return code is still the
suite's: code under test can end its own process with status 0 (for example `os._exit(0)`) and
appear to pass. The boundary contains what the code can do; it does not certify that a hostile
suite ran its tests.

## Limits of this boundary

Docker containers share the host kernel: a kernel or container-runtime escape is outside this
design, and genuinely hostile code would need gVisor or a VM. The suite's output is returned
to the caller, so anything it prints leaves the sandbox. The model still selects only reviewed
patches; allowing model-written code needs its own review ([ADR 0004](../../docs/adr/0004-container-isolation-for-tool-execution.md)).
Docker tests run when `AGENTWATCH_TEST_DOCKER=1`, which CI sets; with it set, an unusable Docker
fails the tests instead of skipping them. Without it they skip, which is incomplete
verification.
