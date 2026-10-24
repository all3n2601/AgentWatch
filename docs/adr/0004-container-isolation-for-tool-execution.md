# 0004: Run tool test suites in a hardened, disposable container

Date: October 24, 2026. Status: proposed for A03 review.

The agent's `run_tests` tool executes a fixture project's test suite in a host subprocess
(`agentwatch_sandbox.project_tests`). That boundary has a timeout, a scrubbed environment and
process-group cleanup, but the tests run with the caller's filesystem, network and
privileges. AGENTS.md therefore restricts it to reviewed fixtures, and ADR 0002 limits the
model to choosing reviewed patches. Before any model-influenced code can run, execution needs
a real isolation boundary whose properties are verified, not assumed.

Add `agentwatch_sandbox.container.run_project_tests_isolated`, a drop-in for
`run_project_tests` that returns the same `ProjectTestResult`. It runs the suite in a
throwaway Docker container, using the Docker CLI rather than a new Python dependency.

| Boundary | Control | Verified by |
| --- | --- | --- |
| Filesystem | The project is copied to a private (`0700`) staging directory that the container mounts read-only at `/workspace`; symlinks and special files are rejected before and after the copy. The root filesystem is read-only. `/tmp` is a `noexec,nosuid,nodev` tmpfs with a size cap, and `--ipc none` removes `/dev/shm`. No host path other than the staging copy is mounted. | Tests that the host is invisible, the workspace and root are read-only, `/tmp` cannot run binaries, writes beyond the cap fail, and `/dev/shm` is absent |
| Network | `--network none`: only loopback is up, and there are no routes. | Test that outbound connections and DNS fail |
| Privilege | Runs as `65534:65534` (nobody) with all capabilities dropped and `no-new-privileges`. The default seccomp profile stays on. | Tests that the uid is 65534, the effective, permitted and bounding capability sets are empty, and NoNewPrivs is set |
| Resources | Memory equals the memory and swap cap (no swap), plus a CPU quota, a process limit and an open-file limit. Exceeding memory is reported as `resource_limit`, not as a test failure. | Tests that the memory cap stops a large allocation, process creation stops at the limit, and the CPU quota is applied |
| Execution time | The container is created, then started, so the host deadline or cancellation can always `docker kill` it, including during start-up. A wrapper running as PID 1 sets a backstop timer that ends the suite if the host stops supervising. As PID 1 it ignores signals it has no handler for, and its SIGINT handler does nothing, so the suite cannot stop or end it. The host drains output as it arrives, stores only a bounded tail, and stops a suite that writes more than 4 MiB. | Tests for the host timeout, cancellation before and after start, the backstop with no host intervention, signals (`SIGSTOP`, `SIGINT`, `SIGTERM` and others) and an early `SIGALRM` sent to the wrapper, an output flood, and the bounded capture |
| Supply chain | The image is pinned by digest and run with `--pull never`, so execution never fetches code over the network. | Unit test that an unpinned image reference is rejected |

Resource release is part of the contract: on every outcome (passed, failed, timed out,
cancelled, resource limit, rejected request, container start failure), the container and the
staging copy are removed, and their removal is checked. A failed removal raises
`SandboxCleanupError` instead of returning a result. Containers carry an
`agentwatch.sandbox` label and a deadline label (backstop plus a margin).
`remove_orphaned_sandboxes` runs before each isolated run and removes only containers past
their deadline, so it cleans up after a runner killed with SIGKILL without touching live runs.

Measurement: `wall_seconds` is host-measured and includes container start-up. The test suite's
own wall time and CPU time (its process and children) are measured inside the container by a
small wrapper and reported separately. An unavailable measurement stays `None`, never zero.

Outcome: host decisions (cancellation, deadline, output cap) come first, then memory: the
runtime's `OOMKilled` flag or the wrapper's count of OOM kills in the container's cgroup. The
flag is set by an asynchronous event and was observed missing at inspect on Linux CI, so it
cannot be the only evidence. Then comes the wrapper's report. The wrapper is PID 1 and non-dumpable, and reads a random
nonce from stdin, so the suite cannot read, signal or forge its report. A backstop timeout
also needs exit code 124 and enough host wall time. `passed` needs the suite's return code 0
and container exit code 0. The suite's return code still belongs to the suite: code under test
can exit 0 itself. This boundary contains what the code can do; it does not prove that a
hostile suite ran its tests.

Consequences: Docker becomes a requirement for isolated execution, and the existing
subprocess runner remains the trusted-fixture path until the agent runner switches executors
(Benoy's integration task). The container adds roughly a second of start-up time, which is
reported, not hidden. This ADR does not authorize free-form model-written code: that still
needs its own review of the tool allowlist and of what a compromised test may exfiltrate
through its outputs. Docker's isolation is shared-kernel. A kernel or runtime escape is out of
scope, and gVisor or a VM would be required for hostile code. See
[the sandbox README](../../services/sandbox/README.md) and
[the threat model](../threat-model/README.md).
