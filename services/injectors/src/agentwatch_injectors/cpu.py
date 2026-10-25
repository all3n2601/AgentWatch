"""CPU contention: busy child processes hashing in a loop, one per requested worker.

The workers compete with everything else scheduled on the host's CPUs. That is host-wide
contention (in WSL2, the whole VM, including Docker containers such as the model server), not
contention of one process: the log says so, and the labels must not claim otherwise.
"""

import hashlib
import os
import time
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event, Semaphore

from agentwatch_injectors.base import (
    BACKSTOP_GRACE_SECONDS,
    BoundedInjector,
    InjectionClock,
    require_count,
)
from agentwatch_injectors.processes import ChildGroup, finished, halt_children, spawn_children

CPU_METHOD = "cpu-spin-processes-v1"
# Hashes between stop checks: about a millisecond of work, so a worker stops promptly.
SPIN_BATCH = 2000


def _spin(
    stop_event: Event, ready: Semaphore, reports: Connection, backstop_seconds: float
) -> None:
    started, cpu_started = time.monotonic(), time.process_time()
    deadline = started + backstop_seconds
    value, rounds = b"agentwatch-cpu-injector", 0
    while True:
        for _ in range(SPIN_BATCH):
            value = hashlib.sha256(value).digest()
        rounds += SPIN_BATCH
        if rounds == SPIN_BATCH:
            ready.release()  # ready once load is actually being applied
        if finished(stop_event, deadline):
            break
    reports.send(
        {
            "hash_rounds": rounds,
            "cpu_seconds": time.process_time() - cpu_started,
            "wall_seconds": time.monotonic() - started,
            "stopped_by_backstop": not stop_event.is_set(),
        }
    )
    reports.close()


def available_cpus() -> int:
    """CPUs this process may run on (its affinity where supported), at least one."""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:  # macOS has no affinity call
        return os.cpu_count() or 1


class CpuSpinInjector(BoundedInjector):
    kind = "cpu"
    method = CPU_METHOD
    resource_class = "cpu_sandbox"

    def __init__(self, clock: InjectionClock, max_duration_seconds: float, workers: int):
        super().__init__(clock, max_duration_seconds)
        # Hard resource bound: never more busy workers than CPUs this process may use.
        self.workers = require_count("workers", workers, available_cpus())
        self._group: ChildGroup | None = None

    def bounds(self) -> dict:
        return {
            "workers": self.workers,
            "available_cpus": available_cpus(),
            "backstop_seconds": self._backstop_seconds(),
        }

    def target(self) -> dict:
        return {
            "scope": "host CPUs",
            "detail": "busy processes compete with every process on the host or VM",
        }

    def _backstop_seconds(self) -> float:
        return self.max_duration_seconds + BACKSTOP_GRACE_SECONDS

    def _launch(self) -> None:
        self._group = spawn_children(
            _spin, [()] * self.workers, self._backstop_seconds(), "agentwatch-cpu-injector"
        )

    def _halt(self) -> tuple[dict, dict]:
        if self._group is None:
            return {"workers_started": 0}, {"confirmed": True, "children": []}
        reports, cleanup = halt_children(self._group)
        received = [report for report in reports if report is not None]
        activity = {
            "workers_started": len(self._group.children),
            "workers_reported": len(received),
            # Totals over reporting workers only; a missing report is not counted as zero.
            "hash_rounds": sum(report["hash_rounds"] for report in received),
            "worker_cpu_seconds": sum(report["cpu_seconds"] for report in received),
            "workers_stopped_by_backstop": sum(
                1 for report in received if report["stopped_by_backstop"]
            ),
        }
        return activity, cleanup
