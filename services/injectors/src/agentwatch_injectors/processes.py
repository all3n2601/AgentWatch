"""Load-generating child processes that always end: on request, at a deadline, or by force.

Each child receives a shared stop event, a shared ready semaphore, the sending end of its own
pipe and a backstop in seconds. It must release the semaphore once it applies load, then send
one report through the pipe and exit when the event is set or its backstop passes. Starting
waits for every child to be ready. Halting escalates: set the event and wait for reports, then
terminate, then kill. Cleanup is confirmed only when every child has exited.

The caller creates the group before starting children into it and halts it on every path,
including a failed start, so the cleanup it reports always covers every child started.
"""

import multiprocessing
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event, Semaphore

from agentwatch_injectors.base import (
    KILL_GRACE_SECONDS,
    READY_TIMEOUT_SECONDS,
    STOP_GRACE_SECONDS,
    InjectorError,
)

# Spawn starts every child from a clean interpreter on Linux and macOS alike, so children never
# inherit the parent's threads, locks or open sockets.
CONTEXT = multiprocessing.get_context("spawn")


@dataclass
class Child:
    process: BaseProcess
    reports: Connection


@dataclass
class ChildGroup:
    stop_event: Event
    ready: Semaphore
    children: list[Child]


def new_group() -> ChildGroup:
    return ChildGroup(CONTEXT.Event(), CONTEXT.Semaphore(0), [])


def start_children(
    group: ChildGroup,
    target: Callable[..., None],
    arguments: Sequence[tuple],
    backstop_seconds: float,
    name: str,
) -> None:
    """Start one daemon child per argument tuple into `group`; wait until each applies load.

    Raises if a child cannot start or is not ready in time. Children already started stay in
    the group, and the caller halts it, so their cleanup is reported, never assumed.
    """
    for index, extra in enumerate(arguments):
        receiver, sender = CONTEXT.Pipe(duplex=False)
        process = CONTEXT.Process(
            target=target,
            args=(group.stop_event, group.ready, sender, backstop_seconds, *extra),
            name=f"{name}-{index}",
            # A daemon child is also terminated when the parent exits normally.
            daemon=True,
        )
        group.children.append(Child(process, receiver))
        try:
            process.start()
        finally:
            # The child holds its own copy; closing ours lets recv() see EOF if it dies.
            sender.close()
    _wait_ready(group)


def _wait_ready(group: ChildGroup) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    for _ in group.children:
        if not group.ready.acquire(timeout=max(0.0, deadline - time.monotonic())):
            exited = [c.process.exitcode for c in group.children if not c.process.is_alive()]
            raise InjectorError(
                f"Injector workers did not start within {READY_TIMEOUT_SECONDS:g} s "
                f"(exit codes of exited workers: {exited})"
            )


def _receive(connection: Connection, timeout_seconds: float) -> dict | None:
    try:
        if connection.poll(max(0.0, timeout_seconds)):
            report = connection.recv()
            return report if isinstance(report, dict) else None
    except (EOFError, OSError):
        pass  # The child exited or was stopped before reporting: its report is missing.
    return None


def halt_children(group: ChildGroup) -> tuple[list[dict | None], dict]:
    """Stop every child; return each child's report (None when missing) and cleanup evidence."""
    alive_at_stop = [child.process.is_alive() for child in group.children]
    group.stop_event.set()
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    reports: list[dict | None] = []
    for child in group.children:
        started = child.process.pid is not None
        reports.append(_receive(child.reports, deadline - time.monotonic()) if started else None)
        if started:
            child.process.join(max(0.0, deadline - time.monotonic()))
    exits = []
    for child, alive in zip(group.children, alive_at_stop, strict=True):
        process = child.process
        if process.pid is None:
            exits.append({"stopped_by": "never_started", "exitcode": None})
            continue
        stopped_by = "exited" if alive else "exited_before_stop"
        for escalation, action in (("terminated", process.terminate), ("killed", process.kill)):
            if not process.is_alive():
                break
            action()
            process.join(KILL_GRACE_SECONDS)
            stopped_by = escalation
        if process.is_alive():
            stopped_by = "still_running"
        exits.append({"stopped_by": stopped_by, "exitcode": process.exitcode})
    for child in group.children:
        child.reports.close()
        if child.process.pid is not None and not child.process.is_alive():
            child.process.close()
    cleanup = {
        "confirmed": all(entry["stopped_by"] != "still_running" for entry in exits),
        "children": exits,
    }
    return reports, cleanup


def finished(stop_event: Event, deadline_monotonic: float) -> bool:
    """For use inside a child: stop requested, or the child's own backstop has passed."""
    return stop_event.is_set() or time.monotonic() >= deadline_monotonic
