"""Measure the runner instrumentation's overhead on the tool workload it profiles.

Protocol (declared before measurement, see `PROTOCOL`): the correctly patched fixture test
suite of one task is the workload. Warm-up repetitions are discarded, then uninstrumented and
instrumented repetitions alternate (ABAB) so drift affects both conditions equally. Each
instrumented repetition runs exactly what a live run adds around a tool step: a CPU sampler at
the run's interval, background live delivery at its flush interval, and the step's three
events. Uninstrumented repetitions run the workload alone.

Reported per condition: median and 90th-percentile workload wall time and the runner
process's own CPU time (the workload runs in a child process, so runner CPU is the
instrumentation plus orchestration). No pass/fail threshold is applied: the project has not
declared an overhead budget, and this measurement does not choose one.
"""

import hashlib
import json
import platform
import resource
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agentwatch_sandbox.project_tests import run_project_tests
from agentwatch_sdk.live import ResourceRegistration, RunCreate

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import (
    CONTRACT_VERSION,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    RUNNER_VERSION,
)
from agentwatch_agent.delivery import (
    COMPLETION_REFUSED,
    TRANSIENT_ERRORS,
    Journal,
    LiveDelivery,
    LiveTransport,
    PeriodicFlusher,
    RetryPolicy,
    is_retryable,
)
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.resources import CpuProbe, ResourceSampler
from agentwatch_agent.session import TERMINAL_RUN_STATES, CancellationState, RunSession
from agentwatch_agent.stats import summarize
from agentwatch_agent.steps import Step, StepRecorder, new_trace_id
from agentwatch_agent.tasks import Task
from agentwatch_agent.workspace import Workspace

PROTOCOL = {
    "version": "agent-overhead-v1",
    "workload": "correctly patched fixture test suite via run_project_tests",
    "warmup_repetitions": 3,
    "repetitions_per_condition": 30,
    "order": "alternating uninstrumented/instrumented after warm-up",
    "instrumented": "CPU sampler + background live delivery + one step lifecycle",
    "statistics": "median and p90 workload wall time; runner process CPU per repetition",
    "threshold": None,
}
FLUSH_INTERVAL_SECONDS = 0.5
OVERHEAD_SERVICE_ID = "agentwatch-agent-overhead"
# Reported live state when completion was refused and the run was closed as failed instead.
COMPLETION_REFUSED_STATE = "failed_completion_refused"
TEST_TIMEOUT_SECONDS = 60
CONDITIONS = ("uninstrumented", "instrumented")


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def merged_availability(summaries: list[dict]) -> dict[str, dict[str, int]]:
    """Availability counts across every repetition's sampler, not just the last one."""
    merged: dict[str, dict[str, int]] = {}
    for summary in summaries:
        for metric, counts in summary["availability_counts"].items():
            for availability, count in counts.items():
                bucket = merged.setdefault(metric, {})
                bucket[availability] = bucket.get(availability, 0) + count
    return merged


class OverheadCancelled(Exception):
    """The operator cancelled the measurement (SIGINT/SIGTERM)."""


class OverheadFailed(Exception):
    """The measurement did not finish. `closure` says how its live run was left:

    "closed" (failed or cancelled and delivered), or "deferred" (journaled for `resend`).
    The original error is the exception's __cause__.
    """

    def __init__(self, message: str, closure: str, journal: Path, refused: int = 0):
        super().__init__(message)
        self.closure = closure
        self.journal = journal
        # Records the service refused permanently before the measurement stopped.
        self.refused = refused


def _close_completed(delivery: LiveDelivery, session: RunSession) -> str:
    """Complete the run, or defer the closure for `resend` while evidence is undelivered.

    Transient delivery failures make `flush()` return False; anything else raises and fails
    the measurement, which then defers its closure in `_close_unfinished`.
    """
    if not delivery.flush():
        session.defer_change("completed", None)
        return "completed_locally_undelivered"
    try:
        state = session.change_state("completed", None)
    except TRANSIENT_ERRORS as error:
        if not is_retryable(error):
            raise
        # The change is journaled, so `resend` delivers it; the measurement itself is kept.
        return "completed_locally_undelivered"
    if state in TERMINAL_RUN_STATES:
        return state
    # Everything was delivered, so the refusal is permanent (refused or expired evidence).
    if session.change_state("failed", COMPLETION_REFUSED) in TERMINAL_RUN_STATES:
        return COMPLETION_REFUSED_STATE
    return "completed_locally_undelivered"


def _close_unfinished(
    recorder: StepRecorder,
    open_step: Step | None,
    delivery: LiveDelivery,
    session: RunSession,
    state: str,
    reason: str,
    cause: BaseException,
) -> str:
    """End the open step and the run as `state`; return "closed" or "deferred"."""
    if open_step is not None and not open_step.terminal:
        if state == "cancelled":
            recorder.cancel(open_step, reason)
        else:
            recorder.fail(open_step, reason)
    try:
        delivered = delivery.flush()
    # Already failing: defer the closure rather than send it early, and keep this error on
    # the measurement's own error instead of discarding it.
    except Exception as flush_error:  # noqa: BLE001
        cause.add_note(f"Delivery failed while closing the run: {flush_error!r}")
        delivered = False
    if not delivered:
        session.defer_change(state, reason)
        return "deferred"
    try:
        result = session.change_state(state, reason)
    except TRANSIENT_ERRORS as error:
        if not is_retryable(error):
            raise
        return "deferred"  # journaled by change_state before it was sent
    return "closed" if result in TERMINAL_RUN_STATES else "deferred"


def protocol_digest(task: Task, sampling_interval_seconds: float) -> str:
    canonical = json.dumps(
        {
            "protocol": PROTOCOL,
            "task_content_sha256": task.content_sha256,
            "sampling_interval_seconds": sampling_interval_seconds,
            "runner_version": RUNNER_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def measure_overhead(
    task: Task,
    correct_patch_id: str,
    transport: LiveTransport,
    output_dir: Path,
    *,
    repetitions: int = PROTOCOL["repetitions_per_condition"],
    warmup: int = PROTOCOL["warmup_repetitions"],
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
    proc: Path = Path("/proc"),
    cancel: CancellationState | None = None,
) -> dict:
    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup nonnegative")
    patch = task.patch(correct_patch_id)
    if patch is None:
        raise ValueError(f"Task {task.name} has no patch {correct_patch_id!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # The task identity stays the task's own: dataset splits group by task_id, and the
    # overhead protocol is identified by the registered configuration digest instead.
    task_id = task.task_id
    clock = MonotonicClock()
    host = ResourceRegistration(
        resource_id=f"tool-host-{uuid4().hex[:12]}",
        # A distinct service lets consumers exclude overhead runs without runner internals.
        service_id=OVERHEAD_SERVICE_ID,
        resource_class="cpu_sandbox",
    )
    journal_path = output_dir / "overhead-journal.jsonl"
    journal = Journal(journal_path)
    retry = RetryPolicy()
    cancel = cancel or CancellationState()
    try:
        session = RunSession.open(
            transport,
            RunCreate(
                contract_version=CONTRACT_VERSION,
                request_id=uuid4(),
                task_id=task_id,
                configuration_digest=protocol_digest(task, sampling_interval_seconds),
                clock_ids=[clock.clock_id],
                resources=[host],
            ),
            journal,
            retry,
        )
    except Exception as error:
        journal.close()
        unconfirmed = is_retryable(error)
        raise RunNotStarted(
            f"{type(error).__name__}: {error}", "registration", closure_journaled=unconfirmed
        ) from error
    delivery = LiveDelivery(transport, session.run_id, journal, retry=retry)
    recorder = StepRecorder(
        run_id=session.run_id,
        task_id=task_id,
        trace_id=new_trace_id(),
        clock=clock,
        emit=delivery.emit,
    )
    walls: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    runner_cpu: dict[str, float] = dict.fromkeys(CONDITIONS, 0.0)
    sampling: list[dict] = []
    workspace: Workspace | None = None
    open_step: Step | None = None

    def workload() -> float:
        result = run_project_tests(
            workspace.root, timeout_seconds=TEST_TIMEOUT_SECONDS, cancel=cancel.event
        )
        if result.outcome == "cancelled" or cancel.event.is_set():
            raise OverheadCancelled(cancel.request.reason if cancel.request else "Cancelled")
        if result.outcome != "passed":
            raise RuntimeError(f"Overhead workload did not pass: {result.outcome}")
        return result.wall_seconds

    def instrumented(index: int) -> float:
        nonlocal open_step
        sampler = ResourceSampler(
            delivery.emit,
            run_id=session.run_id,
            task_id=task_id,
            clock=clock,
            probes=[CpuProbe(host, proc)],
            interval_seconds=sampling_interval_seconds,
        )
        flusher = PeriodicFlusher(delivery, FLUSH_INTERVAL_SECONDS)
        with flusher, sampler:
            open_step = recorder.submit(f"overhead-{index}", "tool", parent=None, resource=host)
            recorder.start(open_step)
            wall = workload()
            recorder.complete(open_step)
            open_step = None
        # A dead flusher or sampler would silently remove instrumentation work and understate
        # the overhead, so either one ends the measurement.
        for component in (flusher, sampler):
            if component.error is not None:
                raise component.error
        sampling.append(sampler.summary())
        return wall

    try:
        workspace = Workspace.create(task)
        workspace.apply(patch)
        session.change_state("running", None)
        for _ in range(warmup):
            workload()
        for index in range(2 * repetitions):
            condition = CONDITIONS[index % 2]
            cpu_before = _process_cpu_seconds()
            wall = instrumented(index) if condition == "instrumented" else workload()
            runner_cpu[condition] += _process_cpu_seconds() - cpu_before
            walls[condition].append(wall)
        final_state = _close_completed(delivery, session)
    except BaseException as error:
        # Close the live run so it is not left running, then report how it was left.
        # KeyboardInterrupt counts as cancellation for callers without signal handlers.
        if isinstance(error, (OverheadCancelled, KeyboardInterrupt)):
            state, reason = "cancelled", str(error) or "Cancelled by the operator"
        else:
            state, reason = "failed", f"Overhead measurement failed ({type(error).__name__})"
        try:
            closure = _close_unfinished(
                recorder, open_step, delivery, session, state, reason, error
            )
        # The measurement's own error stays the cause; the closure failure is attached.
        except Exception as closure_error:  # noqa: BLE001
            error.add_note(f"The live run could not be closed: {closure_error!r}")
            closure = "deferred"
        raise OverheadFailed(
            reason, closure, journal_path, refused=delivery.stats().rejected
        ) from error
    finally:
        workspace_removed = workspace.remove() if workspace is not None else True
        journal.close()
    baseline, measured = summarize(walls["uninstrumented"]), summarize(walls["instrumented"])
    gaps = [
        s["observed_interval_max_seconds"] for s in sampling if s["observed_interval_max_seconds"]
    ]
    stats = delivery.stats()
    return {
        "protocol": PROTOCOL,
        "recorded_at": datetime.now(UTC).isoformat(),
        "runner_version": RUNNER_VERSION,
        "task_id": task.task_id,
        "host": {"system": platform.system(), "machine": platform.machine()},
        "sampling_interval_seconds": sampling_interval_seconds,
        "flush_interval_seconds": FLUSH_INTERVAL_SECONDS,
        "live_run_id": str(session.run_id),
        "live_run_state": final_state,
        "delivery": {
            "delivered": stats.delivered,
            "complete": stats.complete and final_state in TERMINAL_RUN_STATES,
            # Only an undelivered closure needs `resend`; a refused completion was closed.
            "resend_needed": final_state.endswith("_locally_undelivered"),
            "refused": stats.rejected,
            "completion_refused": final_state == COMPLETION_REFUSED_STATE,
        },
        "journal": str(output_dir / "overhead-journal.jsonl"),
        "sampling": {
            "rounds": sum(s["sample_rounds"] for s in sampling),
            "max_observed_interval_seconds": max(gaps) if gaps else None,
            "availability_counts": merged_availability(sampling),
        },
        "workload_wall_seconds": {"uninstrumented": baseline, "instrumented": measured},
        "median_difference_seconds": measured["median"] - baseline["median"],
        "median_difference_fraction": (measured["median"] - baseline["median"])
        / baseline["median"],
        "runner_cpu_seconds_per_repetition": {
            condition: total / repetitions for condition, total in runner_cpu.items()
        },
        "workspace_removed": workspace_removed,
        "limitations": [
            "One host and one short subprocess workload; this is not an overhead budget",
            (
                "Overhead runs share the task's task_id; identify them by the agent-overhead-v1 "
                "configuration digest and exclude them from training datasets"
            ),
            "Model-call overhead is not measured: model requests run outside the runner",
            (
                "Each instrumented repetition starts its own sampler and flusher threads, so "
                "thread start-up cost is included"
            ),
        ],
    }
