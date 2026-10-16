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
import statistics
import threading
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
    Journal,
    LiveDelivery,
    LiveTransport,
    PeriodicFlusher,
    RetryPolicy,
)
from agentwatch_agent.resources import CpuResourceSampler
from agentwatch_agent.session import RunSession
from agentwatch_agent.steps import StepRecorder, new_trace_id
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
TEST_TIMEOUT_SECONDS = 60
CONDITIONS = ("uninstrumented", "instrumented")


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[round(0.9 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
    }


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
) -> dict:
    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup nonnegative")
    patch = task.patch(correct_patch_id)
    if patch is None:
        raise ValueError(f"Task {task.name} has no patch {correct_patch_id!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"{task.task_id}-overhead"
    clock = MonotonicClock()
    host = ResourceRegistration(
        resource_id=f"tool-host-{uuid4().hex[:12]}",
        service_id="agentwatch-agent-runner",
        resource_class="cpu_sandbox",
    )
    journal = Journal(output_dir / "overhead-journal.jsonl")
    retry = RetryPolicy()
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
    delivery = LiveDelivery(transport, session.run_id, journal, retry=retry)
    recorder = StepRecorder(
        run_id=session.run_id,
        task_id=task_id,
        trace_id=new_trace_id(),
        clock=clock,
        emit=delivery.emit,
    )
    workspace = Workspace.create(task)
    workspace.apply(patch)
    never_cancel = threading.Event()
    walls: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    runner_cpu: dict[str, float] = dict.fromkeys(CONDITIONS, 0.0)
    sampling: list[dict] = []

    def workload() -> float:
        result = run_project_tests(
            workspace.root, timeout_seconds=TEST_TIMEOUT_SECONDS, cancel=never_cancel
        )
        if result.outcome != "passed":
            raise RuntimeError(f"Overhead workload did not pass: {result.outcome}")
        return result.wall_seconds

    def instrumented(index: int) -> float:
        sampler = CpuResourceSampler(
            delivery.emit,
            run_id=session.run_id,
            task_id=task_id,
            clock=clock,
            resource=host,
            interval_seconds=sampling_interval_seconds,
            proc=proc,
        )
        with PeriodicFlusher(delivery, FLUSH_INTERVAL_SECONDS), sampler:
            step = recorder.submit(f"overhead-{index}", "tool", parent=None, resource=host)
            recorder.start(step)
            wall = workload()
            recorder.complete(step)
        if sampler.error is not None:
            raise sampler.error
        sampling.append(sampler.summary())
        return wall

    try:
        session.change_state("running", None)
        for _ in range(warmup):
            workload()
        for index in range(2 * repetitions):
            condition = CONDITIONS[index % 2]
            cpu_before = _process_cpu_seconds()
            wall = instrumented(index) if condition == "instrumented" else workload()
            runner_cpu[condition] += _process_cpu_seconds() - cpu_before
            walls[condition].append(wall)
        delivery.flush()
        final_state = session.change_state("completed", None)
    finally:
        workspace_removed = workspace.remove()
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
        "delivery": {"delivered": stats.delivered, "complete": stats.complete},
        "sampling": {
            "rounds": sum(s["sample_rounds"] for s in sampling),
            "max_observed_interval_seconds": max(gaps) if gaps else None,
            "availability_counts": sampling[-1]["availability_counts"] if sampling else {},
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
            "Model-call overhead is not measured: model requests run outside the runner",
            (
                "Each instrumented repetition starts its own sampler and flusher threads, so "
                "thread start-up cost is included"
            ),
        ],
    }
