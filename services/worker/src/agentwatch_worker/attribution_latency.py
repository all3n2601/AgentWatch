"""Step completion to visible attribution latency, measured through the live API.

The workload is synthetic: each step is a client-side sleep with deterministic made-up resource
readings, registered as task `synthetic-latency-v1`. It exercises the real ingest, background
attribution and cursor-polled read path; it is not recorded-run evidence.

One `time.monotonic()` clock times each step, from just before the request that delivers its
terminal record to the poll response that contains its attribution. The first poll is sent one
poll interval after that request starts, as for a dashboard whose previous poll had just
returned, so the figure includes a whole poll interval. That figure's nearest-rank p95 is
compared with eval-contract-v3's limit. The server-side figure, `committed_at -
step_terminal_received_at`, comes from the database clock and is reported separately; client
and server clocks are never compared. A step that never becomes visible within the declared
timeout is a failure and makes the gate inconclusive (`null`), never a pass. The report does not
change any promotion state.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from agentwatch_sdk.attribution import AttributionItem, AttributionPage, ModelStatusView
from agentwatch_sdk.live import (
    NANOSECONDS_PER_SECOND,
    BatchResult,
    ChangeResult,
    CreateResult,
    LiveBatch,
    LiveModel,
    ResourceRegistration,
    ResourceSample,
    ResponseModel,
    RunChange,
    RunCreate,
    StepEvent,
)

REPORT_VERSION = "attribution-latency-v1"
# eval-contract-v3, [overhead] max_attribution_latency_p95_seconds; a test pins the two equal.
LIMIT_P95_SECONDS = 15.0
LIMIT_SOURCE = (
    "pipelines/model/evaluation-contract.toml (eval-contract-v3) "
    "[overhead] max_attribution_latency_p95_seconds"
)
LIMIT_PERCENT = 95
MEDIAN_PERCENT = 50
MAX_STEPS = 1000  # The live API's steps-per-run bound.
SAMPLES_PER_STEP = 4
PAGE_LIMIT = 256
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 8_388_608
GIT_TIMEOUT_SECONDS = 5.0
TASK_ID = "synthetic-latency-v1"
CLOCK_ID = "latency-client-monotonic"
RESOURCE_ID = "synthetic-latency-worker"
SERVICE_ID = "synthetic-latency-service"
PRESSURE_METRIC = "cpu.psi.some.avg10.v1"
THROTTLED_METRIC = "cpu.cgroup.throttled_usec.v1"
# Made-up readings, so the feature row has measured inputs; they describe no host.
SYNTHETIC_PRESSURE_PERCENT = 5.0
SYNTHETIC_THROTTLED_USEC_PER_SAMPLE = 100
NOTE = (
    "Synthetic workload through the real live API on one machine. This report is one input to "
    "the eval-contract-v3 attribution-latency gate and does not change any promotion state; "
    "model_status is copied as the worker reported it."
)

type Statistics = dict[str, int | float | None]


class Transport(Protocol):
    """Send one request to the live API and return the response body; raise on a non-2xx."""

    def __call__(self, method: str, path: str, payload: bytes | None) -> bytes: ...


class HttpTransport:
    """The standard library HTTP client, bounded by a socket inactivity timeout per request."""

    def __init__(self, base_url: str, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def __call__(self, method: str, path: str, payload: bytes | None) -> bytes:
        headers = {} if payload is None else {"Content-Type": "application/json"}
        request = Request(self.base_url + path, data=payload, headers=headers, method=method)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("Live API response exceeds the measurement's size limit")
        return body


def nearest_rank(ordered: Sequence[float], percent: int) -> float:
    """The smallest observed value with at least `percent`% of observations at or below it."""
    rank = -(-percent * len(ordered) // 100)
    return ordered[max(rank, 1) - 1]


def summarize(latencies_seconds: list[float]) -> Statistics:
    """count, nearest-rank p50 and p95, and max; None (never 0) for no observations."""
    if any(not math.isfinite(value) for value in latencies_seconds):
        raise ValueError("Latencies must be finite")
    if not latencies_seconds:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(latencies_seconds)
    return {
        "count": len(ordered),
        "p50": nearest_rank(ordered, MEDIAN_PERCENT),
        "p95": nearest_rank(ordered, LIMIT_PERCENT),
        "max": ordered[-1],
    }


def p95_within_limit(client_latency: Statistics, never_visible: int) -> bool | None:
    """True or false against the declared limit (inclusive); None when inconclusive."""
    p95 = client_latency["p95"]
    if never_visible or p95 is None:
        return None
    return p95 <= LIMIT_P95_SECONDS


def code_revision() -> dict[str, object]:
    """This checkout's commit and whether tracked files differ from it."""
    here = Path(__file__).resolve().parent
    try:
        sha = git(here, "rev-parse", "HEAD")
        changes = git(here, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError) as error:
        return {"git_sha": None, "dirty": None, "unavailable_reason": type(error).__name__}
    return {"git_sha": sha, "dirty": bool(changes), "dirty_scope": "tracked files"}


def git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=True,
    )
    return result.stdout.strip()


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def validate_settings(
    base_url: str,
    steps: int,
    poll_interval_seconds: float,
    step_duration_seconds: float,
    visible_timeout_seconds: float,
) -> None:
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("base_url must be an http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("base_url must not carry credentials")
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be an integer from 1 to {MAX_STEPS}")
    for name, value in (
        ("poll_interval_seconds", poll_interval_seconds),
        ("step_duration_seconds", step_duration_seconds),
        ("visible_timeout_seconds", visible_timeout_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


class Ticks:
    """Strictly increasing nanosecond ticks on the measurement clock, from 1 at its origin."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.origin = clock()
        self.last = 0

    def __call__(self) -> int:
        tick = int((self.clock() - self.origin) * NANOSECONDS_PER_SECOND)
        self.last = max(tick, self.last + 1)
        return self.last


@dataclass(frozen=True)
class StepIdentity:
    step_id: str
    trace_id: str
    span_id: str


@dataclass(frozen=True)
class StepResult:
    """One step's outcome: latencies are None when its attribution never became visible."""

    step_id: str
    terminal_disposition: str | None
    client_latency_seconds: float | None
    server_side_seconds: float | None
    polls: int

    def as_report(self) -> dict[str, object]:
        return asdict(self) | {"visible": self.client_latency_seconds is not None}


class LatencyRun:
    """One live run of synthetic steps, each timed until its attribution is visible."""

    def __init__(
        self,
        api: Transport,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        poll_interval_seconds: float,
        step_duration_seconds: float,
        visible_timeout_seconds: float,
    ) -> None:
        self.api = api
        self.clock = clock
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.visible_timeout_seconds = visible_timeout_seconds
        self.sampling_interval_seconds = step_duration_seconds / SAMPLES_PER_STEP
        self.ticks = Ticks(clock)
        self.run_id: UUID | None = None
        self.revision = 0
        self.cursor = 0
        self.seen: dict[str, AttributionItem] = {}
        self.model_states_seen: set[str] = set()
        self.serving_keys_seen: set[str] = set()
        self.first_poll_delays: list[float] = []
        self.poll_spacings: list[float] = []

    def send[Result: ResponseModel](
        self, method: str, path: str, payload: LiveModel | None, result: type[Result]
    ) -> Result:
        body = None if payload is None else payload.model_dump_json().encode()
        return result.model_validate_json(self.api(method, path, body))

    def path(self, suffix: str) -> str:
        if self.run_id is None:
            raise RuntimeError("The run is not registered")
        return f"/v1/live/runs/{self.run_id}{suffix}"

    def register(self, settings: dict[str, object]) -> None:
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        registration = RunCreate(
            contract_version="live-contracts-v1",
            request_id=uuid4(),
            task_id=TASK_ID,
            configuration_digest=digest,
            clock_ids=[CLOCK_ID],
            resources=[
                ResourceRegistration(
                    resource_id=RESOURCE_ID, service_id=SERVICE_ID, resource_class="cpu_sandbox"
                )
            ],
            sampling_interval_seconds=self.sampling_interval_seconds,
        )
        self.run_id = self.send("POST", "/v1/live/runs", registration, CreateResult).run_id
        self.change("running", None)

    def change(self, state: str, reason: str | None) -> None:
        command = RunChange.model_validate(
            {
                "change_id": uuid4(),
                "expected_revision": self.revision,
                "state": state,
                "reason": reason,
            }
        )
        self.revision = self.send("POST", self.path("/state"), command, ChangeResult).revision

    def ingest(self, records: list[StepEvent | ResourceSample]) -> BatchResult:
        return self.send("POST", self.path("/records"), LiveBatch(records=records), BatchResult)

    def event(
        self,
        step: StepIdentity,
        sequence: int,
        state: str,
        submitted: int,
        started: int | None = None,
        finished: int | None = None,
    ) -> StepEvent:
        return StepEvent.model_validate(
            {
                "contract_version": "live-contracts-v1",
                "record_type": "step_event",
                "run_id": self.run_id,
                "task_id": TASK_ID,
                "event_id": uuid4(),
                "step_id": step.step_id,
                "step_type": "tool",
                "parent_step_id": None,
                "trace_id": step.trace_id,
                "span_id": step.span_id,
                "resource_id": RESOURCE_ID,
                "service_id": SERVICE_ID,
                "sequence": sequence,
                "state": state,
                "observed_at": datetime.now(UTC),
                "clock_id": CLOCK_ID,
                "submitted_tick_ns": submitted,
                "started_tick_ns": started,
                "finished_tick_ns": finished,
                "cpu_seconds": None,
                "cpu_scope": None,
                "reason": None,
            }
        )

    def sample(self, metric: str, unit: str, tick: int, value: float) -> ResourceSample:
        return ResourceSample.model_validate(
            {
                "contract_version": "live-contracts-v1",
                "record_type": "resource_sample",
                "run_id": self.run_id,
                "task_id": TASK_ID,
                "sample_id": uuid4(),
                "resource_id": RESOURCE_ID,
                "service_id": SERVICE_ID,
                "resource_class": "cpu_sandbox",
                "observed_at": datetime.now(UTC),
                "clock_id": CLOCK_ID,
                "tick_ns": tick,
                "metric": metric,
                "unit": unit,
                "value": value,
                "availability": "measured",
                "reason": None,
            }
        )

    def run_step(self, step_id: str, step_index: int) -> StepResult:
        # A step keeps its trace and span across revisions; uuid4 hex is never all zeros.
        step = StepIdentity(step_id, uuid4().hex, uuid4().hex[:16])
        submitted = self.ticks()
        started = self.ticks()
        self.ingest(
            [
                self.event(step, 0, "queued", submitted),
                self.event(step, 1, "running", submitted, started),
            ]
        )
        samples: list[StepEvent | ResourceSample] = []
        for sample_index in range(SAMPLES_PER_STEP):
            tick = self.ticks()
            throttled = SYNTHETIC_THROTTLED_USEC_PER_SAMPLE * (
                step_index * SAMPLES_PER_STEP + sample_index
            )
            samples.append(
                self.sample(PRESSURE_METRIC, "percent", tick, SYNTHETIC_PRESSURE_PERCENT)
            )
            samples.append(self.sample(THROTTLED_METRIC, "microseconds", tick, throttled))
            self.sleep(self.sampling_interval_seconds)
        finished = self.ticks()
        self.ingest(samples)
        terminal = self.event(step, 2, "completed", submitted, started, finished)

        terminal_sent = self.clock()
        result = self.ingest([terminal])
        disposition = next(
            (a.disposition for a in result.acknowledgements if a.record_id == terminal.event_id),
            None,
        )
        latency, polls = self.poll_until_visible(step_id, terminal_sent)
        item = self.seen.get(step_id) if latency is not None else None
        return StepResult(
            step_id=step_id,
            terminal_disposition=disposition,
            client_latency_seconds=latency,
            server_side_seconds=None if item is None else server_side_seconds(item),
            polls=polls,
        )

    def poll_until_visible(self, step_id: str, terminal_sent: float) -> tuple[float | None, int]:
        next_poll = terminal_sent + self.poll_interval_seconds
        previous_poll: float | None = None
        polls = 0
        while True:
            self.sleep(max(0.0, next_poll - self.clock()))
            poll_sent = self.clock()
            if previous_poll is None:
                self.first_poll_delays.append(poll_sent - terminal_sent)
            else:
                self.poll_spacings.append(poll_sent - previous_poll)
            previous_poll = poll_sent
            polls += 1
            visible_at = self.drain(step_id)
            if visible_at is not None:
                return visible_at - terminal_sent, polls
            if self.clock() - terminal_sent >= self.visible_timeout_seconds:
                return None, polls
            next_poll += self.poll_interval_seconds

    def drain(self, step_id: str) -> float | None:
        """Read every new page; return when the response holding `step_id`'s row arrived."""
        visible_at: float | None = None
        while True:
            path = self.path(f"/attributions?after={self.cursor}&limit={PAGE_LIMIT}")
            page = AttributionPage.model_validate_json(self.api("GET", path, None))
            received = self.clock()
            if page.run_id != self.run_id:
                raise ValueError("Attribution page is for another run")
            self.model_states_seen.add(page.model.state)
            self.serving_keys_seen.add(page.model.serving_key)
            for item in page.attributions:
                self.cursor = max(self.cursor, item.sequence)
                item_step = item.record.step_id
                if item_step not in self.seen:
                    self.seen[item_step] = item
                    if item_step == step_id and visible_at is None:
                        visible_at = received
            if page.next_cursor is None:
                return visible_at
            self.cursor = max(self.cursor, page.next_cursor)

    def abort(self, error: Exception) -> None:
        """Cancel the run after a failed request, so it is not left running."""
        if self.run_id is None:
            return
        try:
            self.change("cancelled", "Latency measurement aborted after a failed request")
        except (OSError, ValueError) as cancel_error:
            error.add_note(f"Cancelling the run also failed: {type(cancel_error).__name__}")


def server_side_seconds(item: AttributionItem) -> float:
    return (item.committed_at - item.step_terminal_received_at).total_seconds()


def measure(
    base_url: str,
    steps: int,
    poll_interval_seconds: float,
    step_duration_seconds: float,
    visible_timeout_seconds: float,
    *,
    transport: Transport | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    revision: Callable[[], dict[str, object]] = code_revision,
) -> dict[str, object]:
    """Drive `steps` synthetic steps through the live API and report their visible latency."""
    validate_settings(
        base_url, steps, poll_interval_seconds, step_duration_seconds, visible_timeout_seconds
    )
    api = transport if transport is not None else HttpTransport(base_url)
    started_at = datetime.now(UTC)
    status = ModelStatusView.model_validate_json(api("GET", "/v1/live/attribution/model", None))
    run = LatencyRun(
        api, clock, sleep, poll_interval_seconds, step_duration_seconds, visible_timeout_seconds
    )
    settings: dict[str, object] = {
        "description": (
            "synthetic steps: a client-side sleep per step with made-up resource readings; "
            "no agent, model call or real resource contention"
        ),
        "task_id": TASK_ID,
        "steps": steps,
        "step_duration_seconds": step_duration_seconds,
        "samples_per_step": SAMPLES_PER_STEP,
        "sampling_interval_seconds": run.sampling_interval_seconds,
        "metrics": [PRESSURE_METRIC, THROTTLED_METRIC],
        "steps_in_sequence": True,
        "first_poll": "one poll interval after the terminal request is sent",
        "poll_interval_seconds": poll_interval_seconds,
        "visible_timeout_seconds": visible_timeout_seconds,
    }
    per_step: list[StepResult] = []
    try:
        run.register(settings)
        for index in range(steps):
            per_step.append(run.run_step(f"latency-step-{index:04d}", index))
        run.change("completed", None)
    except (OSError, ValueError) as error:
        run.abort(error)
        raise
    finished_at = datetime.now(UTC)

    never_visible = [step.step_id for step in per_step if step.client_latency_seconds is None]
    client = summarize(
        [s.client_latency_seconds for s in per_step if s.client_latency_seconds is not None]
    )
    server = summarize(
        [s.server_side_seconds for s in per_step if s.server_side_seconds is not None]
    )
    outcomes = Counter(
        run.seen[s.step_id].record.model.outcome
        for s in per_step
        if s.client_latency_seconds is not None
    )
    return {
        "report_version": REPORT_VERSION,
        "synthetic_workload": True,
        "workload": settings,
        "steps": steps,
        "poll_interval_seconds": poll_interval_seconds,
        "step_duration_seconds": step_duration_seconds,
        "visible_timeout_seconds": visible_timeout_seconds,
        "base_url": base_url,
        "run_id": str(run.run_id),
        "model_status": status.model_dump(mode="json"),
        "model_states_seen": sorted(run.model_states_seen),
        "serving_key_changed": run.serving_keys_seen - {status.serving_key} != set(),
        "machine": machine(),
        "code_revision": revision(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "clock": (
            "client time.monotonic(): from just before the terminal record's request to the "
            "poll response holding the step's attribution, poll interval included"
        ),
        "limit_p95_seconds": LIMIT_P95_SECONDS,
        "limit_source": LIMIT_SOURCE,
        "statistic": "nearest-rank p95 of client_latency_seconds",
        "p95_within_limit": p95_within_limit(client, len(never_visible)),
        "client_latency_seconds": client,
        "server_side_seconds": server,
        "server_side_clock": (
            "database clock: committed_at - step_terminal_received_at; never compared with the "
            "client clock"
        ),
        "first_poll_after_terminal_seconds": summarize(run.first_poll_delays),
        "later_poll_spacing_seconds": summarize(run.poll_spacings),
        "steps_never_visible": {"count": len(never_visible), "step_ids": never_visible},
        "steps_visible_after_timeout": [
            step_id for step_id in never_visible if step_id in run.seen
        ],
        "model_outcomes": dict(sorted(outcomes.items())),
        "per_step": [step.as_report() for step in per_step],
        "note": NOTE,
    }


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure step completion to visible attribution latency on a synthetic "
        "workload through a running worker's live API."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, required=True)
    parser.add_argument("--step-duration-seconds", type=float, required=True)
    parser.add_argument("--visible-timeout-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        report = measure(
            arguments.base_url,
            arguments.steps,
            arguments.poll_interval_seconds,
            arguments.step_duration_seconds,
            arguments.visible_timeout_seconds,
        )
    except (OSError, ValueError) as error:
        print(f"Latency measurement failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    client = report["client_latency_seconds"]
    never = report["steps_never_visible"]
    print(
        f"{arguments.output}: client latency {client}, never visible {never}, "
        f"p95_within_limit={report['p95_within_limit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
