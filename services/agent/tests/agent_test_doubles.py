"""Agent test doubles. Their output is synthetic and never recorded-run evidence.

Kept out of conftest.py so tests import them by a unique module name: several test
directories have a conftest.py, and `from conftest import ...` resolves ambiguously.
"""

import io
import json
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from uuid import UUID, uuid4

from agentwatch_sdk.live import (
    BatchResult,
    ChangeResult,
    CreateResult,
    StepEvent,
    validate_transition,
)
from agentwatch_worker.nvml import GpuIdentity, NvmlError

TERMINAL = {"completed", "failed", "cancelled"}


def http_error(status: int, detail: str) -> HTTPError:
    body = io.BytesIO(json.dumps({"detail": detail}).encode())
    return HTTPError("http://fake-live", status, detail, {}, body)


@dataclass
class FakeRun:
    run_id: UUID
    registration: object
    state: str = "queued"
    revision: int = 0
    reason: str | None = None
    cleanup_requested: bool = False
    steps: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)


class FakeLiveServer:
    """In-memory stand-in for the B01 live API that enforces the rules the runner relies on:
    idempotent identities, registered clocks/resources, contiguous step transitions, revision
    checks, no new steps on closed runs and no completion with unfinished steps."""

    def __init__(self):
        self.runs: dict[UUID, FakeRun] = {}
        self.by_request: dict[UUID, UUID] = {}
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.failures: list[BaseException] = []  # raised by the next calls, in order
        self.down = False
        self.unavailable_calls: set[str] = set()  # e.g. {"ingest"}: only those calls fail
        self.refuse_completion = False  # answer completion with 409 as for unfinished steps
        self.cancel_after_events: int | None = None
        self.calls: list[str] = []

    def _maybe_fail(self, name: str) -> None:
        self.calls.append(name)
        if self.down or name in self.unavailable_calls:
            raise URLError("connection refused")
        if self.failures:
            raise self.failures.pop(0)

    def create_run(self, registration):
        with self.lock:
            self._maybe_fail("create_run")
            if registration.request_id in self.by_request:
                return CreateResult(run_id=self.by_request[registration.request_id], created=False)
            run_id = uuid4()
            self.by_request[registration.request_id] = run_id
            self.runs[run_id] = FakeRun(run_id, registration)
            return CreateResult(run_id=run_id, created=True)

    def ingest(self, run_id, batch):
        with self.lock:
            self._maybe_fail("ingest")
            run = self.runs[run_id]
            staged_steps = dict(run.steps)
            added = []
            registration = run.registration
            resources = {r.resource_id: r for r in registration.resources}
            for record in batch.records:
                record_id = getattr(record, "event_id", None) or record.sample_id
                saved = run.records.get(record_id)
                if saved is not None:
                    if saved != record.model_dump(mode="json"):
                        raise http_error(409, "Record UUID reused with different evidence")
                    continue
                if record.clock_id not in registration.clock_ids:
                    raise http_error(422, "Clock domain is not registered for this run")
                if record.resource_id is not None and record.resource_id not in resources:
                    raise http_error(422, "Resource is not registered for this run")
                if isinstance(record, StepEvent):
                    previous = staged_steps.get(record.step_id)
                    if previous is None and run.state in TERMINAL:
                        raise http_error(409, "Cannot add a new step to a closed run")
                    if record.parent_step_id and record.parent_step_id not in staged_steps:
                        raise http_error(422, "Parent step must exist in the same run/batch")
                    try:
                        validate_transition(previous, record)
                    except ValueError as error:
                        raise http_error(409, f"Step evidence conflicts: {error}") from None
                    staged_steps[record.step_id] = record
                    if self.cancel_after_events is not None:
                        self.cancel_after_events -= 1
                        if self.cancel_after_events == 0:
                            self._close(run, "cancelled", "Cancelled from the dashboard")
                added.append((record_id, record.model_dump(mode="json")))
            run.steps = staged_steps
            run.records.update(added)
            self.changed.notify_all()
            return BatchResult(
                run_id=run_id,
                added=len(added),
                acknowledgements=[
                    {"record_id": record_id, "disposition": "applied"} for record_id, _ in added
                ],
                gaps=[],
            )

    def change_state(self, run_id, change):
        with self.lock:
            self._maybe_fail("change_state")
            run = self.runs[run_id]
            if change.change_id in run.changes:
                return ChangeResult(
                    run_id=run_id, revision=run.changes[change.change_id], changed=False
                )
            if change.expected_revision != run.revision:
                raise http_error(409, "Run revision conflict; refresh the selected run")
            allowed = {"queued": {"running", "failed", "cancelled"}, "running": TERMINAL}
            if change.state not in allowed.get(run.state, set()):
                raise http_error(409, "Invalid run transition")
            unfinished = [s for s, e in run.steps.items() if e.state not in TERMINAL]
            if change.state == "completed" and (unfinished or self.refuse_completion):
                raise http_error(409, "Cannot complete a run with unfinished steps")
            self._close(run, change.state, change.reason)
            run.changes[change.change_id] = run.revision
            return ChangeResult(run_id=run_id, revision=run.revision, changed=True)

    def _close(self, run, state, reason):
        run.revision += 1
        run.state = state
        run.reason = reason
        run.cleanup_requested = run.cleanup_requested or state == "cancelled"

    def get_run(self, run_id, after=0, limit=256):
        with self.lock:
            self._maybe_fail("get_run")
            run = self.runs[run_id]
            return SimpleNamespace(
                state=run.state,
                revision=run.revision,
                reason=run.reason,
                cleanup_requested=run.cleanup_requested,
            )

    def cancel(self, run_id, reason="Cancelled from the dashboard"):
        with self.lock:
            self._close(self.runs[run_id], "cancelled", reason)

    def wait_for_step(self, step_id: str, timeout: float = 10) -> bool:
        """Block until some run has received `step_id`; no fixed sleeps."""
        with self.changed:
            return self.changed.wait_for(
                lambda: any(step_id in run.steps for run in self.runs.values()), timeout
            )

    def only_run(self) -> FakeRun:
        assert len(self.runs) == 1
        return next(iter(self.runs.values()))


def chat_response(content="", tool_calls=None, **metrics):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = [
            {"function": {"name": name, "arguments": arguments}} for name, arguments in tool_calls
        ]
    return {"message": message, "done_reason": "stop", "eval_count": 7, **metrics}


class ScriptedModel:
    """Test double: replays fixed chat responses. Never a substitute for a real model run."""

    name = "scripted-test-double"
    endpoint = "test://scripted"

    def __init__(
        self, responses, block: threading.Event | None = None, block_on: int | None = None
    ):
        self.responses = list(responses)
        self.payloads: list[dict] = []
        self.block = block
        self.block_on = block_on
        self.blocked = threading.Event()  # set once the blocking request is in progress
        self.placement_block: threading.Event | None = None
        self.placement_delay_seconds = 0.0
        self.placement_started = threading.Event()

    def chat(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        if self.block is not None and len(self.payloads) == self.block_on:
            self.blocked.set()
            self.block.wait(30)
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def observe_placement(self):
        if self.placement_block is not None:
            self.placement_started.set()
            self.placement_block.wait(30)
        if self.placement_delay_seconds:
            # A slow but successful /api/ps read; must never count as model request time.
            threading.Event().wait(self.placement_delay_seconds)
        return {"source": "test double", "status": "unavailable", "reason": "not a model server"}


SOLVE_MEAN = [
    chat_response(tool_calls=[("search_docs", {"query": "mean of an empty input"})]),
    chat_response(tool_calls=[("read_file", {"path": "stats.py"})]),
    chat_response(tool_calls=[("apply_patch", {"patch_id": "return-none-for-empty"})]),
    chat_response(tool_calls=[("run_tests", {})]),
    chat_response("Applied return-none-for-empty; all 5 tests pass."),
]


class FakeGpu:
    """Test double for an open NVML device; its readings are synthetic, never evidence."""

    identity = GpuIdentity(
        index=0,
        device_count=1,
        name="Test GPU",
        driver_version="0.0",
        memory_total_bytes=8 * 2**30,
    )

    def __init__(self, utilization=(0, 0), used=2**30, power=4.2, clock=180, failures=None):
        self.utilization, self.used, self.power, self.clock = utilization, used, power, clock
        self.failures = failures or {}
        self.closed = 0

    def _maybe_fail(self, name):
        if name in self.failures:
            raise NvmlError(f"nvml-{name}", self.failures[name])

    def utilization_percent(self):
        self._maybe_fail("utilization")
        return self.utilization

    def memory_bytes(self):
        self._maybe_fail("memory")
        return self.used, self.identity.memory_total_bytes

    def power_watts(self):
        self._maybe_fail("power")
        return self.power

    def sm_clock_mhz(self):
        self._maybe_fail("clock")
        return self.clock

    def close(self):
        self.closed += 1
