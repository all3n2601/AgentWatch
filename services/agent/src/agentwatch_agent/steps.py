"""Build live-contracts-v1 step lifecycles: queued (0) → running (1) → terminal (2).

A step that ends before it starts (a rejected tool call, a cancellation while queued) goes
straight from queued to failed/cancelled at revision 1, with null start, CPU and CPU scope.
Every event is built and validated once, so a delivery retry resends identical content.
"""

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from agentwatch_sdk.live import ResourceRegistration, StepEvent

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import CONTRACT_VERSION
from agentwatch_agent.redaction import redact_reason

StepType = Literal["model_call", "tool", "retrieval", "other"]
CpuScope = Literal["process", "process_and_children"]
TERMINAL = {"completed", "failed", "cancelled"}


def _nonzero_hex(byte_count: int) -> str:
    while True:
        value = secrets.token_hex(byte_count)
        if int(value, 16):
            return value


def new_trace_id() -> str:
    return _nonzero_hex(16)


def new_span_id() -> str:
    return _nonzero_hex(8)


@dataclass
class Step:
    step_id: str
    step_type: StepType
    parent_step_id: str | None
    span_id: str
    resource_id: str | None
    service_id: str | None
    submitted_tick_ns: int
    started_tick_ns: int | None = None
    sequence: int = 0
    state: str = "queued"

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


class StepRecorder:
    def __init__(
        self,
        *,
        run_id: UUID,
        task_id: str,
        trace_id: str,
        clock: MonotonicClock,
        emit: Callable[[StepEvent], None],
    ):
        self.run_id = run_id
        self.task_id = task_id
        self.trace_id = trace_id
        self.clock = clock
        self._emit_event = emit
        self._steps: dict[str, Step] = {}

    @property
    def steps(self) -> list[Step]:
        return list(self._steps.values())

    def submit(
        self,
        step_id: str,
        step_type: StepType,
        *,
        parent: Step | None,
        resource: ResourceRegistration | None,
    ) -> Step:
        if step_id in self._steps:
            raise ValueError(f"Step {step_id!r} already exists in this run")
        if parent is not None and parent.terminal:
            raise ValueError("Cannot add a child to a finished step")
        step = Step(
            step_id=step_id,
            step_type=step_type,
            parent_step_id=parent.step_id if parent else None,
            span_id=new_span_id(),
            resource_id=resource.resource_id if resource else None,
            service_id=resource.service_id if resource else None,
            submitted_tick_ns=self.clock.tick(),
        )
        self._steps[step_id] = step
        self._emit(step, "queued", finished=None, cpu_seconds=None, cpu_scope=None, reason=None)
        return step

    def start(self, step: Step) -> None:
        self._require(step, {"queued"})
        self._emit(
            step,
            "running",
            started=self.clock.tick(),
            finished=None,
            cpu_seconds=None,
            cpu_scope=None,
            reason=None,
        )

    def complete(
        self, step: Step, *, cpu_seconds: float | None = None, cpu_scope: CpuScope | None = None
    ) -> None:
        self._require(step, {"running"})
        self._emit(
            step,
            "completed",
            finished=self.clock.tick(),
            cpu_seconds=cpu_seconds,
            cpu_scope=cpu_scope,
            reason=None,
        )

    def fail(
        self,
        step: Step,
        reason: str,
        *,
        cpu_seconds: float | None = None,
        cpu_scope: CpuScope | None = None,
    ) -> None:
        self._end(step, "failed", reason, cpu_seconds, cpu_scope)

    def cancel(self, step: Step, reason: str) -> None:
        self._end(step, "cancelled", reason, None, None)

    def unfinished(self) -> list[Step]:
        """Unfinished steps, children before parents, for orderly cancellation."""
        return [step for step in reversed(self._steps.values()) if not step.terminal]

    def _end(
        self,
        step: Step,
        state: str,
        reason: str,
        cpu_seconds: float | None,
        cpu_scope: CpuScope | None,
    ) -> None:
        self._require(step, {"queued", "running"})
        if step.started_tick_ns is None:
            # Nothing executed, so no CPU time can belong to this step.
            cpu_seconds = cpu_scope = None
        self._emit(
            step,
            state,
            finished=self.clock.tick(),
            cpu_seconds=cpu_seconds,
            cpu_scope=cpu_scope,
            reason=redact_reason(reason),
        )

    @staticmethod
    def _require(step: Step, states: set[str]) -> None:
        if step.state not in states:
            raise ValueError(f"Step {step.step_id!r} is {step.state}, expected {sorted(states)}")

    def _emit(
        self,
        step: Step,
        state: str,
        *,
        finished: int | None,
        cpu_seconds: float | None,
        cpu_scope: CpuScope | None,
        reason: str | None,
        started: int | None = None,
    ) -> None:
        sequence = 0 if state == "queued" else step.sequence + 1
        started = started if started is not None else step.started_tick_ns
        event = StepEvent(
            contract_version=CONTRACT_VERSION,
            record_type="step_event",
            run_id=self.run_id,
            task_id=self.task_id,
            event_id=uuid4(),
            step_id=step.step_id,
            step_type=step.step_type,
            parent_step_id=step.parent_step_id,
            trace_id=self.trace_id,
            span_id=step.span_id,
            resource_id=step.resource_id,
            service_id=step.service_id,
            sequence=sequence,
            state=state,
            observed_at=self.clock.now(),
            clock_id=self.clock.clock_id,
            submitted_tick_ns=step.submitted_tick_ns,
            started_tick_ns=started,
            finished_tick_ns=finished,
            cpu_seconds=cpu_seconds,
            cpu_scope=cpu_scope,
            reason=reason,
        )
        # Update local state only after the event validated, so a rejected event cannot
        # leave the recorder believing a transition happened.
        step.sequence = sequence
        step.state = state
        step.started_tick_ns = started
        self._emit_event(event)
