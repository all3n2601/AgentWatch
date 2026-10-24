"""Strict live-contracts-v1 producer records, shared with the ingestion service."""

import re
from datetime import UTC, datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError


def postgres_text(value: str) -> str:
    if "\x00" in value:
        raise ValueError("Text must not contain NUL characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("Text must contain valid Unicode characters") from error
    return value


Identifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"\S"), AfterValidator(postgres_text)
]
Reason = Annotated[
    str, Field(min_length=1, max_length=512, pattern=r"\S"), AfterValidator(postgres_text)
]
MAX_CURSOR = 2**63 - 1
Tick = Annotated[int, Field(strict=True, ge=0, le=9007199254740991)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Seconds = Annotated[Number, Field(ge=0)]
State = Literal["queued", "running", "completed", "failed", "cancelled"]
ResourceClass = Literal["cpu_sandbox", "inference", "retrieval"]
TERMINAL_STATES = {"completed", "failed", "cancelled"}


class LiveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("observed_at", mode="before", check_fields=False)
    @classmethod
    def timestamp_input(cls, value: object) -> str | datetime:
        if not isinstance(value, (str, datetime)):
            raise PydanticCustomError("timestamp_type", "observed_at must be an RFC 3339 timestamp")
        if isinstance(value, str) and not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})",
            value,
        ):
            raise ValueError("observed_at must be an RFC 3339 timestamp with offset")
        return value

    @field_validator("observed_at", check_fields=False)
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        try:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("observed_at must include a timezone")
            return value.astimezone(UTC)
        except OverflowError as error:
            raise ValueError("observed_at is out of range after UTC normalization") from error


class ResourceRegistration(LiveModel):
    resource_id: Identifier
    service_id: Identifier
    resource_class: ResourceClass


class RunCreate(LiveModel):
    contract_version: Literal["live-contracts-v1"]
    request_id: UUID
    task_id: Identifier
    configuration_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    clock_ids: list[Identifier] = Field(min_length=1, max_length=64)
    resources: list[ResourceRegistration] = Field(max_length=64)

    @model_validator(mode="after")
    def registrations(self) -> Self:
        ids = [r.resource_id for r in self.resources]
        if len(ids) != len(set(ids)) or len(self.clock_ids) != len(set(self.clock_ids)):
            raise ValueError("Resource and clock registrations must be unique")
        return self


class RunChange(LiveModel):
    change_id: UUID
    expected_revision: Annotated[int, Field(strict=True, ge=0)]
    state: State
    reason: Reason | None

    @model_validator(mode="after")
    def terminal_reason(self) -> Self:
        if (self.state in {"failed", "cancelled"}) != (self.reason is not None):
            raise ValueError("Only failure/cancellation requires a reason")
        return self


class Record(LiveModel):
    contract_version: Literal["live-contracts-v1"]
    run_id: UUID
    task_id: Identifier
    observed_at: datetime
    clock_id: Identifier


class StepEventFields(Record):
    record_type: Literal["step_event"]
    event_id: UUID
    step_id: Identifier
    step_type: Literal["model_call", "tool", "retrieval", "other"]
    parent_step_id: Identifier | None
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    resource_id: Identifier | None
    service_id: Identifier | None
    sequence: Annotated[int, Field(strict=True, ge=0, le=1023)]
    state: State
    submitted_tick_ns: Tick
    started_tick_ns: Tick | None
    finished_tick_ns: Tick | None
    cpu_seconds: Seconds | None
    cpu_scope: Literal["process", "process_and_children"] | None
    reason: Reason | None


class StepEvent(StepEventFields):
    @model_validator(mode="after")
    def boundaries(self) -> Self:
        revisions = {
            "queued": {0},
            "running": {1},
            "completed": {2},
            "failed": {1, 2},
            "cancelled": {1, 2},
        }
        if self.sequence not in revisions[self.state]:
            raise ValueError("Step state is incompatible with its revision")
        if self.sequence == 2 and self.started_tick_ns is None:
            raise ValueError("Revision two must follow running and include a start tick")
        if int(self.trace_id, 16) == 0 or int(self.span_id, 16) == 0:
            raise ValueError("Trace/span IDs must be nonzero")
        if self.parent_step_id == self.step_id:
            raise ValueError("A step cannot parent itself")
        if (self.cpu_seconds is None) != (self.cpu_scope is None):
            raise ValueError("CPU scope is required iff CPU time is measured")
        if self.started_tick_ns is not None and self.started_tick_ns < self.submitted_tick_ns:
            raise ValueError("Start precedes submission")
        if self.finished_tick_ns is not None and self.finished_tick_ns < (
            self.started_tick_ns if self.started_tick_ns is not None else self.submitted_tick_ns
        ):
            raise ValueError("Finish precedes start/submission")
        if self.state == "queued" and any(
            v is not None
            for v in (self.started_tick_ns, self.finished_tick_ns, self.cpu_seconds, self.reason)
        ):
            raise ValueError("Queued boundaries, CPU and reason must be null")
        if self.state == "running" and (
            self.started_tick_ns is None
            or self.finished_tick_ns is not None
            or self.cpu_seconds is not None
            or self.reason is not None
        ):
            raise ValueError("Running requires a start and null finish, CPU and reason")
        if self.state in TERMINAL_STATES and self.finished_tick_ns is None:
            raise ValueError("Terminal steps require a finish")
        if self.state == "completed" and (self.started_tick_ns is None or self.reason is not None):
            raise ValueError("Completed requires a start and null reason")
        if self.state in {"failed", "cancelled"} and self.reason is None:
            raise ValueError("Failure/cancellation requires a reason")
        if self.started_tick_ns is None and self.cpu_seconds is not None:
            raise ValueError("Unexecuted steps cannot have CPU time")
        return self


# metric: (resource class, unit, minimum, maximum). Additions need new versioned names; a
# definition never changes meaning in place. Semantics are in docs/live-ingestion.md.
METRICS = {
    "cpu.psi.some.avg10.v1": ("cpu_sandbox", "percent", 0, 100),
    "cpu.cgroup.throttled_usec.v1": ("cpu_sandbox", "microseconds", 0, None),
    # A04: whole-device NVML gauges for the GPU serving inference, never per process.
    "gpu.nvml.utilization.gpu.v1": ("inference", "percent", 0, 100),
    "gpu.nvml.utilization.memory.v1": ("inference", "percent", 0, 100),
    "gpu.nvml.memory.used_bytes.v1": ("inference", "bytes", 0, None),
    "gpu.nvml.power.usage_watts.v1": ("inference", "watts", 0, None),
    "gpu.nvml.clock.sm_mhz.v1": ("inference", "megahertz", 0, None),
}


class ResourceSampleFields(Record):
    record_type: Literal["resource_sample"]
    sample_id: UUID
    resource_id: Identifier
    service_id: Identifier
    resource_class: ResourceClass
    tick_ns: Tick
    metric: Identifier
    unit: Identifier
    value: Number | None
    availability: Literal["measured", "missing", "unsupported", "malformed", "unavailable"]
    reason: Reason | None

    @model_validator(mode="after")
    def availability_value(self) -> Self:
        if self.availability == "measured":
            if self.value is None or self.reason is not None:
                raise ValueError("Measured samples require a number and null reason")
        elif self.value is not None or self.reason is None:
            raise ValueError("Unavailable readings require null value and a reason")
        return self


class ResourceSample(ResourceSampleFields):
    @model_validator(mode="after")
    def reading(self) -> Self:
        definition = METRICS.get(self.metric)
        if definition is None:
            raise ValueError("Metric is not registered in live-contracts-v1")
        resource_class, unit, minimum, maximum = definition
        if self.resource_class != resource_class or self.unit != unit:
            raise ValueError("Metric resource class/unit mismatch")
        if self.value is not None and (
            self.value < minimum or (maximum is not None and self.value > maximum)
        ):
            raise ValueError("Metric value is outside registered bounds")
        return self


LiveRecord = Annotated[StepEvent | ResourceSample, Field(discriminator="record_type")]


class LiveBatch(LiveModel):
    records: list[LiveRecord] = Field(min_length=1, max_length=256)


def validate_step_identity(previous: StepEventFields, event: StepEventFields) -> None:
    identity = (
        "run_id",
        "task_id",
        "step_id",
        "step_type",
        "parent_step_id",
        "trace_id",
        "span_id",
        "resource_id",
        "service_id",
        "clock_id",
        "submitted_tick_ns",
    )
    if any(getattr(previous, key) != getattr(event, key) for key in identity):
        raise ValueError("Step identity/clock/submission changed")
    if (
        previous.started_tick_ns is not None
        and event.started_tick_ns is not None
        and (previous.started_tick_ns != event.started_tick_ns)
    ):
        raise ValueError("Recorded start tick changed")


def validate_transition(previous: StepEventFields | None, event: StepEventFields) -> None:
    if previous is None:
        if event.sequence != 0 or event.state != "queued":
            raise ValueError("A step must begin with queued revision zero")
        return
    validate_step_identity(previous, event)
    allowed = {
        "queued": {"running", "failed", "cancelled"},
        "running": TERMINAL_STATES,
    }
    if event.sequence != previous.sequence + 1 or event.state not in allowed.get(
        previous.state, set()
    ):
        raise ValueError("Invalid step transition")
    if previous.started_tick_ns is not None and previous.started_tick_ns != event.started_tick_ns:
        raise ValueError("Recorded start tick changed")


def step_timings(event: StepEventFields) -> dict[str, float | None]:
    start, finish, submit = (event.started_tick_ns, event.finished_tick_ns, event.submitted_tick_ns)
    duration = (finish - submit) / 1e9 if finish is not None else None
    queue = (start - submit) / 1e9 if start is not None else None
    execution = (finish - start) / 1e9 if start is not None and finish is not None else None
    return {
        "queue_seconds": queue,
        "execution_seconds": execution,
        "duration_seconds": duration,
        "cpu_seconds": event.cpu_seconds,
        "queue_fraction": queue / duration if queue is not None and duration else None,
    }


class ResponseModel(BaseModel):
    """Validate known response fields while tolerating compatible server additions."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class ResourceRegistrationSnapshot(ResourceRegistration):
    model_config = ConfigDict(extra="ignore")


class RunRegistrationSnapshot(RunCreate):
    model_config = ConfigDict(extra="ignore")
    resources: list[ResourceRegistrationSnapshot] = Field(max_length=64)


class RunChangeSnapshot(RunChange):
    model_config = ConfigDict(extra="ignore")


class StepEventSnapshot(StepEventFields):
    # Pending evidence accepted by older writers must remain readable, even when a
    # tightened writer invariant would reject it today. Applied prefixes are server-validated.
    model_config = ConfigDict(extra="ignore")


class ResourceSampleSnapshot(ResourceSampleFields):
    # Readers must not reapply an older producer's closed metric registry to saved evidence.
    model_config = ConfigDict(extra="ignore")


SavedPayload = Annotated[
    StepEventSnapshot | ResourceSampleSnapshot, Field(discriminator="record_type")
]


class CreateResult(ResponseModel):
    run_id: UUID
    created: bool


class ChangeResult(ResponseModel):
    run_id: UUID
    revision: int
    changed: bool


class Gap(ResponseModel):
    step_id: Identifier
    missing_sequence: int
    first_pending_sequence: int
    expired: bool


class RecordAcknowledgement(ResponseModel):
    record_id: UUID
    disposition: Literal["pending", "applied", "expired"]


class BatchResult(ResponseModel):
    run_id: UUID
    added: int
    acknowledgements: list[RecordAcknowledgement]
    gaps: list[Gap]


class Timings(ResponseModel):
    queue_seconds: Seconds | None
    execution_seconds: Seconds | None
    duration_seconds: Seconds | None
    cpu_seconds: Seconds | None
    queue_fraction: Annotated[Number, Field(ge=0, le=1)] | None


class StepView(ResponseModel):
    event: StepEventSnapshot
    timings: Timings


class SavedRecord(RecordAcknowledgement):
    sequence: int
    payload: SavedPayload
    received_at: datetime


class SavedChange(ResponseModel):
    payload: RunChangeSnapshot
    revision: int


class RunSummary(ResponseModel):
    run_id: UUID
    registration: RunRegistrationSnapshot
    state: State
    revision: int
    cleanup_requested: bool
    created_at: datetime
    updated_at: datetime
    sequence: int


class RunSnapshot(RunSummary):
    request_id: UUID
    reason: Reason | None
    cleanup_confirmed: None
    unfinished_steps: list[Identifier]
    steps: list[StepView]
    changes: list[SavedChange]
    gaps: list[Gap]
    records: list[SavedRecord]
    next_cursor: int | None


class RunPage(ResponseModel):
    runs: list[RunSummary]
    next_cursor: int | None


class ValidationIssue(ResponseModel):
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(ResponseModel):
    detail: str | list[ValidationIssue]
