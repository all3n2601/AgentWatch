"""Validated completed-run and OpenTelemetry ingestion contracts (not OTLP)."""

import math
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def validate_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Evidence cannot contain nonfinite values")
    if isinstance(value, dict):
        for child in value.values():
            validate_finite(child)
    if isinstance(value, list):
        for child in value:
            validate_finite(child)


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    @model_validator(mode="after")
    def finite_evidence(self):
        validate_finite(self.model_dump())
        return self


class Step(EvidenceModel):
    step_id: str = Field(min_length=1, max_length=128)
    submitted_ns: int = Field(ge=0)
    started_ns: int = Field(ge=0)
    finished_ns: int = Field(ge=0)
    duration_seconds: Positive
    queue_seconds: Nonnegative
    execution_seconds: Positive
    cpu_seconds: Nonnegative
    queue_fraction: float = Field(ge=0, le=1)
    resource: Literal["none", "cpu_sandbox"]
    checksum: str
    evidence: str
    rule_version: str
    threshold_seconds: Nonnegative

    @model_validator(mode="after")
    def timing(self):
        if not self.submitted_ns <= self.started_ns < self.finished_ns:
            raise ValueError("Invalid step timing order")
        if abs(self.duration_seconds - self.queue_seconds - self.execution_seconds) > 1e-5:
            raise ValueError("Duration does not match timing breakdown")
        if abs(self.queue_fraction - self.queue_seconds / self.duration_seconds) > 1e-5:
            raise ValueError("Queue fraction does not match queue delay")
        if abs(self.queue_seconds - (self.started_ns - self.submitted_ns) / 1e9) > 1e-5:
            raise ValueError("Queue delay does not match timestamps")
        if abs(self.execution_seconds - (self.finished_ns - self.started_ns) / 1e9) > 1e-5:
            raise ValueError("Execution time does not match timestamps")
        return self


class ExperimentConfig(EvidenceModel):
    worker_capacity: int = Field(gt=0)
    iterations: int = Field(gt=0)
    baseline_repeats: int = Field(ge=3)
    workload: str = Field(min_length=1)
    tests_per_step: int | None = None


class Baseline(EvidenceModel):
    execution_cv: Nonnegative
    queue_threshold_seconds: Nonnegative
    steps: list[Step] = Field(min_length=3, max_length=1000)


class RunResult(EvidenceModel):
    run_id: UUID
    experiment: ExperimentConfig
    baseline: Baseline
    clean: Step
    contended: Step
    checks: dict[str, bool]
    limitations: list[str]

    @model_validator(mode="after")
    def consistency(self):
        steps = [*self.baseline.steps, self.clean, self.contended]
        if len({s.step_id for s in steps}) != len(steps):
            raise ValueError("Step IDs must be unique within a run")
        if self.experiment.baseline_repeats != len(self.baseline.steps):
            raise ValueError("Baseline repeat count does not match recorded steps")
        if "passed" not in self.checks or self.checks["passed"] != all(
            value for key, value in self.checks.items() if key != "passed"
        ):
            raise ValueError("Acceptance status must agree with individual checks")
        return self


class SpanContext(EvidenceModel):
    trace_id: str = Field(pattern=r"^0x[0-9a-fA-F]{32}$")
    span_id: str = Field(pattern=r"^0x[0-9a-fA-F]{16}$")


class Span(EvidenceModel):
    name: str = Field(min_length=1, max_length=256)
    context: SpanContext
    start_time: datetime
    end_time: datetime
    attributes: dict[str, Any]

    @model_validator(mode="after")
    def timing(self):
        if not self.start_time.tzinfo or not self.end_time.tzinfo:
            raise ValueError("Span timestamps need timezones")
        if self.end_time < self.start_time:
            raise ValueError("Span end precedes start")
        if not self.attributes.get("agentwatch.step_id"):
            raise ValueError("Missing agentwatch.step_id")
        return self


class Sample(EvidenceModel):
    timestamp_ns: int = Field(ge=0, le=9223372036854775807)
    cpu_pressure: dict[str, dict[str, Nonnegative]] | None
    cgroup_cpu_stat: dict[str, Annotated[int, Field(ge=0)]] | None
    unavailable: list[str]


class TelemetryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spans: list[Span] = Field(default_factory=list, max_length=2000)
    samples: list[Sample] = Field(default_factory=list, max_length=20000)

    def validate_run(self, run_id: str, step_ids: set[str]):
        for span in self.spans:
            if str(span.attributes.get("agentwatch.run_id")) != run_id:
                raise ValueError("Span run ID does not match its run")
            if span.attributes["agentwatch.step_id"] not in step_ids:
                raise ValueError("Span step ID is not recorded in this run")


class RunIngest(TelemetryBatch):
    kind: Literal["coding", "cpu"]
    result: RunResult

    @model_validator(mode="after")
    def telemetry_matches(self):
        steps = [*self.result.baseline.steps, self.result.clean, self.result.contended]
        self.validate_run(str(self.result.run_id), {s.step_id for s in steps})

        return self
