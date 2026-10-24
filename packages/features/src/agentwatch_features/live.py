"""`step-features-v1`: live step windows joined to their registered resource samples.

Implements the window and FeatureRow rules in docs/shared-event-contracts.md. A step's window
is `[started_tick_ns, finished_tick_ns)` in the step's own clock domain, and only samples from
the same run, resource, service and clock domain join it. Samples from another clock domain
stay unaligned: no clock mapping exists. Feature values are immutable for this version, so
only the metrics named here are read; newer metrics never change a v1 row.
"""

import math
import statistics
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby, pairwise
from typing import Annotated, Literal, Self
from uuid import UUID

from agentwatch_sdk.live import (
    TERMINAL_STATES,
    Identifier,
    Number,
    ResourceSampleFields,
    Seconds,
    StepEventFields,
    Tick,
    step_timings,
    validate_transition,
)
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from agentwatch_features.legacy import throttled_delta_usec

FEATURE_VERSION = "step-features-v1"
PSI_METRIC = "cpu.psi.some.avg10.v1"
THROTTLED_METRIC = "cpu.cgroup.throttled_usec.v1"
NANOSECONDS_PER_SECOND = 1_000_000_000

TIMING_NAMES = ("queue_seconds", "execution_seconds", "duration_seconds", "cpu_seconds")

TelemetryStatus = Literal["present", "empty", "missing", "no_overlap", "unaligned"]
# No same-domain v1 stream exists for the step's target, so lost samples cannot be judged:
# an unsampled target must not look like a sampler that lost every reading.
UNSAMPLED_STATUSES = {"missing", "empty", "unaligned"}
V1_METRICS = {PSI_METRIC, THROTTLED_METRIC}
Fraction = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
PositiveSeconds = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
QUEUE_FRACTION_TOLERANCE = 1e-6  # absolute, as for contract timing breakdowns


class IncompleteStepError(ValueError):
    """The step's applied history is not terminal, so its window is not final."""


class FeatureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Window(FeatureModel):
    clock_id: Identifier
    started_tick_ns: Tick | None
    finished_tick_ns: Tick | None

    @model_validator(mode="after")
    def boundaries(self) -> Self:
        start, finish = self.started_tick_ns, self.finished_tick_ns
        if (start is None) != (finish is None):
            raise ValueError("Window boundaries must both exist or both be null")
        if start is not None and finish is not None and finish < start:
            raise ValueError("Window finish precedes its start")
        return self


class StepTimings(FeatureModel):
    queue_seconds: Seconds | None
    execution_seconds: Seconds | None
    duration_seconds: Seconds | None
    cpu_seconds: Seconds | None


class Telemetry(FeatureModel):
    status: TelemetryStatus
    sample_count: StrictInt = Field(ge=0)
    coverage: Fraction | None
    declared_interval_seconds: PositiveSeconds | None
    observed_interval_seconds: PositiveSeconds | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.status == "present") != (self.sample_count > 0):
            raise ValueError("Only present telemetry has samples inside the window")
        if self.status in UNSAMPLED_STATUSES and self.coverage is not None:
            raise ValueError("Coverage needs a sampler stream for the target in its clock domain")
        if self.sample_count < 2 and self.observed_interval_seconds is not None:
            raise ValueError("An observed interval needs at least two samples")
        return self


class StepFeatures(FeatureModel):
    """The complete, ordered model-input allowlist for FEATURE_VERSION.

    Identity, lineage, run configuration and labels are deliberately absent. Changing a field,
    its type or its meaning needs a new feature version.
    """

    queue_seconds: Seconds | None
    execution_seconds: Seconds | None
    duration_seconds: Seconds | None
    cpu_seconds: Seconds | None
    queue_fraction: Fraction | None
    cpu_pressure_max_avg10: Annotated[Number, Field(ge=0, le=100)] | None
    cpu_pressure_available: StrictBool
    cgroup_throttled_usec_delta: Annotated[Number, Field(ge=0)] | None
    cgroup_available: StrictBool
    telemetry_sample_count: StrictInt = Field(ge=0)
    telemetry_coverage: Fraction | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.cpu_pressure_available != (self.cpu_pressure_max_avg10 is not None):
            raise ValueError("cpu_pressure_available must say whether the maximum was measured")
        if self.cgroup_available != (self.cgroup_throttled_usec_delta is not None):
            raise ValueError("cgroup_available must say whether the delta was measured")
        queue, duration = self.queue_seconds, self.duration_seconds
        expected = queue / duration if queue is not None and duration else None
        if (expected is None) != (self.queue_fraction is None) or (
            expected is not None and abs(self.queue_fraction - expected) > QUEUE_FRACTION_TOLERANCE
        ):
            raise ValueError("queue_fraction must equal queue_seconds / duration_seconds")
        return self


FEATURE_NAMES = tuple(StepFeatures.model_fields)
RESOURCE_AGGREGATES = ("cpu_pressure_max_avg10", "cgroup_throttled_usec_delta")


class FeatureRow(FeatureModel):
    contract_version: Literal["live-contracts-v1"]
    record_type: Literal["feature_row"]
    run_id: UUID
    task_id: Identifier
    step_id: Identifier
    step_type: Literal["model_call", "tool", "retrieval", "other"]
    feature_version: Literal["step-features-v1"]
    source_event_ids: list[UUID] = Field(min_length=1)
    source_sample_ids: list[UUID]
    window: Window
    timings: StepTimings
    telemetry: Telemetry
    features: StepFeatures

    @model_validator(mode="after")
    def consistent(self) -> Self:
        for ids in (self.source_event_ids, self.source_sample_ids):
            if len(ids) != len(set(ids)):
                raise ValueError("Source lineage IDs must be unique")
        # Features repeat envelope values so the model input stays self-contained; they must agree.
        features = self.features.model_dump()
        if any(features[name] != getattr(self.timings, name) for name in TIMING_NAMES):
            raise ValueError("Timing features differ from the row timings")
        if (
            features["telemetry_sample_count"] != self.telemetry.sample_count
            or features["telemetry_coverage"] != self.telemetry.coverage
        ):
            raise ValueError("Telemetry features differ from the row telemetry")
        if self.telemetry.status != "present" and (
            any(features[name] is not None for name in RESOURCE_AGGREGATES)
            or self.source_sample_ids
        ):
            raise ValueError("Resource aggregates need samples inside the aligned window")
        if self.telemetry.status == "present" and not self.source_sample_ids:
            raise ValueError("Present telemetry needs its joined sample lineage")
        if self.window.started_tick_ns is None and self.telemetry.status != "missing":
            raise ValueError("A step without an executed window has missing telemetry")
        return self


@dataclass(frozen=True)
class FeatureRows:
    rows: list[FeatureRow]
    # Steps whose applied history has not reached a terminal state. Their windows are not
    # final, so they get no row yet; they are reported rather than silently dropped.
    incomplete_step_ids: list[str]


def applied_history(events: Iterable[StepEventFields]) -> list[StepEventFields]:
    """The step's revisions in order, validated as one contiguous terminal lifecycle."""
    ordered = sorted(events, key=lambda event: event.sequence)
    if not ordered:
        raise ValueError("A feature row needs the step's applied events")
    previous = None
    for event in ordered:
        validate_transition(previous, event)
        previous = event
    if ordered[-1].state not in TERMINAL_STATES:
        raise IncompleteStepError(f"Step {ordered[-1].step_id!r} is still {ordered[-1].state}")
    return ordered


def distinct_samples(
    samples: Iterable[ResourceSampleFields], run_id: UUID
) -> list[ResourceSampleFields]:
    """Exact retries collapse to one observation; conflicting readings are rejected."""
    by_id: dict[UUID, ResourceSampleFields] = {}
    keys: set[tuple] = set()
    for sample in samples:
        if sample.run_id != run_id:
            raise ValueError("A sample belongs to a different run")
        existing = by_id.get(sample.sample_id)
        if existing is not None:
            if existing.model_dump(mode="json") != sample.model_dump(mode="json"):
                raise ValueError("A sample UUID was reused with different evidence")
            continue
        key = (
            sample.resource_id,
            sample.service_id,
            sample.clock_id,
            sample.metric,
            sample.tick_ns,
        )
        if key in keys:
            raise ValueError("Two readings share one resource/service/clock/metric/tick")
        keys.add(key)
        by_id[sample.sample_id] = sample
    return list(by_id.values())


class SampleIndex:
    """A run's distinct v1 samples by target and clock domain, sorted by tick.

    Built once per run so each step window is two binary searches, not a scan of the run.
    """

    def __init__(self, samples: list[ResourceSampleFields]):
        streams: dict[tuple[str, str, str], list[ResourceSampleFields]] = defaultdict(list)
        for sample in samples:
            if sample.metric in V1_METRICS:
                streams[(sample.resource_id, sample.service_id, sample.clock_id)].append(sample)
        self.targets = {(resource, service) for resource, service, _ in streams}
        self.streams = {
            key: sorted(stream, key=lambda s: (s.tick_ns, s.metric))
            for key, stream in streams.items()
        }
        self.ticks = {key: [s.tick_ns for s in stream] for key, stream in self.streams.items()}

    def join(self, step: StepEventFields) -> tuple[TelemetryStatus, list[ResourceSampleFields]]:
        """The step's telemetry status and its samples inside `[start, finish)`, by tick."""
        if step.started_tick_ns is None or step.resource_id is None or step.service_id is None:
            return "missing", []  # nothing executed, or no measured target to join
        if (step.resource_id, step.service_id) not in self.targets:
            return "empty", []
        key = (step.resource_id, step.service_id, step.clock_id)
        if key not in self.streams:
            return "unaligned", []
        ticks = self.ticks[key]
        first = bisect_left(ticks, step.started_tick_ns)
        inside = self.streams[key][first : bisect_left(ticks, step.finished_tick_ns)]
        return ("present" if inside else "no_overlap"), inside


def gauge_max(readings: list[ResourceSampleFields]) -> float | None:
    """Maximum reading, or None if any reading in the window was not measured.

    A maximum over only the readings that happened to be measured is not the step's maximum.
    """
    if not readings or any(r.availability != "measured" for r in readings):
        return None
    return max(r.value for r in readings)


def validate_interval(declared_interval_seconds: float | None) -> None:
    if declared_interval_seconds is None:
        return
    if (
        isinstance(declared_interval_seconds, bool)
        or not math.isfinite(declared_interval_seconds)
        or not math.isfinite(declared_interval_seconds * NANOSECONDS_PER_SECOND)
        or declared_interval_seconds * NANOSECONDS_PER_SECOND < 1
    ):
        raise ValueError("Declared sampling interval must be finite and at least 1 ns")


def window_coverage(
    observations: int, window_ns: int, declared_interval_seconds: float | None
) -> float | None:
    """Observations ÷ whole declared intervals in the window, capped at 1.

    Judged against the declared interval because a measured one cannot reveal uniform
    sample loss. A window shorter than one interval cannot show whether samples were lost.
    """
    if declared_interval_seconds is None:
        return None
    expected = window_ns // round(declared_interval_seconds * NANOSECONDS_PER_SECOND)
    return min(1.0, observations / expected) if expected > 0 else None


def compute_feature_row(
    events: Iterable[StepEventFields],
    samples: Iterable[ResourceSampleFields],
    declared_interval_seconds: float | None,
) -> FeatureRow:
    """One step's `step-features-v1` row from its applied events and the run's samples.

    `declared_interval_seconds` is the producer's configured sampling interval, or None when
    it was not recorded; coverage is then null rather than guessed from observed gaps.
    """
    validate_interval(declared_interval_seconds)
    history = applied_history(events)
    index = SampleIndex(distinct_samples(samples, history[-1].run_id))
    return row_from_history(history, index, declared_interval_seconds)


def row_from_history(
    history: list[StepEventFields], index: SampleIndex, declared_interval_seconds: float | None
) -> FeatureRow:
    step = history[-1]
    status, inside = index.join(step)
    ticks = sorted({s.tick_ns for s in inside})
    gaps = [(b - a) / NANOSECONDS_PER_SECOND for a, b in pairwise(ticks)]
    coverage = None
    if status not in UNSAMPLED_STATUSES:
        coverage = window_coverage(
            len(ticks), step.finished_tick_ns - step.started_tick_ns, declared_interval_seconds
        )
    pressure = gauge_max([s for s in inside if s.metric == PSI_METRIC])
    # Cumulative counter: a reading gap or a decrease (reset) leaves the window's delta unknown.
    # Readings inside the window bound the whole step's throttling from below.
    throttled = throttled_delta_usec([s.value for s in inside if s.metric == THROTTLED_METRIC])
    # JSON readings such as 5 and 5.0 are the same measurement; keep one feature type.
    pressure, throttled = (None if v is None else float(v) for v in (pressure, throttled))
    timings = step_timings(step)
    return FeatureRow(
        contract_version="live-contracts-v1",
        record_type="feature_row",
        run_id=step.run_id,
        task_id=step.task_id,
        step_id=step.step_id,
        step_type=step.step_type,
        feature_version=FEATURE_VERSION,
        source_event_ids=[event.event_id for event in history],
        source_sample_ids=[s.sample_id for s in inside],
        window=Window(
            clock_id=step.clock_id,
            started_tick_ns=step.started_tick_ns,
            finished_tick_ns=step.finished_tick_ns if step.started_tick_ns is not None else None,
        ),
        timings=StepTimings(**{name: timings[name] for name in TIMING_NAMES}),
        telemetry=Telemetry(
            status=status,
            sample_count=len(ticks),
            coverage=coverage,
            declared_interval_seconds=declared_interval_seconds,
            observed_interval_seconds=statistics.median(gaps) if gaps else None,
        ),
        features=StepFeatures(
            **{name: timings[name] for name in TIMING_NAMES},
            queue_fraction=timings["queue_fraction"],
            cpu_pressure_max_avg10=pressure,
            cpu_pressure_available=pressure is not None,
            cgroup_throttled_usec_delta=throttled,
            cgroup_available=throttled is not None,
            telemetry_sample_count=len(ticks),
            telemetry_coverage=coverage,
        ),
    )


def compute_feature_rows(
    events: Iterable[StepEventFields],
    samples: Iterable[ResourceSampleFields],
    declared_interval_seconds: float | None,
) -> FeatureRows:
    """Rows for every terminal step of one run, in step_id order; incomplete steps listed."""
    validate_interval(declared_interval_seconds)
    rows, incomplete = [], []
    by_step = sorted(events, key=lambda event: (event.step_id, event.sequence))
    run_ids = {event.run_id for event in by_step}
    if len(run_ids) > 1:
        raise ValueError("Step events from different runs cannot share one window join")
    if not run_ids:
        return FeatureRows(rows, incomplete)
    index = SampleIndex(distinct_samples(samples, run_ids.pop()))
    for step_id, history in groupby(by_step, key=lambda event: event.step_id):
        try:
            rows.append(
                row_from_history(applied_history(history), index, declared_interval_seconds)
            )
        except IncompleteStepError:
            incomplete.append(step_id)
    return FeatureRows(rows, incomplete)
