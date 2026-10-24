"""Synthetic test-only fixtures for step-features-v1 window joins; not recorded-run evidence."""

from uuid import UUID, uuid4

import pytest
from agentwatch_features import (
    FEATURE_NAMES,
    FeatureRow,
    IncompleteStepError,
    compute_feature_row,
    compute_feature_rows,
)
from agentwatch_features.legacy import throttled_delta_usec
from agentwatch_sdk.live import ResourceSample, StepEvent
from pydantic import ValidationError

RUN = UUID("00000000-0000-4000-8000-000000000001")
SECOND_NS = 1_000_000_000
START_NS, FINISH_NS = 1 * SECOND_NS, 3 * SECOND_NS
PSI, THROTTLED = "cpu.psi.some.avg10.v1", "cpu.cgroup.throttled_usec.v1"


def event(sequence, state, step_id="tool-1", **changes):
    started = START_NS if sequence and state != "cancelled" else None
    finished = FINISH_NS if state in {"completed", "failed", "cancelled"} else None
    payload = {
        "contract_version": "live-contracts-v1",
        "record_type": "step_event",
        "run_id": str(RUN),
        "task_id": "fixture-task-v1",
        "event_id": str(uuid4()),
        "step_id": step_id,
        "step_type": "tool",
        "parent_step_id": None,
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "resource_id": "worker-1",
        "service_id": "sandbox-1",
        "sequence": sequence,
        "state": state,
        "observed_at": "2026-10-24T00:00:00Z",
        "clock_id": "clock-1",
        "submitted_tick_ns": 0,
        "started_tick_ns": started,
        "finished_tick_ns": finished,
        "cpu_seconds": 1.5 if state == "completed" else None,
        "cpu_scope": "process" if state == "completed" else None,
        "reason": "Fixture stop" if state in {"failed", "cancelled"} else None,
    }
    return StepEvent.model_validate(payload | changes)


def completed(step_id="tool-1", **changes):
    return [
        event(0, "queued", step_id, **changes),
        event(1, "running", step_id, **changes),
        event(2, "completed", step_id, **changes),
    ]


def sample(tick_ns, metric=PSI, value=10.0, availability="measured", **changes):
    unit = "percent" if metric == PSI else "microseconds"
    payload = {
        "contract_version": "live-contracts-v1",
        "record_type": "resource_sample",
        "run_id": str(RUN),
        "task_id": "fixture-task-v1",
        "sample_id": str(uuid4()),
        "resource_id": "worker-1",
        "service_id": "sandbox-1",
        "resource_class": "cpu_sandbox",
        "observed_at": "2026-10-24T00:00:01Z",
        "clock_id": "clock-1",
        "tick_ns": tick_ns,
        "metric": metric,
        "unit": unit,
        "value": value if availability == "measured" else None,
        "availability": availability,
        "reason": None if availability == "measured" else "Fixture reading not available",
    }
    return ResourceSample.model_validate(payload | changes)


def every_half_second(metric=PSI, values=(10.0, 20.0, 30.0, 40.0)):
    return [sample(START_NS + i * SECOND_NS // 2, metric, v) for i, v in enumerate(values)]


def test_measured_zero_is_a_reading_and_unsupported_is_null():
    zero = compute_feature_row(completed(), [sample(START_NS, value=0.0)], 0.5)
    assert zero.features.cpu_pressure_max_avg10 == 0.0
    assert zero.features.cpu_pressure_available is True
    unsupported = compute_feature_row(
        completed(), [sample(START_NS, availability="unsupported")], 0.5
    )
    assert unsupported.features.cpu_pressure_max_avg10 is None
    assert unsupported.features.cpu_pressure_available is False
    assert unsupported.telemetry.status == "present"  # an attempted reading is an observation


def test_one_unavailable_reading_voids_the_window_maximum():
    samples = every_half_second() + [sample(START_NS + 1, availability="malformed")]
    row = compute_feature_row(completed(), samples, 0.5)
    assert row.features.cpu_pressure_max_avg10 is None
    assert row.features.cpu_pressure_available is False


def test_units_are_seconds_from_nanosecond_ticks_and_counters_in_microseconds():
    counters = every_half_second(THROTTLED, (100.0, 150.0, 175.0, 400.0))
    row = compute_feature_row(completed(), every_half_second() + counters, 0.5)
    assert row.timings.queue_seconds == 1.0
    assert row.timings.execution_seconds == 2.0
    assert row.timings.duration_seconds == 3.0
    assert row.timings.cpu_seconds == 1.5
    assert row.features.queue_fraction == pytest.approx(1 / 3)
    assert row.features.cgroup_throttled_usec_delta == 300.0
    assert row.features.cpu_pressure_max_avg10 == 40.0
    assert row.telemetry.observed_interval_seconds == 0.5
    assert row.telemetry.sample_count == 4  # distinct ticks; two metrics per tick count once
    assert row.features.telemetry_coverage == 1.0
    assert list(row.features.model_dump()) == list(FEATURE_NAMES)


@pytest.mark.parametrize(
    "values",
    [(100.0, 50.0, 60.0, 70.0), (100.0, 200.0, 10.0, 20.0)],
    ids=["reset-at-second-reading", "reset-mid-window"],
)
def test_counter_reset_is_an_unknown_delta_not_negative_load(values):
    row = compute_feature_row(completed(), every_half_second(THROTTLED, values), 0.5)
    assert row.features.cgroup_throttled_usec_delta is None
    assert row.features.cgroup_available is False


def test_counter_needs_two_measured_readings():
    one = compute_feature_row(completed(), [sample(START_NS, THROTTLED, 5.0)], 0.5)
    assert one.features.cgroup_throttled_usec_delta is None
    gap = every_half_second(THROTTLED)[:2] + [sample(FINISH_NS - 1, THROTTLED, None, "missing")]
    assert compute_feature_row(completed(), gap, 0.5).features.cgroup_available is False
    assert throttled_delta_usec([5, 9]) == 4  # legacy integer counters keep their type


def test_window_is_half_open_in_the_step_clock_domain():
    at_start, before = sample(START_NS, value=1.0), sample(START_NS - 1, value=99.0)
    at_finish = sample(FINISH_NS, value=98.0)
    row = compute_feature_row(completed(), [before, at_start, at_finish], 0.5)
    assert row.source_sample_ids == [at_start.sample_id]
    assert row.features.cpu_pressure_max_avg10 == 1.0
    assert row.window.model_dump() == {
        "clock_id": "clock-1",
        "started_tick_ns": START_NS,
        "finished_tick_ns": FINISH_NS,
    }


def test_sparse_samples_give_partial_coverage_for_an_incomplete_window():
    row = compute_feature_row(completed(), [sample(START_NS), sample(START_NS + SECOND_NS)], 0.5)
    assert row.telemetry.coverage == 0.5  # 2 observations in 4 declared intervals
    assert row.telemetry.observed_interval_seconds == 1.0
    assert compute_feature_row(completed(), [sample(START_NS)], None).telemetry.coverage is None
    short = compute_feature_row(completed(), [sample(START_NS)], 5.0)
    assert short.telemetry.coverage is None  # window shorter than one declared interval


def test_running_step_has_no_row_yet_and_is_reported():
    running = completed()[:2]
    with pytest.raises(IncompleteStepError):
        compute_feature_row(running, every_half_second(), 0.5)
    result = compute_feature_rows(running + completed("tool-2"), every_half_second(), 0.5)
    assert [row.step_id for row in result.rows] == ["tool-2"]
    assert result.incomplete_step_ids == ["tool-1"]


def test_cancelled_before_start_keeps_timing_without_a_window():
    events = [event(0, "queued"), event(1, "cancelled")]
    row = compute_feature_row(events, every_half_second(), 0.5)
    assert row.window.started_tick_ns is None and row.window.finished_tick_ns is None
    assert row.telemetry.status == "missing"
    assert row.telemetry.coverage is None
    assert row.timings.queue_seconds is None and row.timings.execution_seconds is None
    assert row.timings.duration_seconds == 3.0  # submission to terminal notification
    assert row.features.cpu_seconds is None


def test_failed_step_after_start_keeps_its_executed_window():
    events = [event(0, "queued"), event(1, "running"), event(2, "failed")]
    row = compute_feature_row(events, every_half_second(), 0.5)
    assert row.telemetry.status == "present"
    assert row.features.cpu_seconds is None  # no process measurement was recorded


def test_zero_duration_step_has_null_fraction_and_no_samples():
    tick = {"submitted_tick_ns": START_NS, "started_tick_ns": START_NS}
    events = [
        event(0, "queued", submitted_tick_ns=START_NS),
        event(1, "running", **tick),
        event(2, "completed", **tick, finished_tick_ns=START_NS),
    ]
    row = compute_feature_row(events, [sample(START_NS)], 0.5)
    assert row.features.queue_fraction is None
    assert row.telemetry.status == "no_overlap"
    assert row.telemetry.coverage is None


@pytest.mark.parametrize(
    ("samples", "status"),
    [
        ([], "empty"),
        ([sample(START_NS, resource_id="worker-2")], "empty"),
        ([sample(START_NS, service_id="sandbox-2")], "empty"),
        ([sample(START_NS, clock_id="clock-2")], "unaligned"),
        ([sample(FINISH_NS + SECOND_NS)], "no_overlap"),
    ],
)
def test_samples_join_only_the_registered_target_and_clock(samples, status):
    row = compute_feature_row(completed(), samples, 0.5)
    assert row.telemetry.status == status
    assert row.source_sample_ids == []
    assert row.features.cpu_pressure_max_avg10 is None
    assert row.telemetry.sample_count == 0
    # Only a same-domain stream for the target shows samples were lost; otherwise null.
    assert row.telemetry.coverage == (0.0 if status == "no_overlap" else None)


def test_target_sampled_only_by_unread_metrics_is_empty_not_zero_coverage():
    gpu = sample(START_NS, value=50.0).model_copy(
        update={"metric": "gpu.nvml.utilization.gpu.v1", "resource_class": "inference"}
    )
    row = compute_feature_row(completed(), [gpu], 0.5)
    assert row.telemetry.status == "empty"
    assert row.features.telemetry_coverage is None
    assert row.features.telemetry_sample_count == 0


def test_step_without_a_measured_target_is_missing():
    row = compute_feature_row(completed(resource_id=None, service_id=None), [sample(START_NS)], 0.5)
    assert row.telemetry.status == "missing"


def test_unregistered_future_metrics_never_change_a_v1_row():
    events, samples = completed(), every_half_second()
    future = sample(START_NS + 1, value=5.0).model_copy(update={"metric": "gpu.util.v9"})
    with_future = compute_feature_row(events, [*samples, future], 0.5)
    assert with_future == compute_feature_row(events, samples, 0.5)


def test_exact_retries_do_not_inflate_coverage_and_conflicts_are_rejected():
    samples = every_half_second()[:2]
    row = compute_feature_row(completed(), samples + samples, 0.5)
    assert row.telemetry.sample_count == 2
    changed = samples[0].model_copy(update={"value": 55.0})
    with pytest.raises(ValueError, match="reused"):
        compute_feature_row(completed(), [samples[0], changed], 0.5)
    same_tick = samples[0].model_copy(update={"sample_id": uuid4(), "value": 55.0})
    with pytest.raises(ValueError, match="share one"):
        compute_feature_row(completed(), [samples[0], same_tick], 0.5)


def test_samples_and_events_must_belong_to_one_run():
    other = UUID("00000000-0000-4000-8000-000000000009")
    with pytest.raises(ValueError, match="different run"):
        compute_feature_row(completed(), [sample(START_NS, run_id=str(other))], 0.5)
    with pytest.raises(ValueError, match="different runs"):
        compute_feature_rows(completed() + completed("tool-2", run_id=str(other)), [], 0.5)


def test_history_must_be_a_contiguous_lifecycle():
    queued, _, done = completed()
    with pytest.raises(ValueError, match="Invalid step transition"):
        compute_feature_row([queued, done], [], 0.5)
    with pytest.raises(ValueError, match="applied events"):
        compute_feature_row([], [], 0.5)


@pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan"), 1e-10, True])
def test_declared_interval_must_be_a_positive_finite_duration(interval):
    with pytest.raises(ValueError, match="interval"):
        compute_feature_row(completed(), [], interval)


def test_feature_row_round_trips_and_rejects_unknown_features():
    row = compute_feature_row(completed(), every_half_second(), 0.5)
    assert FeatureRow.model_validate_json(row.model_dump_json()) == row
    payload = row.model_dump(mode="json")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FeatureRow.model_validate(payload | {"features": payload["features"] | {"run_id": 1}})
    with pytest.raises(ValidationError, match="Field required"):
        FeatureRow.model_validate(
            payload
            | {"features": {k: v for k, v in payload["features"].items() if k != "cpu_seconds"}}
        )
    with pytest.raises(ValidationError):
        FeatureRow.model_validate(payload | {"feature_version": "proposal-only"})


NO_SAMPLES = {"sample_count": 0, "observed_interval_seconds": None}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"features": {"telemetry_sample_count": 3}}, "Telemetry features differ"),
        ({"features": {"execution_seconds": 9.0}}, "Timing features differ"),
        ({"telemetry": {"status": "unaligned"}}, "Only present telemetry"),
        ({"telemetry": {"status": "unaligned", **NO_SAMPLES}}, "Coverage needs a sampler stream"),
        ({"telemetry": {"sample_count": 1}}, "two samples"),
        ({"telemetry": {"coverage": 1.5}}, "less than or equal to 1"),
        ({"window": {"finished_tick_ns": None}}, "both exist"),
        ({"window": {"finished_tick_ns": 0}}, "precedes"),
        ({"source_event_ids": []}, "at least 1 item"),
        ({"source_sample_ids": [RUN, RUN]}, "unique"),
        ({"features": {"queue_seconds": float("nan")}}, "finite"),
        ({"features": {"cpu_pressure_available": "yes"}}, "valid boolean"),
        ({"features": {"cpu_pressure_available": False}}, "whether the maximum was measured"),
        (
            {"features": {"cpu_pressure_max_avg10": None}},
            "whether the maximum was measured",
        ),
        (
            {"features": {"cgroup_available": True}},
            "whether the delta was measured",
        ),
        ({"features": {"cpu_pressure_max_avg10": True}}, "valid number"),
        ({"features": {"cpu_pressure_max_avg10": 101.0}}, "less than or equal to 100"),
        ({"features": {"telemetry_sample_count": 4.0}}, "valid integer"),
        ({"features": {"queue_fraction": 7.0}}, "less than or equal to 1"),
        ({"features": {"queue_fraction": 0.9}}, "queue_seconds / duration_seconds"),
        ({"features": {"queue_fraction": None}}, "queue_seconds / duration_seconds"),
        ({"source_sample_ids": []}, "joined sample lineage"),
        (
            {"window": {"started_tick_ns": None, "finished_tick_ns": None}},
            "without an executed window",
        ),
        (
            {
                "telemetry": {"status": "no_overlap", "coverage": 0.0, **NO_SAMPLES},
                "features": {"telemetry_sample_count": 0, "telemetry_coverage": 0.0},
            },
            "need samples inside",
        ),
    ],
)
def test_feature_row_rejects_inconsistent_envelopes(change, message):
    payload = compute_feature_row(completed(), every_half_second(), 0.5).model_dump()
    for key, value in change.items():
        payload[key] = {**payload[key], **value} if isinstance(value, dict) else value
    with pytest.raises(ValidationError, match=message):
        FeatureRow.model_validate(payload)


def test_run_without_step_events_has_no_rows():
    result = compute_feature_rows([], every_half_second(), 0.5)
    assert result.rows == [] and result.incomplete_step_ids == []
