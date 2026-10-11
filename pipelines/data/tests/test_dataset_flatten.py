import json

import pytest
from agentwatch_data.extract import validate
from agentwatch_data.flatten import (
    FlattenError,
    flatten_run,
    sampling_interval_seconds,
    window_readings,
)
from agentwatch_data.schema import COLUMNS
from dataset_fixtures import BASE_NS, SAMPLE, make_result


def sample(offset, avg10=None, throttled=None):
    return {
        "timestamp_ns": BASE_NS + offset,
        "cpu_pressure": None if avg10 is None else {"some": {"avg10": avg10}},
        "cgroup_cpu_stat": None if throttled is None else {"throttled_usec": throttled},
        "unavailable": [],
    }


@pytest.fixture
def rows():
    return flatten_run(validate(make_result(), [SAMPLE], "cpu"))


def test_one_row_per_step_with_conditions(rows):
    assert [row["step_id"] for row in rows] == [
        "baseline-0",
        "baseline-1",
        "baseline-2",
        "clean",
        "contended",
    ]
    assert [row["condition"] for row in rows] == ["baseline"] * 3 + ["clean", "contended"]


def test_rule_output_is_kept_apart_from_labels(rows):
    assert rows[-1]["rule_prediction"] == "cpu_sandbox"
    assert not any(key.startswith("label_") for row in rows for key in row)


def test_rows_only_use_schema_columns(rows):
    names = {column.name for column in COLUMNS}
    assert set(rows[0]) <= names


def test_shared_run_fields(rows):
    row = rows[0]
    assert row["task_id"] == "cpu:fixed SHA-256 CPU task:10"
    assert (row["source"], row["workload_iterations"], row["worker_capacity"]) == (
        "measured",
        10,
        1,
    )


def test_unavailable_readings_stay_null_not_zero(rows):
    row = rows[0]
    assert row["cpu_pressure_max_avg10"] is None and row["cpu_pressure_available"] is False
    assert row["cgroup_throttled_usec_delta"] is None and row["cgroup_available"] is False


def test_sample_error_text_never_reaches_rows():
    run = validate(make_result(), [SAMPLE | {"unavailable": ["/proc/pressure/cpu missing"]}], "cpu")
    assert "/proc" not in json.dumps(flatten_run(run), default=str)


def test_window_includes_only_samples_inside_the_step():
    samples = [
        sample(-1, avg10=90.0, throttled=0),
        sample(0, avg10=2.0, throttled=100),
        sample(500, avg10=7.5, throttled=400),
        sample(2_000_000_000, avg10=99.0, throttled=900),
    ]
    readings = window_readings(samples, BASE_NS, BASE_NS + 1_000_000_000)
    assert readings == {
        "cpu_pressure_max_avg10": 7.5,
        "cpu_pressure_available": True,
        "cgroup_throttled_usec_delta": 300,
        "cgroup_available": True,
        "telemetry_sample_count": 2,
        "telemetry_coverage": None,
    }


def test_counter_reset_is_not_a_measurement():
    samples = [sample(0, throttled=500), sample(10, throttled=20)]
    readings = window_readings(samples, BASE_NS, BASE_NS + 100)
    assert readings["cgroup_throttled_usec_delta"] is None
    assert readings["cgroup_available"] is False


def test_measured_zero_is_kept():
    samples = [sample(0, avg10=0.0, throttled=50), sample(10, avg10=0.0, throttled=50)]
    readings = window_readings(samples, BASE_NS, BASE_NS + 100)
    assert readings["cpu_pressure_max_avg10"] == 0.0 and readings["cpu_pressure_available"]
    assert readings["cgroup_throttled_usec_delta"] == 0 and readings["cgroup_available"]


def test_intermediate_reset_is_not_a_measurement():
    samples = [sample(0, throttled=100), sample(10, throttled=10), sample(20, throttled=200)]
    readings = window_readings(samples, BASE_NS, BASE_NS + 100)
    assert readings["cgroup_throttled_usec_delta"] is None
    assert readings["cgroup_available"] is False


def test_counters_are_compared_in_timestamp_order():
    samples = [sample(20, throttled=300), sample(0, throttled=100), sample(10, throttled=150)]
    readings = window_readings(samples, BASE_NS, BASE_NS + 100)
    assert readings["cgroup_throttled_usec_delta"] == 200


def test_missing_counter_in_window_is_not_a_measurement():
    samples = [sample(0, throttled=100), sample(10, avg10=1.0), sample(20, throttled=200)]
    samples[1]["cgroup_cpu_stat"] = {"usage_usec": 5}
    readings = window_readings(samples, BASE_NS, BASE_NS + 100)
    assert readings["cgroup_throttled_usec_delta"] is None


def test_coerced_numeric_strings_flatten_with_contract_types():
    result = make_result()
    result["clean"]["submitted_ns"] = str(result["clean"]["submitted_ns"])
    result["experiment"]["iterations"] = "10"
    rows = flatten_run(validate(result, [SAMPLE | {"timestamp_ns": str(BASE_NS)}], "cpu"))
    assert rows[0]["workload_iterations"] == 10
    assert rows[0]["telemetry_sample_count"] == 1


# Conditions come from the result structure


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda r: r["contended"].update(step_id="baseline-9"),
            "'baseline-9' is recorded in the contended",
        ),
        (
            lambda r: (
                r["clean"].update(step_id="contended"),
                r["contended"].update(step_id="clean"),
            ),
            "clean position",
        ),
        (lambda r: r["baseline"]["steps"][0].update(step_id="warmup-0"), "baseline position"),
        (lambda r: r["baseline"]["steps"][0].update(step_id="baseline-x"), "baseline position"),
    ],
)
def test_step_ids_must_match_their_position(change, message):
    result = make_result()
    change(result)
    with pytest.raises(FlattenError, match=message):
        flatten_run(validate(result, [SAMPLE], "cpu"))


# Counter chains


@pytest.mark.parametrize(
    "middle",
    [None, {}, {"usage_usec": 3}],
    ids=["unreadable", "empty", "no-throttle-key"],
)
def test_any_gap_breaks_the_counter_chain(middle):
    samples = [sample(0, throttled=1), sample(10), sample(20, throttled=9)]
    samples[1]["cgroup_cpu_stat"] = middle
    readings = window_readings(samples, BASE_NS, BASE_NS + 100)
    assert readings["cgroup_throttled_usec_delta"] is None
    assert readings["cgroup_available"] is False


# Telemetry coverage and status


def every(interval_ns, count, start=0):
    return [sample(start + i * interval_ns, avg10=1.0) for i in range(count)]


def test_sampling_interval_is_the_median_gap():
    samples = every(250_000_000, 4) + [sample(10_000_000_000, avg10=1.0)]
    assert sampling_interval_seconds(samples) == 0.25
    assert sampling_interval_seconds(every(250_000_000, 1)) is None


def test_coverage_compares_received_to_expected_samples():
    full = window_readings(every(250_000_000, 5), BASE_NS, BASE_NS + 1_000_000_000, 0.25)
    assert full["telemetry_coverage"] == 1.0
    sparse = window_readings(every(500_000_000, 2), BASE_NS, BASE_NS + 1_000_000_000, 0.25)
    assert sparse["telemetry_coverage"] == 0.5
    assert window_readings([], BASE_NS, BASE_NS + 1_000_000_000, 0.25)["telemetry_coverage"] == 0
    assert window_readings([], BASE_NS, BASE_NS + 1_000_000_000)["telemetry_coverage"] is None


def test_samples_outside_every_step_mean_no_overlap(rows):
    assert {row["telemetry_status"] for row in rows} == {"no_overlap"}
    assert {row["telemetry_coverage"] for row in rows} == {None}


@pytest.mark.parametrize(
    ("samples", "status"), [([], "empty"), (every(250_000_000, 40), "present")]
)
def test_run_telemetry_status(samples, status):
    rows = flatten_run(validate(make_result(), samples, "cpu"))
    assert {row["telemetry_status"] for row in rows} == {status}


def test_missing_samples_file_is_reported_as_missing():
    run = validate(make_result(), [], "cpu", telemetry_status="missing")
    assert {row["telemetry_status"] for row in flatten_run(run)} == {"missing"}


# Run-level lineage


def test_passing_run_lineage(rows):
    row = rows[0]
    assert (row["run_passed"], row["failed_checks"], row["agent_model"]) == (True, None, None)
    assert row["schema_version"] == "steps-v1.1"


def test_failed_checks_are_recorded_in_order():
    result = make_result()
    result["checks"] = {"stable_baseline": False, "queue_increased": False, "passed": False}
    row = flatten_run(validate(result, [SAMPLE], "cpu"))[0]
    assert (row["run_passed"], row["failed_checks"]) == (False, "queue_increased,stable_baseline")


@pytest.mark.parametrize(
    ("agent", "model"),
    [({"model": "qwen3:1.7b"}, "qwen3:1.7b"), ({"model": " "}, None), ({}, None), ("x", None)],
)
def test_agent_model_identifies_llm_runs(agent, model):
    result = make_result() | {"agent": agent}
    assert flatten_run(validate(result, [SAMPLE], "coding"))[0]["agent_model"] == model


def between_steps():
    """Samples after baseline-0 finishes (+1 s) and before baseline-1 is submitted (+2 s)."""
    return [sample(offset, avg10=1.0) for offset in (1_250_000_000, 1_500_000_000, 1_750_000_000)]


def test_samples_only_between_steps_mean_no_overlap():
    rows = flatten_run(validate(make_result(), between_steps(), "cpu"))
    assert {row["telemetry_status"] for row in rows} == {"no_overlap"}
    assert {row["telemetry_coverage"] for row in rows} == {None}
    assert {row["telemetry_sample_count"] for row in rows} == {0}


def test_one_sample_inside_one_step_means_present():
    inside_clean = sample(6_500_000_000, avg10=1.0)  # clean runs from +6 s to +7.000999 s
    rows = flatten_run(validate(make_result(), between_steps() + [inside_clean], "cpu"))
    assert {row["telemetry_status"] for row in rows} == {"present"}
    assert [row["telemetry_sample_count"] for row in rows] == [0, 0, 0, 1, 0]
    assert all(row["telemetry_coverage"] is not None for row in rows)
