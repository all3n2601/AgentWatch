import json

import pytest
from agentwatch_data.extract import validate
from agentwatch_data.flatten import flatten_run, window_readings
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
