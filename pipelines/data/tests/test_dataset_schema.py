from datetime import UTC, datetime
from pathlib import Path

import pytest
from agentwatch_data.extract import TELEMETRY_EMPTY, TELEMETRY_MISSING, TELEMETRY_PRESENT
from agentwatch_data.flatten import NO_OVERLAP
from agentwatch_data.schema import (
    COLUMNS,
    LEAKY_COLUMNS,
    RELEASE_COLUMNS,
    TELEMETRY_STATUSES,
    Role,
    SchemaError,
    feature_columns,
    validate_rows,
)
from test_dataset_end_to_end import build_rows

DATA_CARD = Path(__file__).resolve().parents[3] / "docs" / "data-card.md"


def test_column_names_are_unique():
    names = [column.name for column in COLUMNS]
    assert len(names) == len(set(names))


def test_every_column_is_documented():
    assert all(column.description.strip() for column in COLUMNS)
    card = DATA_CARD.read_text(encoding="utf-8")
    missing = [column.name for column in COLUMNS if f"`{column.name}`" not in card]
    assert not missing, f"Columns missing from docs/data-card.md: {missing}"


def test_leaky_columns_are_never_features():
    names = {column.name for column in COLUMNS}
    assert LEAKY_COLUMNS <= names
    assert not LEAKY_COLUMNS & set(feature_columns())


def test_labels_and_rule_output_are_not_features():
    features = set(feature_columns())
    assert not features & {c.name for c in COLUMNS if c.role in (Role.LABEL, Role.RULE)}


def test_optional_readings_have_availability_flags():
    names = {column.name for column in COLUMNS}
    assert {"cpu_pressure_available", "cgroup_available"} <= names
    nullable = {column.name for column in COLUMNS if column.nullable}
    assert {"cpu_pressure_max_avg10", "cgroup_throttled_usec_delta"} <= nullable


# Row validation


@pytest.fixture(scope="module")
def valid_row():
    return build_rows()[-1]


def test_stage_status_values_match_the_schema():
    produced = {TELEMETRY_PRESENT, TELEMETRY_EMPTY, TELEMETRY_MISSING, NO_OVERLAP}
    assert produced == set(TELEMETRY_STATUSES)


def test_release_columns_are_required_when_requested(valid_row):
    every = frozenset(column.name for column in COLUMNS)
    with pytest.raises(SchemaError, match=r"missing \['dataset_version'\]"):
        validate_rows([valid_row], every)
    validate_rows([valid_row | {"dataset_version": "v1"}], every)
    assert RELEASE_COLUMNS == {"dataset_version"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"queue_fraction": 1.5}, "queue_fraction must be at most 1"),
        ({"queue_seconds": -0.1}, "queue_seconds must be at least 0"),
        ({"cpu_seconds": float("nan")}, "cpu_seconds must be finite"),
        ({"workload_iterations": True}, "workload_iterations must be a number"),
        ({"workload_iterations": 2**70}, "outside the int64 range"),
        ({"workload_iterations": 1.5}, "workload_iterations must be an integer"),
        ({"tests_per_step": -3}, "tests_per_step must be at least 0"),
        ({"worker_capacity": 0}, "worker_capacity must be at least 1"),
        ({"cgroup_throttled_usec_delta": 2**64}, "outside the int64 range"),
        ({"cpu_pressure_max_avg10": 150.0}, "at most 100"),
        ({"telemetry_coverage": 1.2}, "telemetry_coverage must be at most 1"),
        ({"label_resource": "gpu"}, "label_resource must be one of"),
        ({"rule_prediction": "maybe"}, "rule_prediction must be one of"),
        ({"condition": "mixed"}, "condition must be one of"),
        ({"telemetry_status": "partial"}, "telemetry_status must be one of"),
        ({"label_basis": "guess"}, "label_basis must be one of"),
        ({"kind": "gpu"}, "kind must be one of"),
        ({"source": "scraped"}, "source must be one of"),
        ({"run_id": None}, "run_id is null"),
        ({"workload": "  "}, "workload must be a nonempty string"),
        ({"cpu_pressure_available": 1}, "cpu_pressure_available must be a boolean"),
        # A naive datetime is the invalid value under test.
        ({"recorded_at": datetime(2026, 10, 5)}, "timezone-aware"),  # noqa: DTZ001
        ({"recorded_at": "2026-10-05T00:00:00Z"}, "timezone-aware"),
    ],
)
def test_invalid_values_are_rejected_with_column_and_step(valid_row, change, message):
    with pytest.raises(SchemaError, match=message) as raised:
        validate_rows([valid_row | change])
    assert "(contended)" in str(raised.value)


def test_nullable_columns_accept_null(valid_row):
    nullable = {column.name: None for column in COLUMNS if column.nullable}
    validate_rows([valid_row | nullable])


@pytest.mark.parametrize("change", [{"extra": 1}, "drop"])
def test_rows_must_have_exactly_the_schema_columns(valid_row, change):
    row = dict(valid_row)
    if change == "drop":
        row.pop("telemetry_coverage")
    else:
        row |= change
    with pytest.raises(SchemaError, match="missing|unexpected"):
        validate_rows([row])


def test_timezone_aware_saved_time_is_valid(valid_row):
    validate_rows([valid_row | {"recorded_at": datetime(2026, 10, 9, tzinfo=UTC)}])
