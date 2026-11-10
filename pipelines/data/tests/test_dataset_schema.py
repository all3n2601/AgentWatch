import json
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import pytest
from agentwatch_data.extract import TELEMETRY_EMPTY, TELEMETRY_MISSING, TELEMETRY_PRESENT
from agentwatch_data.files import json_text
from agentwatch_data.flatten import NO_OVERLAP
from agentwatch_data.schema import (
    COLUMNS,
    COLUMNS_BY_NAME,
    KINDS,
    LABEL_BASES,
    LABEL_RESOURCE_VALUES,
    LEAKY_COLUMNS,
    RELEASE_COLUMNS,
    REPLAY_CONDITIONS,
    REPLAY_LABEL_BASES,
    REPLAY_ROLES,
    SCHEMA_VERSION,
    TELEMETRY_STATUSES,
    Column,
    Role,
    SchemaError,
    check_value,
    columns_from,
    contract,
    feature_columns,
    finite_number,
    validate_rows,
)
from agentwatch_features.live import TelemetryStatus
from test_dataset_end_to_end import build_rows

DATA_CARD = Path(__file__).resolve().parents[3] / "docs" / "data-card.md"
CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "src" / "agentwatch_data" / "contracts"


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
    # Replay rows carry the step-features-v1 status, which adds unaligned.
    assert produced | set(get_args(TelemetryStatus)) == set(TELEMETRY_STATUSES)


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
        ({"cpu_seconds": 10**400}, r"too large for float64 \(401-digit integer\)"),
        ({"cpu_seconds": 2**53 + 1}, "float64 cannot hold exactly"),  # Arrow would refuse it
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.25, 0.25), (3, 3.0), (True, None), ("1", None), (float("nan"), None), (10**400, None)],
)
def test_finite_number_never_raises(value, expected):
    assert finite_number(value) == expected


# The TFDV quality gate (pipelines/validation) runs in its own environment, so these checks
# keep its committed inputs in step with this contract without importing TensorFlow.

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def test_frozen_tfdv_schema_lists_exactly_the_contract_columns():
    import re

    frozen = (VALIDATION / "schema" / f"{SCHEMA_VERSION}.pbtxt").read_text(encoding="utf-8")
    assert frozen.startswith(f"# Frozen TFDV schema for {SCHEMA_VERSION}.")
    names = re.findall(r'^feature \{\n  name: "([^"]+)"', frozen, flags=re.MULTILINE)
    assert sorted(names) == sorted(column.name for column in COLUMNS), (
        f"Refresh pipelines/validation/schema/{SCHEMA_VERSION}.pbtxt with freeze-schema after a "
        "schema change"
    )


def test_validation_fixture_carries_the_current_contract():
    [fixture] = sorted((VALIDATION / "tests" / "fixtures").iterdir())
    shipped = json.loads((fixture / "schema.json").read_text(encoding="utf-8"))
    assert shipped == contract(), (
        "Rebuild pipelines/validation/tests/fixtures from the golden run after a schema change"
    )


def test_the_current_contract_is_frozen_for_integrity_checks():
    """Integrity holds every release to the contract of its schema version, read from
    agentwatch_data/contracts. Changing the contract means bumping SCHEMA_VERSION and
    freezing the new contract beside the old ones, which must never change."""
    from agentwatch_data.build import contract_text
    from agentwatch_data.integrity import CONTRACTS
    from agentwatch_data.schema import columns_from

    assert CONTRACTS.get(SCHEMA_VERSION) == contract_text(), (
        f"Freeze the new contract as agentwatch_data/contracts/{SCHEMA_VERSION}.json and keep "
        "the old ones; changing a column without a new schema version is not allowed"
    )
    assert columns_from(contract()) == tuple(
        Column(
            c.name, c.dtype, c.role, "", c.nullable, c.allowed, c.minimum, c.maximum, c.allow_empty
        )
        for c in COLUMNS
    )


# steps-v1.3: CPU/coding rows and agent replay rows in one release

V1_2_FEATURES = [
    "queue_seconds",
    "execution_seconds",
    "duration_seconds",
    "cpu_seconds",
    "queue_fraction",
    "workload_iterations",
    "tests_per_step",
    "worker_capacity",
    "cpu_pressure_max_avg10",
    "cpu_pressure_available",
    "cgroup_throttled_usec_delta",
    "cgroup_available",
    "telemetry_sample_count",
    "telemetry_coverage",
]
NEW_IN_V1_3 = [
    "feature_version",
    "task_group",
    "experiment",
    "plan_sha256",
    "replay_index",
    "step_type",
    "replay_role",
    "replay_condition",
    "label_resources",
    "control_exceeded_noise",
    "configuration_sha256",
]


def v1_2_columns() -> tuple[Column, ...]:
    return columns_from(json.loads((CONTRACTS_DIR / "steps-v1.2.json").read_text("utf-8")))


def test_schema_version_is_v1_3():
    assert SCHEMA_VERSION == "steps-v1.3"
    assert contract()["schema_version"] == "steps-v1.3"


def test_empty_string_is_allowed_only_where_declared():
    assert check_value(COLUMNS_BY_NAME["label_resources"], "") is None
    assert check_value(COLUMNS_BY_NAME["label_resources"], "cpu_sandbox,inference") is None
    assert check_value(COLUMNS_BY_NAME["label_resources"], None) is not None  # empty is not null
    assert check_value(COLUMNS_BY_NAME["workload"], "") is not None
    assert [column.name for column in COLUMNS if column.allow_empty] == ["label_resources"]


def test_v1_3_keeps_every_v1_2_column_name_type_and_role():
    old = {column.name: column for column in v1_2_columns()}
    for name, column in old.items():
        new = COLUMNS_BY_NAME[name]
        assert (new.dtype, new.role) == (column.dtype, column.role), name


def test_frozen_v1_2_contract_still_parses_without_allow_empty():
    assert not any(column.allow_empty for column in v1_2_columns())
    assert len(v1_2_columns()) == 40


def test_v1_3_appends_the_new_columns_after_the_v1_2_ones():
    names = [column.name for column in COLUMNS]
    assert names == [column.name for column in v1_2_columns()] + NEW_IN_V1_3


def test_v1_3_column_changes():
    by_name = COLUMNS_BY_NAME
    for name in ("cpu_seconds", "workload_iterations", "worker_capacity"):
        assert by_name[name].nullable, name
    for name in ("rule_prediction", "rule_version"):
        assert by_name[name].nullable, name
    assert KINDS == ("cpu", "coding", "agent_replay")
    assert by_name["kind"].allowed == KINDS
    assert by_name["condition"].allowed == ("baseline", "clean", "contended")
    assert "unaligned" in by_name["telemetry_status"].allowed
    assert by_name["label_resource"].allowed == LABEL_RESOURCE_VALUES
    assert by_name["label_basis"].allowed == tuple(dict.fromkeys(LABEL_BASES + REPLAY_LABEL_BASES))
    # Rule output stays limited to resource classes; mixed is a label value only.
    assert "mixed" not in by_name["rule_prediction"].allowed


@pytest.mark.parametrize(
    ("name", "dtype", "role", "nullable", "allowed"),
    [
        ("feature_version", "string", Role.LINEAGE, False, ("steps-v1.2", "step-features-v1")),
        ("task_group", "string", Role.ID, False, None),
        ("experiment", "string", Role.ID, True, None),
        ("plan_sha256", "string", Role.ID, True, None),
        ("replay_index", "int64", Role.ID, True, None),
        ("step_type", "string", Role.ID, True, None),
        ("replay_role", "string", Role.LINEAGE, True, REPLAY_ROLES),
        ("replay_condition", "string", Role.LINEAGE, True, REPLAY_CONDITIONS),
        ("label_resources", "string", Role.LABEL, False, None),
        ("control_exceeded_noise", "bool", Role.LINEAGE, True, None),
        ("configuration_sha256", "string", Role.LINEAGE, True, None),
    ],
)
def test_new_v1_3_columns(name, dtype, role, nullable, allowed):
    column = COLUMNS_BY_NAME[name]
    assert (column.dtype, column.role, column.nullable, column.allowed) == (
        dtype,
        role,
        nullable,
        allowed,
    )


def test_replay_index_is_never_negative():
    assert check_value(COLUMNS_BY_NAME["replay_index"], 0) is None
    assert check_value(COLUMNS_BY_NAME["replay_index"], -1) is not None


def test_experiment_design_columns_are_leaky():
    design = {"replay_role", "replay_condition", "replay_index", "control_exceeded_noise"}
    assert design | {"experiment"} <= LEAKY_COLUMNS


def test_feature_columns_are_the_fourteen_v1_2_features():
    assert feature_columns() == V1_2_FEATURES
    assert feature_columns() == [c.name for c in v1_2_columns() if c.role == Role.FEATURE]


def test_frozen_contract_matches_contract():
    frozen = (CONTRACTS_DIR / f"{SCHEMA_VERSION}.json").read_text(encoding="utf-8")
    assert frozen == json_text(contract())
    assert all("allow_empty" in column for column in json.loads(frozen)["columns"])
