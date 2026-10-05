from pathlib import Path

from agentwatch_data.schema import COLUMNS, LEAKY_COLUMNS, Role, feature_columns

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
