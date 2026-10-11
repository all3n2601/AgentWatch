"""Step dataset columns: one row per measured agent step.

Only FEATURE columns may be model inputs. The other roles keep rows traceable and
labels reproducible without leaking experiment design into predictions.

steps-v1.1 adds columns to steps-v1 without changing any existing column, so v1 readers
remain valid for the columns they know.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

SCHEMA_VERSION = "steps-v1.1"

RESOURCE_CLASSES = ("none", "cpu_sandbox", "inference", "retrieval")
SOURCES = ("measured", "simulated", "rcaeval")
KINDS = ("cpu", "coding")
CONDITIONS = ("baseline", "clean", "contended")
TELEMETRY_STATUSES = ("present", "empty", "missing", "no_overlap")
LABEL_BASES = (
    "baseline_reference",
    "clean_within_noise",
    "clean_exceeded_noise",
    "contention_exceeded_threshold",
    "contention_below_threshold",
)
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


class Role(StrEnum):
    ID = "id"  # joins, grouping, and split keys
    FEATURE = "feature"  # observable at prediction time; the only model inputs
    LABEL = "label"  # training targets
    RULE = "rule"  # rule baseline output, kept for comparison
    LINEAGE = "lineage"  # provenance and label evidence


class SchemaError(ValueError):
    """A row that does not satisfy the dataset schema."""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # Arrow type name, applied when the Parquet file is written
    role: Role
    description: str
    nullable: bool = False
    allowed: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None


def seconds(name: str, role: Role, description: str) -> Column:
    return Column(name, "float64", role, description, minimum=0)


def fraction(name: str, role: Role, description: str, nullable: bool = False) -> Column:
    return Column(name, "float64", role, description, nullable, minimum=0, maximum=1)


COLUMNS = (
    # Identity
    Column("run_id", "string", Role.ID, "Experiment run UUID"),
    Column("step_id", "string", Role.ID, "Step name within the run; reveals the condition"),
    Column("task_id", "string", Role.ID, "Task identity; splits are grouped by this key"),
    Column(
        "kind", "string", Role.ID, "Experiment kind recorded by the ingestion API", allowed=KINDS
    ),
    Column("source", "string", Role.ID, f"Data origin: {', '.join(SOURCES)}", allowed=SOURCES),
    Column("recorded_at", "timestamp[us, tz=UTC]", Role.ID, "When the run was saved", True),
    # Timing features, from worker enqueue/start/finish timestamps
    seconds("queue_seconds", Role.FEATURE, "Measured wait before a worker started"),
    seconds("execution_seconds", Role.FEATURE, "Worker start to finish wall time"),
    seconds("duration_seconds", Role.FEATURE, "Submission to finish wall time"),
    seconds("cpu_seconds", Role.FEATURE, "Process and child CPU time"),
    fraction("queue_fraction", Role.FEATURE, "queue_seconds / duration_seconds"),
    # Workload features
    Column("workload_iterations", "int64", Role.FEATURE, "Fixed work size of the step", minimum=1),
    Column(
        "tests_per_step", "int64", Role.FEATURE, "Tests executed by the tool step", True, minimum=0
    ),
    Column("worker_capacity", "int64", Role.FEATURE, "Sandbox workers available", minimum=1),
    # Resource-window features; missing readings stay null, never zero
    Column(
        "cpu_pressure_max_avg10",
        "float64",
        Role.FEATURE,
        "Max host CPU PSI some avg10 (%) during the step",
        True,
        minimum=0,
        maximum=100,
    ),
    Column(
        "cpu_pressure_available", "bool", Role.FEATURE, "Whether PSI was readable in the window"
    ),
    Column(
        "cgroup_throttled_usec_delta",
        "int64",
        Role.FEATURE,
        "cgroup v2 throttled time gained during the step",
        True,
        minimum=0,
    ),
    Column("cgroup_available", "bool", Role.FEATURE, "Whether a throttling delta was measured"),
    Column(
        "telemetry_sample_count",
        "int64",
        Role.FEATURE,
        "Resource samples inside the window",
        minimum=0,
    ),
    fraction(
        "telemetry_coverage",
        Role.FEATURE,
        "Samples received / samples expected at the run's measured sampling interval",
        nullable=True,
    ),
    # Targets
    Column(
        "label_resource",
        "string",
        Role.LABEL,
        f"Resource that caused excess waiting: {', '.join(RESOURCE_CLASSES)}",
        allowed=RESOURCE_CLASSES,
    ),
    fraction(
        "label_excess_fraction",
        Role.LABEL,
        "max(0, min(1, (T - B) / T)) against the clean baseline mean B",
    ),
    # Rule baseline output
    Column(
        "rule_prediction",
        "string",
        Role.RULE,
        "Resource named by the threshold rule",
        allowed=RESOURCE_CLASSES,
    ),
    Column("rule_version", "string", Role.RULE, "Version of the rule that produced it"),
    # Provenance and label evidence
    Column(
        "condition",
        "string",
        Role.LINEAGE,
        "Experiment condition: baseline, clean, contended",
        allowed=CONDITIONS,
    ),
    Column("workload", "string", Role.LINEAGE, "Human-readable workload description"),
    Column(
        "injection_method",
        "string",
        Role.LINEAGE,
        "How contention was applied to the step, or none",
    ),
    Column("agent_model", "string", Role.LINEAGE, "Model that invoked the tool, if any", True),
    Column("run_passed", "bool", Role.LINEAGE, "Whether every experiment check passed"),
    Column(
        "failed_checks", "string", Role.LINEAGE, "Comma-separated failed experiment checks", True
    ),
    Column(
        "telemetry_status",
        "string",
        Role.LINEAGE,
        "Run telemetry: present, empty, missing, or no_overlap",
        allowed=TELEMETRY_STATUSES,
    ),
    seconds("baseline_mean_seconds", Role.LINEAGE, "Clean baseline mean duration B"),
    seconds("noise_threshold_seconds", Role.LINEAGE, "Frozen slowdown threshold"),
    Column(
        "label_basis",
        "string",
        Role.LINEAGE,
        "Why the step received its label",
        allowed=LABEL_BASES,
    ),
    Column("label_version", "string", Role.LINEAGE, "Labeling rules version and fingerprint"),
    Column("schema_version", "string", Role.LINEAGE, "Dataset schema version of the row"),
    Column("dataset_version", "string", Role.LINEAGE, "Dataset release this row belongs to"),
    Column("source_sha256", "string", Role.LINEAGE, "SHA-256 of the run's results and samples"),
)

COLUMNS_BY_NAME = {column.name: column for column in COLUMNS}

# Columns that encode the experiment design or its answer; never model inputs.
LEAKY_COLUMNS = frozenset(
    {
        "step_id",
        "condition",
        "rule_prediction",
        "baseline_mean_seconds",
        "label_basis",
        "injection_method",
        "failed_checks",
        "run_passed",
    }
)

# Added by the release build, not by the per-run stages.
RELEASE_COLUMNS = frozenset({"dataset_version"})


def columns_with(role: Role) -> list[str]:
    return [column.name for column in COLUMNS if column.role == role]


def feature_columns() -> list[str]:
    return columns_with(Role.FEATURE)


def check_value(column: Column, value) -> str | None:
    """Return why value violates the column, or None when it is valid."""
    if value is None:
        return None if column.nullable else "is null"
    if column.dtype == "string":
        if not isinstance(value, str) or not value.strip():
            return f"must be a nonempty string, got {value!r}"
        if column.allowed is not None and value not in column.allowed:
            return f"must be one of {', '.join(column.allowed)}, got {value!r}"
        return None
    if column.dtype == "bool":
        return None if isinstance(value, bool) else f"must be a boolean, got {value!r}"
    if column.dtype.startswith("timestamp"):
        if not isinstance(value, datetime) or value.tzinfo is None:
            return f"must be a timezone-aware datetime, got {value!r}"
        return None
    # Numeric columns. bool is an int subclass, so it is rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, int | float):
        return f"must be a number, got {value!r}"
    if column.dtype == "int64":
        if not isinstance(value, int):
            return f"must be an integer, got {value!r}"
        if not INT64_MIN <= value <= INT64_MAX:
            return f"is outside the int64 range: {value}"
    elif not math.isfinite(value):
        return f"must be finite, got {value!r}"
    if column.minimum is not None and value < column.minimum:
        return f"must be at least {column.minimum}, got {value}"
    if column.maximum is not None and value > column.maximum:
        return f"must be at most {column.maximum}, got {value}"
    return None


def validate_rows(rows: list[dict], expected: frozenset[str] | None = None) -> None:
    """Require every row to have exactly the expected columns, each with a valid value.

    By default the expected columns are the full schema minus release-only columns, which
    is what the per-run stages produce.
    """
    expected = expected or frozenset(COLUMNS_BY_NAME) - RELEASE_COLUMNS
    for index, row in enumerate(rows):
        missing, extra = expected - row.keys(), row.keys() - expected
        if missing or extra:
            raise SchemaError(f"Row {index}: missing {sorted(missing)}, unexpected {sorted(extra)}")
        for name in expected:
            problem = check_value(COLUMNS_BY_NAME[name], row[name])
            if problem:
                raise SchemaError(f"Row {index} ({row.get('step_id')}): {name} {problem}")
