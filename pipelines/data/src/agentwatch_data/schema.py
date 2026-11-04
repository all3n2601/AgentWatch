"""Step dataset columns: one row per measured agent step.

Only FEATURE columns may be model inputs. The other roles keep rows traceable and
labels reproducible without leaking experiment design into predictions.

steps-v1.1 and steps-v1.2 add columns to steps-v1 without changing any existing column, so
v1 readers remain valid for the columns they know.

REPLAY_LABEL_COLUMNS describes the rows of replay-experiment labels (replay_labels.py). It is
not part of the release schema: COLUMNS, contract() and schema.json are unchanged, and D03 adds
these columns with steps-v1.3.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

SCHEMA_VERSION = "steps-v1.2"

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
# Replay-experiment labels (labels-v2). `mixed` is a label value only: it is not a
# resource class, so rule predictions and the model package keep RESOURCE_CLASSES.
MIXED = "mixed"
LABEL_RESOURCE_VALUES = (*RESOURCE_CLASSES, MIXED)
REPLAY_CONDITIONS = ("clean", "cpu", "inference", "retrieval", "mixed")
REPLAY_ROLES = ("baseline", "contended", "clean_control")
REPLAY_LABEL_BASES = (
    "baseline_reference",
    "clean_within_noise",
    "clean_exceeded_noise",
    "contention_below_threshold",
    "contention_exceeded_threshold",
    "mixed_below_threshold",
    "mixed_confirmed",
    "mixed_partial",
    "mixed_joint_only",
)
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
FLOAT64_EXACT_INTEGER = 2**53


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
    Column(
        "sampling_interval_seconds",
        "float64",
        Role.LINEAGE,
        "Sampler interval declared by the run; telemetry_coverage is judged against it",
        True,
        minimum=0,
    ),
    Column(
        "measured_sampling_interval_seconds",
        "float64",
        Role.LINEAGE,
        "Median gap between the run's samples, for comparison with the declared interval",
        True,
        minimum=0,
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

# Replay label rows (replay_labels.ROW_KEYS, in that order). A name shared with COLUMNS reuses
# its definition unless the replay row's values differ; the replay variants below widen
# label_resource to include `mixed` and extend condition and label_basis. Every column here
# is an ID, LABEL or LINEAGE except the three shared timings, which keep their release FEATURE
# role; the experiment design (condition, role, injection, label metadata) is never a feature.
REPLAY_LABEL_COLUMNS = (
    Column("experiment", "string", Role.ID, "Experiment name; replays are grouped by it"),
    COLUMNS_BY_NAME["task_id"],
    Column("task_group", "string", Role.ID, "Task family; splits keep related tasks together"),
    Column("plan_sha256", "string", Role.ID, "SHA-256 of the experiment plan the replays ran"),
    COLUMNS_BY_NAME["agent_model"],
    COLUMNS_BY_NAME["run_id"],
    Column("replay_index", "int64", Role.ID, "Position of the replay in the experiment", minimum=0),
    Column(
        "role",
        "string",
        Role.LINEAGE,
        "Replay role: baseline, contended, clean_control",
        allowed=REPLAY_ROLES,
    ),
    Column(
        "condition",
        "string",
        Role.LINEAGE,
        f"Injected condition: {', '.join(REPLAY_CONDITIONS)}",
        allowed=REPLAY_CONDITIONS,
    ),
    COLUMNS_BY_NAME["step_id"],
    Column("step_type", "string", Role.ID, "Kind of step recorded by the agent run"),
    COLUMNS_BY_NAME["duration_seconds"],
    COLUMNS_BY_NAME["queue_seconds"],
    COLUMNS_BY_NAME["execution_seconds"],
    Column(
        "label_resource",
        "string",
        Role.LABEL,
        f"Resource that caused excess waiting: {', '.join(LABEL_RESOURCE_VALUES)}",
        allowed=LABEL_RESOURCE_VALUES,
    ),
    Column(
        "label_resources",
        "string",
        Role.LABEL,
        "Sorted comma-separated confirmed contended resources; the empty string means none. "
        "D03 must allow the empty string when merging this into steps-v1.3: the release "
        "value check rejects empty strings",
    ),
    COLUMNS_BY_NAME["label_excess_fraction"],
    Column(
        "label_basis",
        "string",
        Role.LINEAGE,
        "Why the step received its label",
        allowed=REPLAY_LABEL_BASES,
    ),
    COLUMNS_BY_NAME["injection_method"],
    COLUMNS_BY_NAME["baseline_mean_seconds"],
    COLUMNS_BY_NAME["noise_threshold_seconds"],
    Column(
        "control_exceeded_noise",
        "bool",
        Role.LINEAGE,
        "Whether the clean control exceeded noise on this step; null when no usable control",
        True,
    ),
    COLUMNS_BY_NAME["failed_checks"],
    COLUMNS_BY_NAME["label_version"],
)

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


def contract() -> dict:
    """The column contract as JSON-ready data, shipped with every release as schema.json.

    Tools that cannot import this package (the TFDV quality gate runs in its own
    environment) read the contract from the release instead.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "role": str(column.role),
                "nullable": column.nullable,
                "allowed": list(column.allowed) if column.allowed is not None else None,
                "minimum": column.minimum,
                "maximum": column.maximum,
            }
            for column in COLUMNS
        ],
    }


def columns_from(document: dict) -> tuple[Column, ...]:
    """The columns a contract() document describes (descriptions are not shipped)."""
    return tuple(
        Column(
            entry["name"],
            entry["dtype"],
            Role(entry["role"]),
            "",
            entry["nullable"],
            tuple(entry["allowed"]) if entry["allowed"] is not None else None,
            entry["minimum"],
            entry["maximum"],
        )
        for entry in document["columns"]
    )


def columns_with(role: Role) -> list[str]:
    return [column.name for column in COLUMNS if column.role == role]


def feature_columns() -> list[str]:
    return columns_with(Role.FEATURE)


def finite_number(value) -> float | None:
    """value as a finite float64, or None when it is not one.

    Booleans are not numbers here, and an integer too large for float64 (JSON and TOML both
    allow arbitrarily large integers) is rejected instead of raising OverflowError.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def brief(value) -> str:
    """A short repr for error messages; a huge integer is described, not printed."""
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 10**18:
        return f"an integer with {len(str(abs(value)))} digits"
    return repr(value)


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
    else:
        try:
            finite = math.isfinite(value)
        except OverflowError:  # an integer too large to be a float64
            return f"is too large for float64 ({len(str(abs(value)))}-digit integer)"
        if not finite:
            return f"must be finite, got {value!r}"
        # Arrow refuses to store an integer in a float64 column unless it converts exactly.
        if isinstance(value, int) and abs(value) > FLOAT64_EXACT_INTEGER:
            return f"is an integer float64 cannot hold exactly: {brief(value)}"
    if column.minimum is not None and value < column.minimum:
        return f"must be at least {column.minimum}, got {value}"
    if column.maximum is not None and value > column.maximum:
        return f"must be at most {column.maximum}, got {value}"
    return None


def validate_rows(
    rows: list[dict], expected: frozenset[str] | None = None, columns: tuple[Column, ...] = COLUMNS
) -> None:
    """Require every row to have exactly the expected columns, each with a valid value.

    By default the expected columns are the full schema minus release-only columns, which
    is what the per-run stages produce. columns defaults to the current contract; a release
    built under an earlier one is checked against that one.
    """
    by_name = {column.name: column for column in columns}
    expected = expected or frozenset(by_name) - RELEASE_COLUMNS
    for index, row in enumerate(rows):
        missing, extra = expected - row.keys(), row.keys() - expected
        if missing or extra:
            raise SchemaError(f"Row {index}: missing {sorted(missing)}, unexpected {sorted(extra)}")
        for name in expected:
            problem = check_value(by_name[name], row[name])
            if problem:
                raise SchemaError(f"Row {index} ({row.get('step_id')}): {name} {problem}")
