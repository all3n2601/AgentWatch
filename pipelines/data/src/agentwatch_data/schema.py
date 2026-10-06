"""Step dataset v1 columns: one row per measured agent step.

Only FEATURE columns may be model inputs. The other roles keep rows traceable and
labels reproducible without leaking experiment design into predictions.
"""

from dataclasses import dataclass
from enum import StrEnum

SCHEMA_VERSION = "steps-v1"

RESOURCE_CLASSES = ("none", "cpu_sandbox", "inference", "retrieval")
SOURCES = ("measured", "simulated", "rcaeval")


class Role(StrEnum):
    ID = "id"  # joins, grouping, and split keys
    FEATURE = "feature"  # observable at prediction time; the only model inputs
    LABEL = "label"  # training targets
    RULE = "rule"  # rule baseline output, kept for comparison
    LINEAGE = "lineage"  # provenance and label evidence


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # Arrow type name, applied when the Parquet file is written
    role: Role
    description: str
    nullable: bool = False


COLUMNS = (
    # Identity
    Column("run_id", "string", Role.ID, "Experiment run UUID"),
    Column("step_id", "string", Role.ID, "Step name within the run; reveals the condition"),
    Column("task_id", "string", Role.ID, "Task identity; splits are grouped by this key"),
    Column("kind", "string", Role.ID, "Experiment kind recorded by the ingestion API"),
    Column("source", "string", Role.ID, f"Data origin: {', '.join(SOURCES)}"),
    Column("recorded_at", "timestamp[us, tz=UTC]", Role.ID, "When the run was saved", True),
    # Timing features, from worker enqueue/start/finish timestamps
    Column("queue_seconds", "float64", Role.FEATURE, "Measured wait before a worker started"),
    Column("execution_seconds", "float64", Role.FEATURE, "Worker start to finish wall time"),
    Column("duration_seconds", "float64", Role.FEATURE, "Submission to finish wall time"),
    Column("cpu_seconds", "float64", Role.FEATURE, "Process and child CPU time"),
    Column("queue_fraction", "float64", Role.FEATURE, "queue_seconds / duration_seconds"),
    # Workload features
    Column("workload_iterations", "int64", Role.FEATURE, "Fixed work size of the step"),
    Column("tests_per_step", "int64", Role.FEATURE, "Tests executed by the tool step", True),
    Column("worker_capacity", "int64", Role.FEATURE, "Sandbox workers available"),
    # Resource-window features; missing readings stay null, never zero
    Column(
        "cpu_pressure_max_avg10",
        "float64",
        Role.FEATURE,
        "Max host CPU PSI some avg10 (%) during the step",
        True,
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
    ),
    Column("cgroup_available", "bool", Role.FEATURE, "Whether a throttling delta was measured"),
    Column("telemetry_sample_count", "int64", Role.FEATURE, "Resource samples inside the window"),
    # Targets
    Column(
        "label_resource",
        "string",
        Role.LABEL,
        f"Resource that caused excess waiting: {', '.join(RESOURCE_CLASSES)}",
    ),
    Column(
        "label_excess_fraction",
        "float64",
        Role.LABEL,
        "max(0, min(1, (T - B) / T)) against the clean baseline mean B",
    ),
    # Rule baseline output
    Column("rule_prediction", "string", Role.RULE, "Resource named by the threshold rule"),
    Column("rule_version", "string", Role.RULE, "Version of the rule that produced it"),
    # Provenance and label evidence
    Column("condition", "string", Role.LINEAGE, "Experiment condition: baseline, clean, contended"),
    Column("workload", "string", Role.LINEAGE, "Human-readable workload description"),
    Column("baseline_mean_seconds", "float64", Role.LINEAGE, "Clean baseline mean duration B"),
    Column("noise_threshold_seconds", "float64", Role.LINEAGE, "Frozen slowdown threshold"),
    Column("label_version", "string", Role.LINEAGE, "Version of the labeling rules"),
    Column("dataset_version", "string", Role.LINEAGE, "Dataset release this row belongs to"),
    Column("source_sha256", "string", Role.LINEAGE, "Hash of the source evidence file"),
)

# Columns that encode the experiment design or its answer; never model inputs.
LEAKY_COLUMNS = frozenset({"step_id", "condition", "rule_prediction", "baseline_mean_seconds"})


def columns_with(role: Role) -> list[str]:
    return [column.name for column in COLUMNS if column.role == role]


def feature_columns() -> list[str]:
    return columns_with(Role.FEATURE)
