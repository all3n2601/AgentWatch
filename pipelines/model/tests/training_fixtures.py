"""Synthetic, fully featured step rows for LightGBM training tests.

The rows are made up so each class is learnable from a different feature. They are not
recorded evidence and never appear in a dataset release.
"""

import pytest
from agentwatch_data.schema import feature_columns
from rule_fixtures import row


def require_lightgbm() -> None:
    """Skip the calling test module when LightGBM cannot load (macOS without libomp)."""
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError) as error:
        pytest.skip(
            f"LightGBM unavailable ({type(error).__name__}); on macOS it needs OpenMP (libomp)",
            allow_module_level=True,
        )


def featured_row(label: str, index: int, task_id: str = "task-a", **overrides) -> dict:
    """A row whose features point clearly at its label, with every schema feature present."""
    base = dict.fromkeys(feature_columns())
    base |= {
        "queue_seconds": 0.01,
        "execution_seconds": 0.3,
        "duration_seconds": 0.31,
        "cpu_seconds": 0.25,
        "queue_fraction": 0.03,
        "workload_iterations": 600_000,
        "tests_per_step": 4,
        "worker_capacity": 1,
        "cpu_pressure_max_avg10": None,
        "cpu_pressure_available": False,
        "cgroup_throttled_usec_delta": None,
        "cgroup_available": False,
        "telemetry_sample_count": 3,
        "telemetry_coverage": 1.0,
    }
    jitter = (index % 7) * 0.001
    signals = {
        "none": {},
        "cpu_sandbox": {"queue_seconds": 2.0, "queue_fraction": 0.85 + jitter},
        "inference": {"execution_seconds": 3.0 + jitter, "duration_seconds": 3.01},
        "retrieval": {"tests_per_step": 40, "telemetry_sample_count": 20},
    }
    excess = {"none": 0.0, "cpu_sandbox": 0.8, "inference": 0.6, "retrieval": 0.4}[label]
    condition = "clean" if label == "none" else "contended"
    return (
        row(label, condition, task_id, excess=excess, run_id=f"run-{task_id}")
        | base
        | signals[label]
        | overrides
    )


def dataset(
    per_class: int, task_prefix: str, classes=("none", "cpu_sandbox", "inference", "retrieval")
):
    rows = []
    for label in classes:
        for index in range(per_class):
            rows.append(featured_row(label, index, task_id=f"{task_prefix}-{index % 5}"))
    return rows
