"""Model inputs: the 14 features, encoded identically for training and serving.

A missing reading is passed to LightGBM as NaN and never imputed, so it stays distinct from a
measured zero. This module imports no training or dataset code, so a server reproduces exactly
what training fed the models.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from agentwatch_attribution.classes import is_number

# Order of the "Inputs" table in docs/model-serving-contract.md. A drift test in the model
# package asserts it equals the dataset schema's feature columns.
MODEL_INPUTS = (
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
)


class FeatureError(ValueError):
    """A row whose features cannot be encoded as recorded."""


def encode(value: object, name: str, index: int) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if is_number(value):
        return float(value)
    raise FeatureError(f"Row {index} feature {name} is not a finite number: {value!r}")


def feature_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    """One float64 row per step, columns in MODEL_INPUTS order."""
    matrix = np.empty((len(rows), len(MODEL_INPUTS)), dtype=np.float64)
    for index, row in enumerate(rows):
        missing = [name for name in MODEL_INPUTS if name not in row]
        if missing:
            raise FeatureError(f"Row {index} is missing feature {', '.join(missing)}")
        matrix[index] = [encode(row[name], name, index) for name in MODEL_INPUTS]
    return matrix
