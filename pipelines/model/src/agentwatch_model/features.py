"""Model inputs: the schema's feature columns, encoded identically for training and serving.

Only columns the schema marks as features are inputs, so labels, rule output and lineage can
never leak into a model. A missing reading is passed to LightGBM as NaN and never imputed, so
it stays distinct from a measured zero.
"""

import math

import numpy as np
from agentwatch_data.schema import feature_columns

from agentwatch_model.metrics import is_number

FEATURES = tuple(feature_columns())
MISSING_VALUES = "NaN passed to LightGBM, which routes it natively; never imputed"
BOOLEANS = "true is 1.0, false is 0.0"


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


def feature_matrix(rows: list[dict]) -> np.ndarray:
    """One float64 row per step, columns in FEATURES order."""
    matrix = np.empty((len(rows), len(FEATURES)), dtype=np.float64)
    for index, row in enumerate(rows):
        missing = [name for name in FEATURES if name not in row]
        if missing:
            raise FeatureError(f"Row {index} is missing feature {', '.join(missing)}")
        matrix[index] = [encode(row[name], name, index) for name in FEATURES]
    return matrix


def preprocessing_contract(
    class_order: tuple[str, ...], training_version: str, lineage: dict
) -> dict:
    """What serving must reproduce to feed the trained models the same inputs."""
    return {
        "features": list(FEATURES),
        "missing_values": MISSING_VALUES,
        "booleans": BOOLEANS,
        "class_order": list(class_order),
        "training_version": training_version,
        **lineage,
    }
