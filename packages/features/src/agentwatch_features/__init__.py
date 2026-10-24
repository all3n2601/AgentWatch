"""Versioned step feature definitions shared by AgentWatch training and serving."""

from agentwatch_features.legacy import LEGACY_FEATURE_VERSION, window_readings
from agentwatch_features.live import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    FeatureRow,
    FeatureRows,
    IncompleteStepError,
    compute_feature_row,
    compute_feature_rows,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "LEGACY_FEATURE_VERSION",
    "FeatureRow",
    "FeatureRows",
    "IncompleteStepError",
    "compute_feature_row",
    "compute_feature_rows",
    "window_readings",
]
