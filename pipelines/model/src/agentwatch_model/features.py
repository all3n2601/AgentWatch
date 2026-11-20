"""Model inputs: the schema's feature columns, encoded identically for training and serving.

The encoding lives in agentwatch_attribution.inputs so serving reproduces it exactly.

Only columns the schema marks as features are inputs, so labels, rule output and lineage can
never leak into a model. A missing reading is passed to LightGBM as NaN and never imputed, so
it stays distinct from a measured zero.
"""

from agentwatch_attribution.inputs import MODEL_INPUTS, FeatureError, encode, feature_matrix

FEATURES = MODEL_INPUTS
MISSING_VALUES = "NaN passed to LightGBM, which routes it natively; never imputed"
BOOLEANS = "true is 1.0, false is 0.0"

__all__ = [
    "BOOLEANS",
    "FEATURES",
    "MISSING_VALUES",
    "FeatureError",
    "encode",
    "feature_matrix",
    "preprocessing_contract",
]


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
