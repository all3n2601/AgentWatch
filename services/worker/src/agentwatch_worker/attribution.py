"""One terminal step's attribution record: the rules verdict and the served model's prediction.

No database, files or clock. The record follows docs/model-serving-contract.md ("Outputs"
and "What the dashboard must show") and attributes the resource the step likely waited on; it
does not establish a cause.

- The model inputs are the row's `step-features-v1` features plus the three inputs that no live
  feature measures, passed as missing (NaN to the model, never 0) and listed in
  `missing_inputs`, as on the replay rows the pair was trained on. A measured zero or a `False`
  flag is a value, not a missing input.
- The rules always give a verdict; they never abstain and carry no confidence.
- The model scores only `completed` steps, the only kind it was trained on. `failed` and
  `cancelled` steps get outcome `not_scored_step_state`.
- An abstention names no class (`predicted_class` None): it is uncertain, never `none`.
- `excess_fraction_estimate` is the regressor's estimated share of the step's duration spent as
  excess waiting, not a measured queue delay; `queue_seconds` stays in the feature row.
- A prediction with a non-finite number is never emitted: its outcome is `unavailable` with
  reason `non_finite_prediction`, and the rules verdict is still recorded.
- A prediction call that rejects the row's own values (`ValueError`, which includes
  `FeatureError`) leaves that step `unavailable` with reason `prediction_failed` (the error type
  is logged, never its message): it would fail the same way again, and the rules verdict is
  still recorded. Any other error (memory pressure, a fault in the booster call or in our code)
  may not repeat, so it raises `PredictionDeferredError` and the step stays pending to be
  scored again instead of getting a permanent `unavailable` row.
"""

import logging
import math
from collections.abc import Mapping, Sequence

from agentwatch_attribution.inputs import MODEL_INPUTS
from agentwatch_attribution.rules import Rules
from agentwatch_attribution.served_model import ModelIdentity, ModelPrediction, ServedModel
from agentwatch_features import FeatureRow

from agentwatch_worker.attribution_model import ModelStatus

# Model inputs with no live step-features-v1 counterpart; always passed as missing.
NOT_LIVE_INPUTS: tuple[str, ...] = ("workload_iterations", "tests_per_step", "worker_capacity")
SCORED_STEP_STATE = "completed"
STEP_STATES = frozenset({"completed", "failed", "cancelled"})
NON_FINITE_PREDICTION = "non_finite_prediction"
PREDICTION_FAILED = "prediction_failed"
logger = logging.getLogger(__name__)


class PredictionDeferredError(Exception):
    """Scoring one step failed for a reason that may not repeat; the step is left pending."""


PREDICTION_FIELDS = (
    "predicted_class",
    "confidence",
    "abstention_threshold",
    "abstained",
    "calibrated_probabilities",
    "excess_fraction_estimate",
)


def model_input_row(row: FeatureRow) -> dict[str, object]:
    """The row's model inputs in MODEL_INPUTS order; inputs no live feature measures are None."""
    values: dict[str, object] = {**row.features.model_dump(), **dict.fromkeys(NOT_LIVE_INPUTS)}
    if set(values) != set(MODEL_INPUTS) or len(values) != len(MODEL_INPUTS):
        raise ValueError(
            f"{row.feature_version} features plus {list(NOT_LIVE_INPUTS)} do not equal "
            f"MODEL_INPUTS {list(MODEL_INPUTS)}"
        )
    return {name: values[name] for name in MODEL_INPUTS}


def build_attribution(
    row: FeatureRow,
    step_state: str,
    rules: Rules,
    status: ModelStatus,
    model: ServedModel | None,
    prediction: ModelPrediction | None = None,
) -> dict[str, object]:
    """The JSON-ready attribution record for one terminal step.

    `step_state` is the step's terminal state, which the feature row does not carry. `model` is
    the served pair, or None when `status` says none is served. `prediction` is the model's
    prediction for this row when the caller already scored it in a batch (`predict_batch`); None
    makes the model score the row here.
    """
    if step_state not in STEP_STATES:
        raise ValueError(f"step_state must be one of {sorted(STEP_STATES)}, got {step_state!r}")
    inputs = model_input_row(row)
    return {
        "run_id": str(row.run_id),
        "step_id": row.step_id,
        "step_state": step_state,
        "feature_version": row.feature_version,
        "feature_row": row.model_dump(mode="json"),
        "missing_inputs": [name for name in MODEL_INPUTS if inputs[name] is None],
        "telemetry_status": row.telemetry.status,
        "rules": {"version": rules.version, "verdict": rules.predict(row.features.model_dump())},
        "model": model_attribution(inputs, step_state, status, model, prediction),
    }


def model_attribution(
    inputs: Mapping[str, object],
    step_state: str,
    status: ModelStatus,
    model: ServedModel | None,
    prediction: ModelPrediction | None = None,
) -> dict[str, object]:
    """The model half of the record: serving identity, then the outcome and its numbers."""
    served: dict[str, object] = {
        "state": status.state,
        "reason": status.reason,
        **identity_labels(status.identity),
    }
    if model is None:
        # A refused or unconfigured pair has no check code; its state is the reason.
        return served | {"reason": status.reason or status.state} | no_prediction("unavailable")
    if step_state != SCORED_STEP_STATE:
        return served | no_prediction("not_scored_step_state")
    if prediction is None:
        try:
            [prediction] = model.predict([inputs])
        except ValueError as error:
            logger.warning("Model prediction failed for one step: type=%s", type(error).__name__)
            return served | {"reason": PREDICTION_FAILED} | no_prediction("unavailable")
        # LightGBM and numpy raise assorted types and some are transient: leave the step pending
        # rather than store a permanent `unavailable` row for an error that may not repeat.
        except Exception as error:
            raise PredictionDeferredError(type(error).__name__) from error
    if not is_finite_prediction(prediction):
        return served | {"reason": NON_FINITE_PREDICTION} | no_prediction("unavailable")
    return served | {
        "outcome": "abstained" if prediction.abstained else "predicted",
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "abstention_threshold": model.identity.abstention_threshold,
        "abstained": prediction.abstained,
        "calibrated_probabilities": dict(prediction.calibrated_probabilities),
        "excess_fraction_estimate": prediction.excess_fraction,
    }


def predict_batch(model: ServedModel, rows: Sequence[FeatureRow]) -> dict[str, ModelPrediction]:
    """One model call for many rows, by step id; empty when the call raises.

    A batch costs one LightGBM call instead of one per step. If it raises, no prediction is
    returned and each row is scored alone by `build_attribution`, so one bad row fails only
    its own step.
    """
    if not rows:
        return {}
    try:
        predictions = model.predict([model_input_row(row) for row in rows])
    # As in model_attribution: any prediction error leaves the rows to be scored one by one.
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "Batch model prediction failed; scoring steps one by one: type=%s", type(error).__name__
        )
        return {}
    return {row.step_id: prediction for row, prediction in zip(rows, predictions, strict=True)}


def identity_labels(identity: ModelIdentity | None) -> dict[str, object]:
    """Versions and the not-promoted label of a loaded pair; all None when none was loaded."""
    if identity is None:
        return {"training_version": None, "calibration_version": None, "not_promoted": None}
    return {
        "training_version": identity.training_version,
        "calibration_version": identity.calibration_version,
        "not_promoted": identity.blocks_promotion,
    }


def no_prediction(outcome: str) -> dict[str, object]:
    return {"outcome": outcome, **dict.fromkeys(PREDICTION_FIELDS)}


def is_finite_prediction(prediction: ModelPrediction) -> bool:
    numbers = [
        prediction.confidence,
        prediction.excess_fraction,
        *prediction.calibrated_probabilities.values(),
    ]
    return all(math.isfinite(number) for number in numbers)
