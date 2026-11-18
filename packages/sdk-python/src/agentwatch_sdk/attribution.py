"""Live step attribution responses (B03): the serving model's status and stored attributions.

A record attributes the resource a step likely waited on; it does not establish a cause. See
docs/live-ingestion.md ("Attribution") and docs/model-serving-contract.md.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from agentwatch_sdk.live import ResponseModel

# not_configured: no pair is configured. load_failed: a configured pair failed a load check,
# named by `reason`. not_promoted_refused: the pair loaded but its gates did not pass and serving
# an unpromoted pair is not allowed. ready_not_promoted: that pair is served, labelled not
# promoted. ready_promoted: a pair whose gates passed is served.
ModelState = Literal[
    "not_configured",
    "not_promoted_refused",
    "ready_not_promoted",
    "ready_promoted",
    "load_failed",
]


class ModelIdentityView(ResponseModel):
    """What was loaded: versions, artifact hashes and calibration parameters."""

    training_version: str
    calibration_version: str
    classifier_sha256: str
    regressor_sha256: str
    preprocessing_sha256: str
    calibration_sha256: str
    temperature: float
    abstention_threshold: float
    class_order: list[str]
    blocks_promotion: bool


class ModelStatusView(ResponseModel):
    """The serving model's status. `identity` is set whenever a pair was loaded, refused ones
    included; `not_promoted` is None when none was. `serving_key` identifies the rules version
    and model configuration that rows written now carry."""

    state: ModelState
    reason: str | None
    identity: ModelIdentityView | None
    not_promoted: bool | None
    serving_key: str


# A terminal step state; only `completed` steps are scored by the model.
StepState = Literal["completed", "failed", "cancelled"]
# How a step's window joined resource samples: `agentwatch_features.live.TelemetryStatus`, which
# a worker test pins equal (the features package depends on this one, not the reverse).
TelemetryStatus = Literal["present", "empty", "missing", "no_overlap", "unaligned"]
# predicted / abstained: a completed step was scored. not_scored_step_state: a failed or cancelled
# step, which the model does not score. unavailable: no pair is served, or the prediction was not
# finite; `reason` says which.
ModelOutcome = Literal["predicted", "abstained", "not_scored_step_state", "unavailable"]


class RulesVerdictView(ResponseModel):
    """The rules baseline's verdict. The rules never abstain and carry no confidence."""

    version: str
    verdict: str


class ModelPredictionView(ResponseModel):
    """The model half of a record: the serving labels at write time, then the outcome.

    `reason` is the load check code for `load_failed`, the state itself for `not_configured` and
    `not_promoted_refused`, `non_finite_prediction` for an unusable prediction,
    `prediction_failed` when the served pair raised on the step, otherwise None.
    The prediction fields are None unless `outcome` is predicted or abstained; an abstention has
    `predicted_class` None, never `none`. `excess_fraction_estimate` is the regressor's estimated
    share of the step's duration spent as excess waiting, not a measured queue delay.
    """

    state: ModelState
    reason: str | None
    training_version: str | None
    calibration_version: str | None
    not_promoted: bool | None
    outcome: ModelOutcome
    predicted_class: str | None
    confidence: float | None
    abstention_threshold: float | None
    abstained: bool | None
    calibrated_probabilities: dict[str, float] | None
    excess_fraction_estimate: float | None


class AttributionRecord(ResponseModel):
    """One terminal step's attribution record, as the worker built and stored it.

    `feature_row` is the stored `step-features-v1` FeatureRow (`agentwatch_features.FeatureRow`,
    docs/step-features.md) and is left untyped here. `missing_inputs` lists the model inputs
    passed as missing (NaN, never 0). A record attributes the resource a step likely waited on;
    it does not establish a cause.
    """

    run_id: UUID
    step_id: str
    step_state: StepState
    feature_version: str
    feature_row: dict[str, object]
    missing_inputs: list[str]
    telemetry_status: TelemetryStatus
    rules: RulesVerdictView
    model: ModelPredictionView


class AttributionItem(ResponseModel):
    """One stored step attribution. Both times are database-clock times; `evidence_changed` is
    set when a later audit computed a different feature row, possibly after the row was
    served."""

    sequence: int
    serving_key: str
    committed_at: datetime
    step_terminal_received_at: datetime
    evidence_changed: bool
    record: AttributionRecord


class AttributionPage(ResponseModel):
    """Stored attributions of the selected run after the `after` cursor, oldest first. Rows of
    every serving key are returned; the current one is `model.serving_key`."""

    run_id: UUID
    model: ModelStatusView
    attributions: list[AttributionItem]
    next_cursor: int | None
