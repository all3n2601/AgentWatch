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


class AttributionItem(ResponseModel):
    """One stored step attribution. `record` is the attribution record: feature row, missing
    inputs, telemetry status, rules verdict and model outcome. Both times are database-clock
    times; `evidence_changed` is set when a later audit computed a different feature row."""

    sequence: int
    serving_key: str
    committed_at: datetime
    step_terminal_received_at: datetime
    evidence_changed: bool
    record: dict[str, object]


class AttributionPage(ResponseModel):
    """Stored attributions of the selected run after the `after` cursor, oldest first. Rows of
    every serving key are returned; the current one is `model.serving_key`."""

    run_id: UUID
    model: ModelStatusView
    attributions: list[AttributionItem]
    next_cursor: int | None
