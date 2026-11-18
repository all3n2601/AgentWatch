"""Live step attributions and the serving model's status, polled by UUID.

Attribution runs after the evidence commits: a background pass follows an ingest that holds a
terminal step event and a run closing, and every read first attributes terminal steps that lack
a row for the current serving configuration. A failed pass, in the background or before a read,
therefore delays a row until the next read; it never fails or rolls back the ingest, and never
stops a read from returning the rows already stored.
"""

import logging
from uuid import UUID

import psycopg
from agentwatch_sdk.attribution import AttributionPage, ModelStatusView
from agentwatch_sdk.live import MAX_CURSOR, ErrorResponse
from fastapi import APIRouter, HTTPException, Query, Request

from agentwatch_worker.attribution_store import AttributionStore
from agentwatch_worker.live_store import MissingRunError

MAX_PAGE_ROWS = 256
MISSING_RUN = {404: {"model": ErrorResponse, "description": "Selected live run does not exist"}}
ATTRIBUTION_NOTE = (
    "A record attributes the resource a step likely waited on; it does not establish a cause. "
    "See docs/live-ingestion.md (Attribution)."
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/live",
    tags=["live attribution"],
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Invalid request or semantic validation failure",
        },
        500: {"model": ErrorResponse, "description": "Unexpected internal storage failure"},
        503: {
            "model": ErrorResponse,
            "description": "Database unavailable or statement/lock timeout",
        },
    },
)


def store(request: Request) -> AttributionStore:
    return request.app.state.attribution


def attribute_run_logged(store: AttributionStore, run_id: UUID, **options: bool) -> None:
    """Attribute the run's terminal steps; a failure is logged and dropped, never raised.

    The failures dropped are expected ones: storage (`psycopg.Error`) and evidence that cannot
    become a feature row (`ValueError`, which includes the feature encoding errors). The log
    carries the exception type and, for storage, the SQLSTATE only, since both kinds of message
    can quote producer values. A vanished run propagates as `MissingRunError`. Other errors are
    faults and propagate too.
    """
    try:
        store.attribute_run(run_id, **options)
    except psycopg.Error as error:
        logger.warning(
            "Attribution pass failed; the next read reconciles: type=%s sqlstate=%s",
            type(error).__name__,
            error.sqlstate,
        )
    except ValueError as error:
        logger.warning(
            "Attribution pass found unusable evidence; the next read retries: type=%s",
            type(error).__name__,
        )


def attribute_after_ingest(
    store: AttributionStore, run_id: UUID, *, recheck_evidence: bool = False
) -> None:
    """Attribute the run's terminal steps after its evidence committed.

    The evidence is already committed and the next read reconciles, so failures are logged and
    dropped (`attribute_run_logged`), as is a vanished run. Only a run closing re-checks the
    stored rows' evidence (`recheck_evidence`); a pass after a terminal ingest computes features
    for the steps still lacking a row.
    """
    try:
        attribute_run_logged(store, run_id, recheck_evidence=recheck_evidence)
    except MissingRunError as error:
        logger.warning(
            "Attribution after ingest found no run; nothing to reconcile: type=%s",
            type(error).__name__,
        )


@router.get(
    "/runs/{run_id}/attributions",
    response_model=AttributionPage,
    responses=MISSING_RUN,
    description="Stored step attributions of the selected run after the `after` cursor, oldest "
    "first, with the serving model's status. Terminal steps lacking a row for the current "
    "serving key are attributed before the page is read. A run without terminal steps returns "
    f"an empty page. {ATTRIBUTION_NOTE}",
)
def get_attributions(
    run_id: UUID,
    request: Request,
    after: int = Query(0, ge=0, le=MAX_CURSOR),
    limit: int = Query(50, ge=1, le=MAX_PAGE_ROWS),
) -> dict:
    attribution = store(request)
    try:
        # Only the pending steps are computed here; the evidence re-check is left to the pass
        # that follows a run closing, so a poll costs no more than the steps it attributes.
        if attribution.has_pending_terminal_steps(run_id):
            attribute_run_logged(attribution, run_id, recheck_evidence=False)
        return attribution.page(run_id, after, limit)
    except MissingRunError as error:
        raise HTTPException(404, str(error)) from error


@router.get(
    "/attribution/model",
    response_model=ModelStatusView,
    description="The serving model's state, reason and identity, as every attribution page "
    f"reports it. {ATTRIBUTION_NOTE}",
)
def get_model_status(request: Request) -> dict[str, object]:
    return store(request).model_status()
