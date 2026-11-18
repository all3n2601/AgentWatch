"""Live transport: bounded JSON batches and cursor-based polling, separate from imports."""

from uuid import UUID

from agentwatch_sdk.live import (
    MAX_CURSOR,
    TERMINAL_STATES,
    BatchResult,
    ChangeResult,
    CreateResult,
    ErrorResponse,
    LiveBatch,
    RunChange,
    RunCreate,
    RunPage,
    RunSnapshot,
    StepEvent,
)
from agentwatch_sdk.policy import RunPolicy
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, Response

from agentwatch_worker.attribution_api import attribute_after_ingest
from agentwatch_worker.live_limits import BODY_TIMEOUT_SECONDS, MAX_BODY_BYTES
from agentwatch_worker.live_store import LiveStore, MissingRunError
from agentwatch_worker.policy_store import PolicyStore
from agentwatch_worker.store import ConflictError

WRITE_ERRORS = {
    400: {"model": ErrorResponse, "description": "Unparseable body or invalid Content-Length"},
    409: {
        "model": ErrorResponse,
        "description": "Identity, revision or lifecycle conflict; transaction rolled back",
    },
    413: {"model": ErrorResponse, "description": f"Live body exceeds {MAX_BODY_BYTES} bytes"},
    408: {
        "model": ErrorResponse,
        "description": f"Live body upload exceeds {BODY_TIMEOUT_SECONDS} seconds",
    },
}
MISSING_RUN = {404: {"model": ErrorResponse, "description": "Selected live run does not exist"}}

router = APIRouter(
    prefix="/v1/live/runs",
    tags=["live lifecycle"],
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


def store(request: Request) -> LiveStore:
    return LiveStore(request.app.state.store)


@router.post(
    "",
    status_code=201,
    response_model=CreateResult,
    responses={**WRITE_ERRORS, 200: {"model": CreateResult}},
)
def create_run(registration: RunCreate, request: Request, response: Response):
    try:
        result = store(request).create(registration)
    except ConflictError as error:
        raise HTTPException(409, str(error)) from error
    response.status_code = 201 if result["created"] else 200
    return result


@router.get("", response_model=RunPage)
def list_runs(
    request: Request,
    before: int | None = Query(None, gt=0, le=MAX_CURSOR),
    limit: int = Query(50, ge=1, le=100),
):
    return store(request).list_runs(before, limit)


@router.get("/{run_id}", response_model=RunSnapshot, responses=MISSING_RUN)
def get_run(
    run_id: UUID,
    request: Request,
    after: int = Query(0, ge=0, le=MAX_CURSOR),
    limit: int = Query(256, ge=1, le=256),
):
    try:
        return store(request).get(run_id, after, limit)
    except MissingRunError as error:
        raise HTTPException(404, str(error)) from error


@router.get(
    "/{run_id}/policy",
    response_model=RunPolicy,
    responses=MISSING_RUN,
    description="Concurrency policy versions active while this run was open (queued or "
    "running), oldest first. Links record configuration only and are not evidence of "
    "enforcement. `versions` is empty for runs that are not coding-agent runs or that were "
    "registered before policy tracking.",
)
def get_run_policy(run_id: UUID, request: Request):
    try:
        return PolicyStore(request.app.state.store).run_policy(run_id)
    except MissingRunError as error:
        raise HTTPException(404, str(error)) from error


@router.post(
    "/{run_id}/records", response_model=BatchResult, responses={**WRITE_ERRORS, **MISSING_RUN}
)
def ingest_records(run_id: UUID, batch: LiveBatch, request: Request, background: BackgroundTasks):
    try:
        result = store(request).ingest(run_id, batch)
    except MissingRunError as error:
        raise HTTPException(404, str(error)) from error
    except ConflictError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if holds_terminal_step_event(batch):
        # After the commit: an attribution failure must not fail the acknowledged ingest.
        background.add_task(attribute_after_ingest, request.app.state.attribution, run_id)
    return result


@router.post(
    "/{run_id}/state", response_model=ChangeResult, responses={**WRITE_ERRORS, **MISSING_RUN}
)
def change_run(run_id: UUID, change: RunChange, request: Request, background: BackgroundTasks):
    try:
        result = store(request).change(run_id, change)
    except MissingRunError as error:
        raise HTTPException(404, str(error)) from error
    except ConflictError as error:
        raise HTTPException(409, str(error)) from error
    if change.state in TERMINAL_STATES:
        # A closing run gets a final audit, which flags rows whose window gained late samples.
        background.add_task(
            attribute_after_ingest, request.app.state.attribution, run_id, recheck_evidence=True
        )
    return result


def holds_terminal_step_event(batch: LiveBatch) -> bool:
    """A terminal event that arrives before its predecessors and is applied by a later
    non-terminal batch triggers nothing here; the next attribution read reconciles it."""
    return any(
        isinstance(record, StepEvent) and record.state in TERMINAL_STATES
        for record in batch.records
    )
