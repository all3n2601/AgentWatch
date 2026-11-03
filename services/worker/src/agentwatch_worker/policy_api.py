"""Versioned manual concurrency policy: read the active policy, list history, record changes."""

from agentwatch_sdk.live import MAX_CURSOR, ErrorResponse
from agentwatch_sdk.policy import PolicyChange, PolicyChangeResult, PolicyPage, PolicyVersion
from fastapi import APIRouter, HTTPException, Query, Request, Response

from agentwatch_worker.live_limits import BODY_TIMEOUT_SECONDS, MAX_BODY_BYTES
from agentwatch_worker.policy_store import (
    InvalidRestoreError,
    PolicyStore,
    UnsupportedTargetError,
)
from agentwatch_worker.store import ConflictError

UNSUPPORTED_TARGET = {
    404: {"model": ErrorResponse, "description": "Unsupported policy target"},
}
WRITE_ERRORS = {
    400: {"model": ErrorResponse, "description": "Unparseable body or invalid Content-Length"},
    408: {
        "model": ErrorResponse,
        "description": f"Policy body upload exceeds {BODY_TIMEOUT_SECONDS} seconds",
    },
    409: {
        "model": ErrorResponse,
        "description": "Stale expected_version or request_id reused with a different change",
    },
    413: {"model": ErrorResponse, "description": f"Policy body exceeds {MAX_BODY_BYTES} bytes"},
}

router = APIRouter(
    prefix="/v1/policies/concurrency",
    tags=["concurrency policy"],
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Invalid request, limits or restore source",
        },
        500: {"model": ErrorResponse, "description": "Unexpected internal storage failure"},
        503: {
            "model": ErrorResponse,
            "description": "Database unavailable or statement/lock timeout",
        },
    },
)


def store(request: Request) -> PolicyStore:
    return PolicyStore(request.app.state.store)


NOT_ENFORCED = "Policy is stored and versioned only; nothing enforces these limits yet."


@router.get(
    "/{target}",
    response_model=PolicyVersion,
    responses=UNSUPPORTED_TARGET,
    description=f"The active policy version; version 0 is the no-limit baseline. {NOT_ENFORCED}",
)
def get_active_policy(target: str, request: Request):
    try:
        return store(request).active(target)
    except UnsupportedTargetError as error:
        raise HTTPException(404, str(error)) from error


@router.get(
    "/{target}/versions",
    response_model=PolicyPage,
    responses=UNSUPPORTED_TARGET,
    description=f"Policy history newest first, ending with the version 0 baseline. {NOT_ENFORCED}",
)
def list_policy_versions(
    target: str,
    request: Request,
    before: int | None = Query(None, ge=0, le=MAX_CURSOR),
    limit: int = Query(50, ge=1, le=100),
):
    try:
        return store(request).history(target, before, limit)
    except UnsupportedTargetError as error:
        raise HTTPException(404, str(error)) from error


@router.post(
    "/{target}/versions",
    status_code=201,
    response_model=PolicyChangeResult,
    responses={**WRITE_ERRORS, **UNSUPPORTED_TARGET, 200: {"model": PolicyChangeResult}},
    description="Append a set, disable or restore version and link it to open coding-agent "
    f"runs; an identical retry returns 200. {NOT_ENFORCED}",
)
def change_policy(target: str, change: PolicyChange, request: Request, response: Response):
    try:
        result = store(request).change(target, change)
    except UnsupportedTargetError as error:
        raise HTTPException(404, str(error)) from error
    except ConflictError as error:
        raise HTTPException(409, str(error)) from error
    except InvalidRestoreError as error:
        raise HTTPException(422, str(error)) from error
    response.status_code = 201 if result["created"] else 200
    return result
