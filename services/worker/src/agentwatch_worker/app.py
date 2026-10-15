import json
import logging
import os
from contextlib import asynccontextmanager
from uuid import UUID

import psycopg
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agentwatch_worker.contracts import RunIngest, TelemetryBatch
from agentwatch_worker.live_api import router as live_router
from agentwatch_worker.live_limits import LiveBodyLimit
from agentwatch_worker.store import ConflictError, RunStore


@asynccontextmanager
async def lifespan(app):
    app.state.store = RunStore(
        os.environ.get(
            "DATABASE_URL", "postgresql://agentwatch:agentwatch_dev@localhost:5432/agentwatch"
        )
    )
    app.state.store.migrate()
    yield


app = FastAPI(title="AgentWatch Ingestion API", version="0.3.0", lifespan=lifespan)
app.include_router(live_router)
app.add_middleware(LiveBodyLimit)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    if request.url.path.startswith("/v1/live/"):
        # Validation input can contain nonfinite numbers or sensitive producer evidence.
        # Return actionable locations/messages without reflecting the entire supplied record.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
                    for item in error.errors()
                ]
            },
        )
    return await request_validation_exception_handler(request, error)


@app.exception_handler(psycopg.Error)
async def unavailable_storage(request, error):
    if request.url.path.startswith("/v1/live/"):
        if isinstance(error, psycopg.DataError) and request.method == "POST":
            return JSONResponse(status_code=422, content={"detail": "Invalid live database value"})
        if not isinstance(error, (psycopg.OperationalError, psycopg.InterfaceError)):
            # PostgreSQL exception messages can contain producer values; retain only diagnostics.
            logging.getLogger(__name__).error(
                "Unexpected live storage failure: type=%s sqlstate=%s",
                type(error).__name__,
                error.sqlstate,
            )
            return JSONResponse(status_code=500, content={"detail": "Internal storage failure"})
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "worker"}


@app.get("/readyz")
def ready(request: Request):
    try:
        request.app.state.store.ready()
    except psycopg.Error:
        raise HTTPException(503, "Database unavailable") from None
    return {"status": "ready", "storage": "postgresql"}


@app.post("/v1/runs", status_code=201)
def ingest(batch: RunIngest, request: Request, response: Response):
    try:
        result = request.app.state.store.ingest(batch)
    except ConflictError as error:
        raise HTTPException(409, str(error)) from error
    except psycopg.Error:
        raise HTTPException(503, "Database unavailable") from None
    response.status_code = 201 if result["created"] else 200
    return result


@app.get("/v1/runs")
def history(
    request: Request, limit: int = Query(50, ge=1, le=100), before: int | None = Query(None, gt=0)
):
    return request.app.state.store.list_runs(limit, before)


@app.get("/v1/runs/{run_id}")
def get_run(run_id: UUID, request: Request):
    run = request.app.state.store.get(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


@app.post("/v1/runs/{run_id}/telemetry")
def telemetry(run_id: UUID, batch: TelemetryBatch, request: Request):
    try:
        added = request.app.state.store.append_telemetry(run_id, batch)
    except ConflictError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except psycopg.Error:
        raise HTTPException(503, "Database unavailable") from None
    if added is None:
        raise HTTPException(404, "Run not found")
    return {"run_id": str(run_id), "added": added}


@app.get("/v1/runs/{run_id}/artifacts/{filename}")
def artifact(run_id: UUID, filename: str, request: Request):
    if filename not in {"results.json", "spans.jsonl", "samples.jsonl"}:
        raise HTTPException(404, "Unknown artifact")
    content = request.app.state.store.artifact(run_id, filename)
    if content is None:
        raise HTTPException(404, "Run not found")
    body = (
        json.dumps(content, indent=2) + "\n"
        if filename.endswith(".json")
        else "".join(json.dumps(row) + "\n" for row in content)
    )
    return Response(
        body,
        media_type="application/json" if filename.endswith(".json") else "application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{run_id}-{filename}"'},
    )
