"""Synchronous producer client. Retry with the same identities after uncertain outcomes."""

import math
from urllib.request import Request, urlopen
from uuid import UUID

from agentwatch_sdk.live import (
    MAX_CURSOR,
    BatchResult,
    ChangeResult,
    CreateResult,
    LiveBatch,
    LiveModel,
    ResponseModel,
    RunChange,
    RunCreate,
    RunPage,
    RunSnapshot,
)

MAX_RESPONSE_BYTES = 8_388_608


class LiveClient:
    """timeout_seconds bounds socket inactivity, not total request elapsed time.

    A peer continuously delivering small chunks can take longer. Response bytes are bounded;
    orchestration that needs a total deadline must enforce one outside this synchronous client.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 10):
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request[Model: ResponseModel](
        self, path: str, response_type: type[Model], payload: LiveModel | None = None
    ) -> Model:
        body = payload.model_dump_json().encode() if payload is not None else None
        request = Request(
            self.base_url + path, data=body, headers={"Content-Type": "application/json"}
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            result = response.read(MAX_RESPONSE_BYTES + 1)
        if len(result) > MAX_RESPONSE_BYTES:
            raise ValueError("Live API response exceeds SDK limit")
        return response_type.model_validate_json(result)

    def create_run(self, registration: RunCreate) -> CreateResult:
        return self._request("/v1/live/runs", CreateResult, registration)

    def ingest(self, run_id: UUID, batch: LiveBatch) -> BatchResult:
        return self._request(f"/v1/live/runs/{UUID(str(run_id))}/records", BatchResult, batch)

    def change_state(self, run_id: UUID, change: RunChange) -> ChangeResult:
        return self._request(f"/v1/live/runs/{UUID(str(run_id))}/state", ChangeResult, change)

    def get_run(self, run_id: UUID, after: int = 0, limit: int = 256) -> RunSnapshot:
        if (
            type(after) is not int
            or type(limit) is not int
            or not 0 <= after <= MAX_CURSOR
            or not 1 <= limit <= 256
        ):
            raise ValueError("Invalid live record cursor/page size")
        return self._request(
            f"/v1/live/runs/{UUID(str(run_id))}?after={after}&limit={limit}", RunSnapshot
        )

    def list_runs(self, before: int | None = None, limit: int = 50) -> RunPage:
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or (before is not None and (type(before) is not int or not 1 <= before <= MAX_CURSOR))
        ):
            raise ValueError("Invalid live history cursor/page size")
        query = f"?limit={limit}" + (f"&before={before}" if before is not None else "")
        return self._request("/v1/live/runs" + query, RunPage)
