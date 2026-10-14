"""SDK and published API schemas must describe the final live router."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from agentwatch_sdk.client import LiveClient
from agentwatch_sdk.live import LiveBatch, StepEvent
from agentwatch_worker.export_live_contracts import openapi_document, typescript_document
from pydantic import ValidationError
from test_live_api import event

ROOT = Path(__file__).resolve().parents[3]


def test_published_live_contracts_match_router():
    document = openapi_document()
    assert json.loads((ROOT / "api/openapi/live.json").read_text()) == document
    assert (ROOT / "packages/sdk-typescript/src/live.ts").read_text() == typescript_document(
        document
    )


def test_nonfinite_and_strict_producer_records():
    record = event(str(uuid4()), 2, "completed", cpu_seconds=float("inf"), cpu_scope="process")
    with pytest.raises(ValidationError):
        StepEvent.model_validate(record)
    with pytest.raises(ValidationError):
        LiveBatch.model_validate({"records": [record]})


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_sdk_rejects_unbounded_timeouts(timeout):
    with pytest.raises(ValueError):
        LiveClient("http://localhost:8090", timeout)


def test_sdk_validates_response_and_propagates_transport_failure(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    from agentwatch_sdk import client as sdk_client
    from agentwatch_sdk.live import RunCreate

    registration = RunCreate.model_validate(
        {
            "contract_version": "live-contracts-v1",
            "request_id": str(uuid4()),
            "task_id": "fixture",
            "configuration_digest": "a" * 64,
            "clock_ids": ["clock"],
            "resources": [],
        }
    )
    run_id = str(uuid4())
    sent = []

    def response(request, timeout):
        sent.append((request.full_url, json.loads(request.data), timeout))
        return BytesIO(json.dumps({"run_id": run_id, "created": True}).encode())

    monkeypatch.setattr(sdk_client, "urlopen", response)
    sdk = LiveClient("http://localhost:8090", 2)
    assert str(sdk.create_run(registration).run_id) == run_id
    assert sent == [("http://localhost:8090/v1/live/runs", registration.model_dump(mode="json"), 2)]

    def malformed(request, timeout):
        return BytesIO(b'{"created":true}')

    monkeypatch.setattr(sdk_client, "urlopen", malformed)
    with pytest.raises(ValidationError):
        sdk.create_run(registration)

    def failure(request, timeout):
        raise HTTPError(request.full_url, 409, "Conflict", {}, None)

    monkeypatch.setattr(sdk_client, "urlopen", failure)
    with pytest.raises(HTTPError) as caught:
        sdk.create_run(registration)
    assert caught.value.code == 409


def test_sdk_rejects_path_injection_before_network():
    sdk = LiveClient("http://localhost:8090")
    with pytest.raises(ValueError):
        sdk.get_run("../other")
