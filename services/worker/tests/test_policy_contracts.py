"""SDK concurrency policy contract: strict limits and action shapes."""

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from uuid import uuid4

import pytest
from agentwatch_sdk import client as sdk_client
from agentwatch_sdk.client import LiveClient
from agentwatch_sdk.policy import (
    PolicyChange,
    PolicyChangeResult,
    PolicyLimits,
    PolicyPage,
    PolicyVersion,
    RunPolicy,
)
from agentwatch_worker.export_live_contracts import (
    openapi_document,
    policy_openapi_document,
    policy_typescript_document,
)
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = "http://localhost:8090"


def change(**overrides):
    record = {
        "request_id": str(uuid4()),
        "expected_version": 0,
        "action": "set",
        "limits": {"test_jobs": 2},
        "operator": "alice",
        "reason": "cap runner contention",
    }
    record.update(overrides)
    return {key: value for key, value in record.items() if value is not ...}


def test_limits_accept_positive_integers_and_absence():
    assert PolicyLimits.model_validate({"test_jobs": 2}).model_dump(exclude_none=True) == {
        "test_jobs": 2
    }
    assert PolicyLimits.model_validate({}).model_dump(exclude_none=True) == {}


def test_limits_omit_absent_controls_when_serialized():
    # "No limit" is an absent key on the wire; null is rejected on input and never emitted.
    limits = PolicyLimits(test_jobs=2)
    assert limits.model_dump() == {"test_jobs": 2}
    assert json.loads(limits.model_dump_json()) == {"test_jobs": 2}
    assert PolicyLimits().model_dump() == {}
    assert PolicyLimits().model_dump_json() == "{}"


@pytest.mark.parametrize(
    "limits",
    [
        {"test_jobs": 0},
        {"test_jobs": -1},
        {"test_jobs": None},
        {"inference_requests": None},
        {"test_jobs": 2.0},
        {"test_jobs": True},
        {"test_jobs": "2"},
        {"test_jobs": 1025},
        {"gpu": 1},
    ],
)
def test_limits_reject_invalid(limits):
    with pytest.raises(ValidationError):
        PolicyLimits.model_validate(limits)


@pytest.mark.parametrize(
    "record",
    [
        change(limits=...),
        change(limits={}),
        change(restored_from=1),
        change(action="disable"),
        change(action="disable", limits=..., restored_from=1),
        change(action="restore", limits=...),
        change(action="restore", limits={"test_jobs": 2}, restored_from=1),
    ],
)
def test_change_rejects_wrong_action_shape(record):
    with pytest.raises(ValidationError):
        PolicyChange.model_validate(record)


def test_set_without_limits_points_to_disable():
    with pytest.raises(ValidationError, match="disable"):
        PolicyChange.model_validate(change(limits={}))


@pytest.mark.parametrize(
    "record",
    [
        change(),
        change(action="disable", limits=...),
        change(action="restore", limits=..., restored_from=1),
    ],
)
def test_change_accepts_each_valid_action_shape(record):
    assert PolicyChange.model_validate(record).action == record["action"]


@pytest.mark.parametrize("field,value", [("operator", " "), ("reason", "")])
def test_operator_and_reason_reject_blank(field, value):
    with pytest.raises(ValidationError):
        PolicyChange.model_validate(change(**{field: value}))


def baseline_policy() -> dict:
    return {
        "target": "coding-agent",
        "version": 0,
        "action": "baseline",
        "limits": {},
        "restored_from": None,
        "operator": None,
        "reason": None,
        "request_id": None,
        "created_at": None,
        "controlled_services": {
            "inference_requests": {"service_id": "ollama", "resource_class": "inference"},
            "test_jobs": {"service_id": "agentwatch-agent-runner", "resource_class": "cpu_sandbox"},
        },
        "enforcement_scope": "runner_process",
    }


def test_published_policy_contracts_match_router():
    document = policy_openapi_document()
    assert json.loads((ROOT / "api/openapi/policy.json").read_text()) == document
    assert (ROOT / "packages/sdk-typescript/src/policy.ts").read_text() == (
        policy_typescript_document(document, openapi_document())
    )


def test_policy_typescript_declares_only_new_schemas_and_imports_shared_ones():
    text = policy_typescript_document(policy_openapi_document(), openapi_document())
    assert 'import type { PolicyLimits, PolicyVersion } from "./live";' in text
    assert "export interface PolicyChange {" in text
    assert "export interface PolicyVersion" not in text


def test_sdk_policy_methods_validate_responses(monkeypatch):
    run_id = uuid4()
    change = PolicyChange.model_validate(
        {
            "request_id": str(uuid4()),
            "expected_version": 0,
            "action": "set",
            "limits": {"inference_requests": 2},
            "operator": "operator",
            "reason": "bound inference",
        }
    )
    bodies = {
        f"{BASE_URL}/v1/policies/concurrency/coding-agent": baseline_policy(),
        f"{BASE_URL}/v1/policies/concurrency/coding-agent/versions?limit=50": {
            "versions": [baseline_policy()],
            "next_before": None,
        },
        f"{BASE_URL}/v1/policies/concurrency/coding-agent/versions?limit=5&before=3": {
            "versions": [],
            "next_before": None,
        },
        f"{BASE_URL}/v1/policies/concurrency/coding-agent/versions": {
            "created": True,
            "policy": baseline_policy(),
        },
        f"{BASE_URL}/v1/live/runs/{run_id}/policy": {
            "run_id": str(run_id),
            "target": "coding-agent",
            "applicable": True,
            "versions": [baseline_policy()],
        },
    }
    sent = []

    def respond(request, timeout):
        sent.append((request.full_url, request.data and json.loads(request.data), timeout))
        return BytesIO(json.dumps(bodies[request.full_url]).encode())

    monkeypatch.setattr(sdk_client, "urlopen", respond)
    sdk = LiveClient(BASE_URL, 2)
    assert isinstance(sdk.get_policy(), PolicyVersion)
    assert isinstance(sdk.list_policy_versions(), PolicyPage)
    assert sdk.list_policy_versions(before=3, limit=5).versions == []
    assert isinstance(sdk.change_policy(change), PolicyChangeResult)
    assert isinstance(sdk.get_run_policy(run_id), RunPolicy)
    assert sent[3] == (
        f"{BASE_URL}/v1/policies/concurrency/coding-agent/versions",
        change.model_dump(mode="json"),
        2,
    )

    monkeypatch.setattr(sdk_client, "urlopen", lambda request, timeout: BytesIO(b'{"version":0}'))
    with pytest.raises(ValidationError):
        sdk.get_policy()

    def conflict(request, timeout):
        raise HTTPError(request.full_url, 409, "Conflict", {}, None)

    monkeypatch.setattr(sdk_client, "urlopen", conflict)
    with pytest.raises(HTTPError) as caught:
        sdk.change_policy(change)
    assert caught.value.code == 409


@pytest.mark.parametrize("target", ["../other", "a/b", "", "x?y=1"])
def test_sdk_policy_rejects_path_injection_before_network(target):
    sdk = LiveClient(BASE_URL)
    with pytest.raises(ValueError):
        sdk.get_policy(target)
    with pytest.raises(ValueError):
        sdk.list_policy_versions(target)


@pytest.mark.parametrize("before,limit", [(None, 0), (None, 101), (-1, 5), (True, 5)])
def test_sdk_policy_history_rejects_invalid_cursor(before, limit):
    with pytest.raises(ValueError):
        LiveClient(BASE_URL).list_policy_versions(before=before, limit=limit)
