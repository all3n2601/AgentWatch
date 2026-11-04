"""HTTP behavior of the concurrency policy API against isolated PostgreSQL schemas."""

from uuid import uuid4

import pytest
from agentwatch_sdk.live import MAX_CURSOR
from agentwatch_sdk.policy import RUNNER_SERVICE_ID

BASE = "/v1/policies/concurrency/coding-agent"


def body(expected_version, **fields):
    return {
        "request_id": str(uuid4()),
        "expected_version": expected_version,
        "action": "set",
        "limits": {"test_jobs": 2},
        "operator": "fixture-operator",
        "reason": "fixture policy change",
    } | fields


def register_run(client, service_id):
    response = client.post(
        "/v1/live/runs",
        json={
            "contract_version": "live-contracts-v1",
            "request_id": str(uuid4()),
            "task_id": "policy-fixture",
            "configuration_digest": "a" * 64,
            "clock_ids": ["fixture-clock"],
            "resources": [
                {
                    "resource_id": "fixture-resource",
                    "service_id": service_id,
                    "resource_class": "cpu_sandbox",
                }
            ],
        },
    )
    assert response.status_code == 201
    return response.json()["run_id"]


def test_get_baseline(client):
    response = client.get(BASE)
    assert response.status_code == 200
    assert response.json()["version"] == 0


def test_post_set_then_retry(client):
    payload = body(0)
    created = client.post(f"{BASE}/versions", json=payload)
    assert created.status_code == 201
    retried = client.post(f"{BASE}/versions", json=payload)
    assert retried.status_code == 200
    assert retried.json()["policy"]["version"] == created.json()["policy"]["version"] == 1
    assert client.get(BASE).json()["version"] == 1


def test_post_invalid_limits_422_without_echo(client):
    response = client.post(
        f"{BASE}/versions",
        json=body(0, limits={"test_jobs": 0}, operator="secret-operator-string"),
    )
    assert response.status_code == 422
    assert "secret-operator-string" not in response.text


def test_post_stale_409(client):
    assert client.post(f"{BASE}/versions", json=body(0)).status_code == 201
    assert client.post(f"{BASE}/versions", json=body(0)).status_code == 409


def test_unsupported_target_404(client):
    path = "/v1/policies/concurrency/cpu-demo"
    for response in (
        client.get(path),
        client.get(f"{path}/versions"),
        client.post(f"{path}/versions", json=body(0)),
    ):
        assert response.status_code == 404
        assert response.json()["detail"] == "Unsupported policy target"


def test_restore_missing_version_422(client):
    response = client.post(
        f"{BASE}/versions",
        json={
            "request_id": str(uuid4()),
            "expected_version": 0,
            "action": "restore",
            "restored_from": 5,
            "operator": "fixture-operator",
            "reason": "fixture restore",
        },
    )
    assert response.status_code == 422


def test_history_pagination(client):
    for version in range(3):
        assert client.post(f"{BASE}/versions", json=body(version)).status_code == 201
    first = client.get(f"{BASE}/versions", params={"limit": 2}).json()
    assert [item["version"] for item in first["versions"]] == [3, 2]
    second = client.get(
        f"{BASE}/versions", params={"limit": 2, "before": first["next_before"]}
    ).json()
    assert [item["version"] for item in second["versions"]] == [1, 0]
    assert second["next_before"] is None


@pytest.mark.parametrize("before", [2**31, MAX_CURSOR])
def test_history_accepts_cursor_beyond_integer_range(client, before):
    for version in range(2):
        assert client.post(f"{BASE}/versions", json=body(version)).status_code == 201
    response = client.get(f"{BASE}/versions", params={"before": before})
    assert response.status_code == 200
    page = response.json()
    assert [item["version"] for item in page["versions"]] == [2, 1, 0]
    assert page["next_before"] is None


def test_history_rejects_out_of_range_limit(client):
    for limit in (0, 101):
        assert client.get(f"{BASE}/versions", params={"limit": limit}).status_code == 422


def test_run_policy_endpoint(client):
    run_id = register_run(client, RUNNER_SERVICE_ID)
    response = client.get(f"/v1/live/runs/{run_id}/policy")
    assert response.status_code == 200
    assert response.json()["applicable"] is True
    assert client.get(f"/v1/live/runs/{uuid4()}/policy").status_code == 404


def test_limits_never_serialize_null(client):
    run_id = register_run(client, RUNNER_SERVICE_ID)
    baseline = client.get(BASE).json()
    assert baseline["limits"] == {}
    created = client.post(f"{BASE}/versions", json=body(0, limits={"test_jobs": 2}))
    assert created.status_code == 201
    assert created.json()["policy"]["limits"] == {"test_jobs": 2}
    responses = [
        client.get(BASE).json(),
        *client.get(f"{BASE}/versions").json()["versions"],
        *client.get(f"/v1/live/runs/{run_id}/policy").json()["versions"],
    ]
    assert [version["limits"] for version in responses] == [
        {"test_jobs": 2},
        {"test_jobs": 2},
        {},
        {},
        {"test_jobs": 2},
    ]
