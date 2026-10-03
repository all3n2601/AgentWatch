from copy import deepcopy
from uuid import uuid4


def test_import_is_idempotent_and_history_retains_runs(client, evidence):
    first = client.post("/v1/runs", json=evidence)
    assert first.status_code == 201
    repeated = client.post("/v1/runs", json=evidence)
    assert repeated.status_code == 200
    assert repeated.json()["added"] == {"spans": 0, "samples": 0}
    second = deepcopy(evidence)
    second_id = str(uuid4())
    second["result"]["run_id"] = second_id
    second["spans"][0]["attributes"]["agentwatch.run_id"] = second_id
    assert client.post("/v1/runs", json=second).status_code == 201
    history = client.get("/v1/runs?limit=1").json()
    assert history["runs"][0]["run_id"] == second_id
    older = client.get(f"/v1/runs?limit=1&before={history['next_cursor']}").json()
    assert older["runs"][0]["run_id"] == evidence["result"]["run_id"]
    assert older["next_cursor"] is None
    assert client.get("/readyz").json()["storage"] == "postgresql"


def test_conflicting_results_cannot_overwrite_saved_evidence(client, evidence):
    client.post("/v1/runs", json=evidence)
    different = deepcopy(evidence)
    different["result"]["clean"]["checksum"] = "different"
    assert client.post("/v1/runs", json=different).status_code == 409
    saved = client.get(f"/v1/runs/{evidence['result']['run_id']}").json()
    assert saved["result"]["clean"]["checksum"] == "fixed"


def test_telemetry_conflict_rolls_back_the_entire_batch(client, evidence):
    client.post("/v1/runs", json=evidence)
    run_id = evidence["result"]["run_id"]
    new = deepcopy(evidence["spans"][0])
    new["context"]["span_id"] = "0x" + "c" * 16
    conflict = deepcopy(evidence["spans"][0])
    conflict["name"] = "different"
    response = client.post(f"/v1/runs/{run_id}/telemetry", json={"spans": [new, conflict]})
    assert response.status_code == 409
    exported = client.get(f"/v1/runs/{run_id}/artifacts/spans.jsonl")
    assert len(exported.text.strip().splitlines()) == 1
    assert (
        client.post(f"/v1/runs/{run_id}/telemetry", json={"spans": [new]}).json()["added"]["spans"]
        == 1
    )
    assert (
        client.post(f"/v1/runs/{run_id}/telemetry", json={"spans": [new]}).json()["added"]["spans"]
        == 0
    )


def test_invalid_ingestion_is_rejected_before_storage(client, evidence):
    evidence["result"]["clean"]["queue_fraction"] = 0.9
    assert client.post("/v1/runs", json=evidence).status_code == 422
    assert client.get("/v1/runs").json()["runs"] == []
    assert client.get("/v1/runs/not-a-uuid").status_code == 422
    assert client.get(f"/v1/runs/{uuid4()}").status_code == 404
    assert client.get("/v1/runs?limit=101").status_code == 422


def test_artifacts_belong_to_the_selected_saved_run(client, evidence):
    client.post("/v1/runs", json=evidence)
    run_id = evidence["result"]["run_id"]
    exported = client.get(f"/v1/runs/{run_id}/artifacts/results.json")
    assert exported.json() == evidence["result"]
    assert "attachment" in exported.headers["content-disposition"]
    assert client.get(f"/v1/runs/{run_id}/artifacts/unknown").status_code == 404
    assert client.get(f"/v1/runs/{uuid4()}/artifacts/results.json").status_code == 404
