"""PR 25 regression cases use synthetic evidence in isolated PostgreSQL schemas."""

import json
from io import BytesIO
from uuid import UUID

import pytest
from agentwatch_sdk import LiveClient
from agentwatch_sdk.live import MAX_CURSOR, ErrorResponse, ResourceSample, RunCreate, StepEvent
from agentwatch_worker.export_live_contracts import openapi_document
from psycopg.types.json import Jsonb
from pydantic import ValidationError
from test_live_api import change, event, post, sample, snapshot


@pytest.mark.parametrize("field", ["task_id", "clock_ids", "resource_id", "service_id"])
@pytest.mark.parametrize("invalid_text", ["a\x00b", "\x00", "a\x00\n", "a\ud800b"])
def test_invalid_registration_text_is_not_a_storage_outage(
    client, registration, field, invalid_text
):
    if field in {"resource_id", "service_id"}:
        registration["resources"][0][field] = invalid_text
    else:
        registration[field] = [invalid_text] if field == "clock_ids" else invalid_text
    response = client.post(
        "/v1/live/runs",
        content=json.dumps(registration),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert client.get("/v1/live/runs").json()["runs"] == []
    assert client.get("/readyz").status_code == 200


@pytest.mark.parametrize(
    "field", ["task_id", "step_id", "parent_step_id", "clock_id", "resource_id", "service_id"]
)
def test_invalid_event_text_rejects_whole_batch_before_storage(client, live, field):
    assert post(client, live, sample(live), event(live, **{field: "x\x00y"})).status_code == 422
    assert snapshot(client, live)["records"] == []


@pytest.mark.parametrize("kind", ["sample", "state", "step"])
def test_nul_reason_is_a_validation_error(client, live, kind):
    if kind == "sample":
        response = post(
            client,
            live,
            sample(live, value=None, availability="unavailable", reason="redacted\x00reason"),
        )
    elif kind == "state":
        response = change(client, live, "cancelled", 0, reason="redacted\x00reason")
    else:
        response = post(client, live, event(live, 1, "failed", reason="redacted\x00reason"))
    assert response.status_code == 422
    assert snapshot(client, live)["state"] == "queued"
    assert snapshot(client, live)["records"] == []


@pytest.mark.parametrize("timestamp", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
@pytest.mark.parametrize("record_kind", ["step", "sample"])
def test_utc_normalization_overflow_is_422(client, live, timestamp, record_kind):
    record = (
        event(live, observed_at=timestamp)
        if record_kind == "step"
        else sample(live, observed_at=timestamp)
    )
    assert post(client, live, record).status_code == 422
    assert snapshot(client, live)["records"] == []


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_impossible_unstarted_revision_two_is_rejected_immediately(client, live, state):
    assert post(client, live, event(live, 2, state)).status_code == 422
    assert snapshot(client, live)["records"] == []
    assert (
        post(
            client, live, event(live), event(live, 1, "running"), event(live, 2, "completed")
        ).status_code
        == 200
    )


@pytest.mark.parametrize("terminal_first", [True, False])
def test_pending_adjacent_contradiction_rolls_back_only_arriving_batch(
    client, live, terminal_first
):
    completed, failed = event(live, 2, "completed"), event(live, 1, "failed")
    first, contradictory = (completed, failed) if terminal_first else (failed, completed)
    assert post(client, live, first).status_code == 200
    response = post(client, live, contradictory, sample(live), event(live, step_id="unrelated"))
    assert response.status_code == 409
    view = snapshot(client, live)
    assert [r["payload"]["event_id"] for r in view["records"]] == [first["event_id"]]
    assert view["steps"] == []
    if terminal_first:
        assert post(client, live, event(live), event(live, 1, "running")).status_code == 200
        assert snapshot(client, live)["steps"][0]["event"]["state"] == "completed"
    else:
        assert post(client, live, event(live)).status_code == 200
        assert snapshot(client, live)["steps"][0]["event"]["state"] == "failed"


def test_pre_fix_pending_evidence_remains_readable_and_can_be_closed_as_failed(client, live):
    pending = event(live, 2, "completed")
    assert post(client, live, pending).status_code == 200
    # Simulate the impossible pending event accepted by the original PR head.
    with client.app.state.store.connect() as conn:
        conn.execute(
            "UPDATE live_records SET payload=payload || %s WHERE run_id=%s",
            (
                Jsonb(
                    {
                        "state": "failed",
                        "started_tick_ns": None,
                        "reason": "Synthetic pre-fix malformed pending evidence",
                    }
                ),
                live,
            ),
        )
    view = snapshot(client, live)
    assert view["steps"] == []
    assert view["records"][0]["disposition"] == "pending"
    assert view["gaps"][0]["missing_sequence"] == 0
    assert change(client, live, "failed", 0).status_code == 200
    assert snapshot(client, live)["unfinished_steps"] == ["tool-1"]


def test_duplicate_index_upgrade_preserves_populated_live_history(client, live):
    queued = event(live)
    assert post(client, live, queued, sample(live)).status_code == 200
    before = snapshot(client, live)
    with client.app.state.store.connect() as conn:
        # Reconstruct only the pre-0003 index/migration state in this isolated test schema.
        conn.execute(
            "CREATE INDEX live_records_steps ON live_records(run_id,step_id,step_revision)"
        )
        conn.execute(
            "DELETE FROM schema_migrations WHERE version='0003_remove_duplicate_live_step_index.sql'"
        )
    client.app.state.store.migrate()
    assert snapshot(client, live) == before
    assert post(client, live, queued).json()["added"] == 0
    assert post(client, live, event(live)).status_code == 409
    with client.app.state.store.connect() as conn:
        indexes = conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname=current_schema() "
            "AND tablename='live_records'"
        ).fetchall()
    names = {index["indexname"] for index in indexes}
    assert "live_records_steps" not in names
    assert "live_records_run_id_step_id_step_revision_key" in names


@pytest.mark.parametrize("value", [1e16, 1.2345678901234567e19, 1e23, 1.7976931348623157e308])
def test_large_float_retries_use_jsonb_numeric_semantics(client, live, value):
    reading = sample(live, metric="cpu.cgroup.throttled_usec.v1", unit="microseconds", value=value)
    completed = event(live, 2, "completed", cpu_seconds=value, cpu_scope="process")
    assert (
        post(client, live, reading, event(live), event(live, 1, "running"), completed).status_code
        == 200
    )
    assert post(client, live, reading, completed).json()["added"] == 0
    assert post(client, live, reading | {"value": 0}).status_code == 409
    assert len(snapshot(client, live)["records"]) == 4


def test_response_and_storage_additions_do_not_break_cancellation_polling(
    client, live, monkeypatch
):
    from agentwatch_sdk import client as sdk_client

    assert post(client, live, event(live), sample(live)).status_code == 200
    with client.app.state.store.connect() as conn:
        conn.execute(
            "ALTER TABLE live_runs ADD COLUMN future_note TEXT DEFAULT 'new server metadata'"
        )
        conn.execute(
            "UPDATE live_records SET payload=payload || %s WHERE run_id=%s AND record_type='resource_sample'",
            (Jsonb({"metric": "cpu.future.metric.v2", "future_sample_field": "metadata"}), live),
        )
        conn.execute(
            "UPDATE live_records SET payload=payload || %s WHERE run_id=%s AND record_type='step_event'",
            (Jsonb({"future_step_field": "metadata"}), live),
        )
    assert change(client, live, "cancelled", 0).status_code == 200
    view = snapshot(client, live)
    assert "future_note" not in view
    assert view["cleanup_requested"] is True
    assert client.get("/v1/live/runs").status_code == 200

    def add_future_fields(value):
        if isinstance(value, dict):
            return {
                **{key: add_future_fields(child) for key, child in value.items()},
                "future_response_field": "metadata",
            }
        if isinstance(value, list):
            return [add_future_fields(child) for child in value]
        return value

    response_body = json.dumps(add_future_fields(view)).encode()
    monkeypatch.setattr(sdk_client, "urlopen", lambda request, timeout: BytesIO(response_body))
    saved = LiveClient("http://testserver").get_run(UUID(live))
    assert saved.cleanup_requested is True and saved.cleanup_confirmed is None
    readings = [r.payload for r in saved.records if r.payload.record_type == "resource_sample"]
    assert readings[0].metric == "cpu.future.metric.v2"
    assert readings[0].value == 0
    # Producer writes still enforce the old registry and reject extra fields.
    with pytest.raises(ValidationError):
        ResourceSample.model_validate(sample(live, metric="cpu.future.metric.v2"))
    with pytest.raises(ValidationError):
        StepEvent.model_validate(event(live, future_step_field="metadata"))
    with pytest.raises(ValidationError):
        RunCreate.model_validate(view["registration"] | {"future_response_field": "metadata"})


@pytest.mark.parametrize("cursor", [MAX_CURSOR + 1, 99999999999999999999])
def test_cursor_overflow_is_rejected_at_api_and_sdk_boundaries(client, live, cursor):
    assert client.get(f"/v1/live/runs?before={cursor}").status_code == 422
    assert client.get(f"/v1/live/runs/{live}?after={cursor}").status_code == 422
    sdk = LiveClient("http://testserver")
    with pytest.raises(ValueError):
        sdk.list_runs(before=cursor)
    with pytest.raises(ValueError):
        sdk.get_run(UUID(live), after=cursor)
    assert client.get(f"/v1/live/runs?before={MAX_CURSOR}").status_code == 200
    assert client.get(f"/v1/live/runs/{live}?after={MAX_CURSOR}").status_code == 200


def test_error_contract_covers_semantic_and_structural_422(client, live):
    semantic = post(client, live, sample(live, clock_id="unknown"))
    structural = post(client, live)
    assert semantic.status_code == structural.status_code == 422
    assert isinstance(ErrorResponse.model_validate(semantic.json()).detail, str)
    assert isinstance(ErrorResponse.model_validate(structural.json()).detail, list)
    assert "records[0]" in semantic.json()["detail"]
    for operations in openapi_document()["paths"].values():
        for operation in operations.values():
            schema = operation["responses"]["422"]["content"]["application/json"]["schema"]
            assert schema == {"$ref": "#/components/schemas/ErrorResponse"}


@pytest.mark.parametrize(
    "mutation", [{"value": None}, {"value": True}, {"availability": "missing", "value": 0}]
)
def test_new_metric_readers_still_validate_known_evidence_shape(live, mutation):
    from agentwatch_sdk.live import ResourceSampleSnapshot

    with pytest.raises(ValidationError):
        ResourceSampleSnapshot.model_validate(
            sample(live, metric="cpu.future.metric.v2", **mutation)
        )


@pytest.mark.parametrize(
    "kind,statuses",
    [
        ("integer", {400}),
        ("nesting", {400, 422}),
        ("gzip", {400}),
    ],
)
def test_unparseable_bodies_return_documented_errors(client, live, kind, statuses):
    import gzip

    bodies = {
        "integer": b'{"records":' + b"9" * 5000 + b"}",
        "nesting": b"[" * 5000 + b"]" * 5000,
        "gzip": gzip.compress(b'{"records":[]}'),
    }
    headers = {"Content-Type": "application/json"}
    if kind == "gzip":
        headers["Content-Encoding"] = "gzip"
    response = client.post(
        f"/v1/live/runs/{live}/records",
        content=bodies[kind],
        headers=headers,
    )
    # Python JSON parsers differ on nesting limits: reject during parsing or envelope validation.
    assert response.status_code in statuses
    ErrorResponse.model_validate(response.json())
    assert snapshot(client, live)["records"] == []


@pytest.mark.parametrize(
    "exception,status",
    [
        ("OperationalError", 503),
        ("DataError", 422),
        ("ProgrammingError", 500),
    ],
)
def test_live_storage_errors_are_classified_without_exposing_values(
    client, registration, monkeypatch, caplog, exception, status
):
    import psycopg
    from agentwatch_worker.live_store import LiveStore

    def fail(self):
        raise getattr(psycopg, exception)("sensitive producer data must not be logged")

    monkeypatch.setattr(LiveStore, "connect", fail)
    response = client.post("/v1/live/runs", json=registration)
    assert response.status_code == status
    ErrorResponse.model_validate(response.json())
    assert "sensitive producer data" not in response.text
    assert "sensitive producer data" not in caplog.text
