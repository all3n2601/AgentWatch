"""Live attribution API over isolated PostgreSQL history.

Every run, step, sample and model pair here is a synthetic test fixture, never recorded-run
evidence. Fixture ticks are nanoseconds: a step's window is [10, 20). TestClient runs a
response's background tasks before returning it, so a scheduled attribution pass has finished
when a POST returns.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest
from agentwatch_attribution.rules import load_rules
from agentwatch_sdk.attribution import AttributionPage, ModelStatusView
from agentwatch_sdk.live import MAX_CURSOR
from agentwatch_worker.attribution_model import ModelService
from agentwatch_worker.attribution_store import AttributionStore
from agentwatch_worker.export_live_contracts import (
    attribution_openapi_document,
    attribution_typescript_document,
    openapi_document,
)
from agentwatch_worker.store import RunStore
from fastapi.testclient import TestClient
from pair_fixtures import CALIBRATION_VERSION, TRAINING_VERSION, write_synthetic_pair
from test_live_api import change, event, post, sample

ROOT = Path(__file__).resolve().parents[3]
CANDIDATES = "AGENTWATCH_MODEL_CANDIDATES_DIR"
CALIBRATION_DIR = "AGENTWATCH_MODEL_CALIBRATION_DIR"
ALLOW_UNPROMOTED = "AGENTWATCH_ALLOW_UNPROMOTED_MODEL"
SECRET = "producer-secret-7f3a"
IDENTITY_KEYS = {
    "training_version",
    "calibration_version",
    "classifier_sha256",
    "regressor_sha256",
    "preprocessing_sha256",
    "calibration_sha256",
    "temperature",
    "abstention_threshold",
    "class_order",
    "blocks_promotion",
}


def attributions_url(run_id: object) -> str:
    return f"/v1/live/runs/{run_id}/attributions"


def store_of(client: TestClient) -> RunStore:
    store = client.app.state.store
    assert isinstance(store, RunStore)
    return store


def install(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, model: ModelService
) -> AttributionStore:
    """Serve `model` through the app for the rest of the test."""
    attribution = AttributionStore(store_of(client), load_rules(), model)
    monkeypatch.setattr(client.app.state, "attribution", attribution)
    return attribution


@pytest.fixture
def unconfigured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> AttributionStore:
    """A not_configured store, whatever model variables the test environment carries."""
    return install(client, monkeypatch, ModelService.from_environ({}))


def new_run(client: TestClient, registration: dict) -> str:
    response = client.post("/v1/live/runs", json=registration | {"request_id": str(uuid4())})
    assert response.status_code == 201
    return response.json()["run_id"]


def finish_step(client: TestClient, run_id: str, step_id: str = "tool-1") -> None:
    steps = [
        event(run_id, step_id=step_id),
        event(run_id, 1, "running", step_id=step_id),
        event(run_id, 2, "completed", step_id=step_id),
    ]
    assert post(client, run_id, *steps).status_code == 200


def stored(client: TestClient, run_id: str) -> list[dict]:
    with store_of(client).connect() as conn:
        return conn.execute(
            "SELECT step_id,sequence,evidence_changed FROM live_step_attributions "
            "WHERE run_id=%s ORDER BY sequence",
            (run_id,),
        ).fetchall()


def poll(client: TestClient, run_id: str, **params: int) -> AttributionPage:
    response = client.get(attributions_url(run_id), params=params)
    assert response.status_code == 200
    return AttributionPage.model_validate(response.json())


def count_calls(monkeypatch: pytest.MonkeyPatch, attribution: AttributionStore) -> list[UUID]:
    """Record each attribute_run call on `attribution`, still running the real pass."""
    calls: list[UUID] = []
    real = attribution.attribute_run

    def counted(run_id: UUID, **options: bool) -> int:
        calls.append(run_id)
        return real(run_id, **options)

    monkeypatch.setattr(attribution, "attribute_run", counted)
    return calls


def pair_environment(tmp_path: Path, blocks_promotion: bool, **extra: str) -> dict[str, str]:
    candidates, calibration = write_synthetic_pair(tmp_path, blocks_promotion=blocks_promotion)
    return {CANDIDATES: str(candidates), CALIBRATION_DIR: str(calibration), **extra}


def test_lifespan_serves_an_attribution_store(client: TestClient) -> None:
    attribution = client.app.state.attribution
    assert isinstance(attribution, AttributionStore)
    assert attribution.store is client.app.state.store


def test_terminal_event_makes_its_attribution_visible_on_the_next_poll(
    client: TestClient, live: str, unconfigured: AttributionStore
) -> None:
    assert post(client, live, event(live), event(live, 1, "running")).status_code == 200
    assert stored(client, live) == []

    assert post(client, live, event(live, 2, "completed")).status_code == 200

    # The background pass committed the row before the POST returned, before any read.
    assert [row["step_id"] for row in stored(client, live)] == ["tool-1"]
    page = poll(client, live)
    assert page.run_id == UUID(live)
    [item] = page.attributions
    assert item.serving_key == unconfigured.serving_key == page.model.serving_key
    assert item.evidence_changed is False
    assert item.committed_at >= item.step_terminal_received_at
    assert item.record.step_id == "tool-1"
    assert item.record.run_id == UUID(live)
    assert page.model.state == "not_configured"
    assert page.next_cursor is None


def test_poll_with_cursor_returns_only_new_rows(
    client: TestClient, live: str, unconfigured: AttributionStore
) -> None:
    for step_id in ("tool-1", "tool-2", "tool-3"):
        finish_step(client, live, step_id)

    first = poll(client, live, limit=2)
    assert [a.record.step_id for a in first.attributions] == ["tool-1", "tool-2"]
    assert first.next_cursor == first.attributions[-1].sequence
    second = poll(client, live, after=first.next_cursor, limit=2)
    assert [a.record.step_id for a in second.attributions] == ["tool-3"]
    assert second.next_cursor is None
    last_seen = second.attributions[-1].sequence

    assert poll(client, live, after=last_seen).attributions == []
    finish_step(client, live, "tool-4")
    newer = poll(client, live, after=last_seen)
    assert [a.record.step_id for a in newer.attributions] == ["tool-4"]
    assert newer.attributions[0].sequence > last_seen


def test_unknown_run_is_404_and_run_without_terminal_steps_is_an_empty_page(
    client: TestClient, live: str, unconfigured: AttributionStore
) -> None:
    unknown = client.get(attributions_url(uuid4()))
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "Live run not found"}

    assert post(client, live, event(live), event(live, 1, "running")).status_code == 200
    empty = client.get(attributions_url(live))
    assert empty.status_code == 200
    assert empty.json() == {
        "run_id": live,
        "model": unconfigured.model_status(),
        "attributions": [],
        "next_cursor": None,
    }
    assert empty.json() != unknown.json()


def test_model_status_endpoint_for_each_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocked = pair_environment(tmp_path / "blocked", blocks_promotion=True)
    promoted = pair_environment(tmp_path / "promoted", blocks_promotion=False)
    cases = [
        ({}, "not_configured", None, None),
        ({CANDIDATES: blocked[CANDIDATES]}, "load_failed", "artifact_unreadable", None),
        (blocked, "not_promoted_refused", None, True),
        (blocked | {ALLOW_UNPROMOTED: "1"}, "ready_not_promoted", None, True),
        (promoted, "ready_promoted", None, False),
    ]
    for environ, state, reason, not_promoted in cases:
        attribution = install(client, monkeypatch, ModelService.from_environ(environ))
        response = client.get("/v1/live/attribution/model")
        assert response.status_code == 200, state
        body = response.json()
        assert body == attribution.model_status(), state
        status = ModelStatusView.model_validate(body)
        assert status.state == state
        assert status.reason == reason
        assert status.not_promoted is not_promoted
        assert status.serving_key == attribution.serving_key
        if not_promoted is None:
            assert body["identity"] is None, state
            continue
        assert set(body["identity"]) == IDENTITY_KEYS, state
        assert status.identity is not None
        assert status.identity.training_version == TRAINING_VERSION
        assert status.identity.calibration_version == CALIBRATION_VERSION
        assert status.identity.blocks_promotion is not_promoted


def test_a_ready_model_scores_the_step_the_poll_returns(
    client: TestClient, live: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environ = pair_environment(tmp_path, blocks_promotion=True, **{ALLOW_UNPROMOTED: "1"})
    install(client, monkeypatch, ModelService.from_environ(environ))
    finish_step(client, live)

    page = poll(client, live)
    assert page.model.state == "ready_not_promoted"
    assert page.model.not_promoted is True
    [item] = page.attributions
    model = item.record.model
    assert model.state == "ready_not_promoted"
    assert model.outcome in {"predicted", "abstained"}
    assert model.training_version == TRAINING_VERSION
    assert model.not_promoted is True


def test_ingest_succeeds_and_next_read_reconciles_when_attribution_fails_after_commit(
    client: TestClient,
    live: str,
    unconfigured: AttributionStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real = unconfigured.attribute_run
    failures: list[UUID] = []

    def fail_once(run_id: UUID, **options: bool) -> int:
        if not failures:
            failures.append(run_id)
            raise psycopg.OperationalError(f"connection lost while storing {SECRET}")
        return real(run_id, **options)

    monkeypatch.setattr(unconfigured, "attribute_run", fail_once)
    steps = [event(live), event(live, 1, "running"), event(live, 2, "completed")]
    with caplog.at_level(logging.WARNING):
        response = post(client, live, *steps)

    assert response.status_code == 200
    assert response.json()["added"] == 3
    assert failures == [UUID(live)]
    assert stored(client, live) == []
    snapshot = client.get(f"/v1/live/runs/{live}").json()
    assert [step["event"]["state"] for step in snapshot["steps"]] == ["completed"]
    assert "OperationalError" in caplog.text
    assert SECRET not in caplog.text

    page = poll(client, live)
    assert [a.record.step_id for a in page.attributions] == ["tool-1"]
    assert [row["step_id"] for row in stored(client, live)] == ["tool-1"]


@pytest.mark.parametrize(
    "failure",
    [
        ValueError(f"Row 0 feature cpu_seconds is not a finite number: {SECRET}"),
        psycopg.errors.LockNotAvailable(f"canceling statement due to lock timeout {SECRET}"),
    ],
    ids=["unusable_evidence", "lock_timeout"],
)
def test_a_poll_still_returns_stored_rows_when_the_reconcile_pass_fails(
    client: TestClient,
    live: str,
    unconfigured: AttributionStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    finish_step(client, live, "tool-1")
    assert [a.record.step_id for a in poll(client, live).attributions] == ["tool-1"]

    def broken(run_id: UUID, **options: bool) -> int:
        raise failure

    monkeypatch.setattr(unconfigured, "attribute_run", broken)
    with caplog.at_level(logging.WARNING):
        # The pass after this ingest fails too, so tool-2 is still pending when polled.
        finish_step(client, live, "tool-2")
        page = poll(client, live)

    assert [a.record.step_id for a in page.attributions] == ["tool-1"]
    assert type(failure).__name__ in caplog.text
    assert SECRET not in caplog.text


def test_a_background_pass_logs_unusable_evidence_by_type_only(
    client: TestClient,
    live: str,
    unconfigured: AttributionStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def unusable(run_id: UUID, **options: bool) -> int:
        raise ValueError(f"Row 0 feature cpu_seconds is not a finite number: {SECRET}")

    monkeypatch.setattr(unconfigured, "attribute_run", unusable)
    with caplog.at_level(logging.WARNING):
        response = post(
            client, live, event(live), event(live, 1, "running"), event(live, 2, "completed")
        )

    assert response.status_code == 200
    assert "ValueError" in caplog.text
    assert SECRET not in caplog.text


def test_a_poll_computes_only_the_pending_steps_and_leaves_evidence_to_the_close_pass(
    client: TestClient, live: str, unconfigured: AttributionStore
) -> None:
    finish_step(client, live, "tool-1")
    # A late sample inside tool-1's window, then another step finishing in a batch of its own.
    assert post(client, live, sample(live, tick_ns=15, value=40)).status_code == 200
    finish_step(
        client,
        live,
        "tool-2",
    )

    page = poll(client, live)

    assert [a.record.step_id for a in page.attributions] == ["tool-1", "tool-2"]
    assert [a.evidence_changed for a in page.attributions] == [False, False]


def test_ingest_does_not_swallow_programming_faults_in_the_background_pass(
    client: TestClient, live: str, unconfigured: AttributionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(run_id: UUID, **options: bool) -> int:
        raise TypeError("record builder fault")

    monkeypatch.setattr(unconfigured, "attribute_run", broken)
    # TestClient re-raises an exception escaping a background task; a server would log it.
    with pytest.raises(TypeError, match="record builder fault"):
        post(client, live, event(live), event(live, 1, "running"), event(live, 2, "completed"))
    # The evidence was committed before the background pass ran.
    snapshot = client.get(f"/v1/live/runs/{live}").json()
    assert [step["event"]["state"] for step in snapshot["steps"]] == ["completed"]


def test_non_terminal_batches_do_not_trigger_attribution(
    client: TestClient, live: str, unconfigured: AttributionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = count_calls(monkeypatch, unconfigured)

    assert post(client, live, event(live), event(live, 1, "running")).status_code == 200
    assert post(client, live, sample(live, tick_ns=12, value=7)).status_code == 200
    assert change(client, live, "running", 0).status_code == 200
    assert calls == []

    # A rejected terminal batch commits nothing and schedules nothing.
    conflicting = event(live, 2, "completed", task_id="other-task")
    assert post(client, live, conflicting).status_code in {409, 422}
    assert calls == []
    assert stored(client, live) == []

    assert post(client, live, event(live, 2, "completed")).status_code == 200
    assert calls == [UUID(live)]


def test_invalid_cursor_and_limit_are_422(
    client: TestClient, live: str, unconfigured: AttributionStore
) -> None:
    for params in (
        {"after": -1},
        {"after": MAX_CURSOR + 1},
        {"limit": 0},
        {"limit": 257},
        {"after": "next"},
    ):
        assert client.get(attributions_url(live), params=params).status_code == 422, params
    assert poll(client, live, after=MAX_CURSOR, limit=256).attributions == []
    assert client.get(attributions_url("not-a-uuid")).status_code == 422


def test_attribution_error_bodies_do_not_echo_producer_values(
    client: TestClient, live: str, unconfigured: AttributionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = [
        client.get(attributions_url(SECRET)),
        client.get(attributions_url(live), params={"after": SECRET}),
        client.get(attributions_url(live), params={"limit": SECRET}),
    ]
    for response in invalid:
        assert response.status_code == 422
        assert SECRET not in response.text
        assert all(set(issue) == {"loc", "msg", "type"} for issue in response.json()["detail"])

    finish_step(client, live)

    def leak(error: type[psycopg.Error]) -> Callable[[UUID], list[str]]:
        def pending(run_id: UUID) -> list[str]:
            raise error(f"value {SECRET} violates a constraint")

        return pending

    for error, status, detail in (
        (psycopg.DatabaseError, 500, "Internal storage failure"),
        (psycopg.OperationalError, 503, "Database unavailable"),
    ):
        monkeypatch.setattr(unconfigured, "has_attribution_work", leak(error))
        response = client.get(attributions_url(live))
        assert response.status_code == status
        assert response.json() == {"detail": detail}


def test_run_close_triggers_attribution_audit(
    client: TestClient, live: str, unconfigured: AttributionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    finish_step(client, live)
    assert [row["evidence_changed"] for row in stored(client, live)] == [False]
    # A late sample inside the step's window, in a batch that triggers nothing.
    assert post(client, live, sample(live, tick_ns=15, value=40)).status_code == 200
    assert change(client, live, "running", 0).status_code == 200
    assert [row["evidence_changed"] for row in stored(client, live)] == [False]
    calls = count_calls(monkeypatch, unconfigured)

    assert change(client, live, "completed", 1).status_code == 200

    assert calls == [UUID(live)]
    assert [row["evidence_changed"] for row in stored(client, live)] == [True]
    [item] = poll(client, live).attributions
    assert item.evidence_changed is True
    assert item.record.feature_row["source_sample_ids"] == []


def test_attributions_stay_within_the_selected_run(
    client: TestClient, live: str, registration: dict, unconfigured: AttributionStore
) -> None:
    other = new_run(client, registration)
    finish_step(client, live, "live-step")
    finish_step(client, other, "other-step")

    for run_id, step_id in ((live, "live-step"), (other, "other-step")):
        page = poll(client, run_id)
        assert page.run_id == UUID(run_id)
        assert [a.record.step_id for a in page.attributions] == [step_id]
        assert [a.record.run_id for a in page.attributions] == [UUID(run_id)]
    missing = client.get(attributions_url(uuid4()))
    assert missing.status_code == 404


def test_published_attribution_contracts_match_router() -> None:
    document = attribution_openapi_document()
    assert json.loads((ROOT / "api/openapi/attribution.json").read_text()) == document
    assert (ROOT / "packages/sdk-typescript/src/attribution.ts").read_text() == (
        attribution_typescript_document(document, openapi_document())
    )
    assert set(document["paths"]) == {
        "/v1/live/runs/{run_id}/attributions",
        "/v1/live/attribution/model",
    }


def test_attribution_typescript_declares_only_new_schemas() -> None:
    text = attribution_typescript_document(attribution_openapi_document(), openapi_document())
    assert "export interface AttributionPage {" in text
    assert "  record: AttributionRecord;" in text
    assert "export interface AttributionRecord {" in text
    assert "  calibrated_probabilities: Record<string, number> | null;" in text
    assert '  outcome: "predicted" | "abstained" | "not_scored_step_state" | "unavailable";' in text
    assert "export interface ErrorResponse" not in text
    assert "export interface ValidationIssue" not in text
