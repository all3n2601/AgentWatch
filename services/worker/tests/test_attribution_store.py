"""Append-only step attributions over isolated PostgreSQL history.

Every run, step, sample and model pair here is a synthetic test fixture, never recorded-run
evidence. Fixture ticks are nanoseconds: a step's window is [10, 20).
"""

import json
import logging
import math
import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest
from agentwatch_attribution.rules import Rules, load_rules
from agentwatch_features import FEATURE_VERSION, FeatureRow
from agentwatch_worker import attribution_store, live_api
from agentwatch_worker.attribution_model import ModelService
from agentwatch_worker.attribution_store import AttributionStore, StoredRow
from agentwatch_worker.live_features import live_feature_rows
from agentwatch_worker.live_store import MissingRunError
from agentwatch_worker.store import RunStore
from fastapi.testclient import TestClient
from pair_fixtures import CALIBRATION_VERSION, TRAINING_VERSION, write_synthetic_pair
from psycopg import sql
from psycopg.conninfo import make_conninfo
from test_live_api import change, event, post, sample

INTERVAL_SECONDS = 5e-9  # a 10 ns window holds two whole intervals
MIGRATION = "0005_step_attributions.sql"
# An ingest or poll that waited behind an attribution transaction would take the 3 s lock_timeout.
PROMPT_SECONDS = 2.0
SECRET = "producer-secret-7f3a"
EARLIER_MIGRATIONS = (
    "0001_run_history.sql",
    "0002_live_lifecycle.sql",
    "0003_remove_duplicate_live_step_index.sql",
    "0004_concurrency_policy.sql",
)
ATTRIBUTION_KEYS = {
    "sequence",
    "serving_key",
    "committed_at",
    "step_terminal_received_at",
    "evidence_changed",
    "record",
}
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


@pytest.fixture(autouse=True)
def no_background_attribution(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests drive AttributionStore passes directly, so ingest schedules none.

    test_attribution_api.py covers the passes the live API schedules.
    """

    def skip(store: AttributionStore, run_id: UUID, **options: bool) -> None:
        return None

    monkeypatch.setattr(live_api, "attribute_after_ingest", skip)


@pytest.fixture
def rules() -> Rules:
    return load_rules()


@pytest.fixture
def unavailable() -> ModelService:
    service = ModelService.from_environ({})
    assert service.status.state == "not_configured"
    return service


@pytest.fixture
def ready(tmp_path: Path) -> ModelService:
    candidates, calibration = write_synthetic_pair(tmp_path, blocks_promotion=True)
    service = ModelService.from_environ(
        {
            "AGENTWATCH_MODEL_CANDIDATES_DIR": str(candidates),
            "AGENTWATCH_MODEL_CALIBRATION_DIR": str(calibration),
            "AGENTWATCH_ALLOW_UNPROMOTED_MODEL": "1",
        }
    )
    assert service.status.state == "ready_not_promoted"
    return service


def store_of(client: TestClient) -> RunStore:
    store = client.app.state.store
    assert isinstance(store, RunStore)
    return store


def attribution_for(client: TestClient, rules: Rules, model: ModelService) -> AttributionStore:
    return AttributionStore(store_of(client), rules, model)


def new_run(client: TestClient, registration: dict, **changes: object) -> str:
    response = client.post(
        "/v1/live/runs", json=registration | {"request_id": str(uuid4())} | changes
    )
    assert response.status_code == 201
    return response.json()["run_id"]


def finish_step(
    client: TestClient, run_id: str, step_id: str = "tool-1", state: str = "completed"
) -> list[dict]:
    """Post a step's whole applied lifecycle ending in `state`; return its events."""
    if state == "cancelled":
        steps = [event(run_id, step_id=step_id), event(run_id, 1, "cancelled", step_id=step_id)]
    else:
        steps = [
            event(run_id, step_id=step_id),
            event(run_id, 1, "running", step_id=step_id),
            event(run_id, 2, state, step_id=step_id, started_tick_ns=10),
        ]
    assert post(client, run_id, *steps).status_code == 200
    return steps


def stored(client: TestClient, run_id: str) -> list[dict]:
    with store_of(client).connect() as conn:
        return conn.execute(
            "SELECT * FROM live_step_attributions WHERE run_id=%s ORDER BY sequence", (run_id,)
        ).fetchall()


def received_at(client: TestClient, record_id: str) -> object:
    with store_of(client).connect() as conn:
        row = conn.execute(
            "SELECT received_at FROM live_records WHERE record_id=%s", (record_id,)
        ).fetchone()
    assert row is not None
    return row["received_at"]


def model_of(record: dict) -> dict:
    model = record["model"]
    assert isinstance(model, dict)
    return model


def test_terminal_step_gets_one_row_with_rules_and_unavailable_model(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    steps = finish_step(client, live)
    assert post(client, live, sample(live, tick_ns=12, value=7)).status_code == 200
    attribution = attribution_for(client, rules, unavailable)

    assert attribution.attribute_run(UUID(live)) == 1

    [row] = stored(client, live)
    assert row["step_id"] == "tool-1"
    assert row["feature_version"] == FEATURE_VERSION
    assert row["serving_key"] == attribution.serving_key
    assert row["evidence_changed"] is False
    assert row["step_terminal_received_at"] == received_at(client, steps[-1]["event_id"])
    assert row["committed_at"] >= row["step_terminal_received_at"]
    record = row["record"]
    json.dumps(record, allow_nan=False)
    assert record["run_id"] == live
    assert record["step_state"] == "completed"
    assert record["telemetry_status"] == "present"
    # Queue 10 ns of a 20 ns step: a queue fraction of 0.5 meets the rules-v1 threshold.
    assert record["feature_row"]["features"]["queue_fraction"] == 0.5
    assert record["rules"] == {"version": rules.version, "verdict": "cpu_sandbox"}
    model = model_of(record)
    assert model["state"] == "not_configured"
    assert model["outcome"] == "unavailable"
    assert model["reason"] == "not_configured"
    assert model["predicted_class"] is None


def test_attribute_run_is_idempotent(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    finish_step(client, live)
    attribution = attribution_for(client, rules, unavailable)
    assert attribution.attribute_run(UUID(live)) == 1
    [first] = stored(client, live)

    assert attribution.attribute_run(UUID(live)) == 0
    assert attribution_for(client, rules, unavailable).attribute_run(UUID(live)) == 0

    assert stored(client, live) == [first]


def test_incomplete_steps_get_no_row(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    running = [event(live), event(live, 1, "running")]
    gapped = [event(live, step_id="tool-2"), event(live, 2, "completed", step_id="tool-2")]
    assert post(client, live, *running, *gapped).status_code == 200
    attribution = attribution_for(client, rules, unavailable)

    assert not attribution.has_pending_terminal_steps(UUID(live))
    assert attribution.attribute_run(UUID(live)) == 0
    assert stored(client, live) == []

    assert post(client, live, event(live, 2, "completed")).status_code == 200
    assert attribution.has_pending_terminal_steps(UUID(live))
    assert attribution.attribute_run(UUID(live)) == 1
    assert not attribution.has_pending_terminal_steps(UUID(live))
    assert [row["step_id"] for row in stored(client, live)] == ["tool-1"]


def test_two_threads_attributing_one_step_leave_one_row(
    client: TestClient,
    live: str,
    rules: Rules,
    unavailable: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish_step(client, live)
    passes = [attribution_for(client, rules, unavailable) for _ in range(2)]
    started, read_nothing = threading.Barrier(2), threading.Barrier(2, timeout=5)
    real_read = AttributionStore.stored_rows

    def read_then_wait(
        self: AttributionStore, conn: psycopg.Connection, run_id: UUID
    ) -> dict[str, StoredRow]:
        # Both passes see no stored row before either inserts, so the unique key must decide.
        rows = real_read(self, conn, run_id)
        read_nothing.wait()
        return rows

    monkeypatch.setattr(AttributionStore, "stored_rows", read_then_wait)

    def attribute(attribution: AttributionStore) -> int:
        started.wait()
        return attribution.attribute_run(UUID(live))

    with ThreadPoolExecutor(max_workers=2) as pool:
        inserted = list(pool.map(attribute, passes))

    assert sorted(inserted) == [0, 1]
    assert len(stored(client, live)) == 1


def test_late_sample_sets_evidence_changed_and_keeps_the_stored_row(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    finish_step(client, live)
    first_sample = sample(live, tick_ns=12, value=7)
    assert post(client, live, first_sample).status_code == 200
    assert change(client, live, "running", 0).status_code == 200
    assert change(client, live, "completed", 1).status_code == 200
    attribution = attribution_for(client, rules, unavailable)
    assert attribution.attribute_run(UUID(live)) == 1
    [before] = stored(client, live)

    assert post(client, live, sample(live, tick_ns=15, value=40)).status_code == 200
    assert attribution.attribute_run(UUID(live)) == 0

    [after] = stored(client, live)
    assert after["evidence_changed"] is True
    assert after["record"] == before["record"]
    assert after["sequence"] == before["sequence"]
    assert after["committed_at"] == before["committed_at"]
    feature_row = after["record"]["feature_row"]
    assert feature_row["source_sample_ids"] == [first_sample["sample_id"]]
    assert feature_row["features"]["cpu_pressure_max_avg10"] == 7.0
    [served] = attribution.page(UUID(live), 0, 10)["attributions"]
    assert served["evidence_changed"] is True


def test_new_serving_configuration_appends_a_new_row_and_keeps_the_old(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService, ready: ModelService
) -> None:
    finish_step(client, live)
    old = attribution_for(client, rules, unavailable)
    assert old.attribute_run(UUID(live)) == 1
    [old_row] = stored(client, live)

    new = attribution_for(client, rules, ready)
    assert new.serving_key != old.serving_key
    assert new.has_pending_terminal_steps(UUID(live))
    assert new.attribute_run(UUID(live)) == 1
    other_rules = attribution_for(client, replace(rules, version="rules-v1+synthetic"), ready)
    assert other_rules.serving_key not in {old.serving_key, new.serving_key}
    assert other_rules.attribute_run(UUID(live)) == 1

    rows = stored(client, live)
    assert [row["serving_key"] for row in rows] == [
        old.serving_key,
        new.serving_key,
        other_rules.serving_key,
    ]
    assert rows[0] == old_row
    model = model_of(rows[1]["record"])
    assert model["state"] == "ready_not_promoted"
    assert model["training_version"] == TRAINING_VERSION
    assert model["calibration_version"] == CALIBRATION_VERSION
    assert rows[2]["record"]["rules"]["version"] == "rules-v1+synthetic"
    assert not old.has_pending_terminal_steps(UUID(live))
    served = new.page(UUID(live), 0, 10)
    assert served["model"]["state"] == "ready_not_promoted"
    assert served["model"]["reason"] is None
    assert served["model"]["not_promoted"] is True
    assert set(served["model"]["identity"]) == IDENTITY_KEYS
    assert served["model"]["identity"]["training_version"] == TRAINING_VERSION
    assert isinstance(served["model"]["identity"]["class_order"], list)
    # Every stored serving configuration stays readable: the page is not filtered to one key,
    # and names the current key so superseded rows are never mistaken for current ones.
    for store, current_row in ((old, rows[0]), (new, rows[1]), (other_rules, rows[2])):
        page = store.page(UUID(live), 0, 10)
        assert page["model"]["serving_key"] == store.serving_key
        assert len(page["attributions"]) == 3
        current = [a for a in page["attributions"] if a["serving_key"] == store.serving_key]
        assert [a["sequence"] for a in current] == [current_row["sequence"]]


def test_page_cursor_and_limit(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    for step_id in ("tool-1", "tool-2", "tool-3"):
        finish_step(client, live, step_id)
    attribution = attribution_for(client, rules, unavailable)
    assert attribution.attribute_run(UUID(live)) == 3
    sequences = [row["sequence"] for row in stored(client, live)]

    first = attribution.page(UUID(live), 0, 2)
    assert first["run_id"] == live
    assert first["model"] == {
        "state": "not_configured",
        "reason": None,
        "identity": None,
        "not_promoted": None,
        "serving_key": attribution.serving_key,
    }
    assert [a["sequence"] for a in first["attributions"]] == sequences[:2]
    assert all(set(a) == ATTRIBUTION_KEYS for a in first["attributions"])
    assert first["next_cursor"] == sequences[1]
    second = attribution.page(UUID(live), first["next_cursor"], 2)
    assert [a["sequence"] for a in second["attributions"]] == sequences[2:]
    assert [a["record"]["step_id"] for a in second["attributions"]] == ["tool-3"]
    assert second["next_cursor"] is None
    assert attribution.page(UUID(live), 0, 3)["next_cursor"] is None
    beyond = attribution.page(UUID(live), sequences[-1], 2)
    assert beyond["attributions"] == []
    assert beyond["next_cursor"] is None


def test_unknown_run_raises_missing_run_error_and_empty_run_is_an_empty_page(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    attribution = attribution_for(client, rules, unavailable)
    unknown = uuid4()
    with pytest.raises(MissingRunError):
        attribution.attribute_run(unknown)
    with pytest.raises(MissingRunError):
        attribution.page(unknown, 0, 10)
    with pytest.raises(MissingRunError):
        attribution.has_pending_terminal_steps(unknown)

    assert attribution.attribute_run(UUID(live)) == 0
    assert not attribution.has_pending_terminal_steps(UUID(live))
    empty = attribution.page(UUID(live), 0, 10)
    assert empty["run_id"] == live
    assert empty["attributions"] == []
    assert empty["next_cursor"] is None


def test_attribution_never_mixes_runs(
    client: TestClient,
    live: str,
    registration: dict,
    rules: Rules,
    unavailable: ModelService,
) -> None:
    other = new_run(client, registration)
    finish_step(client, live)
    finish_step(client, other)
    live_sample, other_sample = (
        sample(live, tick_ns=12, value=7),
        sample(other, tick_ns=12, value=80),
    )
    assert post(client, live, live_sample).status_code == 200
    assert post(client, other, other_sample).status_code == 200
    attribution = attribution_for(client, rules, unavailable)

    assert attribution.attribute_run(UUID(live)) == 1
    assert attribution.has_pending_terminal_steps(UUID(other))
    assert attribution.page(UUID(other), 0, 10)["attributions"] == []
    assert attribution.attribute_run(UUID(other)) == 1

    for run_id, expected_sample in ((live, live_sample), (other, other_sample)):
        [served] = attribution.page(UUID(run_id), 0, 10)["attributions"]
        record = served["record"]
        assert record["run_id"] == run_id
        assert record["feature_row"]["run_id"] == run_id
        assert record["feature_row"]["source_sample_ids"] == [expected_sample["sample_id"]]
    assert [row["run_id"] for row in stored(client, live)] == [UUID(live)]


def test_registration_interval_reaches_telemetry_coverage(
    client: TestClient, registration: dict, rules: Rules, unavailable: ModelService
) -> None:
    declared = new_run(client, registration, sampling_interval_seconds=INTERVAL_SECONDS)
    undeclared = new_run(client, registration)
    attribution = attribution_for(client, rules, unavailable)
    features = {}
    for run_id in (declared, undeclared):
        finish_step(client, run_id)
        readings = [sample(run_id, tick_ns=10, value=3), sample(run_id, tick_ns=15, value=4)]
        assert post(client, run_id, *readings).status_code == 200
        assert attribution.attribute_run(UUID(run_id)) == 1
        [row] = stored(client, run_id)
        features[run_id] = row["record"]

    with_interval = features[declared]
    assert with_interval["feature_row"]["features"]["telemetry_coverage"] == 1.0
    assert with_interval["feature_row"]["telemetry"]["declared_interval_seconds"] == (
        INTERVAL_SECONDS
    )
    assert "telemetry_coverage" not in with_interval["missing_inputs"]
    without = features[undeclared]
    assert without["feature_row"]["features"]["telemetry_coverage"] is None
    assert without["feature_row"]["telemetry"]["declared_interval_seconds"] is None
    assert "telemetry_coverage" in without["missing_inputs"]


def test_completed_step_without_samples_is_scored_with_finite_numbers(
    client: TestClient, live: str, rules: Rules, ready: ModelService
) -> None:
    finish_step(client, live)

    assert attribution_for(client, rules, ready).attribute_run(UUID(live)) == 1

    [row] = stored(client, live)
    record = row["record"]
    json.dumps(record, allow_nan=False)
    assert record["telemetry_status"] == "empty"
    assert "cpu_pressure_max_avg10" in record["missing_inputs"]
    model = model_of(record)
    assert model["outcome"] in {"predicted", "abstained"}
    numbers = [
        model["confidence"],
        model["excess_fraction_estimate"],
        *model["calibrated_probabilities"].values(),
    ]
    assert all(isinstance(n, float) and math.isfinite(n) for n in numbers)


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_failed_and_cancelled_steps_get_a_row_that_the_model_does_not_score(
    client: TestClient, live: str, rules: Rules, ready: ModelService, state: str
) -> None:
    finish_step(client, live, state=state)

    assert attribution_for(client, rules, ready).attribute_run(UUID(live)) == 1

    [row] = stored(client, live)
    assert row["record"]["step_state"] == state
    assert row["record"]["rules"]["version"] == rules.version
    model = model_of(row["record"])
    assert model["state"] == "ready_not_promoted"
    assert model["outcome"] == "not_scored_step_state"


def test_a_step_that_cannot_be_built_stays_pending_and_the_steps_after_it_are_attributed(
    client: TestClient,
    live: str,
    rules: Rules,
    unavailable: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for step_id in ("tool-1", "tool-2", "tool-3"):
        finish_step(client, live, step_id)
    real_build = attribution_store.build_attribution

    def fail_on_second_step(row: FeatureRow, *args: object, **kwargs: object) -> dict:
        if row.step_id == "tool-2":
            raise ValueError("synthetic unusable evidence")
        return real_build(row, *args, **kwargs)

    monkeypatch.setattr(attribution_store, "build_attribution", fail_on_second_step)
    attribution = attribution_for(client, rules, unavailable)

    assert attribution.attribute_run(UUID(live)) == 2
    assert [row["step_id"] for row in stored(client, live)] == ["tool-1", "tool-3"]
    assert attribution.has_pending_terminal_steps(UUID(live))

    monkeypatch.setattr(attribution_store, "build_attribution", real_build)
    assert attribution.attribute_run(UUID(live)) == 1
    assert [row["step_id"] for row in stored(client, live)] == ["tool-1", "tool-3", "tool-2"]
    assert not attribution.has_pending_terminal_steps(UUID(live))


def test_a_fault_in_building_a_step_propagates_and_keeps_the_rows_committed_before_it(
    client: TestClient,
    live: str,
    rules: Rules,
    unavailable: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for step_id in ("tool-1", "tool-2"):
        finish_step(client, live, step_id)
    real_build = attribution_store.build_attribution

    def fail_on_second_step(row: FeatureRow, *args: object, **kwargs: object) -> dict:
        if row.step_id == "tool-2":
            raise RuntimeError("synthetic serving fault")
        return real_build(row, *args, **kwargs)

    monkeypatch.setattr(attribution_store, "build_attribution", fail_on_second_step)
    attribution = attribution_for(client, rules, unavailable)

    with pytest.raises(RuntimeError, match="synthetic serving fault"):
        attribution.attribute_run(UUID(live))
    assert [row["step_id"] for row in stored(client, live)] == ["tool-1"]


def test_pending_steps_are_scored_in_one_model_call(
    client: TestClient,
    live: str,
    rules: Rules,
    ready: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for step_id in ("tool-1", "tool-2", "tool-3"):
        finish_step(client, live, step_id)
    served = type(ready.served)
    real_predict = served.predict
    batch_sizes: list[int] = []

    def counted(self: object, rows: list) -> list:
        batch_sizes.append(len(rows))
        return real_predict(self, rows)

    monkeypatch.setattr(served, "predict", counted)

    assert attribution_for(client, rules, ready).attribute_run(UUID(live)) == 3

    assert batch_sizes == [3]
    outcomes = {model_of(row["record"])["outcome"] for row in stored(client, live)}
    assert outcomes <= {"predicted", "abstained"}


def test_a_step_the_model_cannot_score_is_unavailable_and_the_others_are_scored(
    client: TestClient,
    live: str,
    rules: Rules,
    ready: ModelService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for step_id in ("tool-1", "tool-2", "tool-3"):
        finish_step(client, live, step_id)
    served = type(ready.served)
    real_predict = served.predict
    calls: list[int] = []

    def fail_batch_and_second_row(self: object, rows: list) -> list:
        # The batch call fails; the rows are then scored alone, in step order.
        calls.append(len(rows))
        if len(rows) > 1 or calls.count(1) == 2:
            raise RuntimeError(f"synthetic LightGBM failure {SECRET}")
        return real_predict(self, rows)

    monkeypatch.setattr(served, "predict", fail_batch_and_second_row)
    with caplog.at_level(logging.WARNING):
        assert attribution_for(client, rules, ready).attribute_run(UUID(live)) == 3

    assert calls == [3, 1, 1, 1]
    by_step = {row["step_id"]: model_of(row["record"]) for row in stored(client, live)}
    assert by_step["tool-2"]["outcome"] == "unavailable"
    assert by_step["tool-2"]["reason"] == "prediction_failed"
    assert by_step["tool-2"]["predicted_class"] is None
    for step_id in ("tool-1", "tool-3"):
        assert by_step[step_id]["outcome"] in {"predicted", "abstained"}
    # Every step, the failed one included, keeps its rules verdict.
    assert all(row["record"]["rules"]["verdict"] for row in stored(client, live))
    assert "RuntimeError" in caplog.text
    assert SECRET not in caplog.text


def recorded_step_ids(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the steps each feature read of a pass was limited to, and whether it locked."""
    reads: list[list[str]] = []
    real = attribution_store.live_feature_rows

    def recording(*args: object, step_ids: list[str], lock: bool, **kwargs: object) -> object:
        assert lock is False
        reads.append(sorted(step_ids))
        return real(*args, step_ids=step_ids, lock=lock, **kwargs)

    monkeypatch.setattr(attribution_store, "live_feature_rows", recording)
    return reads


def test_a_pass_computes_features_only_for_pending_steps_when_not_rechecking(
    client: TestClient,
    live: str,
    rules: Rules,
    unavailable: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish_step(client, live, "tool-1")
    attribution = attribution_for(client, rules, unavailable)
    assert attribution.attribute_run(UUID(live)) == 1
    finish_step(client, live, "tool-2")
    reads = recorded_step_ids(monkeypatch)

    assert attribution.attribute_run(UUID(live), recheck_evidence=False) == 1
    assert reads == [["tool-2"]]
    # Nothing is pending and nothing is rechecked: no feature read at all.
    assert attribution.attribute_run(UUID(live), recheck_evidence=False) == 0
    assert reads == [["tool-2"]]


def test_a_flagged_row_is_not_recomputed_by_later_passes(
    client: TestClient,
    live: str,
    rules: Rules,
    unavailable: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish_step(client, live, "tool-1")
    finish_step(client, live, "tool-2")
    attribution = attribution_for(client, rules, unavailable)
    assert attribution.attribute_run(UUID(live)) == 2
    assert post(client, live, sample(live, tick_ns=15, value=40)).status_code == 200
    reads = recorded_step_ids(monkeypatch)

    assert attribution.attribute_run(UUID(live)) == 0
    assert reads == [["tool-1", "tool-2"]]
    assert [row["evidence_changed"] for row in stored(client, live)] == [True, True]

    # Both rows are flagged now: later passes neither read features nor open connections to
    # update them.
    assert attribution.attribute_run(UUID(live)) == 0
    assert reads == [["tool-1", "tool-2"]]


def test_a_poll_style_pass_does_not_flag_evidence_but_a_full_pass_does(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    finish_step(client, live, "tool-1")
    attribution = attribution_for(client, rules, unavailable)
    assert attribution.attribute_run(UUID(live)) == 1
    assert post(client, live, sample(live, tick_ns=15, value=40)).status_code == 200
    finish_step(client, live, "tool-2")

    assert attribution.attribute_run(UUID(live), recheck_evidence=False) == 1
    assert [row["evidence_changed"] for row in stored(client, live)] == [False, False]

    assert attribution.attribute_run(UUID(live)) == 0
    # tool-2 was stored after the sample arrived, so its row already holds it.
    assert [row["evidence_changed"] for row in stored(client, live)] == [True, False]


def test_the_feature_read_for_a_pass_does_not_wait_for_the_run_row_lock(
    client: TestClient, live: str
) -> None:
    finish_step(client, live, "tool-1")
    finish_step(client, live, "tool-2")
    store = store_of(client)

    with store.connect() as holder:
        holder.execute("SELECT 1 FROM live_runs WHERE run_id=%s FOR UPDATE", (live,))
        started = time.monotonic()
        limited = live_feature_rows(store, UUID(live), None, step_ids=["tool-2"], lock=False)
        elapsed_seconds = time.monotonic() - started

    assert [row.step_id for row in limited.rows] == ["tool-2"]
    assert elapsed_seconds < PROMPT_SECONDS
    # The default read still takes the lock; it is not asked for here.
    assert [row.step_id for row in live_feature_rows(store, UUID(live), None).rows] == [
        "tool-1",
        "tool-2",
    ]


def test_an_unlocked_feature_read_of_an_unknown_run_is_a_missing_run(client: TestClient) -> None:
    with pytest.raises(MissingRunError):
        live_feature_rows(store_of(client), uuid4(), None, lock=False)


def test_ingest_and_snapshot_proceed_while_a_pass_scores_between_inserts(
    client: TestClient,
    live: str,
    rules: Rules,
    unavailable: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish_step(client, live, "tool-1")
    finish_step(client, live, "tool-2")
    real_build = attribution_store.build_attribution
    scoring_second, release = threading.Event(), threading.Event()

    def pause_on_second_step(row: FeatureRow, *args: object, **kwargs: object) -> dict:
        if row.step_id == "tool-2":
            scoring_second.set()
            assert release.wait(timeout=10)
        return real_build(row, *args, **kwargs)

    monkeypatch.setattr(attribution_store, "build_attribution", pause_on_second_step)
    attribution = attribution_for(client, rules, unavailable)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(attribution.attribute_run, UUID(live))
        try:
            assert scoring_second.wait(timeout=5)
            started = time.monotonic()
            ingested = post(client, live, sample(live, tick_ns=12, value=7))
            polled = client.get(f"/v1/live/runs/{live}")
            elapsed_seconds = time.monotonic() - started
            # tool-1's insert committed before tool-2 was scored.
            committed = [row["step_id"] for row in stored(client, live)]
        finally:
            release.set()
        assert pending.result(timeout=10) == 2

    assert ingested.status_code == 200
    assert polled.status_code == 200
    assert committed == ["tool-1"]
    # Well under the 3 s lock_timeout an ingest would hit behind an open insert transaction.
    assert elapsed_seconds < PROMPT_SECONDS


def test_evidence_flag_takes_no_lock_on_the_run_row_but_an_insert_does(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    finish_step(client, live, "tool-1")
    finish_step(client, live, "tool-2")
    attribution = attribution_for(client, rules, unavailable)
    store = store_of(client)
    tool_1, tool_2 = live_feature_rows(store, UUID(live), None).rows
    record = attribution_store.build_attribution(
        tool_1, "completed", rules, unavailable.status, unavailable.served
    )

    def run_row_is_free() -> bool:
        with store.connect() as other:
            other.execute("SET LOCAL lock_timeout = '100ms'")
            try:
                other.execute("SELECT 1 FROM live_runs WHERE run_id=%s FOR UPDATE", (live,))
            except psycopg.errors.LockNotAvailable:
                return False
            return True

    with store.connect() as conn:
        received = attribution.terminal_steps(conn, UUID(live))
        assert (
            attribution.insert(conn, UUID(live), tool_1, record, received["tool-1"].received_at)
            == 1
        )
        # The foreign-key check holds FOR KEY SHARE on the run row until this commits.
        assert run_row_is_free() is False
    with store.connect() as conn:
        attribution.flag_evidence_changed(conn, UUID(live), tool_1)
        attribution.flag_evidence_changed(conn, UUID(live), tool_2)  # no row: updates nothing
        assert run_row_is_free() is True
    assert [row["evidence_changed"] for row in stored(client, live)] == [True]


def waits_for_an_advisory_lock(store: RunStore, pid: int, pending: Future[int]) -> bool:
    """True once backend `pid` waits for an advisory lock; False if `pending` finishes first."""
    deadline = time.monotonic() + PROMPT_SECONDS
    while time.monotonic() < deadline:
        with store.connect() as conn:
            if conn.execute(
                "SELECT 1 FROM pg_locks WHERE pid=%s AND locktype='advisory' AND NOT granted",
                (pid,),
            ).fetchone():
                return True
        # Doubles as the poll delay: returns at once if the insert already finished.
        if wait([pending], timeout=0.01).done:
            return False
    return False


def test_inserts_for_one_run_commit_in_sequence_order(
    client: TestClient, live: str, rules: Rules, unavailable: ModelService
) -> None:
    """A poller that advanced past a committed sequence must never miss an earlier one."""
    finish_step(client, live, "tool-1")
    finish_step(client, live, "tool-2")
    attribution = attribution_for(client, rules, unavailable)
    store = store_of(client)
    tool_1, tool_2 = live_feature_rows(store, UUID(live), None).rows
    records = [
        attribution_store.build_attribution(
            row, "completed", rules, unavailable.status, unavailable.served
        )
        for row in (tool_1, tool_2)
    ]
    with store.connect() as conn:
        received = attribution.terminal_steps(conn, UUID(live))
    second = store.connect()
    second_pid = second.info.backend_pid

    def insert_second() -> int:
        with second:
            second.execute("SET LOCAL lock_timeout = '5s'")
            return attribution.insert(
                second, UUID(live), tool_2, records[1], received["tool-2"].received_at
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.connect() as first:
            assert attribution.insert(
                first, UUID(live), tool_1, records[0], received["tool-1"].received_at
            )
            pending = pool.submit(insert_second)
            # tool-1's sequence is taken but not committed: tool-2 must not take a later one
            # and commit first.
            blocked = waits_for_an_advisory_lock(store, second_pid, pending)
            finished_before_commit = pending.done()
        assert pending.result(timeout=5) == 1

    assert blocked is True
    assert finished_before_commit is False
    rows = stored(client, live)
    assert [row["step_id"] for row in rows] == ["tool-1", "tool-2"]
    assert rows[0]["committed_at"] <= rows[1]["committed_at"]


def test_inserts_for_different_runs_do_not_wait_for_each_other(
    client: TestClient, live: str, registration: dict, rules: Rules, unavailable: ModelService
) -> None:
    other = new_run(client, registration)
    finish_step(client, live)
    finish_step(client, other)
    attribution = attribution_for(client, rules, unavailable)
    store = store_of(client)
    [live_row] = live_feature_rows(store, UUID(live), None).rows
    [other_row] = live_feature_rows(store, UUID(other), None).rows

    def insert(conn: psycopg.Connection, run_id: str, row: FeatureRow) -> int:
        record = attribution_store.build_attribution(
            row, "completed", rules, unavailable.status, unavailable.served
        )
        received = attribution.terminal_steps(conn, UUID(run_id))[row.step_id].received_at
        return attribution.insert(conn, UUID(run_id), row, record, received)

    with store.connect() as first:
        assert insert(first, live, live_row) == 1
        with store.connect() as second:
            # A wait on the first run's uncommitted insert would raise LockNotAvailable.
            second.execute("SET LOCAL lock_timeout = '500ms'")
            assert insert(second, other, other_row) == 1
    assert len(stored(client, live)) == len(stored(client, other)) == 1


@pytest.fixture
def pre_0005_dsn() -> Iterator[str]:
    """A schema migrated only through 0004 by the same SQL files, as an existing database is."""
    dsn = os.environ.get("AGENTWATCH_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set AGENTWATCH_TEST_DATABASE_URL for isolated PostgreSQL migration tests")
    schema = "aw_test_" + uuid4().hex
    migrations = files("agentwatch_worker").joinpath("migrations")
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    scoped = make_conninfo(dsn, options=f"-csearch_path={schema}")
    try:
        with psycopg.connect(scoped) as conn:
            conn.execute(
                "CREATE TABLE schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
            )
            for name in EARLIER_MIGRATIONS:
                conn.execute(migrations.joinpath(name).read_text(), prepare=False)
                conn.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (name,))
        yield scoped
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def schema_shape(conn: psycopg.Connection) -> dict[str, list[tuple]]:
    """Columns, constraints and indexes of every table except the one 0005 adds."""
    scope = "table_schema=current_schema() AND table_name<>'live_step_attributions'"
    return {
        "columns": conn.execute(
            "SELECT table_name,column_name,data_type,is_nullable,column_default "
            f"FROM information_schema.columns WHERE {scope} ORDER BY 1,2"
        ).fetchall(),
        "constraints": conn.execute(
            "SELECT table_name,constraint_name,constraint_type "
            f"FROM information_schema.table_constraints WHERE {scope} ORDER BY 1,2"
        ).fetchall(),
        "indexes": conn.execute(
            "SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname=current_schema() "
            "AND tablename<>'live_step_attributions' ORDER BY 1,2"
        ).fetchall(),
    }


def test_migration_is_additive_and_applies_after_0004(pre_0005_dsn: str) -> None:
    run_id, request_id = uuid4(), uuid4()
    with psycopg.connect(pre_0005_dsn) as conn:
        conn.execute(
            "INSERT INTO live_runs(run_id,request_id,registration) VALUES (%s,%s,'{}')",
            (run_id, request_id),
        )
        before = schema_shape(conn)
        existing = conn.execute("SELECT * FROM live_runs").fetchall()

    RunStore(pre_0005_dsn).migrate()
    RunStore(pre_0005_dsn).migrate()  # re-running applies nothing twice

    with psycopg.connect(pre_0005_dsn) as conn:
        assert schema_shape(conn) == before
        assert conn.execute("SELECT * FROM live_runs").fetchall() == existing
        versions = [
            row[0]
            for row in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY applied_at,version"
            ).fetchall()
        ]
        assert versions == [*EARLIER_MIGRATIONS, MIGRATION]
        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE "
            "table_schema=current_schema() AND table_name='live_step_attributions' ORDER BY 1"
        ).fetchall()
        assert [c[0] for c in columns] == sorted(
            [
                "sequence",
                "run_id",
                "step_id",
                "feature_version",
                "serving_key",
                "record",
                "step_terminal_received_at",
                "committed_at",
                "evidence_changed",
            ]
        )
        assert conn.execute("SELECT count(*) FROM live_step_attributions").fetchone() == (0,)
