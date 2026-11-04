"""Synthetic live specification fixtures, isolated PostgreSQL history tests."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from uuid import uuid4

import psycopg
import pytest
from agentwatch_sdk.live import RunCreate
from agentwatch_worker.live_store import LiveStore


def event(run_id, sequence=0, state="queued", step_id="tool-1", **changes):
    payload = {
        "contract_version": "live-contracts-v1",
        "record_type": "step_event",
        "run_id": run_id,
        "task_id": "synthetic-tool-v1",
        "event_id": str(uuid4()),
        "step_id": step_id,
        "step_type": "tool",
        "parent_step_id": None,
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "resource_id": "fixture-worker",
        "service_id": "fixture-service",
        "sequence": sequence,
        "state": state,
        "observed_at": "2026-10-14T00:00:00Z",
        "clock_id": "fixture-clock",
        "submitted_tick_ns": 0,
        "started_tick_ns": 10 if state in {"running", "completed"} else None,
        "finished_tick_ns": 20 if state in {"completed", "failed", "cancelled"} else None,
        "cpu_seconds": None,
        "cpu_scope": None,
        "reason": "Fixture failure/cancellation" if state in {"failed", "cancelled"} else None,
    }
    return payload | changes


def sample(run_id, **changes):
    return {
        "contract_version": "live-contracts-v1",
        "record_type": "resource_sample",
        "run_id": run_id,
        "task_id": "synthetic-tool-v1",
        "sample_id": str(uuid4()),
        "resource_id": "fixture-worker",
        "service_id": "fixture-service",
        "resource_class": "cpu_sandbox",
        "observed_at": "2026-10-14T01:00:00+01:00",
        "clock_id": "fixture-clock",
        "tick_ns": 10,
        "metric": "cpu.psi.some.avg10.v1",
        "unit": "percent",
        "value": 0,
        "availability": "measured",
        "reason": None,
    } | changes


def post(client, run_id, *records):
    return client.post(f"/v1/live/runs/{run_id}/records", json={"records": list(records)})


def change(client, run_id, state, revision, **changes):
    payload = {
        "change_id": str(uuid4()),
        "state": state,
        "expected_revision": revision,
        "reason": "Fixture closure" if state in {"failed", "cancelled"} else None,
    } | changes
    return client.post(f"/v1/live/runs/{run_id}/state", json=payload)


def snapshot(client, run_id):
    response = client.get(f"/v1/live/runs/{run_id}")
    assert response.status_code == 200
    return response.json()


def test_create_retry_and_concurrent_identity(client, registration):
    store = LiveStore(client.app.state.store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: store.create(RunCreate.model_validate(registration)), [0, 1])
        )
    assert len({r["run_id"] for r in results}) == 1
    assert sum(r["created"] for r in results) == 1
    retry = client.post("/v1/live/runs", json=registration)
    assert retry.status_code == 200
    assert retry.json()["created"] is False
    assert client.post("/v1/live/runs", json=registration | {"task_id": "other"}).status_code == 409


def test_ordering_prefix_retry_and_timing(client, live):
    queued, running, completed = event(live), event(live, 1, "running"), event(live, 2, "completed")
    response = post(client, live, completed)
    assert response.status_code == 200
    assert response.json()["gaps"][0]["missing_sequence"] == 0
    assert snapshot(client, live)["steps"] == []
    assert post(client, live, queued).status_code == 200
    view = snapshot(client, live)
    assert view["steps"][0]["event"]["state"] == "queued"
    assert view["gaps"][0]["missing_sequence"] == 1
    assert post(client, live, running).status_code == 200
    view = snapshot(client, live)
    assert view["gaps"] == []
    assert view["steps"][0]["event"]["state"] == "completed"
    assert view["steps"][0]["timings"]["execution_seconds"] == 1e-8
    assert view["state"] == "queued"  # Step completion never closes its run.
    assert post(client, live, queued, completed).json()["added"] == 0
    assert change(client, live, "running", 0).status_code == 200
    assert change(client, live, "completed", 1).status_code == 200


def test_conflicts_roll_back_whole_batch(client, live):
    queued = event(live)
    assert post(client, live, queued).status_code == 200
    altered = queued | {"step_type": "retrieval"}
    assert post(client, live, sample(live), altered).status_code == 409
    assert len(snapshot(client, live)["records"]) == 1
    assert post(client, live, event(live)).status_code == 409
    assert post(client, live, event(live, 1, "completed")).status_code == 422
    assert len(snapshot(client, live)["records"]) == 1


def test_identity_scope_and_metric_conflicts(client, live, registration):
    first = sample(live)
    assert post(client, live, first).status_code == 200
    assert post(client, live, first).json()["added"] == 0
    assert post(client, live, first | {"sample_id": str(uuid4()), "value": 1}).status_code == 409
    assert post(client, live, sample(live, task_id="wrong")).status_code == 409
    assert post(client, live, sample(live, clock_id="unknown")).status_code == 422
    assert post(client, live, sample(live, resource_id="unknown")).status_code == 422
    other = client.post("/v1/live/runs", json=registration | {"request_id": str(uuid4())}).json()[
        "run_id"
    ]
    assert post(client, other, first).status_code == 409
    assert post(client, other, first | {"run_id": other}).status_code == 409
    assert snapshot(client, other)["records"] == []


@pytest.mark.parametrize("availability", ["missing", "unsupported", "malformed", "unavailable"])
def test_availability_preserved(client, live, availability):
    record = sample(live, value=None, availability=availability, reason="Fixture unavailable")
    assert post(client, live, record).status_code == 200
    saved = snapshot(client, live)["records"][0]["payload"]
    assert saved["value"] is None and saved["availability"] == availability
    assert saved["observed_at"] == "2026-10-14T00:00:00Z"


@pytest.mark.parametrize(
    "mutation",
    [
        {"sequence": True},
        {"submitted_tick_ns": "0"},
        {"submitted_tick_ns": 9007199254740992},
        {"cpu_seconds": True},
        {"trace_id": "0" * 32},
        {"observed_at": "2026-10-14T00:00:00"},
        {"observed_at": 0},
        {"unknown": "x"},
        {"step_id": " "},
        {"state": "running"},
        {"parent_step_id": "tool-1"},
    ],
)
def test_invalid_events(client, live, mutation):
    assert post(client, live, event(live, **mutation)).status_code == 422
    assert snapshot(client, live)["records"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"value": True},
        {"value": "0"},
        {"value": 101},
        {"value": None},
        {"unit": "seconds"},
        {"metric": "gpu.fake.v1"},
        {"resource_class": "inference"},
        {"availability": "missing", "value": 0, "reason": "missing"},
        {"availability": "unsupported", "value": None, "reason": None},
    ],
)
def test_invalid_samples(client, live, mutation):
    assert post(client, live, sample(live, **mutation)).status_code == 422


def test_vllm_and_dcgm_readings_are_stored_with_their_availability(client, registration):
    registration["resources"] += [
        {"resource_id": "model-server", "service_id": "vllm", "resource_class": "inference"},
        {"resource_id": "gpu-dcgm", "service_id": "nvidia-dcgm", "resource_class": "inference"},
    ]
    run_id = client.post("/v1/live/runs", json=registration).json()["run_id"]

    def inference(metric, unit, value, resource_id="model-server", service_id="vllm", **more):
        return sample(
            run_id,
            resource_id=resource_id,
            service_id=service_id,
            resource_class="inference",
            metric=metric,
            unit=unit,
            value=value,
            **more,
        )

    records = [
        inference("inference.vllm.requests_waiting.v1", "requests", 2),
        inference("inference.vllm.kv_cache_usage.v1", "fraction", 0.25, tick_ns=11),
        inference("inference.vllm.request_queue.seconds_total.v1", "seconds", 0.0, tick_ns=12),
        inference(
            "gpu.dcgm.prof.sm_active.v1",
            "fraction",
            None,
            resource_id="gpu-dcgm",
            service_id="nvidia-dcgm",
            availability="unavailable",
            reason="dcgm-exporter did not export DCGM_FI_PROF_SM_ACTIVE",
        ),
        inference(
            "gpu.dcgm.energy.total_mj.v1",
            "millijoules",
            164718335,
            resource_id="gpu-dcgm",
            service_id="nvidia-dcgm",
            tick_ns=13,
        ),
    ]
    assert post(client, run_id, *records).status_code == 200
    stored = {r["payload"]["metric"]: r["payload"] for r in snapshot(client, run_id)["records"]}
    assert stored["inference.vllm.requests_waiting.v1"]["value"] == 2
    # A measured zero is kept as zero, and an absent reading as unavailable without a value.
    assert stored["inference.vllm.request_queue.seconds_total.v1"]["value"] == 0.0
    assert stored["gpu.dcgm.prof.sm_active.v1"]["availability"] == "unavailable"
    assert stored["gpu.dcgm.prof.sm_active.v1"]["value"] is None
    over = inference("inference.vllm.kv_cache_usage.v1", "fraction", 1.5, tick_ns=20)
    assert post(client, run_id, over).status_code == 422


def test_parent_order_cycles_and_missing_parent(client, live):
    assert post(client, live, event(live, parent_step_id="missing")).status_code == 422
    assert (
        post(
            client,
            live,
            event(live, step_id="a", parent_step_id="b"),
            event(live, step_id="b", parent_step_id="a"),
        ).status_code
        == 422
    )
    assert snapshot(client, live)["records"] == []
    assert (
        post(
            client, live, event(live, parent_step_id="parent"), event(live, step_id="parent")
        ).status_code
        == 200
    )


def test_gap_expiry_preserves_error(client, live):
    pending = event(live, 2, "completed")
    assert post(client, live, pending).status_code == 200
    with client.app.state.store.connect() as conn:
        conn.execute(
            "UPDATE live_records SET received_at=now()-interval '301 seconds' WHERE run_id=%s",
            (live,),
        )
    view = snapshot(client, live)
    assert view["gaps"][0]["expired"] is True
    assert view["records"][0]["disposition"] == "expired"
    assert post(client, live, event(live), event(live, 1, "running")).status_code == 200
    assert snapshot(client, live)["steps"][0]["event"]["state"] == "running"
    assert post(client, live, pending).json()["acknowledgements"][0]["disposition"] == "expired"
    assert change(client, live, "running", 0).status_code == 200
    assert change(client, live, "completed", 1).status_code == 409


def test_cancellation_closure_retry_and_late_evidence(client, live):
    assert post(client, live, event(live)).status_code == 200
    change_id = str(uuid4())
    response = change(client, live, "cancelled", 0, change_id=change_id)
    assert response.status_code == 200
    view = snapshot(client, live)
    assert view["state"] == "cancelled"
    assert view["cleanup_requested"] is True and view["cleanup_confirmed"] is None
    assert view["unfinished_steps"] == ["tool-1"]
    assert change(client, live, "cancelled", 0, change_id=change_id).json()["changed"] is False
    assert (
        change(client, live, "cancelled", 0, change_id=change_id, reason="other").status_code == 409
    )
    assert post(client, live, event(live, 1, "cancelled"), sample(live)).status_code == 200
    view = snapshot(client, live)
    assert view["unfinished_steps"] == ["tool-1"]  # Immutable closure evidence.
    assert view["steps"][0]["timings"]["queue_seconds"] is None
    assert view["steps"][0]["timings"]["execution_seconds"] is None
    assert post(client, live, event(live, step_id="late-new-step")).status_code == 409
    assert change(client, live, "running", 1).status_code == 409


def test_zero_duration_and_failed_run(client, live):
    assert (
        post(
            client,
            live,
            event(live),
            event(live, 1, "running", started_tick_ns=0),
            event(live, 2, "failed", started_tick_ns=0, finished_tick_ns=0),
        ).status_code
        == 200
    )
    timings = snapshot(client, live)["steps"][0]["timings"]
    assert timings["duration_seconds"] == 0 and timings["queue_fraction"] is None
    assert change(client, live, "failed", 0).status_code == 200


def test_batch_body_and_pending_bounds(client, live):
    assert post(client, live).status_code == 422
    assert post(client, live, *[sample(live, tick_ns=i) for i in range(257)]).status_code == 422
    assert client.post(f"/v1/live/runs/{live}/records", content=b" " * 1048577).status_code == 413
    pending = [event(live, 2, "completed", step_id=f"tool-{i}") for i in range(256)]
    assert post(client, live, *pending).status_code == 200
    assert post(client, live, event(live, 2, "completed", step_id="overflow")).status_code == 422
    assert len(snapshot(client, live)["gaps"]) == 256


def test_run_record_and_step_limits_roll_back(client, live, monkeypatch):
    from agentwatch_worker import live_store

    monkeypatch.setattr(live_store, "MAX_RECORDS", 2)
    assert post(client, live, *[sample(live, tick_ns=i) for i in range(3)]).status_code == 422
    assert snapshot(client, live)["records"] == []
    monkeypatch.setattr(live_store, "MAX_STEPS", 1)
    assert post(client, live, event(live), event(live, step_id="other")).status_code == 422
    assert snapshot(client, live)["steps"] == []


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_wire_values_are_rejected_without_reflecting_evidence(client, live, number):
    import json

    body = json.dumps({"records": [sample(live)]}).replace('"value": 0', f'"value": {number}')
    response = client.post(
        f"/v1/live/runs/{live}/records", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert all("input" not in error for error in response.json()["detail"])
    assert snapshot(client, live)["records"] == []


def test_pending_identity_changes_are_rejected_before_prefix_application(client, live):
    assert post(client, live, event(live, 2, "completed")).status_code == 200
    assert (
        post(
            client, live, event(live, 1, "running", clock_id="fixture-clock", started_tick_ns=11)
        ).status_code
        == 409
    )
    assert snapshot(client, live)["steps"] == []
    assert post(client, live, event(live, step_type="other")).status_code == 409
    assert len(snapshot(client, live)["records"]) == 1


def test_applied_terminal_steps_reject_future_evidence(client, live):
    assert post(client, live, event(live), event(live, 1, "cancelled")).status_code == 200
    assert post(client, live, event(live, 2, "completed")).status_code == 409


def test_run_revision_conflict_and_unfinished_completion(client, live):
    assert post(client, live, event(live)).status_code == 200
    assert change(client, live, "running", 0).status_code == 200
    assert change(client, live, "failed", 0).status_code == 409
    assert change(client, live, "completed", 1).status_code == 409
    assert snapshot(client, live)["state"] == "running"


def test_concurrent_step_retries_apply_once(client, live):
    from uuid import UUID

    from agentwatch_sdk.live import LiveBatch

    store = LiveStore(client.app.state.store)
    batch = LiveBatch.model_validate({"records": [event(live)]})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: store.ingest(UUID(live), batch), [0, 1]))
    assert sum(r["added"] for r in results) == 1
    assert len(snapshot(client, live)["records"]) == 1


def test_cursor_polling_and_selected_run(client, live, registration):
    assert post(client, live, event(live), sample(live)).status_code == 200
    first = client.get(f"/v1/live/runs/{live}?limit=1").json()
    second = client.get(f"/v1/live/runs/{live}?after={first['next_cursor']}&limit=1").json()
    assert first["records"][0]["record_id"] != second["records"][0]["record_id"]
    assert second["next_cursor"] is None
    other = client.post("/v1/live/runs", json=registration | {"request_id": str(uuid4())}).json()[
        "run_id"
    ]
    assert snapshot(client, other)["records"] == []
    assert client.get(f"/v1/live/runs/{uuid4()}").status_code == 404
    assert post(client, str(uuid4()), sample(live)).status_code == 404
    assert client.get("/v1/live/runs?limit=1").json()["next_cursor"] is not None


def test_migration_preserves_populated_legacy_history(client, evidence, live):
    assert client.post("/v1/runs", json=evidence).status_code == 201
    before = client.get(f"/v1/runs/{evidence['result']['run_id']}").json()
    client.app.state.store.migrate()
    after = client.get(f"/v1/runs/{evidence['result']['run_id']}").json()
    assert before == after
    assert snapshot(client, live)["state"] == "queued"
    assert client.get(f"/v1/runs/{live}").status_code == 404
    with client.app.state.store.connect() as conn:
        assert len(conn.execute("SELECT version FROM schema_migrations").fetchall()) == 3


def test_migration_upgrades_existing_schema(client, evidence):
    # Create an independent legacy schema populated before 0002 is applied.
    from importlib.resources import files

    from agentwatch_worker.store import RunStore
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    schema = "aw_upgrade_" + uuid4().hex
    dsn = client.app.state.store.dsn
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    upgraded = RunStore(make_conninfo(dsn, options=f"-csearch_path={schema}"))
    try:
        with upgraded.connect() as conn:
            conn.execute(
                files("agentwatch_worker").joinpath("migrations/0001_run_history.sql").read_text(),
                prepare=False,
            )
            conn.execute("CREATE TABLE schema_migrations(version TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO schema_migrations VALUES ('0001_run_history.sql')")
        from agentwatch_worker.contracts import RunIngest

        upgraded.ingest(RunIngest.model_validate(deepcopy(evidence)))
        upgraded.migrate()
        assert upgraded.get(evidence["result"]["run_id"])["result"] == evidence["result"]
        assert (
            LiveStore(upgraded).create(
                RunCreate.model_validate(
                    {
                        "contract_version": "live-contracts-v1",
                        "request_id": str(uuid4()),
                        "task_id": "upgrade-fixture",
                        "configuration_digest": "a" * 64,
                        "clock_ids": ["clock"],
                        "resources": [],
                    }
                )
            )["created"]
            is True
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
