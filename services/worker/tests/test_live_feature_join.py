"""Synthetic live fixtures joined from isolated PostgreSQL history into step-features-v1 rows."""

from uuid import UUID, uuid4

import pytest
from agentwatch_worker.live_features import live_feature_rows
from agentwatch_worker.live_store import MissingRunError
from test_live_api import change, event, post, sample

THROTTLED = {"metric": "cpu.cgroup.throttled_usec.v1", "unit": "microseconds"}
INTERVAL_SECONDS = 5e-9  # fixture ticks are nanoseconds: a 10 ns window holds two intervals


@pytest.fixture
def two_clock_run(client, registration):
    registration["clock_ids"] = ["fixture-clock", "other-clock"]
    response = client.post("/v1/live/runs", json=registration)
    assert response.status_code == 201
    return response.json()["run_id"]


def rows_for(client, run_id):
    return live_feature_rows(client.app.state.store, UUID(run_id), INTERVAL_SECONDS)


def test_applied_steps_join_samples_in_their_half_open_window(client, two_clock_run):
    run = two_clock_run
    inside = [
        sample(run, tick_ns=10, value=0),
        sample(run, tick_ns=15, value=35.5),
        sample(run, tick_ns=10, value=100, **THROTTLED),
        sample(run, tick_ns=15, value=160, **THROTTLED),
    ]
    outside = [
        sample(run, tick_ns=20, value=99),  # exactly at finish: excluded
        sample(run, tick_ns=12, value=98, clock_id="other-clock"),  # no clock mapping
    ]
    steps = [event(run), event(run, 1, "running"), event(run, 2, "completed")]
    assert post(client, run, *steps, *inside, *outside).status_code == 200
    assert post(client, run, *inside).json()["added"] == 0  # exact retries

    result = rows_for(client, run)

    assert result.incomplete_step_ids == []
    [row] = result.rows
    assert row.run_id == UUID(run)
    assert row.source_event_ids == [UUID(e["event_id"]) for e in steps]
    assert sorted(row.source_sample_ids) == sorted(UUID(s["sample_id"]) for s in inside)
    assert row.telemetry.status == "present"
    assert row.telemetry.sample_count == 2
    assert row.telemetry.coverage == 1.0
    assert row.features.cpu_pressure_max_avg10 == 35.5
    assert row.features.cgroup_throttled_usec_delta == 60.0
    assert row.features.execution_seconds == 1e-8


def test_pending_revisions_never_shape_a_window(client, live):
    early_terminal = event(live, 2, "completed", step_id="tool-2")
    records = [event(live), event(live, 1, "running"), event(live, 2, "completed")]
    assert post(client, live, *records, event(live, step_id="tool-2"), early_terminal).json()[
        "gaps"
    ]
    assert post(client, live, sample(live, tick_ns=10)).status_code == 200

    result = rows_for(client, live)

    assert [row.step_id for row in result.rows] == ["tool-1"]
    assert result.incomplete_step_ids == ["tool-2"]


def test_late_samples_after_run_completion_join_their_step(client, live):
    steps = [event(live), event(live, 1, "running"), event(live, 2, "completed")]
    assert post(client, live, *steps).status_code == 200
    assert change(client, live, "running", 0).status_code == 200
    assert change(client, live, "completed", 1).status_code == 200
    assert rows_for(client, live).rows[0].telemetry.status == "empty"

    assert post(client, live, sample(live, tick_ns=12, value=7)).status_code == 200

    [row] = rows_for(client, live).rows
    assert row.telemetry.status == "present"
    assert row.features.cpu_pressure_max_avg10 == 7.0


def test_unavailable_history_is_not_an_empty_feature_set(client):
    with pytest.raises(MissingRunError):
        live_feature_rows(client.app.state.store, uuid4(), INTERVAL_SECONDS)
