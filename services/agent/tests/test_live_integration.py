"""End-to-end runner tests against the real live API, PostgreSQL and the SDK LiveClient.

Each test serves the worker app with uvicorn on a free local port, using an isolated
PostgreSQL schema that is dropped afterwards. The model is a scripted test double, so these
runs verify instrumentation and lifecycle plumbing, not model behaviour.
"""

import os
import socket
import threading
import time
from uuid import UUID, uuid4

import psycopg
import pytest
import uvicorn
from agent_test_doubles import SOLVE_MEAN, ScriptedModel
from agentwatch_agent.delivery import RetryPolicy, replay_journal
from agentwatch_agent.runner import AgentRunner
from agentwatch_sdk import LiveClient, RunChange
from psycopg import sql
from psycopg.conninfo import make_conninfo

NO_SLEEP = RetryPolicy(attempts=2, initial_backoff_seconds=0, max_backoff_seconds=0)


@pytest.fixture
def live_api(monkeypatch):
    dsn = os.environ.get("AGENTWATCH_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set AGENTWATCH_TEST_DATABASE_URL for isolated PostgreSQL API tests")
    schema = "aw_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    monkeypatch.setenv("DATABASE_URL", make_conninfo(dsn, options=f"-csearch_path={schema}"))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    from agentwatch_worker.app import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # uvicorn exposes readiness only as a flag, so this bounded poll waits for startup.
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("Live API did not start")
        time.sleep(0.05)
    try:
        yield LiveClient(f"http://127.0.0.1:{port}", timeout_seconds=10)
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def runner(task, client, model, tmp_path):
    return AgentRunner(
        task=task,
        model=model,
        transport=client,
        output_dir=tmp_path / "run",
        sampling_interval_seconds=0.06,
        cancel_poll_seconds=0.05,
        flush_interval_seconds=0.05,
        retry=NO_SLEEP,
    )


def all_records(client, run_id: UUID) -> list:
    records, cursor = [], 0
    while True:
        page = client.get_run(run_id, after=cursor, limit=256)
        records.extend(page.records)
        if page.next_cursor is None:
            return records
        cursor = page.next_cursor


def test_completed_run_is_correlated_in_live_history(mean_task, live_api, tmp_path):
    outcome = runner(mean_task, live_api, ScriptedModel(SOLVE_MEAN), tmp_path).run()
    assert (outcome.state, outcome.delivery_complete) == ("completed", True)
    snapshot = live_api.get_run(outcome.run_id)
    assert snapshot.state == "completed"
    assert snapshot.gaps == []
    assert snapshot.registration.task_id == "fix-mean-empty-r1"
    steps = {view.event.step_id: view for view in snapshot.steps}
    assert {view.event.state for view in steps.values()} == {"completed"}
    assert len(steps) == 11
    assert {view.event.trace_id for view in steps.values()} == {steps["agent-run"].event.trace_id}
    assert steps["tool-1-search_docs"].event.step_type == "retrieval"
    assert steps["verify-tests"].timings.cpu_seconds > 0
    assert steps["model-call-1"].timings.execution_seconds > 0
    # Children run inside the root step's window on the same clock.
    root = steps["agent-run"].event
    for view in steps.values():
        assert root.submitted_tick_ns <= view.event.submitted_tick_ns
        assert view.event.finished_tick_ns <= root.finished_tick_ns
    samples = [
        r
        for r in all_records(live_api, outcome.run_id)
        if r.payload.record_type == "resource_sample"
    ]
    assert samples, "resource samples must be recorded alongside steps"


def test_dashboard_cancellation_ends_every_step(mean_task, live_api, tmp_path):
    model = ScriptedModel(SOLVE_MEAN, block=threading.Event(), block_on=2)
    agent = runner(mean_task, live_api, model, tmp_path)

    def cancel_from_dashboard():
        assert model.blocked.wait(10), "second model request never started"
        run_id = live_api.list_runs(limit=1).runs[0].run_id
        # A viewer can only see, and cancel, steps that have arrived. Background delivery is
        # asynchronous, so wait (bounded) for the in-flight step to reach the service.
        deadline = time.monotonic() + 10
        while True:
            snapshot = live_api.get_run(run_id)
            if any(view.event.step_id == "model-call-2" for view in snapshot.steps):
                break
            assert time.monotonic() < deadline, "model-call-2 never reached the live API"
        live_api.change_state(
            run_id,
            RunChange(
                change_id=uuid4(),
                expected_revision=snapshot.revision,
                state="cancelled",
                reason="Cancelled from the dashboard",
            ),
        )

    helper = threading.Thread(target=cancel_from_dashboard)
    helper.start()
    try:
        outcome = agent.run()
    finally:
        model.block.set()
        helper.join()
    snapshot = live_api.get_run(outcome.run_id)
    assert (snapshot.state, snapshot.cleanup_requested) == ("cancelled", True)
    assert snapshot.cleanup_confirmed is None
    assert snapshot.reason == "Cancelled from the dashboard"
    assert {view.event.state for view in snapshot.steps} <= {"completed", "cancelled"}
    assert {"agent-run", "model-call-2"} <= set(snapshot.unfinished_steps)
    assert outcome.state == "cancelled"
    assert outcome.delivery_complete, "late terminal evidence for known steps must be accepted"


def test_replaying_a_delivered_journal_adds_nothing(mean_task, live_api, tmp_path):
    outcome = runner(mean_task, live_api, ScriptedModel(SOLVE_MEAN), tmp_path).run()
    before = len(all_records(live_api, outcome.run_id))
    report = replay_journal(outcome.journal_path, live_api, NO_SLEEP)
    assert report.run_id == outcome.run_id
    assert report.rejections == []
    assert report.complete and report.final_state == "completed"
    assert len(all_records(live_api, outcome.run_id)) == before
    assert live_api.get_run(outcome.run_id).state == "completed"
