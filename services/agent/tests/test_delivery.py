import json
from urllib.error import URLError
from uuid import uuid4

import pytest
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.delivery import (
    Journal,
    LiveDelivery,
    LiveRejectedError,
    RetryPolicy,
    replay_journal,
)
from agentwatch_agent.session import RunSession
from agentwatch_agent.steps import StepRecorder, new_trace_id
from agentwatch_sdk.live import ResourceRegistration, RunCreate
from conftest import http_error

NO_SLEEP = RetryPolicy(attempts=3, initial_backoff_seconds=0, max_backoff_seconds=0)
CLOCK = MonotonicClock("delivery-clock")
RESOURCE = ResourceRegistration(
    resource_id="tool-host-1", service_id="agentwatch-agent-runner", resource_class="cpu_sandbox"
)


def registration():
    return RunCreate(
        contract_version="live-contracts-v1",
        request_id=uuid4(),
        task_id="fix-mean-empty-r1",
        configuration_digest="b" * 64,
        clock_ids=[CLOCK.clock_id],
        resources=[RESOURCE],
    )


@pytest.fixture
def setup(server, tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    session = RunSession.open(server, registration(), journal, NO_SLEEP)
    delivery = LiveDelivery(server, session.run_id, journal, retry=NO_SLEEP, sleep=lambda _: None)
    recorder = StepRecorder(
        run_id=session.run_id,
        task_id="fix-mean-empty-r1",
        trace_id=new_trace_id(),
        clock=CLOCK,
        emit=delivery.emit,
    )
    return session, delivery, recorder, journal


def complete_step(recorder, step_id="tool-1"):
    step = recorder.submit(step_id, "tool", parent=None, resource=RESOURCE)
    recorder.start(step)
    recorder.complete(step)
    return step


def test_records_are_delivered_in_order_and_acknowledged(server, setup):
    session, delivery, recorder, _ = setup
    complete_step(recorder)
    assert delivery.flush()
    stats = delivery.stats()
    assert (stats.emitted, stats.delivered, stats.pending) == (3, 3, 0)
    assert stats.complete
    states = [e.state for e in server.runs[session.run_id].steps.values()]
    assert states == ["completed"]


def test_large_queues_are_split_into_bounded_batches(server, setup):
    session, delivery, recorder, _ = setup
    for index in range(100):
        complete_step(recorder, f"tool-{index}")
    assert delivery.flush()
    assert server.calls.count("ingest") == 2  # 300 records, at most 256 per batch
    assert len(server.runs[session.run_id].steps) == 100


def test_transient_failures_retry_identical_content(server, setup):
    _, delivery, recorder, _ = setup
    complete_step(recorder)
    server.failures = [http_error(503, "Database unavailable"), URLError("reset")]
    assert delivery.flush()
    assert server.calls[-3:] == ["ingest", "ingest", "ingest"]
    assert delivery.stats().delivered == 3


def test_outage_keeps_records_queued_and_reports_incomplete_delivery(server, setup):
    _, delivery, recorder, _ = setup
    complete_step(recorder)
    server.down = True
    assert not delivery.flush()
    stats = delivery.stats()
    assert (stats.pending, stats.transient_failures, stats.complete) == (3, 1, False)
    assert "URLError" in stats.last_error
    server.down = False
    assert delivery.flush()
    assert delivery.stats().complete


def unregistered_clock_recorder(recorder, delivery):
    """Records the service must refuse: their clock domain was never registered."""
    return StepRecorder(
        run_id=recorder.run_id,
        task_id=recorder.task_id,
        trace_id=recorder.trace_id,
        clock=MonotonicClock("unregistered-clock"),
        emit=delivery.emit,
    )


def test_rejected_records_are_reported_and_do_not_block_later_records(server, setup):
    _, delivery, recorder, journal = setup
    complete_step(unregistered_clock_recorder(recorder, delivery), "tool-1")
    complete_step(recorder, "tool-2")
    assert delivery.flush()
    stats = delivery.stats()
    assert (stats.rejected, stats.delivered, stats.complete) == (3, 3, False)
    assert {r["status"] for r in stats.rejections} == {422}
    assert set(server.runs[recorder.run_id].steps) == {"tool-2"}
    journal.close()
    kinds = [json.loads(line)["kind"] for line in journal.path.read_text().splitlines()]
    assert "rejected" in kinds


def test_overflow_is_journaled_but_not_queued(server, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    session = RunSession.open(server, registration(), journal, NO_SLEEP)
    delivery = LiveDelivery(server, session.run_id, journal, retry=NO_SLEEP, max_pending=2)
    recorder = StepRecorder(
        run_id=session.run_id,
        task_id="fix-mean-empty-r1",
        trace_id=new_trace_id(),
        clock=CLOCK,
        emit=delivery.emit,
    )
    complete_step(recorder)
    stats = delivery.stats()
    assert (stats.pending, stats.not_queued_overflow, stats.complete) == (2, 1, False)
    journal.close()
    records = [line for line in journal.path.read_text().splitlines() if '"record"' in line]
    assert len(records) == 3


def test_registration_retries_reuse_the_request_identity(server, tmp_path):
    server.failures = [URLError("timeout")]
    session = RunSession.open(server, registration(), Journal(tmp_path / "j.jsonl"), NO_SLEEP)
    assert len(server.runs) == 1
    assert session.run_id in server.runs


def test_non_retryable_status_is_raised_as_rejection(server, tmp_path):
    server.failures = [http_error(422, "Invalid")]
    with pytest.raises(LiveRejectedError) as error:
        RunSession.open(server, registration(), Journal(tmp_path / "j.jsonl"), NO_SLEEP)
    assert (error.value.status, error.value.detail) == (422, "Invalid")


def test_retries_are_bounded(server, tmp_path):
    server.down = True
    with pytest.raises(URLError):
        RunSession.open(server, registration(), Journal(tmp_path / "j.jsonl"), NO_SLEEP)
    assert server.calls.count("create_run") == 3


def test_state_conflict_adopts_a_remote_cancellation(server, setup):
    session, *_ = setup
    assert session.change_state("running", None) == "running"
    server.cancel(session.run_id)
    assert session.change_state("completed", None) == "cancelled"
    assert session.revision == server.runs[session.run_id].revision
    # A closed run is not changed again.
    assert session.change_state("failed", "late") == "cancelled"


def test_journal_replay_is_idempotent_and_recovers_an_outage(server, setup):
    session, delivery, recorder, journal = setup
    session.change_state("running", None)
    complete_step(recorder)
    server.down = True
    assert not delivery.flush()
    server.down = False
    journal.close()
    first = replay_journal(journal.path, server, NO_SLEEP)
    assert first.run_id == session.run_id
    assert first.records_sent == 3 and first.changes_sent == 1
    assert server.runs[session.run_id].steps["tool-1"].state == "completed"
    second = replay_journal(journal.path, server, NO_SLEEP)
    assert second.records_sent == 3
    assert len(server.runs[session.run_id].records) == 3


def test_replay_skips_records_the_service_already_rejected(server, setup):
    _, delivery, recorder, journal = setup
    complete_step(unregistered_clock_recorder(recorder, delivery))
    delivery.flush()
    journal.close()
    report = replay_journal(journal.path, server, NO_SLEEP)
    assert (report.records_sent, report.records_skipped_rejected) == (0, 3)


def test_replay_tolerates_only_an_interrupted_final_line(server, setup):
    *_, recorder, journal = setup
    complete_step(recorder)
    journal.close()
    with journal.path.open("a") as stream:
        stream.write('{"kind": "rec')
    assert replay_journal(journal.path, server, NO_SLEEP).truncated_final_line
    lines = journal.path.read_text().splitlines()
    journal.path.write_text("\n".join([lines[0], "{broken", *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="line 2"):
        replay_journal(journal.path, server, NO_SLEEP)


@pytest.mark.parametrize("attempts", [0, -1, True])
def test_retry_policy_validates_attempts(attempts):
    with pytest.raises(ValueError):
        RetryPolicy(attempts=attempts)


def test_closure_race_isolates_new_steps_and_keeps_late_terminal_evidence(server, setup):
    session, delivery, recorder, _ = setup
    session.change_state("running", None)
    known = recorder.submit("tool-1", "tool", parent=None, resource=RESOURCE)
    recorder.start(known)
    assert delivery.flush()
    # The dashboard cancels before the next step and the terminal events are delivered.
    server.cancel(session.run_id)
    late = recorder.submit("tool-2", "tool", parent=None, resource=RESOURCE)
    recorder.cancel(late, "Cancelled from the dashboard")
    recorder.cancel(known, "Cancelled from the dashboard")
    assert delivery.flush()
    run = server.runs[session.run_id]
    assert run.steps["tool-1"].state == "cancelled"
    assert "tool-2" not in run.steps
    stats = delivery.stats()
    assert (stats.delivered, stats.rejected, stats.complete) == (3, 2, False)
    assert {r["status"] for r in stats.rejections} == {409}
