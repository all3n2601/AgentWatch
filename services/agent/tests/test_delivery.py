import json
import threading
import time
from urllib.error import URLError
from uuid import uuid4

import pytest
from agent_test_doubles import http_error
from agentwatch_agent import cli
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.delivery import (
    COMPLETION_REFUSED,
    UNCLOSED_FAILED,
    Journal,
    LiveDelivery,
    LiveRejectedError,
    RetryPolicy,
    replay_journal,
)
from agentwatch_agent.session import CancellationState, RunSession
from agentwatch_agent.steps import StepRecorder, new_trace_id
from agentwatch_sdk.live import ResourceRegistration, RunChange, RunCreate

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
    assert first.records_sent == 3 and first.changes_applied == 1
    # No closure was journaled, so the run is still open: reported, not counted as done.
    assert (first.outcome, first.final_state) == ("unclosed", "running")
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


# Review regressions (PR #27) --------------------------------------------------------------


def test_stale_revision_closure_is_retried_at_the_refreshed_revision(server, setup):
    session, *_ = setup
    session.change_state("running", None)
    # An earlier change applied but its response was lost: the local revision is behind.
    session.revision, session.state = 0, "queued"
    assert session.change_state("failed", "Model request failed") == "failed"
    assert server.runs[session.run_id].state == "failed"


def test_closure_refused_on_content_is_not_retried(server, setup):
    session, delivery, recorder, _ = setup
    session.change_state("running", None)
    recorder.submit("tool-1", "tool", parent=None, resource=RESOURCE)
    assert delivery.flush()
    # Refused because a step is unfinished, not because of a stale revision: no retry.
    assert session.change_state("completed", None) == "running"
    assert server.calls.count("change_state") == 2


def test_repeated_signals_never_wait_on_cancellation_locks():
    state = CancellationState()
    state.cancel_from_signal("signal", "Cancelled by operator signal SIGINT")
    holding, release = threading.Event(), threading.Event()

    def hold_lock():
        with state._lock:
            holding.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert holding.wait(5)
    started = time.monotonic()
    state.cancel_from_signal("signal", "Cancelled by operator signal SIGTERM")
    assert time.monotonic() - started < 0.5, "a repeated signal blocked on the lock"
    release.set()
    holder.join()
    assert state.request.reason == "Cancelled by operator signal SIGINT"


def test_cancellation_lock_is_reentrant_for_an_interrupted_main_thread():
    state = CancellationState()
    with state._lock:  # A plain Lock would deadlock here, as a signal handler would.
        state.cancel("signal", "inside")
    assert state.event.is_set()


def test_replay_isolates_a_refused_record_instead_of_dropping_its_batch(server, setup):
    session, delivery, recorder, journal = setup
    server.unavailable_calls = {"ingest"}
    complete_step(unregistered_clock_recorder(recorder, delivery), "tool-1")
    complete_step(recorder, "tool-2")
    assert not delivery.flush()
    server.unavailable_calls = set()
    journal.close()
    report = replay_journal(journal.path, server, NO_SLEEP)
    assert report.records_sent == 3
    assert {r["status"] for r in report.rejections} == {422} and len(report.rejections) == 3
    assert set(server.runs[session.run_id].steps) == {"tool-2"}
    assert not report.complete


def test_replay_closes_a_permanently_refused_completion_as_failed(server, setup):
    session, delivery, recorder, journal = setup
    session.change_state("running", None)
    step = recorder.submit("tool-1", "tool", parent=None, resource=RESOURCE)
    recorder.start(step)  # never finished, so completion must be refused
    assert delivery.flush()
    session.defer_change("completed", None)
    journal.close()
    report = replay_journal(journal.path, server, NO_SLEEP)
    # Every record was sent, so retrying cannot help: the run is closed as failed, truthfully.
    assert (report.completion_refused, report.final_state, report.outcome) == (
        True,
        "failed",
        "refused",
    )
    assert server.runs[session.run_id].reason == COMPLETION_REFUSED


def test_replay_retries_a_stale_revision_closure_once(server, setup):
    session, delivery, _, journal = setup
    session.change_state("running", None)
    delivery.flush()
    stale = RunChange(change_id=uuid4(), expected_revision=0, state="failed", reason="late")
    journal.write("run_change", stale, deferred=True)
    journal.close()
    report = replay_journal(journal.path, server, NO_SLEEP)
    assert (report.complete, report.final_state) == (True, "failed")
    replayed = [
        json.loads(line)
        for line in journal.path.read_text().splitlines()
        if '"replayed": true' in line
    ]
    assert replayed[0]["payload"]["expected_revision"] == 1


def test_replay_accepts_a_run_closed_by_another_actor(server, setup):
    session, delivery, _, journal = setup
    session.change_state("running", None)
    delivery.flush()
    session.defer_change("failed", "Model request failed")
    server.cancel(session.run_id)
    journal.close()
    report = replay_journal(journal.path, server, NO_SLEEP)
    assert (report.final_state, report.changes_superseded, report.complete) == (
        "cancelled",
        1,
        True,
    )


def test_resend_exits_five_when_completion_is_permanently_refused(server, setup, monkeypatch):
    session, delivery, recorder, journal = setup
    session.change_state("running", None)
    recorder.start(recorder.submit("tool-1", "tool", parent=None, resource=RESOURCE))
    delivery.flush()
    session.defer_change("completed", None)
    journal.close()
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_RECORDS_REFUSED


def test_flush_is_not_complete_while_records_overflowed(server, tmp_path):
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
    assert delivery.flush() is False, "queue drained but one record exists only in the journal"


def test_resend_exits_five_for_records_refused_during_the_run(server, setup, monkeypatch):
    session, delivery, recorder, journal = setup
    session.change_state("running", None)
    complete_step(unregistered_clock_recorder(recorder, delivery), "tool-1")
    delivery.flush()
    session.change_state("failed", "x")
    journal.close()
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_RECORDS_REFUSED


# Third review round (pre-merge audit) -----------------------------------------------------


def test_resend_while_the_api_is_down_exits_three_without_a_traceback(
    server, setup, monkeypatch, capsys
):
    _, _, _, journal = setup
    journal.close()
    server.down = True
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_INCOMPLETE_DELIVERY
    assert "unreachable" in capsys.readouterr().err


def test_resend_of_an_unusable_journal_is_a_usage_error(tmp_path, server, monkeypatch):
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    assert cli.main(["resend", str(tmp_path / "missing.jsonl")]) == cli.EXIT_USAGE
    (tmp_path / "bad.jsonl").write_text("{not json\n{}\n")
    assert cli.main(["resend", str(tmp_path / "bad.jsonl")]) == cli.EXIT_USAGE


def test_never_registered_journal_is_refused_without_creating_a_run(server, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.write("registration", registration())  # registration refused: nothing else
    journal.close()
    with pytest.raises(ValueError, match="never registered"):
        replay_journal(journal.path, server, NO_SLEEP)
    assert server.runs == {}


def test_unconfirmed_registration_is_closed_by_resend(server, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    request = registration()
    server.create_run(request)  # the service recorded it; the runner never saw the answer
    server.failures = [URLError("timed out")] * 3
    with pytest.raises(URLError):
        RunSession.open(server, request, journal, NO_SLEEP)
    journal.close()
    run = server.only_run()
    assert run.state == "queued"
    report = replay_journal(journal.path, server, NO_SLEEP)
    assert (report.outcome, run.state, report.run_created_by_replay) == ("settled", "failed", False)
    assert run.reason == "Runner could not confirm registration; no task work ran"


def test_journal_without_a_closure_is_unclosed_until_explicitly_failed(server, setup, monkeypatch):
    session, delivery, recorder, journal = setup
    session.change_state("running", None)
    complete_step(recorder)
    delivery.flush()
    journal.close()  # the runner stopped here, e.g. SIGKILL
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_RUN_UNCLOSED
    assert server.runs[session.run_id].state == "running"
    assert cli.main(["resend", "--fail-unclosed", str(journal.path)]) == cli.EXIT_PASSED
    run = server.runs[session.run_id]
    assert (run.state, run.reason) == ("failed", UNCLOSED_FAILED)
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_PASSED  # idempotent after


def test_remote_closure_is_journaled_even_if_the_refresh_fails(server, setup):
    session, _, _, journal = setup
    session.change_state("running", None)
    server.cancel(session.run_id, "Stopped by reviewer")
    server.unavailable_calls = {"get_run"}
    with pytest.raises(URLError):
        session.adopt_remote_closure("cancelled", "Stopped by reviewer")
    server.unavailable_calls = set()
    journal.close()
    report = replay_journal(journal.path, server, NO_SLEEP)
    assert (report.outcome, report.expected_state) == ("settled", "cancelled")


def test_after_an_overflow_later_records_stay_journal_only_to_keep_order(server, tmp_path):
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
    step = recorder.submit("tool-1", "tool", parent=None, resource=RESOURCE)
    recorder.start(step)
    recorder.complete(step)  # third record overflows the 2-record queue
    assert not delivery.flush()  # queue drained, but revision 2 is journal-only
    other = recorder.submit("tool-2", "tool", parent=None, resource=RESOURCE)
    assert delivery.stats().pending == 0, "a later record was queued ahead of an overflowed one"
    assert other.state == "queued"
    assert server.runs[session.run_id].steps["tool-1"].sequence == 1


def test_journal_reopened_after_an_interrupted_write_starts_a_new_line(tmp_path):
    path = tmp_path / "j.jsonl"
    path.write_text('{"kind": "registration", "payload": {}}\n{"kind": "rec')
    journal = Journal(path)
    journal.write("rejected", {"status": 409, "detail": "x"}, record_ids=[])
    journal.close()
    lines = path.read_text().splitlines()
    assert lines[1] == '{"kind": "rec'
    assert json.loads(lines[2])["kind"] == "repaired_interrupted_line"
    assert json.loads(lines[3])["kind"] == "rejected"


# Final review (round-three changes) -------------------------------------------------------


def test_resend_twice_after_a_refused_completion_keeps_reporting_it(server, setup, monkeypatch):
    session, delivery, recorder, journal = setup
    session.change_state("running", None)
    recorder.start(recorder.submit("tool-1", "tool", parent=None, resource=RESOURCE))
    assert delivery.flush()
    session.defer_change("completed", None)
    journal.close()
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_RECORDS_REFUSED
    second = replay_journal(journal.path, server, NO_SLEEP)
    assert (second.outcome, second.completion_refused, second.changes_superseded) == (
        "refused",
        True,
        0,
    )
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_RECORDS_REFUSED


def test_truncated_journal_stays_replayable_after_replay_appends(server, setup, monkeypatch):
    session, delivery, recorder, journal = setup
    session.change_state("running", None)
    complete_step(recorder)
    delivery.flush()
    journal.close()
    with journal.path.open("a") as stream:
        stream.write('{"kind": "rec')  # the runner was killed mid-write
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    assert cli.main(["resend", "--fail-unclosed", str(journal.path)]) == cli.EXIT_PASSED
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_PASSED
    assert cli.main(["resend", str(journal.path)]) == cli.EXIT_PASSED


def test_damaged_lines_without_a_repair_marker_are_still_rejected(tmp_path, server):
    path = tmp_path / "j.jsonl"
    path.write_text('{"kind": "registration", "payload": {}}\n{"kind": "rec\n{"kind": "x"}\n')
    with pytest.raises(ValueError, match="line 2"):
        replay_journal(path, server, NO_SLEEP)
