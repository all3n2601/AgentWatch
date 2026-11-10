"""The runner under runner-process admission (O03): every model request and test job waits for
a permit, records the decision, and returns the permit on every path. Batches share one
controller. The model, live API and policy service are test doubles, never recorded evidence.
"""

import json
import threading
import time
from pathlib import Path
from urllib.error import URLError

import pytest
from agent_test_doubles import (
    SOLVE_MEAN,
    FakeLiveServer,
    FakePolicySource,
    ScriptedModel,
    policy_version,
)
from agentwatch_agent.admission import (
    AdmissionController,
    AdmissionSettings,
    PolicyError,
    PolicyWatcher,
    QueueBounds,
    RunnerAdmission,
)
from agentwatch_agent.batch import BATCH_FILE, MAX_BATCH_RUNS, AgentBatch
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import RUNNER_VERSION, RunLimits
from agentwatch_agent.coverage import coverage_report
from agentwatch_agent.delivery import RetryPolicy
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.runner import ADMISSION_LIMITATION, NO_ADMISSION, AgentRunner

NO_SLEEP = RetryPolicy(attempts=2, initial_backoff_seconds=0, max_backoff_seconds=0)
ADMITTED_STEPS = {
    **{f"model-call-{i}": "inference_requests" for i in range(1, 6)},
    "tool-4-run_tests": "test_jobs",
    "verify-tests": "test_jobs",
}


def make_admission(policy, *, inference_service="ollama", watcher=None, **bounds):
    settings = AdmissionSettings(QueueBounds(**bounds), QueueBounds(**bounds))
    controller = AdmissionController(settings, inference_service_id=inference_service)
    return RunnerAdmission(controller, FakePolicySource(policy), watcher, retry=NO_SLEEP)


def make_runner(task, server, model, output_dir, **overrides):
    settings = {
        "task": task,
        "model": model,
        "transport": server,
        "output_dir": output_dir,
        "sampling_interval_seconds": 0.06,
        "cancel_poll_seconds": 0.05,
        "flush_interval_seconds": 0.05,
        "retry": NO_SLEEP,
        "proc": output_dir.parent / "no-procfs",
    } | overrides
    return AgentRunner(**settings)


def evidence(path):
    return json.loads(Path(path).read_text())


def held_permit(admission, control="test_jobs"):
    """A permit held by other work in this process, so the run must wait or be refused."""
    admission.ensure_policy()
    return admission.controller.acquire(
        control,
        step_id="other-work",
        span_id="other-span",
        resource_id="other-resource",
        service_id=admission.controller.governed_services[control],
        clock=MonotonicClock(),
        cancel=threading.Event(),
        deadline_monotonic=time.monotonic() + 60,
    )


def wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        threading.Event().wait(0.005)


def test_every_model_request_and_test_job_is_admitted_under_the_policy(mean_task, server, tmp_path):
    admission = make_admission(policy_version(2, inference=1, tests=1), max_queue_length=8)
    outcome = make_runner(
        mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path / "run", admission=admission
    ).run()
    assert (outcome.state, outcome.task_passed) == ("completed", True)
    saved = evidence(outcome.evidence_path)
    events = server.only_run().steps
    records = {record["step_id"]: record for record in saved["admissions"]}
    assert {step: record["control"] for step, record in records.items()} == ADMITTED_STEPS
    for step_id, record in records.items():
        event = events[step_id]
        assert record["outcome"] == "admitted"
        assert (record["released"], record["held_by_unfinished_work"]) == (True, False)
        assert (record["policy_version"], record["limit"], record["enforced"]) == (2, 1, True)
        assert (record["span_id"], record["resource_id"]) == (event.span_id, event.resource_id)
        assert record["clock_id"] == saved["clock_id"]
        # The admission wait lies inside the step's queue on the run's own clock.
        assert (
            event.submitted_tick_ns
            <= record["arrived_tick_ns"]
            <= record["admitted_tick_ns"]
            <= event.started_tick_ns
        )
    assert saved["admission"]["status"] == "configured"
    assert saved["admission"]["open_permits"] == 0
    assert [entry["version"] for entry in saved["admission"]["policy_history"]] == [2]
    assert saved["cleanup"]["admission_permits_held"] == 0
    assert ADMISSION_LIMITATION in saved["limitations"]
    configuration = saved["configuration"]
    assert configuration["runner_version"] == RUNNER_VERSION == "agent-runner-v4"
    assert configuration["admission"]["queue_bounds"]["test_jobs"]["max_queue_length"] == 8
    assert configuration["admission"]["enforcement_scope"] == "runner_process"

    report = coverage_report(outcome.evidence_path.parent)
    assert report["method"] == "coverage-v4"
    steps = {entry["step_id"]: entry for entry in report["steps"]}
    timing = steps["verify-tests"]["timing"]
    assert timing["admission"] == "admitted"
    assert timing["admission_wait_seconds"] + timing["dispatch_queue_seconds"] == pytest.approx(
        timing["queue_seconds"]
    )
    assert timing["admission_policy_version"] == 2
    assert steps["tool-1-search_docs"]["timing"]["admission"] == "not_required"


def test_a_run_without_a_policy_says_so_and_admits_nothing(mean_task, server, tmp_path):
    outcome = make_runner(mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path / "run").run()
    saved = evidence(outcome.evidence_path)
    assert saved["admission"] == NO_ADMISSION and saved["admissions"] == []
    assert saved["configuration"]["admission"] is None
    assert ADMISSION_LIMITATION not in saved["limitations"]
    timing = next(
        s["timing"]
        for s in coverage_report(outcome.evidence_path.parent)["steps"]
        if s["step_id"] == "verify-tests"
    )
    assert timing["admission"] == "not_configured"
    assert timing["dispatch_queue_seconds"] == timing["queue_seconds"]


def test_a_full_admission_queue_fails_the_step_and_the_run(mean_task, server, tmp_path):
    admission = make_admission(policy_version(1, tests=1), max_queue_length=0)
    other = held_permit(admission)
    outcome = make_runner(
        mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path / "run", admission=admission
    ).run()
    run = server.only_run()
    reason = "Admission refused: the test_jobs queue is full (0 waiting)"
    assert (outcome.state, run.reason) == ("failed", reason)
    step = run.steps["tool-4-run_tests"]
    # Refused before it started: queued straight to failed, with no start or CPU time.
    assert (step.state, step.started_tick_ns, step.cpu_seconds) == ("failed", None, None)
    refused = [
        r for r in evidence(outcome.evidence_path)["admissions"] if r["outcome"] != "admitted"
    ]
    assert [(r["step_id"], r["outcome"]) for r in refused] == [("tool-4-run_tests", "queue_full")]
    assert admission.controller.open_permits() == 1  # only the other work's permit
    other.release()


def test_cancellation_while_waiting_for_a_permit_cancels_the_queued_step(
    mean_task, server, tmp_path
):
    admission = make_admission(policy_version(1, inference=1))
    other = held_permit(admission, "inference_requests")
    runner = make_runner(
        mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path / "run", admission=admission
    )
    result = {}
    thread = threading.Thread(target=lambda: result.update(outcome=runner.run()))
    thread.start()
    wait_for(lambda: admission.controller.summary()["controls"]["inference_requests"]["queued"])
    runner.request_cancel("operator", "Stopped while queued")
    thread.join(10)
    run = server.only_run()
    assert (result["outcome"].state, run.reason) == ("cancelled", "Stopped while queued")
    step = run.steps["model-call-1"]
    assert (step.state, step.started_tick_ns) == ("cancelled", None)
    record = evidence(result["outcome"].evidence_path)["admissions"][0]
    assert (record["step_id"], record["outcome"]) == ("model-call-1", "cancelled")
    assert admission.controller.open_permits() == 1
    other.release()


def test_the_run_deadline_ends_an_admission_wait(mean_task, server, tmp_path):
    admission = make_admission(policy_version(1, inference=1))
    other = held_permit(admission, "inference_requests")
    outcome = make_runner(
        mean_task,
        server,
        ScriptedModel(SOLVE_MEAN),
        tmp_path / "run",
        admission=admission,
        limits=RunLimits(run_timeout_seconds=0.5),
    ).run()
    run = server.only_run()
    assert (outcome.state, run.reason) == ("failed", "Run exceeded its 0.5 s time limit")
    assert run.steps["model-call-1"].state == "failed"
    record = evidence(outcome.evidence_path)["admissions"][0]
    assert record["outcome"] == "run_deadline"
    other.release()
    assert admission.controller.open_permits() == 0


def test_a_failed_model_request_returns_its_permit(mean_task, server, tmp_path):
    admission = make_admission(policy_version(1, inference=1))
    outcome = make_runner(
        mean_task,
        server,
        ScriptedModel([URLError("connection reset")]),
        tmp_path / "run",
        admission=admission,
    ).run()
    assert outcome.state == "failed"
    record = evidence(outcome.evidence_path)["admissions"][0]
    assert (record["outcome"], record["released"]) == ("admitted", True)
    assert admission.controller.open_permits() == 0


def test_an_abandoned_model_request_keeps_its_permit_until_it_really_ends(
    mean_task, server, tmp_path
):
    admission = make_admission(policy_version(1, inference=1))
    release_model = threading.Event()
    model = ScriptedModel(SOLVE_MEAN, block=release_model, block_on=1)
    runner = make_runner(mean_task, server, model, tmp_path / "run", admission=admission)
    result = {}
    thread = threading.Thread(target=lambda: result.update(outcome=runner.run()))
    thread.start()
    assert model.blocked.wait(10)
    runner.request_cancel("operator", "Stopped mid-request")
    thread.join(10)
    saved = evidence(result["outcome"].evidence_path)
    assert result["outcome"].state == "cancelled"
    # The server may still be generating: the slot is still in use, not handed to new work.
    assert admission.controller.open_permits() == 1
    assert saved["admissions"][0]["held_by_unfinished_work"] is True
    assert saved["cleanup"]["admission_permits_held"] == 1
    release_model.set()
    assert admission.controller.wait_until_idle(10)


def test_an_unreadable_policy_means_nothing_runs(mean_task, server, tmp_path):
    admission = make_admission(policy_version(1))
    admission.source.failures = [URLError("policy service down")] * 2
    runner = make_runner(
        mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path / "run", admission=admission
    )
    with pytest.raises(RunNotStarted) as not_started:
        runner.run()
    assert not_started.value.stage == "policy"
    assert server.runs == {}  # no run was registered that could not be admitted
    saved = evidence(runner.output_dir / "agent-evidence.json")
    assert saved["status"] == "not_started"
    assert saved["error"].startswith("Run policy failed: URLError")
    assert saved["cleanup"]["workspace_removed"] is True


def test_a_policy_watcher_failure_fails_the_run(mean_task, server, tmp_path):
    source = FakePolicySource(policy_version(1))
    admission = make_admission(policy_version(1))
    watcher = PolicyWatcher(source, admission.controller, poll_seconds=60)
    watcher.error = PolicyError("Active policy went from version 3 back to 1")
    admission = RunnerAdmission(admission.controller, admission.source, watcher, NO_SLEEP)
    runner = make_runner(
        mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path / "run", admission=admission
    )
    with pytest.raises(PolicyError):
        runner.run()
    assert server.only_run().state == "failed"
    saved = evidence(runner.output_dir / "agent-evidence.json")
    assert any("PolicyWatcher" in error for error in saved["background_errors"])


def test_a_runner_refuses_a_controller_for_another_inference_service(mean_task, server, tmp_path):
    admission = make_admission(policy_version(1), inference_service="vllm")
    with pytest.raises(ValueError, match="different inference service"):
        make_runner(
            mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path / "run", admission=admission
        )


class TrackingModel(ScriptedModel):
    """Counts model requests in flight across every run of a batch (test double)."""

    def __init__(self, responses, tracker, barrier=None):
        super().__init__(responses)
        self.tracker = tracker
        self.barrier = barrier

    def chat(self, payload, *, trace_context=None, on_event=None):
        with self.tracker["lock"]:
            self.tracker["running"] += 1
            self.tracker["peak"] = max(self.tracker["peak"], self.tracker["running"])
            gated = self.tracker["gate"] is not None and not self.tracker["gated"].is_set()
            if gated:
                self.tracker["gated"].set()
        try:
            if gated:
                # The batch's first request in flight holds its permit until the test opens.
                self.tracker["gate"].wait(10)
            elif self.barrier is not None and len(self.payloads) == 0:
                # Every run's first request must be in flight at once to pass the barrier.
                self.barrier.wait(10)
            else:
                threading.Event().wait(0.01)
            return super().chat(payload, trace_context=trace_context, on_event=on_event)
        finally:
            with self.tracker["lock"]:
                self.tracker["running"] -= 1


def batch_runners(mean_task, server, tmp_path, admission, count, barrier=None, gate=None):
    tracker = {
        "lock": threading.Lock(),
        "running": 0,
        "peak": 0,
        "gate": gate,
        "gated": threading.Event(),
    }
    clock = MonotonicClock()
    runners = [
        make_runner(
            mean_task,
            server,
            TrackingModel(SOLVE_MEAN, tracker, barrier),
            tmp_path / "batch" / "runs" / f"{index:02d}",
            admission=admission,
            clock=clock,
        )
        for index in range(1, count + 1)
    ]
    return runners, tracker


def test_a_batch_shares_one_limit_across_its_runs(mean_task, tmp_path):
    server = FakeLiveServer()
    admission = make_admission(policy_version(3, inference=1, tests=1))
    gate = threading.Event()
    runners, tracker = batch_runners(mean_task, server, tmp_path, admission, 3, gate=gate)
    batch = AgentBatch(runners, admission.controller, tmp_path / "batch")
    result = {}
    thread = threading.Thread(target=lambda: result.update(runs=batch.run()))
    thread.start()
    assert tracker["gated"].wait(10)
    # One request holds the only permit; the other two runs' first requests wait in line.
    wait_for(
        lambda: admission.controller.summary()["controls"]["inference_requests"]["queued"] == 2
    )
    held_seconds = 0.2
    threading.Event().wait(held_seconds)
    assert tracker["running"] == 1
    gate.set()
    thread.join(30)
    runs = result["runs"]
    assert [run.outcome.state for run in runs] == ["completed"] * 3
    assert tracker["peak"] == 1  # never more than one model request in flight in this process
    saved = evidence(tmp_path / "batch" / BATCH_FILE)
    assert saved["permits_returned"] is True
    assert saved["clock_id"] == runners[0].clock.clock_id
    controls = saved["admission"]["controls"]
    assert (controls["inference_requests"]["admitted"], controls["test_jobs"]["admitted"]) == (
        15,
        6,
    )
    assert {run["state"] for run in saved["runs"]} == {"completed"}
    assert len({run["run_id"] for run in saved["runs"]}) == 3
    # The two queued requests each waited at least as long as the permit was held.
    waits = sorted(
        record["admission_wait_seconds"]
        for runner in runners
        for record in evidence(runner.output_dir / "agent-evidence.json")["admissions"]
        if record["step_id"] == "model-call-1"
    )
    assert waits[0] < held_seconds <= waits[1] <= waits[2]


def test_a_batch_at_the_baseline_runs_its_tasks_concurrently(mean_task, tmp_path):
    server = FakeLiveServer()
    admission = make_admission(policy_version(0))
    barrier = threading.Barrier(3)
    runners, tracker = batch_runners(mean_task, server, tmp_path, admission, 3, barrier)
    runs = AgentBatch(runners, admission.controller, tmp_path / "batch").run()
    assert [run.outcome.state for run in runs] == ["completed"] * 3
    assert tracker["peak"] == 3 and not barrier.broken


def test_a_batch_whose_policy_is_unreadable_starts_nothing(mean_task, tmp_path):
    server = FakeLiveServer()
    admission = make_admission(policy_version(1))
    admission.source.failures = [URLError("down")] * 2
    runners, _ = batch_runners(mean_task, server, tmp_path, admission, 2)
    batch = AgentBatch(runners, admission.controller, tmp_path / "batch")
    with pytest.raises(RunNotStarted) as not_started:
        batch.run()
    assert not_started.value.stage == "policy"
    assert server.runs == {}
    saved = evidence(batch.path)
    assert [run["state"] for run in saved["runs"]] == ["not_started"] * 2
    assert saved["runs"][0]["not_started"]["stage"] == "policy"
    assert admission.source.reads == 2  # read once for the batch, with its retry


def test_a_batch_requires_one_clock_one_controller_and_a_bounded_size(mean_task, tmp_path):
    server = FakeLiveServer()
    admission = make_admission(policy_version(0))
    runners, _ = batch_runners(mean_task, server, tmp_path, admission, 2)
    with pytest.raises(ValueError, match="one admission controller"):
        AgentBatch(runners, None, tmp_path / "batch")
    other = make_runner(
        mean_task,
        server,
        ScriptedModel(SOLVE_MEAN),
        tmp_path / "other",
        admission=admission,
    )
    with pytest.raises(ValueError, match="clock"):
        AgentBatch([runners[0], other], admission.controller, tmp_path / "batch")
    with pytest.raises(ValueError, match=str(MAX_BATCH_RUNS)):
        AgentBatch([], admission.controller, tmp_path / "batch")


def test_a_permit_returned_after_its_step_ended_is_recorded_at_closure(
    mean_task, server, tmp_path, monkeypatch
):
    admission = make_admission(policy_version(1, inference=1))
    release_model = threading.Event()
    model = ScriptedModel(SOLVE_MEAN, block=release_model, block_on=1)
    runner = make_runner(mean_task, server, model, tmp_path / "run", admission=admission)
    finish = AgentRunner._finish

    def finish_after_the_request_ended(self, *args):
        # The abandoned request ends between its step's end and the run's closure.
        release_model.set()
        assert admission.controller.wait_until_idle(10)
        return finish(self, *args)

    monkeypatch.setattr(AgentRunner, "_finish", finish_after_the_request_ended)
    result = {}
    thread = threading.Thread(target=lambda: result.update(outcome=runner.run()))
    thread.start()
    assert model.blocked.wait(10)
    runner.request_cancel("operator", "Stopped mid-request")
    thread.join(10)
    saved = evidence(result["outcome"].evidence_path)
    record = saved["admissions"][0]
    assert (record["released"], record["held_by_unfinished_work"]) == (True, False)
    assert record["released_tick_ns"] >= record["admitted_tick_ns"]
    assert saved["cleanup"]["admission_permits_held"] == 0
    assert (record["pid"], record["enforcement_scope"]) == (
        admission.controller.pid,
        "runner_process",
    )
