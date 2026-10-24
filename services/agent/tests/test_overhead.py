import json
from dataclasses import replace
from urllib.error import URLError

import pytest
from agentwatch_agent import overhead as overhead_module
from agentwatch_agent.overhead import (
    PROTOCOL,
    OverheadCancelled,
    OverheadFailed,
    measure_overhead,
)
from agentwatch_agent.session import CancellationState
from agentwatch_agent.stats import summarize


def test_overhead_alternates_conditions_and_delivers_instrumented_steps(
    mean_task, server, tmp_path
):
    report = measure_overhead(
        mean_task,
        "return-none-for-empty",
        server,
        tmp_path,
        repetitions=2,
        warmup=1,
        sampling_interval_seconds=0.01,
        proc=tmp_path / "no-procfs",
    )
    walls = report["workload_wall_seconds"]
    assert walls["uninstrumented"]["count"] == walls["instrumented"]["count"] == 2
    assert report["protocol"] == PROTOCOL and report["protocol"]["threshold"] is None
    assert report["live_run_state"] == "completed"
    assert report["delivery"]["complete"]
    run = server.only_run()
    assert sorted(run.steps) == ["overhead-1", "overhead-3"]
    assert report["sampling"]["rounds"] >= 2
    assert report["workspace_removed"]
    json.dumps(report)


def test_unknown_patch_is_rejected_before_any_run(mean_task, server, tmp_path):
    with pytest.raises(ValueError, match="no patch"):
        measure_overhead(mean_task, "nope", server, tmp_path)
    assert server.runs == {}


def test_summary_statistics():
    assert summarize([3.0, 1.0, 2.0, 4.0]) == {
        "count": 4,
        "median": 2.5,
        "p90": 4.0,
        "min": 1.0,
        "max": 4.0,
    }


# Review regressions (PR #27) --------------------------------------------------------------


def measure(mean_task, server, tmp_path, **overrides):
    settings = {
        "repetitions": 2,
        "warmup": 0,
        "sampling_interval_seconds": 0.01,
        "proc": tmp_path / "no-procfs",
    } | overrides
    return measure_overhead(mean_task, "return-none-for-empty", server, tmp_path, **settings)


def test_overhead_runs_keep_the_tasks_own_identity(mean_task, server, tmp_path):
    measure(mean_task, server, tmp_path)
    assert server.only_run().registration.task_id == mean_task.task_id


def test_availability_is_counted_across_every_repetition(mean_task, server, tmp_path):
    report = measure(mean_task, server, tmp_path, repetitions=3)
    counts = report["sampling"]["availability_counts"]["cpu.psi.some.avg10.v1"]
    assert counts == {"unsupported": report["sampling"]["rounds"]}
    assert report["sampling"]["rounds"] >= 3


def test_failed_workload_closes_the_overhead_run_as_failed(
    mean_task, server, tmp_path, monkeypatch
):
    real = overhead_module.run_project_tests
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(True)
        result = real(*args, **kwargs)
        return replace(result, outcome="failed") if len(calls) == 2 else result

    monkeypatch.setattr(overhead_module, "run_project_tests", fail_second)
    with pytest.raises(OverheadFailed) as failure:
        measure(mean_task, server, tmp_path)
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert failure.value.closure == "closed"
    run = server.only_run()
    assert (run.state, run.reason) == ("failed", "Overhead measurement failed (RuntimeError)")
    assert run.steps["overhead-1"].state == "failed"


def test_dead_flusher_ends_the_measurement_instead_of_understating_it(
    mean_task, server, tmp_path, monkeypatch
):
    # Flush often enough to run during a repetition, so the background flusher really dies.
    monkeypatch.setattr(overhead_module, "FLUSH_INTERVAL_SECONDS", 0.001)
    # A non-retryable delivery error kills the background flusher.
    original_ingest = server.ingest
    state = {"ingests": 0}

    def ingest(run_id, batch):
        state["ingests"] += 1
        if state["ingests"] == 1:
            raise ValueError("response violated the contract")
        return original_ingest(run_id, batch)

    server.ingest = ingest
    with pytest.raises(OverheadFailed) as failure:
        measure(mean_task, server, tmp_path)
    assert isinstance(failure.value.__cause__, ValueError)
    assert state["ingests"] >= 2, "the flusher must have died before the closing flush"
    assert server.only_run().state == "failed"


def test_refused_completion_closes_the_overhead_run_as_failed(mean_task, server, tmp_path):
    server.refuse_completion = True
    report = measure(mean_task, server, tmp_path)
    assert report["live_run_state"] == "failed_completion_refused"
    assert report["delivery"] == report["delivery"] | {
        "complete": False,
        "resend_needed": False,
        "completion_refused": True,
    }
    assert server.only_run().state == "failed"


def test_overhead_runs_use_their_own_service_identity(mean_task, server, tmp_path):
    measure(mean_task, server, tmp_path)
    services = {r.service_id for r in server.only_run().registration.resources}
    assert services == {"agentwatch-agent-overhead"}
    assert "agent-run" not in server.only_run().steps


def test_closing_flush_errors_fail_the_measurement_instead_of_being_swallowed(
    mean_task, server, tmp_path
):
    original_ingest = server.ingest
    state = {"ingests": 0}

    def ingest(run_id, batch):
        state["ingests"] += 1
        if state["ingests"] == 1:  # the closing flush: background flushes do not run
            raise ValueError("response violated the contract")
        return original_ingest(run_id, batch)

    server.ingest = ingest
    with pytest.raises(OverheadFailed) as failure:
        measure(mean_task, server, tmp_path)
    assert isinstance(failure.value.__cause__, ValueError)
    assert server.only_run().state == "failed"


def test_operator_cancellation_closes_the_overhead_run_as_cancelled(
    mean_task, server, tmp_path, monkeypatch
):
    cancel = CancellationState()
    real = overhead_module.run_project_tests

    def cancel_during_second(*args, **kwargs):
        result = real(*args, **kwargs)
        cancel.cancel_from_signal("signal", "Cancelled by operator signal SIGTERM")
        return result

    monkeypatch.setattr(overhead_module, "run_project_tests", cancel_during_second)
    with pytest.raises(OverheadFailed) as failure:
        measure(mean_task, server, tmp_path, cancel=cancel)
    assert isinstance(failure.value.__cause__, OverheadCancelled)
    run = server.only_run()
    assert (run.state, run.reason) == ("cancelled", "Cancelled by operator signal SIGTERM")
    assert all(step.state in {"completed", "cancelled"} for step in run.steps.values())


def test_transient_failure_on_completion_keeps_the_measurement(mean_task, server, tmp_path):
    original = server.change_state

    def completion_times_out(run_id, change):
        if change.state == "completed":
            raise URLError("timed out")
        return original(run_id, change)

    server.change_state = completion_times_out
    report = measure(mean_task, server, tmp_path)
    assert report["live_run_state"] == "completed_locally_undelivered"
    assert report["delivery"]["resend_needed"] is True
    assert report["workload_wall_seconds"]["instrumented"]["count"] == 2
