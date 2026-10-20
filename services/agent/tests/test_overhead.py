import json
from dataclasses import replace

import pytest
from agentwatch_agent import overhead as overhead_module
from agentwatch_agent.overhead import PROTOCOL, measure_overhead, summarize


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
    with pytest.raises(RuntimeError, match="did not pass"):
        measure(mean_task, server, tmp_path)
    run = server.only_run()
    assert (run.state, run.reason) == ("failed", "Overhead measurement failed (RuntimeError)")
    assert run.steps["overhead-1"].state == "failed"


def test_dead_flusher_ends_the_measurement_instead_of_understating_it(mean_task, server, tmp_path):
    # A non-retryable delivery error kills the background flusher.
    original_ingest = server.ingest
    state = {"ingests": 0}

    def ingest(run_id, batch):
        state["ingests"] += 1
        if state["ingests"] == 1:
            raise ValueError("response violated the contract")
        return original_ingest(run_id, batch)

    server.ingest = ingest
    with pytest.raises(ValueError, match="violated the contract"):
        measure(mean_task, server, tmp_path)
    assert server.only_run().state == "failed"
