import json

import pytest
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
