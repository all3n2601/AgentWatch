"""Coverage report over synthetic journals and a test-double run; never recorded evidence."""

import json
from uuid import uuid4

import pytest
from agent_test_doubles import SOLVE_MEAN, FakeGpu, ScriptedModel
from agentwatch_agent.config import RunLimits
from agentwatch_agent.coverage import CoverageError, coverage_report
from agentwatch_agent.runner import AgentRunner
from agentwatch_sdk.live import ResourceSample, StepEvent

RUN_ID = uuid4()
CLOCK = "fixture-clock"
MS = 1_000_000


def step(step_id, sequence, state, submitted, started=None, finished=None, clock_id=CLOCK):
    return StepEvent(
        contract_version="live-contracts-v1",
        record_type="step_event",
        run_id=RUN_ID,
        task_id="fixture-r1",
        event_id=uuid4(),
        step_id=step_id,
        step_type="model_call",
        parent_step_id=None,
        trace_id="a" * 32,
        span_id="b" * 16,
        resource_id="model-server-1",
        service_id="ollama",
        sequence=sequence,
        state=state,
        observed_at="2026-10-24T00:00:00Z",
        clock_id=clock_id,
        submitted_tick_ns=submitted,
        started_tick_ns=started,
        finished_tick_ns=finished,
        cpu_seconds=None,
        cpu_scope=None,
        reason="Rejected" if state == "failed" else None,
    )


def sample(tick, metric, value, *, resource="gpu-1", clock_id=CLOCK, availability="measured"):
    unit, resource_class = {
        "gpu.nvml.utilization.gpu.v1": ("percent", "inference"),
        "cpu.cgroup.throttled_usec.v1": ("microseconds", "cpu_sandbox"),
    }[metric]
    return ResourceSample(
        contract_version="live-contracts-v1",
        record_type="resource_sample",
        run_id=RUN_ID,
        task_id="fixture-r1",
        sample_id=uuid4(),
        resource_id=resource,
        service_id="fixture",
        resource_class=resource_class,
        observed_at="2026-10-24T00:00:00Z",
        clock_id=clock_id,
        tick_ns=tick,
        metric=metric,
        unit=unit,
        value=value if availability == "measured" else None,
        availability=availability,
        reason=None if availability == "measured" else "Not available in this fixture",
    )


def write_run(tmp_path, records, interval=0.1, extra_lines=()):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    evidence = {
        "run_id": str(RUN_ID),
        "status": "completed",
        "task": {"task_id": "fixture-r1"},
        "configuration": {"sampling_interval_seconds": interval},
    }
    (run_dir / "agent-evidence.json").write_text(json.dumps(evidence))
    lines = [json.dumps({"kind": "record", "payload": r.model_dump(mode="json")}) for r in records]
    (run_dir / "live-journal.jsonl").write_text("\n".join([*lines, *extra_lines]) + "\n")
    return run_dir


def steps_by_id(report):
    return {entry["step_id"]: entry for entry in report["steps"]}


def test_window_includes_start_and_excludes_finish(tmp_path):
    records = [
        step("model-call-1", 0, "queued", 0),
        step("model-call-1", 1, "running", 0, 100 * MS),
        step("model-call-1", 2, "completed", 0, 100 * MS, 400 * MS),
        sample(100 * MS, "gpu.nvml.utilization.gpu.v1", 10),  # at start: inside
        sample(250 * MS, "gpu.nvml.utilization.gpu.v1", 90),
        sample(400 * MS, "gpu.nvml.utilization.gpu.v1", 50),  # at finish: outside
        sample(300 * MS, "gpu.nvml.utilization.gpu.v1", 99, clock_id="other-clock"),
    ]
    report = coverage_report(write_run(tmp_path, records))
    entry = steps_by_id(report)["model-call-1"]
    assert entry["state"] == "completed"
    assert (entry["duration_seconds"], entry["expected_rounds"]) == (pytest.approx(0.3), 3)
    gpu = entry["resources"]["gpu-1"]
    assert gpu["rounds_in_window"] == 2
    assert gpu["coverage"] == pytest.approx(2 / 3)
    assert gpu["measured"]["gpu.nvml.utilization.gpu.v1"] == {
        "measured": 2,
        "min": 10,
        "mean": 50,
        "max": 90,
    }
    assert entry["status"] == "present"


def test_unavailable_readings_count_but_do_not_aggregate(tmp_path):
    records = [
        step("model-call-1", 0, "queued", 0),
        step("model-call-1", 1, "running", 0, 0),
        step("model-call-1", 2, "completed", 0, 0, 300 * MS),
        sample(0, "gpu.nvml.utilization.gpu.v1", None, availability="unsupported"),
        sample(100 * MS, "gpu.nvml.utilization.gpu.v1", None, availability="unsupported"),
    ]
    gpu = steps_by_id(coverage_report(write_run(tmp_path, records)))["model-call-1"]["resources"][
        "gpu-1"
    ]
    assert gpu["availability"] == {"gpu.nvml.utilization.gpu.v1": {"unsupported": 2}}
    # Never presented as an idle GPU.
    assert gpu["measured"] == {"gpu.nvml.utilization.gpu.v1": {"measured": 0}}
    assert gpu["coverage"] == pytest.approx(2 / 3)


def test_counter_reports_increase_and_null_after_reset(tmp_path):
    metric = "cpu.cgroup.throttled_usec.v1"
    records = [
        step("a", 0, "queued", 0),
        step("a", 1, "running", 0, 0),
        step("a", 2, "completed", 0, 0, 300 * MS),
        step("b", 0, "queued", 300 * MS),
        step("b", 1, "running", 300 * MS, 300 * MS),
        step("b", 2, "completed", 300 * MS, 300 * MS, 600 * MS),
        sample(0, metric, 100, resource="host-1"),
        sample(200 * MS, metric, 160, resource="host-1"),
        sample(300 * MS, metric, 500, resource="host-1"),
        sample(500 * MS, metric, 20, resource="host-1"),
    ]
    by_id = steps_by_id(coverage_report(write_run(tmp_path, records)))
    assert by_id["a"]["resources"]["host-1"]["measured"][metric]["increase"] == 60
    assert by_id["b"]["resources"]["host-1"]["measured"][metric]["increase"] is None


def test_counter_reset_inside_the_window_has_no_increase(tmp_path):
    metric = "cpu.cgroup.throttled_usec.v1"
    records = [
        step("a", 0, "queued", 0),
        step("a", 1, "running", 0, 0),
        step("a", 2, "completed", 0, 0, 400 * MS),
        sample(0, metric, 100, resource="host-1"),
        sample(100 * MS, metric, 5, resource="host-1"),  # reset
        sample(200 * MS, metric, 300, resource="host-1"),  # climbs back above the first
    ]
    entry = steps_by_id(coverage_report(write_run(tmp_path, records)))["a"]
    assert entry["resources"]["host-1"]["measured"][metric]["increase"] is None


def test_conflicting_records_under_one_identity_are_refused(tmp_path):
    original = sample(0, "gpu.nvml.utilization.gpu.v1", 10)
    changed = original.model_copy(update={"value": 90.0})
    with pytest.raises(CoverageError, match="conflicting records"):
        coverage_report(write_run(tmp_path, [original, changed]))


def test_short_unexecuted_and_unsampled_steps_are_distinguished(tmp_path):
    records = [
        step("short", 0, "queued", 0),
        step("short", 1, "running", 0, 0),
        step("short", 2, "completed", 0, 0, 50 * MS),
        step("rejected", 0, "queued", 60 * MS),
        step("rejected", 1, "failed", 60 * MS, None, 60 * MS),
        step("long", 0, "queued", 100 * MS),
        step("long", 1, "running", 100 * MS, 100 * MS),
        step("long", 2, "completed", 100 * MS, 100 * MS, 900 * MS),
        sample(950 * MS, "gpu.nvml.utilization.gpu.v1", 5),
    ]
    by_id = steps_by_id(coverage_report(write_run(tmp_path, records)))
    assert by_id["short"]["status"] == "shorter_than_interval"
    assert by_id["short"]["resources"]["gpu-1"]["coverage"] is None
    assert (by_id["rejected"]["status"], by_id["rejected"]["window"]) == ("not_executed", None)
    assert by_id["long"]["status"] == "empty"
    assert by_id["long"]["resources"]["gpu-1"]["coverage"] == 0


def test_run_level_intervals_and_unusable_lines(tmp_path):
    ticks = [0, 100, 200, 500, 600]
    records = [sample(t * MS, "gpu.nvml.utilization.gpu.v1", 1) for t in ticks]
    # A duplicated journal line is one observation, not two.
    duplicate = json.dumps({"kind": "record", "payload": records[0].model_dump(mode="json")})
    invalid = json.dumps({"kind": "record", "payload": {"record_type": "resource_sample"}})
    run_dir = write_run(tmp_path, records, extra_lines=(duplicate, invalid, "{not json"))
    report = coverage_report(run_dir)
    gpu = report["sampling"]["gpu-1"]
    assert gpu["sample_rounds"] == 5
    assert gpu["observed_interval_max_seconds"] == pytest.approx(0.3)
    assert (gpu["late_rounds"], gpu["missed_rounds"]) == (1, 2)
    assert report["unusable_journal_lines"] == 2


@pytest.mark.parametrize(
    "evidence",
    [None, {"configuration": {}}, {"configuration": {"sampling_interval_seconds": 0}}],
)
def test_unusable_evidence_is_refused(tmp_path, evidence):
    run_dir = write_run(tmp_path, [])
    if evidence is None:
        (run_dir / "agent-evidence.json").unlink()
    else:
        (run_dir / "agent-evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(CoverageError):
        coverage_report(run_dir)


def test_report_covers_a_test_double_runner_run(mean_task, server, tmp_path):
    outcome = AgentRunner(
        task=mean_task,
        model=ScriptedModel(SOLVE_MEAN),
        transport=server,
        output_dir=tmp_path / "run",
        limits=RunLimits(run_timeout_seconds=60),
        sampling_interval_seconds=0.02,
        flush_interval_seconds=0.05,
        proc=tmp_path / "no-procfs",
        open_gpu=FakeGpu,
    ).run()
    report = coverage_report(outcome.evidence_path.parent)
    assert report["run_id"] == str(outcome.run_id)
    sampled_services = {"agentwatch-agent-runner", "nvidia-nvml"}
    registered = server.only_run().registration.resources
    assert set(report["sampling"]) == {
        r.resource_id for r in registered if r.service_id in sampled_services
    }
    root = steps_by_id(report)["agent-run"]
    assert root["window"]["clock_id"] == json.loads(outcome.evidence_path.read_text())["clock_id"]
    assert all(entry["rounds_in_window"] > 0 for entry in root["resources"].values())
