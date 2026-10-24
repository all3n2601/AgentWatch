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
GPU_UTIL = "gpu.nvml.utilization.gpu.v1"
THROTTLED = "cpu.cgroup.throttled_usec.v1"
SAMPLED = {"gpu-1": "fixture device", "host-1": "fixture host"}


def step(
    step_id,
    sequence,
    state,
    submitted,
    started=None,
    finished=None,
    *,
    clock_id=CLOCK,
    resource_id="model-server-1",
    step_type="model_call",
    run_id=RUN_ID,
):
    return StepEvent(
        contract_version="live-contracts-v1",
        record_type="step_event",
        run_id=run_id,
        task_id="fixture-r1",
        event_id=uuid4(),
        step_id=step_id,
        step_type=step_type,
        parent_step_id=None,
        trace_id="a" * 32,
        span_id="b" * 16,
        resource_id=resource_id,
        service_id=None if resource_id is None else "fixture",
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


def executed(step_id, submitted, started, finished, **fields):
    return [
        step(step_id, 0, "queued", submitted, **fields),
        step(step_id, 1, "running", submitted, started, **fields),
        step(step_id, 2, "completed", submitted, started, finished, **fields),
    ]


def sample(tick, metric, value, *, resource="gpu-1", clock_id=CLOCK, availability="measured"):
    unit, resource_class = {
        GPU_UTIL: ("percent", "inference"),
        THROTTLED: ("microseconds", "cpu_sandbox"),
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


def write_run(tmp_path, records, interval=0.1, extra_lines=(), evidence_extra=None, clocks=None):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    evidence = {
        "run_id": str(RUN_ID),
        "status": "completed",
        "task": {"task_id": "fixture-r1"},
        "configuration": {"sampling_interval_seconds": interval},
        "resource_sampling": {
            "resources": [{"resource_id": rid, "scope": scope} for rid, scope in SAMPLED.items()]
        },
    } | (evidence_extra or {})
    (run_dir / "agent-evidence.json").write_text(json.dumps(evidence))
    registration = {"clock_ids": clocks or [CLOCK], "resources": []}
    lines = [json.dumps({"kind": "registration", "payload": registration})]
    lines += [json.dumps({"kind": "record", "payload": r.model_dump(mode="json")}) for r in records]
    (run_dir / "live-journal.jsonl").write_text("\n".join([*lines, *extra_lines]) + "\n")
    return run_dir


def steps_by_id(report):
    return {entry["step_id"]: entry for entry in report["steps"]}


def test_window_includes_start_and_excludes_finish(tmp_path):
    records = [
        *executed("model-call-1", 0, 100 * MS, 400 * MS),
        sample(100 * MS, GPU_UTIL, 10),  # at start: inside
        sample(250 * MS, GPU_UTIL, 90),
        sample(400 * MS, GPU_UTIL, 50),  # at finish: outside
    ]
    entry = steps_by_id(coverage_report(write_run(tmp_path, records)))["model-call-1"]
    assert (entry["duration_seconds"], entry["expected_rounds"]) == (pytest.approx(0.3), 3)
    gpu = entry["context"]["gpu-1"]
    assert (gpu["status"], gpu["rounds_in_window"]) == ("present", 2)
    assert gpu["coverage"] == pytest.approx(2 / 3)
    assert gpu["scope"] == "fixture device"
    assert gpu["measured"][GPU_UTIL] == {"measured": 2, "min": 10, "mean": 50, "max": 90}


def test_a_step_joins_only_its_own_resource(tmp_path):
    # The model server has no registered metric: its own telemetry is missing, and the GPU
    # readings during the call are context, not a measurement of the call.
    records = [*executed("model-call-1", 0, 0, 300 * MS), sample(100 * MS, GPU_UTIL, 40)]
    entry = steps_by_id(coverage_report(write_run(tmp_path, records)))["model-call-1"]
    assert entry["own"] == {"resource_id": "model-server-1", "status": "missing"}
    assert entry["status"] == "missing"
    assert entry["context"]["gpu-1"]["status"] == "present"


def test_own_resource_status_vocabulary(tmp_path):
    records = [
        *executed("on-host", 0, 0, 300 * MS, resource_id="host-1", step_type="tool"),
        *executed("gpu-step", 400 * MS, 400 * MS, 700 * MS, resource_id="gpu-1"),
        *executed("root", 0, 0, 800 * MS, resource_id=None, step_type="other"),
        sample(100 * MS, GPU_UTIL, 5),
    ]
    by_id = steps_by_id(coverage_report(write_run(tmp_path, records)))
    # host-1 is sampled according to the evidence, but no rounds were journaled.
    assert by_id["on-host"]["own"]["status"] == "empty"
    # gpu-1 has rounds in the domain, but none inside this window.
    assert by_id["gpu-step"]["own"]["status"] == "no_overlap"
    assert by_id["gpu-step"]["own"]["coverage"] == 0
    assert by_id["root"]["own"] == {"status": "no_resource"}


def test_samples_in_another_domain_are_unaligned(tmp_path):
    records = [
        *executed("model-call-1", 0, 0, 300 * MS, resource_id="gpu-1"),
        sample(100 * MS, GPU_UTIL, 99, clock_id="other-clock"),
    ]
    run_dir = write_run(tmp_path, records, clocks=[CLOCK, "other-clock"])
    entry = steps_by_id(coverage_report(run_dir))["model-call-1"]
    assert entry["own"]["status"] == "unaligned"


def test_unregistered_clock_records_are_excluded_and_counted(tmp_path):
    records = [
        *executed("model-call-1", 0, 0, 300 * MS, resource_id="gpu-1"),
        sample(100 * MS, GPU_UTIL, 99, clock_id="never-registered"),
    ]
    report = coverage_report(write_run(tmp_path, records))
    assert report["unregistered_clock_records"] == 1
    assert steps_by_id(report)["model-call-1"]["own"]["status"] == "empty"


def test_records_of_another_run_are_refused(tmp_path):
    records = [*executed("a", 0, 0, 100 * MS, run_id=uuid4())]
    with pytest.raises(CoverageError, match="other run"):
        coverage_report(write_run(tmp_path, records))


def test_unavailable_readings_count_but_do_not_aggregate(tmp_path):
    records = [
        *executed("model-call-1", 0, 0, 300 * MS),
        sample(0, GPU_UTIL, None, availability="unsupported"),
        sample(100 * MS, GPU_UTIL, None, availability="unsupported"),
    ]
    entry = steps_by_id(coverage_report(write_run(tmp_path, records)))["model-call-1"]
    gpu = entry["context"]["gpu-1"]
    assert gpu["availability"] == {GPU_UTIL: {"unsupported": 2}}
    # Never presented as an idle GPU.
    assert gpu["measured"] == {GPU_UTIL: {"measured": 0}}
    assert gpu["coverage"] == pytest.approx(2 / 3)


def test_counter_reports_increase_and_null_after_reset(tmp_path):
    records = [
        *executed("a", 0, 0, 300 * MS, resource_id="host-1", step_type="tool"),
        *executed("b", 300 * MS, 300 * MS, 600 * MS, resource_id="host-1", step_type="tool"),
        sample(0, THROTTLED, 100, resource="host-1"),
        sample(200 * MS, THROTTLED, 160, resource="host-1"),
        sample(300 * MS, THROTTLED, 500, resource="host-1"),
        sample(500 * MS, THROTTLED, 20, resource="host-1"),
    ]
    by_id = steps_by_id(coverage_report(write_run(tmp_path, records)))
    counter = by_id["a"]["own"]["measured"][THROTTLED]
    assert (counter["increase"], counter["increase_span_seconds"]) == (60, pytest.approx(0.2))
    assert by_id["b"]["own"]["measured"][THROTTLED]["increase"] is None


def test_one_counter_reading_has_no_increase_rather_than_zero(tmp_path):
    records = [
        *executed("short-tool", 0, 0, 300 * MS, resource_id="host-1", step_type="tool"),
        sample(100 * MS, THROTTLED, 5_000, resource="host-1"),
    ]
    entry = steps_by_id(coverage_report(write_run(tmp_path, records)))["short-tool"]
    assert entry["own"]["measured"][THROTTLED] == {
        "measured": 1,
        "increase": None,
        "increase_span_seconds": None,
    }


def test_counter_reset_inside_the_window_has_no_increase(tmp_path):
    records = [
        *executed("a", 0, 0, 400 * MS, resource_id="host-1", step_type="tool"),
        sample(0, THROTTLED, 100, resource="host-1"),
        sample(100 * MS, THROTTLED, 5, resource="host-1"),  # reset
        sample(200 * MS, THROTTLED, 300, resource="host-1"),  # climbs back above the first
    ]
    entry = steps_by_id(coverage_report(write_run(tmp_path, records)))["a"]
    assert entry["own"]["measured"][THROTTLED]["increase"] is None


def test_conflicting_records_under_one_identity_are_refused(tmp_path):
    original = sample(0, GPU_UTIL, 10)
    changed = original.model_copy(update={"value": 90.0})
    with pytest.raises(CoverageError, match="conflicting records"):
        coverage_report(write_run(tmp_path, [original, changed]))


def test_short_unexecuted_and_unsampled_steps_are_distinguished(tmp_path):
    records = [
        *executed("short", 0, 0, 50 * MS),
        step("rejected", 0, "queued", 60 * MS, resource_id=None, step_type="tool"),
        step("rejected", 1, "failed", 60 * MS, None, 60 * MS, resource_id=None, step_type="tool"),
        *executed("long", 100 * MS, 100 * MS, 900 * MS),
        sample(950 * MS, GPU_UTIL, 5),
    ]
    by_id = steps_by_id(coverage_report(write_run(tmp_path, records)))
    assert by_id["short"]["shorter_than_interval"] is True
    assert by_id["short"]["context"]["gpu-1"]["coverage"] is None
    assert (by_id["rejected"]["status"], by_id["rejected"]["window"]) == ("not_executed", None)
    assert by_id["long"]["shorter_than_interval"] is False
    assert by_id["long"]["context"]["gpu-1"]["status"] == "no_overlap"


def test_queue_is_kept_apart_from_service_time(tmp_path):
    records = [
        *executed(
            "tool-1-search_docs", 0, 2 * MS, 5 * MS, resource_id="docs-1", step_type="retrieval"
        ),
        *executed("model-call-1", 10 * MS, 11 * MS, 511 * MS),
    ]
    timings = {
        "tool_calls": [
            {
                "step_id": "tool-1-search_docs",
                "evidence": {
                    "timing": {
                        "method": "retrieval-timing-v1",
                        "service_wall_seconds": 0.001,
                        "index_queue_availability": "unsupported",
                        "index_queue_reason": "In-process synchronous index",
                    }
                },
            }
        ],
        "model_calls": [
            {
                "step_id": "model-call-1",
                "timing": {
                    "method": "ollama-request-timing-v1",
                    "server_total_seconds": 0.49,
                    "observed_queue_seconds": None,
                    "observed_queue_availability": "unavailable",
                    "observed_queue_reason": "Ollama reports no request queue wait",
                },
            }
        ],
    }
    by_id = steps_by_id(coverage_report(write_run(tmp_path, records, evidence_extra=timings)))
    search = by_id["tool-1-search_docs"]["timing"]
    assert search["dispatch_queue_seconds"] == pytest.approx(0.002)
    assert search["execution_seconds"] == pytest.approx(0.003)
    assert (search["service_seconds"], search["service_queue_seconds"]) == (0.001, None)
    assert search["service_queue_availability"] == "unsupported"
    assert search["outside_service_seconds"] == pytest.approx(0.002)
    model = by_id["model-call-1"]["timing"]
    assert model["service_queue_availability"] == "unavailable"
    assert model["outside_service_seconds"] == pytest.approx(0.01)


def test_run_level_intervals_and_unusable_lines(tmp_path):
    ticks = [0, 100, 200, 500, 600]
    records = [sample(t * MS, GPU_UTIL, 1) for t in ticks]
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
    [
        None,
        {"configuration": {}},
        {"configuration": {"sampling_interval_seconds": 0}},
        {"configuration": {"sampling_interval_seconds": 0.1}},  # never registered: no run ID
    ],
)
def test_unusable_evidence_is_refused(tmp_path, evidence):
    run_dir = write_run(tmp_path, [])
    if evidence is None:
        (run_dir / "agent-evidence.json").unlink()
    else:
        (run_dir / "agent-evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(CoverageError):
        coverage_report(run_dir)


def test_journal_without_registration_is_refused(tmp_path):
    run_dir = write_run(tmp_path, [])
    (run_dir / "live-journal.jsonl").write_text("")
    with pytest.raises(CoverageError, match="registration"):
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
    registered = server.only_run().registration.resources
    sampled_services = {"agentwatch-agent-runner", "nvidia-nvml"}
    assert set(report["sampling"]) == {
        r.resource_id for r in registered if r.service_id in sampled_services
    }
    by_id = steps_by_id(report)
    root = by_id["agent-run"]
    assert root["window"]["clock_id"] == json.loads(outcome.evidence_path.read_text())["clock_id"]
    assert root["own"] == {"status": "no_resource"}
    assert all(entry["status"] == "present" for entry in root["context"].values())
    # Model calls run on the unsampled model server; retrieval on the unsampled docs index.
    assert by_id["model-call-1"]["own"]["status"] == "missing"
    search = by_id["tool-1-search_docs"]
    assert search["own"]["status"] == "missing"
    assert search["timing"]["service_method"] == "retrieval-timing-v1"
    assert search["timing"]["service_queue_availability"] == "unsupported"
    assert search["timing"]["service_seconds"] > 0
