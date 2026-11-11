"""Sweep level metrics and capacity decision over synthetic fixtures; never recorded evidence."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from agentwatch_agent.model import joined_vllm_timing
from agentwatch_agent.runner import (
    MODEL_CALL_LIMIT_REASON,
    NO_TOOL_CALL_OR_ANSWER,
    TOOL_CALL_LIMIT_REASON,
    VERIFICATION_FAILED,
)
from agentwatch_agent.sweep_analysis import (
    DEVICE_METRICS,
    batch_metrics,
    capacity_decision,
    classify_run,
    level_summary,
    p90,
    server_changes,
)
from agentwatch_agent.sweep_protocol import RAMP_LEVELS, declared_batches
from agentwatch_sdk.live import ResourceSample, StepEvent

RUN_LIMIT_SECONDS = 900.0
RUN_ID = uuid4()
CLOCK = "fixture-clock"
SECOND_NS = 1_000_000_000
SM_CLOCK = "gpu.nvml.clock.sm_mhz.v1"
GPU_POWER = "gpu.nvml.power.usage_watts.v1"


def run_entry(**fields):
    return {
        "index": 0,
        "task_id": "t",
        "state": "completed",
        "reason": None,
        "task_passed": True,
        "delivery_complete": True,
        # As `BatchRun.evidence()` records it: the runner's own folder.
        "output_dir": "/recorded/batch/runs/00-t",
    } | fields


def recorded(batch_dir: Path, name: str) -> str:
    """A run folder as `BatchRun.evidence()` records it: `str(runner.output_dir)`."""
    return str(batch_dir / "runs" / name)


def level(
    throughput,
    pass_rate=1.0,
    p90_seconds=100.0,
    unreadable=0,
    passed=3,
    without_throughput=0,
    missing_durations=0,
    batches=None,
):
    """A minimal level summary in the shape `level_summary` returns.

    `batches` defaults to the most any level declares (N = 1's six), so a level is complete.
    """
    return {
        "batches": declared_batches(1) if batches is None else batches,
        "passed": passed,
        "batches_without_throughput": without_throughput,
        "unreadable_batches": [{"index": i, "error": "x"} for i in range(unreadable)],
        "throughput_per_minute": None if throughput is None else {"mean": throughput},
        "pass_rate": {"pooled": pass_rate},
        "run_duration_seconds": {"p90": p90_seconds, "missing": missing_durations},
    }


def ramp(*throughputs, **fields):
    return {n: level(t, **fields) for n, t in zip(RAMP_LEVELS, throughputs, strict=False)}


def test_capacity_is_the_level_before_the_first_failing_one():
    levels = ramp(1.0, 1.8, 2.5, 2.6, 3.1)
    decision = capacity_decision(levels, {}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "measured"
    assert decision["capacity"] == 3 and decision["first_failing_level"] == 4
    assert decision["criteria"][4]["throughput_gain"]["passed"] is False
    # Later levels are still evaluated, but a later pass does not undo the first failure.
    assert decision["criteria"][6]["throughput_gain"]["passed"] is True


def test_a_hard_stop_fails_its_level():
    decision = capacity_decision(ramp(1.0, 2.0, 3.0), {2: ["vllm_restart"]}, RUN_LIMIT_SECONDS)
    assert decision["capacity"] == 1 and decision["first_failing_level"] == 2
    assert decision["criteria"][2]["no_hard_stop"]["value"] == ["vllm_restart"]


def test_a_level_short_of_its_declared_batches_without_a_hard_stop_is_incomplete():
    # One usable batch of three, and it fails throughput: never a measured knee.
    levels = {1: level(1.0), 2: level(1.0, batches=1)}
    decision = capacity_decision(levels, {}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "incomplete" and decision["capacity"] == 1
    assert decision["first_failing_level"] is None
    assert decision["batch_counts"][2] == {"usable": 1, "declared": 3, "excluded": 0}
    assert "level 2" in decision["incomplete_reason"]
    # The same shortfall with a hard stop is the knee the hard stop declares.
    stopped = capacity_decision(levels, {2: ["infrastructure_failure"]}, RUN_LIMIT_SECONDS)
    assert stopped["result"] == "measured" and stopped["capacity"] == 1
    assert stopped["first_failing_level"] == 2


def test_a_level_whose_batches_were_all_excluded_is_incomplete():
    decision = capacity_decision({1: level(1.0)}, {}, RUN_LIMIT_SECONDS, excluded_batches={2: 1})
    assert decision["result"] == "incomplete" and decision["capacity"] == 1
    assert decision["batch_counts"][2] == {"usable": 0, "declared": 3, "excluded": 1}


def test_n1_short_of_its_six_batches_is_incomplete_not_below_1():
    decision = capacity_decision({1: level(1.0, passed=0, batches=2)}, {}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "incomplete" and decision["capacity"] is None


@pytest.mark.parametrize(
    "fields", [{"unreadable": 1}, {"without_throughput": 1}, {"missing_durations": 1}]
)
def test_a_first_failure_on_evidence_criteria_only_is_incomplete(fields):
    decision = capacity_decision({1: level(1.0), 2: level(2.0, **fields)}, {}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "incomplete" and decision["capacity"] == 1
    assert decision["first_failing_level"] == 2
    assert "evidence" in decision["incomplete_reason"]
    # A measured criterion failing too makes it the knee.
    slow = capacity_decision({1: level(1.0), 2: level(1.0, **fields)}, {}, RUN_LIMIT_SECONDS)
    assert slow["result"] == "measured" and slow["capacity"] == 1


def test_n1_failing_gives_below_1():
    decision = capacity_decision(ramp(1.0, 2.0), {1: ["oom"]}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "below_1" and decision["capacity"] is None
    assert decision["first_failing_level"] == 1


def test_every_level_passing_reports_the_ceiling():
    throughputs = [1.0 * 1.2**step for step in range(len(RAMP_LEVELS))]
    decision = capacity_decision(ramp(*throughputs), {}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "at_least_ceiling"
    assert decision["capacity"] == 16 and decision["first_failing_level"] is None


def test_a_ramp_that_stops_short_without_a_failure_is_incomplete():
    decision = capacity_decision(ramp(1.0, 2.0), {}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "incomplete" and decision["capacity"] == 2
    assert decision["first_failing_level"] is None
    assert capacity_decision({}, {}, RUN_LIMIT_SECONDS)["capacity"] is None


@pytest.mark.parametrize(("pass_rate", "passes"), [(0.80, True), (0.79, False)])
def test_pass_rate_may_drop_twenty_points_not_more(pass_rate, passes):
    levels = {1: level(1.0), 2: level(2.0, pass_rate=pass_rate)}
    criterion = capacity_decision(levels, {}, RUN_LIMIT_SECONDS)["criteria"][2]["pass_rate_drop"]
    assert criterion["passed"] is passes


@pytest.mark.parametrize(("p90_seconds", "passes"), [(450.0, True), (450.1, False)])
def test_p90_duration_limit_is_half_the_run_limit(p90_seconds, passes):
    levels = {1: level(1.0, p90_seconds=p90_seconds)}
    criterion = capacity_decision(levels, {}, RUN_LIMIT_SECONDS)["criteria"][1]
    assert criterion["p90_duration_seconds"]["limit"] == 450.0
    assert criterion["p90_duration_seconds"]["passed"] is passes


def test_p90_uses_the_rounded_index():
    assert p90([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9
    assert p90([]) is None


def test_task_failures_are_not_infrastructure_failures():
    wrong_patch = run_entry(state="failed", reason=VERIFICATION_FAILED, task_passed=False)
    assert classify_run(wrong_patch) == "task_failure"
    assert classify_run(run_entry()) == "passed"
    assert classify_run(run_entry(state="failed", reason="Server error")) == (
        "infrastructure_failure"
    )
    assert classify_run(run_entry(delivery_complete=False)) == "infrastructure_failure"
    assert classify_run(run_entry(delivery_complete=None)) == "infrastructure_failure"
    assert classify_run(run_entry(task_passed=False)) == "infrastructure_failure"


@pytest.mark.parametrize(
    "reason",
    [
        MODEL_CALL_LIMIT_REASON.format(limit=8),
        TOOL_CALL_LIMIT_REASON.format(limit=12),
        NO_TOOL_CALL_OR_ANSWER,
    ],
)
def test_agent_decision_failures_are_neither_passes_nor_infrastructure(reason):
    assert classify_run(run_entry(state="failed", reason=reason, task_passed=False)) == (
        "agent_failure"
    )
    # Only the runner's exact wording: a different or undelivered run is not an agent failure.
    assert classify_run(run_entry(state="failed", reason=reason + " (edited)")) == (
        "infrastructure_failure"
    )
    assert classify_run(run_entry(state="failed", reason=reason, delivery_complete=False)) == (
        "infrastructure_failure"
    )


def test_agent_failures_count_against_the_pass_rate(tmp_path):
    batch_dir = tmp_path / "batch"
    write_run(batch_dir / "runs" / "00-a", 5.0)
    write_run(batch_dir / "runs" / "01-b", 5.0)
    limit = MODEL_CALL_LIMIT_REASON.format(limit=8)
    entries = [
        run_entry(index=0, output_dir=recorded(batch_dir, "00-a")),
        run_entry(index=1, output_dir=recorded(batch_dir, "01-b"), state="failed", reason=limit),
    ]
    write_batch(batch_dir, entries)
    metrics = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)
    assert (metrics["passed"], metrics["agent_failures"], metrics["infrastructure_failures"]) == (
        1,
        1,
        0,
    )
    summary = level_summary([metrics])
    assert summary["agent_failures"] == 1 and summary["pass_rate"]["pooled"] == 0.5


def test_zero_throughput_levels_fail_without_dividing_by_zero():
    decision = capacity_decision(ramp(0.0, 1.0), {}, RUN_LIMIT_SECONDS)
    assert decision["first_failing_level"] == 2
    gain = decision["criteria"][2]["throughput_gain"]
    assert gain["passed"] is False and gain["value"] is None
    assert gain["reason"] == "previous level had zero throughput"

    collapsed = capacity_decision(ramp(1.0, 0.0), {}, RUN_LIMIT_SECONDS)
    assert collapsed["criteria"][2]["throughput_gain"]["value"] == -1.0
    unmeasured = capacity_decision(ramp(1.0, None), {}, RUN_LIMIT_SECONDS)
    assert unmeasured["criteria"][2]["throughput_gain"]["reason"] == "no throughput measured"


def test_unreadable_level_fails_its_readable_criterion():
    levels = {1: level(1.0), 2: level(2.0, unreadable=1)}
    decision = capacity_decision(levels, {}, RUN_LIMIT_SECONDS)
    assert decision["capacity"] == 1
    assert decision["criteria"][2]["readable"]["passed"] is False


def step_event(sequence, state, submitted, started=None, finished=None):
    return StepEvent(
        contract_version="live-contracts-v1",
        record_type="step_event",
        run_id=RUN_ID,
        task_id="fixture-r1",
        event_id=uuid4(),
        step_id="agent-run",
        step_type="other",
        parent_step_id=None,
        trace_id="a" * 32,
        span_id="b" * 16,
        resource_id="runner-1",
        service_id="fixture",
        sequence=sequence,
        state=state,
        observed_at="2026-10-24T00:00:00Z",
        clock_id=CLOCK,
        submitted_tick_ns=submitted,
        started_tick_ns=started,
        finished_tick_ns=finished,
        cpu_seconds=None,
        cpu_scope=None,
        reason=None,
    )


def gauge(tick, metric, value, availability="measured"):
    return ResourceSample(
        contract_version="live-contracts-v1",
        record_type="resource_sample",
        run_id=RUN_ID,
        task_id="fixture-r1",
        sample_id=uuid4(),
        resource_id="gpu-1",
        service_id="fixture",
        resource_class="inference",
        observed_at="2026-10-24T00:00:00Z",
        clock_id=CLOCK,
        tick_ns=tick,
        metric=metric,
        unit="megahertz" if metric == SM_CLOCK else "watts",
        value=value if availability == "measured" else None,
        availability=availability,
        reason=None if availability == "measured" else "Not available in this fixture",
    )


def write_run(run_dir: Path, duration_seconds: float | None, samples=(), evidence_extra=None):
    """A synthetic run folder: the root step runs `duration_seconds` from tick 0."""
    run_dir.mkdir(parents=True)
    evidence = {
        "run_id": str(RUN_ID),
        "status": "completed",
        "task": {"task_id": "fixture-r1"},
        "configuration": {"sampling_interval_seconds": 1.0},
        "resource_sampling": {"resources": [{"resource_id": "gpu-1", "scope": "device"}]},
    } | (evidence_extra or {})
    (run_dir / "agent-evidence.json").write_text(json.dumps(evidence))
    records = []
    if duration_seconds is not None:
        end = int(duration_seconds * SECOND_NS)
        records = [
            step_event(0, "queued", 0),
            step_event(1, "running", 0, 0),
            step_event(2, "completed", 0, 0, end),
        ]
    lines = [json.dumps({"kind": "registration", "payload": {"clock_ids": [CLOCK]}})]
    lines += [
        json.dumps({"kind": "record", "payload": r.model_dump(mode="json")})
        for r in [*records, *samples]
    ]
    (run_dir / "live-journal.jsonl").write_text("\n".join(lines) + "\n")


def write_batch(batch_dir: Path, entries, wall_seconds=60.0):
    batch_dir.mkdir(parents=True, exist_ok=True)
    document = {"batch_wall_seconds": wall_seconds, "runs": entries}
    (batch_dir / "batch.json").write_text(json.dumps(document))


def vllm_call(status, **fields):
    """A model call whose timing is built by the producer, from a synthetic span."""
    runner_side = {"method": "fixture", "client_wall_seconds": 2.0, "status": "pending_trace"}
    trace = {"status": status, **fields}
    if status != "measured":
        trace["problems"] = ["fixture: no usable span"]
    return {"step_id": "s", "span_id": "p", "timing": joined_vllm_timing(runner_side, trace)}


def test_batch_metrics_counts_throughput_durations_spans_and_device(tmp_path):
    batch_dir = tmp_path / "batch"
    evidence = {
        "model_calls": [
            vllm_call(
                "measured",
                queue_seconds=0.5,
                prefill_seconds=0.1,
                decode_seconds=1.0,
                first_token_seconds=0.6,
                e2e_seconds=1.6,
            ),
            vllm_call("unavailable"),
            vllm_call("malformed"),
        ],
        "replay": {"model_outputs": [{"matches_source": True}, {"matches_source": False}]},
    }
    samples = [gauge(t * SECOND_NS, SM_CLOCK, value) for t, value in ((1, 1500.0), (2, 1700.0))]
    samples.append(gauge(1 * SECOND_NS, GPU_POWER, None, availability="unavailable"))
    write_run(batch_dir / "runs" / "00-a", 10.0, samples, evidence)
    write_run(batch_dir / "runs" / "01-b", 20.0)
    write_run(batch_dir / "runs" / "02-c", None)
    entries = [
        run_entry(index=0, output_dir=recorded(batch_dir, "00-a")),
        run_entry(
            index=1,
            output_dir=recorded(batch_dir, "01-b"),
            state="failed",
            reason=VERIFICATION_FAILED,
        ),
        run_entry(
            index=2, output_dir=recorded(batch_dir, "02-c"), state="failed", reason="Server error"
        ),
    ]
    write_batch(batch_dir, entries, wall_seconds=120.0)

    metrics = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)

    assert metrics["status"] == "readable"
    assert (metrics["passed"], metrics["task_failures"], metrics["infrastructure_failures"]) == (
        1,
        1,
        1,
    )
    assert metrics["throughput_per_minute"] == pytest.approx(0.5)
    assert metrics["run_durations"] == [10.0, 20.0]
    assert metrics["run_durations_missing"] == 1
    assert metrics["reply_match_rate"] == 0.5
    spans = metrics["spans"]
    assert spans["calls_by_status"]["measured"] == 1
    assert spans["calls_by_status"]["unavailable"] == 1
    assert spans["calls_by_status"]["malformed"] == 1
    assert spans["fields"]["queue_seconds"]["values"] == [0.5]
    clock = metrics["device"]["metrics"]["gpu_sm_clock_mhz"]
    assert (clock["min"], clock["mean"], clock["max"]) == (1500.0, 1600.0, 1700.0)
    power = metrics["device"]["metrics"]["gpu_power_watts"]
    assert power["runs_measured"] == 0 and power["mean"] is None
    assert power["run_statuses"] == {"unavailable": 1, "not_sampled": 1}
    assert metrics["device"]["metrics"]["gpm_sm_occupancy_percent"]["run_statuses"] == {
        "not_sampled": 2
    }
    # The run with no root step has no device window: unreadable, never zeros.
    assert len(metrics["device"]["runs_unreadable"]) == 1
    assert metrics["device"]["runs_readable"] == 2


def test_runner_time_outside_vllm_is_reported_per_level_from_each_calls_timing(tmp_path):
    batch_dir = tmp_path / "batch"
    # The producer derives it: client wall 2.0 s minus the span's e2e.
    calls = [vllm_call("measured", e2e_seconds=1.5), vllm_call("measured", e2e_seconds=1.0)]
    calls.append(vllm_call("unavailable"))
    assert [c["timing"]["client_outside_server_seconds"] for c in calls] == [0.5, 1.0, None]
    write_run(batch_dir / "runs" / "00-a", 5.0, evidence_extra={"model_calls": calls})
    write_batch(batch_dir, [run_entry(output_dir=recorded(batch_dir, "00-a"))])
    outside = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)["client_outside_server_seconds"]
    assert (outside["count"], outside["unavailable"], outside["values"]) == (2, 1, [0.5, 1.0])
    assert outside["mean"] == 0.75 and outside["max"] == 1.0
    # Never computed here across clocks: the runner's own field, labelled as an estimate.
    assert "estimate" in outside["method"]
    pooled = level_summary([batch_metrics(batch_dir, RUN_LIMIT_SECONDS)] * 2)
    assert pooled["client_outside_server_seconds"]["count"] == 4
    assert pooled["client_outside_server_seconds"]["unavailable"] == 2
    assert pooled["client_outside_server_seconds"]["p90"] == 1.0


def test_a_run_folder_is_found_after_the_sweep_is_copied(tmp_path):
    batch_dir = tmp_path / "copied"
    write_run(batch_dir / "runs" / "00-a", 5.0)
    absolute = str(tmp_path / "other-machine" / "runs" / "00-a")
    write_batch(batch_dir, [run_entry(output_dir=absolute)])
    metrics = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)
    assert metrics["run_durations"] == [5.0]
    assert metrics["run_folders"] == [{"index": 0, "found_by": "batch_relative"}]


def test_a_run_folder_recorded_relative_to_the_working_directory_is_found(tmp_path, monkeypatch):
    # The CLI's default --output-dir is relative, and runners record str(output_dir) as given.
    monkeypatch.chdir(tmp_path)
    batch_dir = Path(".data/capacity-sweeps/cfg-default-x/batches/01-ramp-n1-r0")
    write_run(batch_dir / "runs" / "01-fix-mean-empty", 5.0)
    write_batch(batch_dir, [run_entry(output_dir=recorded(batch_dir, "01-fix-mean-empty"))])
    metrics = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)
    assert metrics["run_durations"] == [5.0]
    assert metrics["run_folders"] == [{"index": 0, "found_by": "recorded"}]


def test_a_relative_run_folder_from_another_directory_is_found_beside_its_batch(
    tmp_path, monkeypatch
):
    batch_dir = tmp_path / "copied" / "batches" / "01-ramp-n1-r0"
    write_run(batch_dir / "runs" / "01-a", 5.0)
    # Recorded relative to a working directory the analysis is no longer run from.
    write_batch(batch_dir, [run_entry(output_dir="sweeps/x/batches/01-ramp-n1-r0/runs/01-a")])
    monkeypatch.chdir(tmp_path)
    metrics = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)
    assert metrics["run_durations"] == [5.0]
    assert metrics["run_folders"] == [{"index": 0, "found_by": "batch_relative"}]


def test_a_missing_run_folder_is_counted_missing_not_zero(tmp_path):
    batch_dir = tmp_path / "batch"
    write_batch(batch_dir, [run_entry(output_dir=recorded(batch_dir, "gone"))])
    metrics = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)
    assert metrics["run_durations"] == [] and metrics["run_durations_missing"] == 1
    assert metrics["device"]["runs_unreadable"] == [{"error": "run folder not found"}]


def test_a_device_failure_is_recorded_per_run(tmp_path):
    batch_dir = tmp_path / "batch"
    write_run(batch_dir / "runs" / "00-a", 5.0)
    (batch_dir / "runs" / "00-a" / "agent-evidence.json").write_text("{}")
    write_batch(batch_dir, [run_entry(output_dir=recorded(batch_dir, "00-a"))])
    device = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)["device"]
    assert device["runs_readable"] == 0 and len(device["runs_unreadable"]) == 1
    assert device["metrics"]["gpu_sm_clock_mhz"]["mean"] is None


def test_zero_wall_seconds_gives_no_throughput(tmp_path):
    write_batch(
        tmp_path / "b", [run_entry(output_dir=recorded(tmp_path / "b", "gone"))], wall_seconds=0.0
    )
    assert batch_metrics(tmp_path / "b", RUN_LIMIT_SECONDS)["throughput_per_minute"] is None


def test_unreadable_batches_are_reported_not_zeroed(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "batch.json").write_text("{not json")
    broken = batch_metrics(tmp_path / "broken", RUN_LIMIT_SECONDS)
    assert broken["status"] == "unreadable" and "batch.json" in broken["error"]
    absent = batch_metrics(tmp_path / "absent", RUN_LIMIT_SECONDS)
    assert absent["status"] == "unreadable"

    write_run(tmp_path / "good" / "runs" / "00-a", 6.0)
    write_batch(
        tmp_path / "good",
        [run_entry(output_dir=recorded(tmp_path / "good", "00-a"))],
        wall_seconds=60.0,
    )
    good = batch_metrics(tmp_path / "good", RUN_LIMIT_SECONDS)

    summary = level_summary([good, broken, absent])
    assert summary["readable_batches"] == 1
    assert [entry["index"] for entry in summary["unreadable_batches"]] == [1, 2]
    # The unreadable batches are not zero-throughput samples.
    assert summary["throughput_per_minute"]["count"] == 1
    assert summary["throughput_per_minute"]["mean"] == pytest.approx(1.0)
    assert level_summary([broken])["throughput_per_minute"] is None
    decision = capacity_decision({1: level_summary([broken])}, {}, RUN_LIMIT_SECONDS)
    assert decision["criteria"][1]["readable"]["passed"] is False


def readable_batch(throughput, durations, clock_mean=None, server=None):
    overrides = {}
    if clock_mean is not None:
        overrides["gpu_sm_clock_mhz"] = {
            "runs_measured": 1,
            "run_statuses": {"measured": 1},
            "min": clock_mean,
            "mean": clock_mean,
            "max": clock_mean,
        }
    return {
        "status": "readable",
        "runs": 2,
        "passed": 1,
        "task_failures": 0,
        "agent_failures": 0,
        "infrastructure_failures": 1,
        "throughput_per_minute": throughput,
        "run_durations": durations,
        "run_durations_missing": 0,
        "runs_without_evidence": 0,
        "reply_match_rate": None,
        "spans": {
            "calls_by_status": {"measured": 0, "unavailable": 2, "malformed": 0, "not_recorded": 0},
            "fields": {
                f: {"values": []}
                for f in (
                    "queue_seconds",
                    "prefill_seconds",
                    "decode_seconds",
                    "first_token_seconds",
                    "e2e_seconds",
                )
            },
        },
        "client_outside_server_seconds": {"count": 0, "unavailable": 2, "values": []},
        "device": {"runs_readable": 1, "runs_unreadable": [], "metrics": _all_metrics(overrides)},
    } | ({"server": server} if server else {})


def _all_metrics(overrides):
    empty = {"runs_measured": 0, "run_statuses": {"not_sampled": 1}}
    metrics = {}
    for name, _metric, kind in DEVICE_METRICS:
        if kind == "counter":
            metrics[name] = empty | {"increase_sum": None, "runs_with_increase": 0}
        else:
            metrics[name] = empty | {"min": None, "mean": None, "max": None}
    return metrics | overrides


def test_level_summary_statistics_pooled_durations_and_sm_clock_drift():
    server = {
        "prefix_cache_queries": 10.0,
        "prefix_cache_hits": 5.0,
        "preemptions": 1.0,
        "request_successes": 4.0,
    }
    batches = [
        readable_batch(1.0, [10.0, 20.0], clock_mean=1400.0, server=server),
        readable_batch(2.0, [30.0], clock_mean=1500.0, server=server),
    ]
    summary = level_summary(batches, baseline_sm_clock_mhz=1450.0)
    assert summary["throughput_per_minute"]["mean"] == 1.5
    assert summary["throughput_per_minute"]["stdev"] == pytest.approx(0.7071067)
    assert summary["run_duration_seconds"]["p50"] == 20.0
    assert summary["run_duration_seconds"]["max"] == 30.0
    assert summary["pass_rate"]["pooled"] == 0.5
    assert summary["infrastructure_failures"] == 2
    assert summary["device"]["metrics"]["gpu_sm_clock_mhz"]["mean"] == 1450.0
    assert summary["sm_clock_drift_mhz"] == 0.0
    assert summary["server_changes"]["prefix_cache_queries"] == 20.0
    assert summary["server_changes"]["prefix_cache_hit_ratio"] == 0.5
    assert summary["spans"]["calls_by_status"]["unavailable"] == 4
    assert level_summary(batches)["sm_clock_drift_mhz"] is None


def snapshot(**counters):
    names = {
        "queries": "vllm:prefix_cache_queries_total",
        "hits": "vllm:prefix_cache_hits_total",
        "preemptions": "vllm:num_preemptions_total",
        "successes": "vllm:request_success_total",
    }
    full = {
        "start_time_seconds": 1.0,
        "cache_config": {},
        "running": 0.0,
        "waiting": 0.0,
        "counters": {metric: 0.0 for metric in names.values()},
    }
    full["counters"] |= {names[key]: value for key, value in counters.items()}
    return full


def test_server_changes_are_increases_with_a_hit_ratio():
    before = snapshot(queries=100.0, hits=40.0, preemptions=1.0, successes=5.0)
    after = snapshot(queries=300.0, hits=140.0, preemptions=3.0, successes=11.0)
    changes = server_changes(before, after)
    assert changes["prefix_cache_queries"] == 200.0 and changes["prefix_cache_hits"] == 100.0
    assert changes["prefix_cache_hit_ratio"] == 0.5
    assert changes["preemptions"] == 2.0 and changes["request_successes"] == 6.0


def test_server_changes_keep_missing_reset_and_zero_distinct():
    before = snapshot(queries=100.0, hits=40.0, preemptions=5.0)
    after = snapshot(queries=100.0, hits=40.0, preemptions=1.0)
    after["counters"]["vllm:request_success_total"] = None
    changes = server_changes(before, after)
    assert changes["request_successes"] is None
    assert changes["preemptions"] == "reset"
    # Zero new queries: no ratio, but the zero increase itself is measured.
    assert changes["prefix_cache_queries"] == 0.0
    assert changes["prefix_cache_hit_ratio"] is None
    del before["counters"]["vllm:prefix_cache_hits_total"]
    assert server_changes(before, after)["prefix_cache_hits"] is None


def test_a_measured_call_contributes_values_and_an_unavailable_one_is_only_counted(tmp_path):
    batch_dir = tmp_path / "batch"
    measured = vllm_call("measured", queue_seconds=0.5, e2e_seconds=1.5)
    assert measured["timing"]["status"] == "reported"
    evidence = {"model_calls": [measured, vllm_call("unavailable"), vllm_call("malformed")]}
    write_run(batch_dir / "runs" / "00-a", 5.0, evidence_extra=evidence)
    write_batch(batch_dir, [run_entry(output_dir=recorded(batch_dir, "00-a"))])
    spans = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)["spans"]
    assert spans["calls_by_status"] == {
        "measured": 1,
        "unavailable": 1,
        "malformed": 1,
        "not_recorded": 0,
    }
    assert spans["fields"]["queue_seconds"]["values"] == [0.5]
    assert spans["fields"]["e2e_seconds"]["values"] == [1.5]
    assert spans["fields"]["prefill_seconds"]["values"] == []


@pytest.mark.parametrize("wall", [None, 0.0, -5.0, float("inf"), float("nan")])
def test_a_batch_without_usable_wall_time_has_no_throughput_and_fails_evidence(tmp_path, wall):
    batch_dir = tmp_path / "batch"
    write_run(batch_dir / "runs" / "00-a", 5.0)
    write_batch(batch_dir, [run_entry(output_dir=recorded(batch_dir, "00-a"))], wall_seconds=wall)
    batch = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)
    assert batch["throughput_per_minute"] is None
    summary = level_summary([batch])
    assert summary["batches_without_throughput"] == 1
    criteria = capacity_decision({1: summary}, {}, RUN_LIMIT_SECONDS)["criteria"][1]
    assert criteria["evidence_complete"]["passed"] is False
    assert "reason" in criteria["evidence_complete"]


def test_a_run_without_an_agent_run_duration_fails_evidence_complete(tmp_path):
    batch_dir = tmp_path / "batch"
    write_run(batch_dir / "runs" / "00-a", None)
    write_batch(batch_dir, [run_entry(output_dir=recorded(batch_dir, "00-a"))])
    # N = 1's six declared batches, so the decision evaluates the level.
    summary = level_summary([batch_metrics(batch_dir, RUN_LIMIT_SECONDS)] * declared_batches(1))
    assert summary["run_duration_seconds"]["missing"] == declared_batches(1)
    decision = capacity_decision({1: summary}, {}, RUN_LIMIT_SECONDS)
    criterion = decision["criteria"][1]["evidence_complete"]
    assert criterion["value"]["runs_missing_duration"] == declared_batches(1)
    # Without durations the p90 criterion fails too, so N = 1 fails on measurement.
    assert decision["result"] == "below_1"


def test_n1_needs_a_passed_run():
    levels = {1: level(0.0, passed=0), 2: level(1.0)}
    decision = capacity_decision(levels, {}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "below_1" and decision["capacity"] is None
    assert decision["criteria"][1]["passed_runs"]["passed"] is False


def test_a_hard_stop_on_a_level_without_a_summary_fails_that_level():
    decision = capacity_decision(ramp(1.0, 2.0), {3: ["unconfirmed_cleanup"]}, RUN_LIMIT_SECONDS)
    assert decision["result"] == "measured"
    assert decision["capacity"] == 2 and decision["first_failing_level"] == 3
    assert decision["criteria"][3]["no_hard_stop"]["passed"] is False


def test_exactly_ten_percent_gain_passes():
    decision = capacity_decision(ramp(3.0, 3.3), {}, RUN_LIMIT_SECONDS)
    assert decision["criteria"][2]["throughput_gain"]["passed"] is True
    below = capacity_decision(ramp(3.0, 3.29), {}, RUN_LIMIT_SECONDS)
    assert below["criteria"][2]["throughput_gain"]["passed"] is False


def test_levels_reloaded_from_json_have_string_keys():
    levels = {str(n): value for n, value in ramp(1.0, 1.8, 1.9).items()}
    decision = capacity_decision(levels, {"3": ["x"]}, RUN_LIMIT_SECONDS)
    assert decision["capacity"] == 2 and decision["first_failing_level"] == 3


def test_runs_without_readable_evidence_are_counted(tmp_path):
    batch_dir = tmp_path / "batch"
    write_run(batch_dir / "runs" / "00-a", 5.0)
    write_run(batch_dir / "runs" / "01-b", 5.0)
    (batch_dir / "runs" / "01-b" / "agent-evidence.json").write_text("{not json")
    entries = [
        run_entry(index=0, output_dir=recorded(batch_dir, "00-a")),
        run_entry(index=1, output_dir=recorded(batch_dir, "01-b")),
        run_entry(index=2, output_dir=recorded(batch_dir, "gone")),
    ]
    write_batch(batch_dir, entries)
    metrics = batch_metrics(batch_dir, RUN_LIMIT_SECONDS)
    # Their spans and replies are absent, not zero: the counts say how many runs are missing.
    assert metrics["runs_without_evidence"] == 2
    assert metrics["spans"]["calls_by_status"]["not_recorded"] == 0
    assert level_summary([metrics, metrics])["runs_without_evidence"] == 4


@pytest.mark.parametrize("missing", ["batches_without_throughput", "run_durations_missing"])
def test_a_summary_without_its_evidence_counts_fails_evidence_complete(missing):
    summary = level(1.0)
    if missing == "batches_without_throughput":
        del summary["batches_without_throughput"]
    else:
        del summary["run_duration_seconds"]["missing"]
    criterion = capacity_decision({1: summary}, {}, RUN_LIMIT_SECONDS)["criteria"][1][
        "evidence_complete"
    ]
    assert criterion["passed"] is False
    assert criterion["reason"] == "evidence counts missing"
