"""Label validation tests. Every run folder here is a synthetic fixture built in tmp_path."""

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from agentwatch_agent.model import joined_vllm_timing
from agentwatch_agent.sweep_labels import (
    baselines,
    label_validation,
    max_running_check,
    noise_threshold,
    spearman,
)
from agentwatch_sdk.live import ResourceSample, StepEvent

RUN_ID = UUID("00000000-0000-4000-8000-000000000004")
CLOCK = "fixture-clock"
SECOND_NS = 1_000_000_000
PLAN = "a" * 64
MODEL_STEP = "model-1"


def span_call(step_id, status="measured", queue=0.0, prefill=0.1, decode=0.2, start_nano=0):
    runner_side = {"method": "fixture", "client_wall_seconds": 2.0, "status": "pending_trace"}
    trace = {"status": status, "server_start_unix_nano": start_nano}
    if status == "measured":
        trace |= {
            "queue_seconds": queue,
            "prefill_seconds": prefill,
            "decode_seconds": decode,
            "first_token_seconds": prefill,
            "e2e_seconds": queue + prefill + decode,
        }
    else:
        trace["problems"] = ["fixture: no usable span"]
    return {"step_id": step_id, "span_id": "p", "timing": joined_vllm_timing(runner_side, trace)}


def event(step_id, step_type, sequence, state, submitted, started=None, finished=None, reason=None):
    return StepEvent(
        contract_version="live-contracts-v1",
        record_type="step_event",
        run_id=RUN_ID,
        task_id="fixture-r1",
        event_id=uuid4(),
        step_id=step_id,
        step_type=step_type,
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
        reason=reason,
    )


def running_sample(tick, value, availability="measured"):
    return ResourceSample(
        contract_version="live-contracts-v1",
        record_type="resource_sample",
        run_id=RUN_ID,
        task_id="fixture-r1",
        sample_id=uuid4(),
        resource_id="vllm-1",
        service_id="fixture",
        resource_class="inference",
        observed_at="2026-10-24T00:00:00Z",
        clock_id=CLOCK,
        tick_ns=tick,
        metric="inference.vllm.requests_running.v1",
        unit="requests",
        value=value if availability == "measured" else None,
        availability=availability,
        reason=None if availability == "measured" else "Not available in this fixture",
    )


def write_run(run_dir: Path, steps, samples=(), plan=PLAN):
    """steps: (step_id, step_type, duration_seconds, model call or None)."""
    run_dir.mkdir(parents=True)
    calls = [call for *_head, call in steps if call is not None]
    evidence = {
        "run_id": str(RUN_ID),
        "model_calls": calls,
        "replay": {"plan_sha256": plan} if plan else {},
    }
    (run_dir / "agent-evidence.json").write_text(json.dumps(evidence))
    records = []
    for step_id, step_type, duration, _call in steps:
        end = int(duration * SECOND_NS)
        records.append(event(step_id, step_type, 0, "queued", 0))
        records.append(event(step_id, step_type, 1, "running", 0, 0))
        records.append(event(step_id, step_type, 2, "completed", 0, 0, end))
    lines = [json.dumps({"kind": "registration", "payload": {"clock_ids": [CLOCK]}})]
    lines += [
        json.dumps({"kind": "record", "payload": r.model_dump(mode="json")})
        for r in [*records, *samples]
    ]
    (run_dir / "live-journal.jsonl").write_text("\n".join(lines) + "\n")


def baseline_runs(tmp_path, durations=(0.9, 1.0, 1.1), plan=PLAN):
    """Three synthetic N = 1 replays of one model step: mean 1.0, stdev 0.1."""
    runs = []
    for index, duration in enumerate(durations):
        run = tmp_path / f"base-{index}"
        write_run(run, [(MODEL_STEP, "model_call", duration, span_call(MODEL_STEP))], plan=plan)
        runs.append(run)
    return runs


def validate_one(tmp_path, duration, call, name="level"):
    reference = baselines(baseline_runs(tmp_path))
    run = tmp_path / name
    write_run(run, [(MODEL_STEP, "model_call", duration, call)])
    return label_validation([run], reference)["overall"]


def test_the_label_and_reference_use_the_same_noise_threshold(tmp_path):
    reference = baselines(baseline_runs(tmp_path))
    assert noise_threshold(reference[(PLAN, MODEL_STEP)]) == pytest.approx(0.3)
    cases = {
        "tp": (1.31, 0.31),
        "fp": (1.31, 0.29),
        "fn": (1.29, 0.31),
        "tn": (1.29, 0.29),
    }
    runs = {}
    for name, (duration, queue) in cases.items():
        run = tmp_path / name
        write_run(run, [(MODEL_STEP, "model_call", duration, span_call(MODEL_STEP, queue=queue))])
        runs[name] = run
    for name, run in runs.items():
        confusion = label_validation([run], reference)["overall"]["confusion"]
        assert confusion == {key: int(key == name) for key in ("tp", "fp", "fn", "tn")}


def test_the_noise_floor_is_ten_milliseconds():
    assert noise_threshold({"stdev_seconds": 0.001}) == 0.01
    assert noise_threshold({"stdev_seconds": None}) == 0.01


def test_a_baseline_needs_three_values_and_the_omission_is_counted(tmp_path):
    runs = baseline_runs(tmp_path)
    assert (PLAN, MODEL_STEP) in baselines(runs)
    short = baselines(runs[:2])
    assert (PLAN, MODEL_STEP) not in short
    assert short.omitted == 1


def test_a_step_without_a_baseline_is_counted_not_labelled(tmp_path):
    run = tmp_path / "level"
    write_run(run, [(MODEL_STEP, "model_call", 5.0, span_call(MODEL_STEP, queue=4.0))])
    overall = label_validation([run], baselines([]))["overall"]
    assert overall["steps_without_baseline"] == 1
    assert overall["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def contended_runs(tmp_path, tp, fp, fn, tn):
    reference = baselines(baseline_runs(tmp_path))
    runs = []
    plan = [(1.31, 0.31)] * tp + [(1.31, 0.29)] * fp + [(1.29, 0.31)] * fn + [(1.29, 0.29)] * tn
    for index, (duration, queue) in enumerate(plan):
        run = tmp_path / f"c-{index}"
        write_run(run, [(MODEL_STEP, "model_call", duration, span_call(MODEL_STEP, queue=queue))])
        runs.append(run)
    return label_validation(runs, reference)["overall"]


def test_the_verdict_needs_both_precision_and_recall_at_090(tmp_path):
    exact = contended_runs(tmp_path / "a", tp=9, fp=1, fn=1, tn=1)
    assert (exact["precision"], exact["recall"]) == (0.9, 0.9)
    assert exact["verdict"] == "validated"
    low_precision = contended_runs(tmp_path / "b", tp=8, fp=1, fn=0, tn=0)
    assert low_precision["precision"] == pytest.approx(8 / 9)
    assert low_precision["verdict"] == "not_validated"
    low_recall = contended_runs(tmp_path / "c", tp=8, fp=0, fn=1, tn=0)
    assert low_recall["verdict"] == "not_validated"


def test_undefined_precision_is_inconclusive(tmp_path):
    overall = contended_runs(tmp_path, tp=0, fp=0, fn=1, tn=2)
    assert overall["precision"] is None
    assert overall["recall"] == 0.0
    assert overall["verdict"] == "inconclusive"


def test_steps_without_a_measured_span_are_excluded_and_counted(tmp_path):
    reference = baselines(baseline_runs(tmp_path))
    runs = []
    for index, status in enumerate(("unavailable", "malformed")):
        run = tmp_path / f"x-{index}"
        write_run(run, [(MODEL_STEP, "model_call", 5.0, span_call(MODEL_STEP, status=status))])
        runs.append(run)
    missing = tmp_path / "x-missing"
    write_run(missing, [(MODEL_STEP, "model_call", 5.0, None)])
    runs.append(missing)
    overall = label_validation(runs, reference)["overall"]
    assert overall["excluded"] == {"unavailable": 1, "malformed": 1, "not_recorded": 1}
    assert overall["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def test_tool_steps_are_reported_unvalidated(tmp_path):
    runs = []
    for index, duration in enumerate((0.5, 0.5, 0.5)):
        run = tmp_path / f"t-base-{index}"
        write_run(run, [("tool-1", "tool", duration, None)])
        runs.append(run)
    reference = baselines(runs)
    slow = tmp_path / "t-slow"
    write_run(slow, [("tool-1", "tool", 2.0, None)])
    overall = label_validation([slow], reference)["overall"]
    assert overall["tool_steps"] == {
        "validation": "unvalidated",
        "steps": 1,
        "contended": 1,
        "label_rate": 1.0,
    }
    assert overall["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def test_span_timestamps_are_never_used(tmp_path):
    first = validate_one(tmp_path / "a", 1.31, span_call(MODEL_STEP, queue=0.31, start_nano=1))
    second = validate_one(
        tmp_path / "b", 1.31, span_call(MODEL_STEP, queue=0.31, start_nano=9_000_000_000_000)
    )
    assert first == second


def test_contended_excess_is_compared_with_queue_and_service_increase(tmp_path):
    overall = validate_one(
        tmp_path,
        2.0,
        span_call(MODEL_STEP, queue=0.5, prefill=0.4, decode=0.5),
    )
    explained = overall["excess_explained"]
    assert explained["contended_steps"] == 1
    assert explained["excess_seconds"] == pytest.approx(1.0)
    assert explained["queue_seconds"] == pytest.approx(0.5)
    # Baseline service is 0.1 + 0.2 = 0.3; this step served in 0.9.
    assert explained["prefill_decode_increase_seconds"] == pytest.approx(0.6)


def test_spearman_averages_tied_ranks_and_needs_three_pairs():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 4], [1, 2, 2, 4]) == pytest.approx(1.0)
    assert spearman([1, 2], [1, 2]) is None
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_levels_are_reported_separately(tmp_path):
    reference = baselines(baseline_runs(tmp_path))
    hit, miss = tmp_path / "hit", tmp_path / "miss"
    write_run(hit, [(MODEL_STEP, "model_call", 1.31, span_call(MODEL_STEP, queue=0.31))])
    write_run(miss, [(MODEL_STEP, "model_call", 1.0, span_call(MODEL_STEP, queue=0.0))])
    result = label_validation({2: [hit], 4: [miss]}, reference)
    assert result["levels"][2]["confusion"]["tp"] == 1
    assert result["levels"][4]["confusion"]["tn"] == 1
    assert result["overall"]["confusion"]["tp"] == 1
    assert result["overall"]["confusion"]["tn"] == 1


def running_run(tmp_path, name, readings):
    run = tmp_path / name
    samples = [running_sample(tick, v) for tick, v in enumerate(readings)]
    write_run(run, [(MODEL_STEP, "model_call", 1.0, None)], samples=samples)
    return run


def test_running_above_max_num_seqs_is_a_contradiction(tmp_path):
    run = running_run(tmp_path, "r", [1, 3, 2])
    check = max_running_check([run], max_num_seqs=2)
    assert check["status"] == "contradicted"
    assert check["max_observed"] == 3.0
    assert max_running_check([run], max_num_seqs=3)["status"] == "consistent"


def test_no_running_samples_is_not_sampled_not_zero(tmp_path):
    run = tmp_path / "none"
    write_run(
        run,
        [(MODEL_STEP, "model_call", 1.0, None)],
        samples=[running_sample(0, None, availability="unavailable")],
    )
    check = max_running_check([run], max_num_seqs=2)
    assert check["status"] == "not_sampled"
    assert check["max_observed"] is None


def write_failed_step_run(run_dir: Path, step_id, state="failed", plan=PLAN):
    """A synthetic run whose model step ended without completing."""
    run_dir.mkdir(parents=True)
    evidence = {"run_id": str(RUN_ID), "model_calls": [], "replay": {"plan_sha256": plan}}
    (run_dir / "agent-evidence.json").write_text(json.dumps(evidence))
    records = [
        event(step_id, "model_call", 0, "queued", 0),
        event(step_id, "model_call", 1, "running", 0, 0),
        event(step_id, "model_call", 2, state, 0, 0, SECOND_NS, reason="fixture: ended early"),
    ]
    lines = [json.dumps({"kind": "registration", "payload": {"clock_ids": [CLOCK]}})]
    lines += [json.dumps({"kind": "record", "payload": r.model_dump(mode="json")}) for r in records]
    (run_dir / "live-journal.jsonl").write_text("\n".join(lines) + "\n")


def test_model_steps_that_did_not_complete_are_counted_by_final_state(tmp_path):
    reference = baselines(baseline_runs(tmp_path))
    failed, cancelled = tmp_path / "f", tmp_path / "c"
    write_failed_step_run(failed, MODEL_STEP, "failed")
    write_failed_step_run(cancelled, MODEL_STEP, "cancelled")
    overall = label_validation([failed, cancelled], reference)["overall"]
    assert overall["excluded_not_completed"] == {"failed": 1, "cancelled": 1}
    assert overall["excluded"]["not_completed"] == 2
    assert overall["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    levels = label_validation({2: [failed]}, reference)["levels"][2]
    assert levels["excluded_not_completed"] == {"failed": 1}


def test_baselines_count_steps_that_did_not_complete(tmp_path):
    runs = baseline_runs(tmp_path)
    failed = tmp_path / "base-failed"
    write_failed_step_run(failed, MODEL_STEP, "failed")
    result = baselines([*runs, failed])
    assert result.excluded_not_completed == {"failed": 1}
    assert result[(PLAN, MODEL_STEP)]["count"] == 3


def test_runs_without_a_plan_are_counted_separately_from_short_baselines(tmp_path):
    runs = baseline_runs(tmp_path)
    no_plan = tmp_path / "base-noplan"
    write_run(no_plan, [(MODEL_STEP, "model_call", 1.0, span_call(MODEL_STEP))], plan=None)
    result = baselines([*runs, no_plan])
    assert result.runs_without_plan == 1
    assert result.omitted == 0
    level = tmp_path / "level-noplan"
    write_run(level, [(MODEL_STEP, "model_call", 5.0, span_call(MODEL_STEP, queue=4.0))], plan=None)
    report = label_validation([level], result)
    assert report["baseline_runs_without_plan"] == 1
    assert report["overall"]["runs_without_plan"] == 1
    assert report["overall"]["steps_without_baseline"] == 0
    assert report["overall"]["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def test_the_baseline_length_is_the_declared_replay_count():
    import agentwatch_agent.sweep_labels as labels
    from agentwatch_agent.sweep_protocol import BASELINE_BATCHES_PER_SOURCE

    assert not hasattr(labels, "MIN_BASELINE_VALUES")
    assert BASELINE_BATCHES_PER_SOURCE == 3


def test_non_finite_or_negative_values_are_malformed_not_classified(tmp_path, monkeypatch):
    reference = baselines(baseline_runs(tmp_path))
    nan_run, negative_span = tmp_path / "nan", tmp_path / "negspan"
    write_run(
        nan_run, [(MODEL_STEP, "model_call", 1.31, span_call(MODEL_STEP, queue=float("nan")))]
    )
    write_run(
        negative_span,
        [(MODEL_STEP, "model_call", 1.31, span_call(MODEL_STEP, queue=0.31, prefill=-0.1))],
    )
    overall = label_validation([nan_run, negative_span], reference)["overall"]
    assert overall["excluded"]["malformed"] == 2
    assert overall["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    bad_duration = tmp_path / "negdur"
    write_run(bad_duration, [(MODEL_STEP, "model_call", 1.0, span_call(MODEL_STEP))])
    import agentwatch_agent.sweep_labels as labels

    monkeypatch.setattr(
        labels,
        "step_timings",
        lambda _run: {
            MODEL_STEP: {
                "step_type": "model_call",
                "state": "completed",
                "total_seconds": -1.0,
            }
        },
    )
    report = label_validation([bad_duration], reference)["overall"]
    assert report["excluded"]["malformed"] == 1
    assert report["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    assert baselines([bad_duration] * 3).excluded_malformed == 3


def test_a_plain_dict_reference_reports_omitted_as_unknown(tmp_path):
    reference = dict(baselines(baseline_runs(tmp_path)))
    run = tmp_path / "level"
    write_run(run, [(MODEL_STEP, "model_call", 1.31, span_call(MODEL_STEP, queue=0.31))])
    report = label_validation([run], reference)
    assert report["overall"]["confusion"]["tp"] == 1
    assert report["baselines_omitted"] is None
    assert report["baseline_runs_without_plan"] is None


def test_level_one_never_enters_validation(tmp_path):
    reference = baselines(baseline_runs(tmp_path))
    one, two = tmp_path / "one", tmp_path / "two"
    write_run(one, [(MODEL_STEP, "model_call", 1.31, span_call(MODEL_STEP, queue=0.31))])
    write_run(two, [(MODEL_STEP, "model_call", 1.0, span_call(MODEL_STEP, queue=0.0))])
    result = label_validation({1: [one], 2: [two]}, reference)
    assert result["levels_skipped"] == [1]
    assert 1 not in result["levels"]
    assert result["overall"]["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 1}
