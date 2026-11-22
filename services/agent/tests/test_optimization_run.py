"""The optimization batch runner (O06): preregistered slots in order, one record per attempted
batch, policies applied and read back, and the baseline restored on every exit.

Models, the live API, the policy service and injectors are test doubles; nothing here is
recorded evidence or a hardware result.
"""

import copy
import dataclasses
import json
import threading
import time
from urllib.error import URLError

import pytest
from agent_test_doubles import SOLVE_MEAN, ScriptedModel
from agentwatch_agent.batch import AgentBatch, BatchRun
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.optimization_experiment import (
    ExperimentError,
    read_experiment,
    recorded_slots,
)
from agentwatch_agent.optimization_run import (
    LOG_FILE,
    ExperimentRunner,
    _longest_waits,
    _task_state,
)
from agentwatch_agent.runner import RunOutcome
from optimization_doubles import (
    DECLARED,
    PREFILL,
    FakePolicyService,
    make_host,
    record,
    slot,
    write_experiment,
)
from test_runner_admission import make_runner
from test_sweep import FakeInjector

CLEAN_PAIR = [slot(0), slot(1, configuration="candidate")]


def run_experiment(root, host, slots=CLEAN_PAIR, **changes):
    write_experiment(root, host, slots, **changes)
    runner = ExperimentRunner(read_experiment(root), host)
    runner.preflight()
    return runner, runner.run()


def test_each_attempted_batch_records_its_runs_policy_and_admission(server, tmp_path):
    policy = FakePolicyService()
    host = make_host(server, policy)
    root = tmp_path / "experiment"
    _runner, summary = run_experiment(root, host)
    preregistered = {
        task["task_id"]: task["configuration_digest"]
        for task in json.loads((root / "manifest.json").read_text())["workload"]["tasks"]
    }

    assert summary.statuses == {0: "completed", 1: "completed"} and summary.stopped is None
    baseline, candidate = record(root, 0), record(root, 1)
    assert baseline["policy"] == {"sha256": "b" * 64, "limits": {}, "version": 0}
    # The candidate's limit was set on the runner's model server and read back before running.
    assert candidate["policy"]["limits"] == {"inference_requests": 1}
    assert candidate["policy"]["version"] == 1
    assert policy.changes[0].inference_service == "ollama"
    for entry in (baseline, candidate):
        assert entry["format"] == "opt-batch-v1"
        assert {key: entry[key] for key in DECLARED} == DECLARED
        assert entry["injection"] == {"active": None, "cleanup_confirmed": None, "injectors": []}
        assert entry["admission"] == {"starved_admissions": 0, "leaked_permits": 0}
        solved, gave_up = entry["tasks"]
        # A failed task is an outcome of a completed batch, recorded as it happened.
        assert (solved["state"], solved["acceptance"]) == ("completed", "passed")
        assert (gave_up["state"], gave_up["acceptance"]) == ("failed", "failed")
        # Submissions follow the preregistered offsets on the batch's one clock.
        offset = (gave_up["submitted_tick_ns"] - solved["submitted_tick_ns"]) / 1e9
        assert offset == pytest.approx(0.2, abs=0.05)
        for task in entry["tasks"]:
            assert task["finished_tick_ns"] > task["submitted_tick_ns"]
            evidence = json.loads((root / task["evidence_dir"] / "agent-evidence.json").read_text())
            assert evidence["run_id"] == task["run_id"]
            assert evidence["clock_id"] == entry["clock_id"]
            # The digest preregistered before any run is the one each run records.
            assert evidence["configuration_digest"] == preregistered[task["task_id"]]
    assert candidate["tasks"][0]["admission_waits"]["inference_requests"] is not None
    # The baseline is restored and read back at the end.
    assert summary.restored is True
    assert policy.get_policy().limits.model_dump(exclude_none=True) == {}
    events = [json.loads(line)["event"] for line in (root / LOG_FILE).read_text().splitlines()]
    assert events[0] == "start" and events[-1] == "stop" and events.count("batch") == 2


def test_records_are_never_overwritten_and_a_rerun_resumes_at_the_next_slot(server, tmp_path):
    policy = FakePolicyService()
    calls = {"n": 0}

    def flaky_identity():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("powercfg.exe timed out")
        return copy.deepcopy(DECLARED)

    root = tmp_path / "experiment"
    _runner, first = run_experiment(root, make_host(server, policy, observe=flaky_identity))
    assert first.attempted == [0] and "Slot 1 not attempted" in first.stopped
    assert first.restored is True
    kept = (root / "batches" / "000.json").read_bytes()

    runner = ExperimentRunner(read_experiment(root), make_host(server, policy))
    second = runner.run()
    assert second.attempted == [1] and second.statuses == {1: "completed"}
    assert (root / "batches" / "000.json").read_bytes() == kept
    assert recorded_slots(root) == 2


def test_a_host_that_no_longer_matches_the_manifest_is_not_attempted(server, tmp_path):
    on_battery = copy.deepcopy(DECLARED) | {"power": {"source": "battery", "profile": "Balanced"}}
    policy = FakePolicyService()
    root = tmp_path / "experiment"
    _runner, summary = run_experiment(
        root, make_host(server, policy, observe=lambda: copy.deepcopy(on_battery))
    )
    assert summary.attempted == [] and "power differ from the manifest" in summary.stopped
    assert not (root / "batches").exists()
    assert policy.changes == [] and summary.restored is True


def test_a_policy_that_cannot_be_applied_is_a_not_started_batch(server, tmp_path):
    policy = FakePolicyService()
    policy.failures = [URLError("policy service down")]
    root = tmp_path / "experiment"
    _runner, summary = run_experiment(
        root, make_host(server, policy), slots=[slot(0, configuration="candidate")]
    )
    entry = record(root, 0)
    assert entry["status"] == "not_started" and "Policy not applied" in entry["reason"]
    assert entry["started_at"] is None and entry["policy"]["version"] is None
    assert all(task["state"] == "not_started" for task in entry["tasks"])
    assert summary.stopped.startswith("Slot 0 not_started") and summary.restored is True


def test_a_policy_read_back_on_another_server_never_runs_the_batch(server, tmp_path):
    # The service stored the limit, but on another model server: it would not be enforced.
    policy = FakePolicyService(ignore_service=True)
    host = make_host(server, policy)
    root = tmp_path / "experiment"
    write_experiment(root, host, [slot(0, configuration="candidate")])
    host = dataclasses.replace(host, inference_service_id="vllm")
    summary = ExperimentRunner(read_experiment(root), host).run()
    entry = record(root, 0)
    assert entry["status"] == "not_started"
    assert "on ollama, not {'inference_requests': 1}" in entry["reason"]
    assert summary.statuses == {0: "not_started"}


def test_a_contended_batch_records_its_injectors_active_and_cleaned_up(server, tmp_path):
    root = tmp_path / "experiment"
    _runner, summary = run_experiment(
        root, make_host(server, FakePolicyService()), slots=[slot(0, condition="contended")]
    )
    entry = record(root, 0)
    assert summary.statuses == {0: "completed"}
    assert entry["injection"] == {"active": True, "cleanup_confirmed": True, "injectors": [PREFILL]}
    assert json.loads((root / "runs" / "000" / "injection.json").read_text())["cleanup_confirmed"]


def test_unconfirmed_injector_cleanup_aborts_the_batch_and_stops(server, tmp_path):
    root = tmp_path / "experiment"
    host = make_host(
        server,
        FakePolicyService(),
        injector=lambda clock, bound: FakeInjector(clock, bound, confirm=False),
    )
    _runner, summary = run_experiment(
        root, host, slots=[slot(0, condition="contended"), slot(1, condition="contended")]
    )
    entry = record(root, 0)
    assert entry["status"] == "aborted" and "cleanup not confirmed" in entry["reason"]
    # The tasks ran; their outcomes stay in the record.
    assert [task["state"] for task in entry["tasks"]] == ["completed", "failed"]
    assert summary.attempted == [0] and summary.restored is True


def test_an_injector_that_never_starts_leaves_the_batch_not_started(server, tmp_path):
    root = tmp_path / "experiment"
    host = make_host(
        server,
        FakePolicyService(),
        injector=lambda clock, bound: FakeInjector(
            clock, bound, start_error=OSError("no loopback")
        ),
    )
    run_experiment(root, host, slots=[slot(0, condition="contended")])
    entry = record(root, 0)
    assert entry["status"] == "not_started" and "Injector did not start" in entry["reason"]


def test_a_cancelled_experiment_attempts_nothing_more_and_restores(server, tmp_path):
    policy = FakePolicyService()
    root = tmp_path / "experiment"
    host = make_host(server, policy)
    write_experiment(root, host, CLEAN_PAIR)
    runner = ExperimentRunner(read_experiment(root), host)
    runner.request_cancel_from_signal("signal", "Cancelled by operator signal SIGINT")
    summary = runner.run()
    assert summary.attempted == [] and summary.stopped.startswith("Cancelled before slot 0")
    assert summary.restored is True


def test_a_signal_during_the_injector_ramp_submits_nothing_and_stops(server, tmp_path):
    root = tmp_path / "experiment"
    host = dataclasses.replace(make_host(server, FakePolicyService()), ramp_seconds=20.0)
    write_experiment(root, host, [slot(0, condition="contended"), slot(1, condition="contended")])
    runner = ExperimentRunner(read_experiment(root), host)
    result = {}
    thread = threading.Thread(target=lambda: result.update(summary=runner.run()))
    thread.start()
    deadline = time.monotonic() + 10
    # The slot's policy is applied just before its injectors start ramping.
    while '"event": "policy"' not in (
        (root / LOG_FILE).read_text() if (root / LOG_FILE).exists() else ""
    ):
        assert time.monotonic() < deadline, "the batch never started"
        time.sleep(0.01)
    runner.request_cancel_from_signal("signal", "Cancelled by operator signal SIGINT")
    thread.join(10)
    assert not thread.is_alive(), "the ramp was not interrupted"
    entry = record(root, 0)
    assert entry["status"] == "not_started" and "ramp" in entry["reason"]
    assert result["summary"].attempted == [0] and result["summary"].restored is True


def test_preflight_refuses_a_configuration_the_manifest_did_not_preregister(server, tmp_path):
    host = make_host(server, FakePolicyService())
    root = tmp_path / "experiment"
    manifest = write_experiment(root, host, CLEAN_PAIR)
    manifest["workload"]["tasks"][0]["configuration_digest"] = "f" * 64
    write_experiment(root, host, CLEAN_PAIR, workload=manifest["workload"])
    with pytest.raises(ExperimentError, match="configuration digest"):
        ExperimentRunner(read_experiment(root), host).preflight()


def test_preflight_refuses_an_injector_it_cannot_apply(server, tmp_path):
    host = make_host(server, FakePolicyService())
    root = tmp_path / "experiment"
    unknown = {
        "kind": "retrieval",
        "method": "in-process-index-query-flood-v1",
        "intensity": {"q": 1},
    }
    write_experiment(
        root,
        host,
        CLEAN_PAIR,
        conditions=[
            {"name": "clean", "injectors": []},
            {"name": "contended", "injectors": [unknown]},
        ],
    )
    with pytest.raises(ExperimentError, match="not one this runner can apply"):
        ExperimentRunner(read_experiment(root), host).preflight()


def test_a_schedule_for_another_manifest_is_refused(server, tmp_path):
    host = make_host(server, FakePolicyService())
    root = tmp_path / "experiment"
    write_experiment(root, host, CLEAN_PAIR)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["created_at"] = "2026-11-23T10:00:00Z"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ExperimentError, match="another manifest"):
        read_experiment(root)


def test_a_manifest_holding_nan_is_refused(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "manifest.json").write_text('{"format": "opt-experiment-manifest-v1", "x": NaN}')
    with pytest.raises(ExperimentError, match="canonical form"):
        read_experiment(root)


def test_records_must_be_the_schedules_first_slots(tmp_path):
    batches = tmp_path / "batches"
    batches.mkdir()
    (batches / "001.json").write_text("{}")
    with pytest.raises(ExperimentError, match="not the schedule's first slots"):
        recorded_slots(tmp_path)


def outcome(state, reason=None, **flags):
    return RunOutcome(
        run_id="6b0c4c55-0000-4000-8000-000000000001",
        state=state,
        reason=reason,
        task_passed=state == "completed",
        delivery_complete=True,
        evidence_path=None,
        journal_path=None,
        **flags,
    )


@pytest.mark.parametrize(
    ("state", "reason", "expected"),
    [
        ("completed", None, "completed"),
        ("completed_locally_undelivered", None, "completed"),
        ("failed", "Run exceeded its 450 s time limit", "timed_out"),
        ("failed", "Tests failed", "failed"),
        ("cancelled", "Cancelled by operator signal SIGINT", "cancelled"),
    ],
)
def test_run_states_map_to_the_batch_task_states(state, reason, expected):
    run = BatchRun(1, runner=None, submitted_tick_ns=5, outcome=outcome(state, reason))
    assert _task_state(run) == expected


def test_runs_that_never_started_or_errored_are_told_apart():
    assert _task_state(BatchRun(1, runner=None)) == "not_started"
    started = BatchRun(
        1, runner=None, submitted_tick_ns=5, not_started=RunNotStarted("x", "registration")
    )
    assert _task_state(started) == "not_started"
    assert (
        _task_state(BatchRun(1, runner=None, submitted_tick_ns=5, error=RuntimeError())) == "failed"
    )


def test_the_longest_wait_per_control_is_kept_and_unrecorded_waits_stay_null():
    evidence = {
        "admissions": [
            {"control": "inference_requests", "admission_wait_seconds": 0.5},
            {"control": "inference_requests", "admission_wait_seconds": 2.0},
            {"control": "inference_requests", "admission_wait_seconds": float("nan")},
            {"control": "other", "admission_wait_seconds": 9.0},
        ]
    }
    assert _longest_waits(evidence) == {"inference_requests": 2.0, "test_jobs": None}
    assert _longest_waits({}) == {"inference_requests": None, "test_jobs": None}


def test_batch_offsets_are_validated_and_a_cancelled_wait_never_starts_the_run(
    mean_task, server, tmp_path
):
    def runners(count):
        clock = MonotonicClock()
        return [
            make_runner(
                mean_task, server, ScriptedModel(list(SOLVE_MEAN)), tmp_path / f"r{i}", clock=clock
            )
            for i in range(count)
        ]

    for bad in ([0.0], [0.0, -1.0], [0.0, float("inf")], [0.0, True]):
        with pytest.raises(ValueError, match="start offset|Start offsets"):
            AgentBatch(runners(2), None, tmp_path / "b", start_offsets_seconds=bad)
    batch = AgentBatch(runners(2), None, tmp_path / "b", start_offsets_seconds=[0.0, 30.0])
    thread = threading.Thread(target=batch.run)
    thread.start()
    first, second = batch.runs
    deadline = time.monotonic() + 20
    while first.outcome is None:
        assert time.monotonic() < deadline, "the first run did not finish"
        time.sleep(0.01)
    second.runner.cancellation.cancel("test", "stop waiting")
    thread.join(20)
    assert not thread.is_alive()
    assert first.state() == "completed" and first.submitted_tick_ns is not None
    assert second.submitted_tick_ns is None and second.not_started.stage == "setup"
