"""Replay experiments with test-double injectors, a scripted model and the in-memory live server.

Nothing here applies real contention or produces experiment evidence; the real injectors are
tested in services/injectors/tests.
"""

import json
import math
from uuid import UUID

import pytest
from agent_test_doubles import SOLVE_MEAN, ScriptedModel
from agentwatch_agent import cli
from agentwatch_agent.experiment import (
    MANIFEST_FILE,
    ExperimentSettings,
    ReplayExperiment,
    baseline_variability,
    build_injectors,
    compare_with_baseline,
    injection_active,
    variability,
)
from agentwatch_agent.replay import ReplayRunner, load_replay_plan
from agentwatch_agent.runner import EVIDENCE_FILE
from agentwatch_injectors.base import BoundedInjector, InjectorCleanupError
from agentwatch_injectors.cpu import CpuSpinInjector
from agentwatch_injectors.inference import CompetingInferenceInjector
from agentwatch_injectors.retrieval import IndexQueryFloodInjector
from test_replay import record_source, runner_settings

ACTIVITY = {"cpu": "hash_rounds", "inference": "requests_completed", "retrieval": "searches"}


class FakeInjector(BoundedInjector):
    """Test double: applies no load; reports one unit of its kind's activity."""

    method = "test-double"
    resource_class = "test"

    def __init__(self, clock, bound_seconds, kind, confirm=True, active=True):
        super().__init__(clock, bound_seconds)
        self.kind = kind
        self.confirm = confirm
        self.active = active

    def bounds(self):
        return {}

    def target(self):
        return {}

    def _launch(self):
        pass

    def _halt(self):
        return {ACTIVITY[self.kind]: int(self.active)}, {"confirmed": self.confirm}


@pytest.fixture
def plan(mean_task, server, tmp_path):
    source, _ = record_source(mean_task, server, tmp_path)
    return load_replay_plan(source, mean_task)


def make_experiment(mean_task, server, plan, tmp_path, settings=None, injector=FakeInjector):
    calls = []

    def make_injectors(kinds, clock, bound_seconds):
        calls.append((tuple(kinds), bound_seconds))
        return [injector(clock, bound_seconds, kind) for kind in kinds]

    def make_runner(replay_dir, clock):
        model = ScriptedModel(SOLVE_MEAN)
        settings = runner_settings(mean_task, server, model, replay_dir)
        return ReplayRunner(plan=plan, clock=clock, **settings)

    experiment = ReplayExperiment(
        plan=plan,
        settings=settings
        or ExperimentSettings(
            baseline_replays=3,
            warmup_replays=1,
            conditions=("cpu", "mixed"),
            ramp_seconds=0,
            cpu_workers=1,
        ),
        output_dir=tmp_path / "experiment",
        run_timeout_seconds=60,
        make_runner=make_runner,
        make_injectors=make_injectors,
        source={"run_id": plan.source_run_id},
    )
    return experiment, calls


def read(path):
    return json.loads(path.read_text())


def test_experiment_runs_the_declared_schedule(mean_task, server, plan, tmp_path):
    experiment, calls = make_experiment(mean_task, server, plan, tmp_path)
    manifest = experiment.run()
    assert manifest["status"] == "completed"
    assert [(r["role"], r["condition"]) for r in manifest["replays"]] == [
        ("warmup", "clean"),
        ("baseline", "clean"),
        ("baseline", "clean"),
        ("baseline", "clean"),
        ("contended", "cpu"),
        ("clean_control", "clean"),
        ("contended", "mixed"),
        ("clean_control", "clean"),
    ]
    # Clean replays get an empty injector set; each injector is bounded by the run limit.
    assert [kinds for kinds, _ in calls if kinds] == [("cpu",), ("cpu", "inference")]
    assert {bound for _, bound in calls} == {60 + 0 + 30}
    assert all(r["state"] == "completed" and r["task_passed"] for r in manifest["replays"])
    checks = manifest["checks"]
    assert {name for name, passed in checks.items() if not passed} <= {"stable_baseline"}
    assert isinstance(checks["stable_baseline"], bool)
    assert manifest["passed"] == all(checks.values())
    assert manifest["protocol"]["version"] == "agent-replay-experiment-v1"
    assert manifest["plan"]["plan_sha256"] == plan.sha256()
    assert read(tmp_path / "experiment" / MANIFEST_FILE) == json.loads(json.dumps(manifest))


def test_baseline_variability_and_comparisons_are_reported(mean_task, server, plan, tmp_path):
    manifest = make_experiment(mean_task, server, plan, tmp_path)[0].run()
    root = manifest["baseline_variability"]["agent-run"]
    assert root["count"] == 3
    assert root["stdev_seconds"] is not None and root["cv"] is not None
    assert root["min_seconds"] <= root["mean_seconds"] <= root["max_seconds"]
    assert "tool-1-search_docs" in manifest["baseline_variability"]
    for entry in manifest["replays"]:
        compared = entry.get("comparison_with_baseline")
        if entry["role"] in ("contended", "clean_control"):
            assert set(compared) == set(manifest["baseline_variability"])
            assert compared["agent-run"]["status"] == "completed"
        else:
            assert compared is None
        assert entry["model_outputs_matching_source"] == entry["model_outputs"] == 5


def test_conditions_stay_out_of_replay_evidence(mean_task, server, plan, tmp_path):
    manifest = make_experiment(mean_task, server, plan, tmp_path)[0].run()
    root = tmp_path / "experiment"
    digests = set()
    for entry in manifest["replays"]:
        assert entry["evidence_dir"] == f"replays/{entry['index']:02d}"
        evidence_text = (root / entry["evidence_dir"] / EVIDENCE_FILE).read_text()
        for leaked in ("injection", "contended", "clean_control", "baseline"):
            assert leaked not in evidence_text
        digests.add(json.loads(evidence_text)["configuration_digest"])
        run = server.runs[UUID(entry["run_id"])]
        assert run.registration.configuration_digest in digests
    # Clean and contended replays register one workload digest.
    assert len(digests) == 1


def test_injection_logs_share_the_replays_clock(mean_task, server, plan, tmp_path):
    manifest = make_experiment(mean_task, server, plan, tmp_path)[0].run()
    root = tmp_path / "experiment"
    contended = [r for r in manifest["replays"] if r["role"] == "contended"]
    assert [r["injection_log"] for r in contended] == ["injections/04.json", "injections/06.json"]
    assert all(r["injection_log"] is None for r in manifest["replays"] if r not in contended)
    assert sorted(p.name for p in (root / "injections").iterdir()) == ["04.json", "06.json"]
    for entry in contended:
        log = read(root / entry["injection_log"])
        evidence = read(root / entry["evidence_dir"] / EVIDENCE_FILE)
        assert log["clock_id"] == evidence["clock_id"]
        assert {i["clock_id"] for i in log["injections"]} == {evidence["clock_id"]}
        assert log["cleanup_confirmed"] is True and entry["injection_active"] is True
    assert read(root / "injections/06.json")["injectors"] == ["cpu", "inference"]


def test_unconfirmed_cleanup_stops_the_experiment(mean_task, server, plan, tmp_path):
    def leaky(clock, bound, kind):
        return FakeInjector(clock, bound, kind, confirm=False)

    experiment, _ = make_experiment(mean_task, server, plan, tmp_path, injector=leaky)
    with pytest.raises(InjectorCleanupError):
        experiment.run()
    manifest = read(tmp_path / "experiment" / MANIFEST_FILE)
    assert manifest["status"] == "stopped_injector_cleanup_unconfirmed"
    # The contended replay itself is kept; nothing ran after it.
    assert len(manifest["replays"]) == 5
    last = manifest["replays"][-1]
    assert (last["role"], last["state"], last["injection_cleanup_confirmed"]) == (
        "contended",
        "completed",
        False,
    )
    assert "cleanup was not confirmed" in last["error"]
    assert read(tmp_path / "experiment/injections/04.json")["cleanup_confirmed"] is False
    assert manifest["passed"] is False


def test_inactive_injectors_fail_their_check(mean_task, server, plan, tmp_path):
    def idle(clock, bound, kind):
        return FakeInjector(clock, bound, kind, active=False)

    experiment, _ = make_experiment(mean_task, server, plan, tmp_path, injector=idle)
    manifest = experiment.run()
    assert manifest["checks"]["injectors_active"] is False
    assert manifest["passed"] is False


def test_cancellation_during_a_ramp_ends_the_experiment(mean_task, server, plan, tmp_path):
    settings = ExperimentSettings(
        baseline_replays=3, warmup_replays=0, conditions=("cpu",), ramp_seconds=30, cpu_workers=1
    )
    holder = {}

    def cancelling(clock, bound, kind):
        holder["experiment"].request_cancel_from_signal("signal", "Cancelled by test")
        return FakeInjector(clock, bound, kind)

    experiment, _ = make_experiment(
        mean_task, server, plan, tmp_path, settings=settings, injector=cancelling
    )
    holder["experiment"] = experiment
    manifest = experiment.run()
    assert manifest["status"] == "cancelled"
    assert [r["state"] for r in manifest["replays"]] == ["completed"] * 3 + ["cancelled"]
    assert read(tmp_path / "experiment/injections/03.json")["injections"][0]["stop_reason"] == (
        "cancelled"
    )


def test_an_unreachable_live_api_stops_before_more_replays(mean_task, server, plan, tmp_path):
    server.down = True
    manifest = make_experiment(mean_task, server, plan, tmp_path)[0].run()
    assert manifest["status"] == "stopped_replay_not_started"
    assert [r["state"] for r in manifest["replays"]] == ["not_started"]
    assert manifest["passed"] is False


def test_real_injectors_are_built_with_the_replays_settings(mean_task, plan):
    from agentwatch_agent.clock import MonotonicClock

    clock = MonotonicClock()
    settings = ExperimentSettings(cpu_workers=1, inference_concurrency=3, retrieval_threads=4)
    cpu, inference, retrieval = build_injectors(
        ("cpu", "inference", "retrieval"),
        clock,
        90.0,
        settings=settings,
        plan=plan,
        task=mean_task,
        ollama_url="http://127.0.0.1:11435",
    )
    assert isinstance(cpu, CpuSpinInjector) and cpu.workers == 1
    assert isinstance(inference, CompetingInferenceInjector)
    assert (inference.model, inference.num_ctx, inference.concurrency) == (
        plan.model,
        plan.chat_options["num_ctx"],
        3,
    )
    assert isinstance(retrieval, IndexQueryFloodInjector)
    assert retrieval.queries == ("mean of an empty input",) and retrieval.threads == 4
    assert retrieval.search("mean of an empty input")  # the task's own index answers it
    assert {i.max_duration_seconds for i in (cpu, inference, retrieval)} == {90.0}
    assert all(i.clock is clock for i in (cpu, inference, retrieval))
    with pytest.raises(ValueError, match="loopback"):
        build_injectors(
            ("inference",),
            clock,
            90.0,
            settings=settings,
            plan=plan,
            task=mean_task,
            ollama_url="http://192.168.1.20:11434",
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"baseline_replays": 2},
        {"warmup_replays": -1},
        {"conditions": ("cpu", "cpu")},
        {"conditions": ("gpu",)},
        {"mixed": ("cpu",)},
        {"mixed": ("cpu", "mixed")},
        {"cpu_workers": 0},
        {"inference_concurrency": 5},
        {"retrieval_threads": 5},
        {"ramp_seconds": math.inf},
    ],
)
def test_invalid_settings_are_refused_before_anything_runs(settings):
    with pytest.raises(ValueError):
        ExperimentSettings(**({"cpu_workers": 1} | settings))


def test_the_injection_bound_must_fit_under_the_hard_cap():
    settings = ExperimentSettings(cpu_workers=1)
    assert settings.injection_bound_seconds(900) == 932
    with pytest.raises(ValueError, match="cap"):
        settings.injection_bound_seconds(1790)


def test_variability_statistics():
    stats = variability([1.0, 2.0, 3.0])
    assert stats["mean_seconds"] == 2.0
    assert stats["stdev_seconds"] == 1.0
    assert stats["cv"] == 0.5
    assert variability([2.0])["cv"] is None


def test_comparisons_mark_missing_steps_and_zero_spread():
    baseline = {
        "a": {"mean_seconds": 1.0, "stdev_seconds": 0.0},
        "b": {"mean_seconds": 1.0, "stdev_seconds": 0.5},
    }
    steps = {"b": {"state": "completed", "total_seconds": 2.0}}
    comparison = compare_with_baseline(steps, baseline)
    assert comparison["a"] == {"status": "missing"}
    assert comparison["b"]["difference_seconds"] == 1.0
    assert comparison["b"]["baseline_stdevs"] == 2.0
    assert comparison["b"]["ratio"] == 2.0
    unstable = compare_with_baseline({"a": {"state": "completed", "total_seconds": 2.0}}, baseline)
    assert unstable["a"]["baseline_stdevs"] is None


def test_baseline_variability_uses_steps_completed_in_every_baseline():
    completed = {"state": "completed", "total_seconds": 1.0}
    baselines = [
        {"steps": {"a": completed, "b": completed}},
        {"steps": {"a": completed, "b": completed | {"state": "failed"}}},
    ]
    assert sorted(baseline_variability(baselines)) == ["a"]
    assert baseline_variability([]) == {}


def test_injection_activity_must_be_measured():
    assert injection_active({"injections": []}) is False
    cpu = {"injector": "cpu", "activity": {"hash_rounds": 4}}
    assert injection_active({"injections": [cpu]}) is True
    idle = {"injector": "inference", "activity": {"report": "missing"}}
    assert injection_active({"injections": [cpu, idle]}) is False


def test_experiment_cli_refuses_invalid_settings(tmp_path, capsys):
    base = ["experiment", "--task", "fix-mean-empty", "--output-dir", str(tmp_path)]
    assert cli.main([*base, "--baseline-replays", "2"]) == 2
    assert "baseline_replays" in capsys.readouterr().err
    assert cli.main([*base, "--run-timeout", "1790"]) == 2
    assert "cap" in capsys.readouterr().err
    assert cli.main([*base, "--source", str(tmp_path / "missing")]) == 2
    assert "Cannot read source evidence" in capsys.readouterr().err
    assert cli.main([*base, "--ollama-url", "http://192.168.1.20:11434"]) == 2
    assert "loopback" in capsys.readouterr().err
    remote_without_inference = [*base, "--ollama-url", "http://192.168.1.20:11434"]
    remote_without_inference += ["--conditions", "cpu", "--source", str(tmp_path / "missing")]
    assert cli.main(remote_without_inference) == 2
    assert "Cannot read source evidence" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*base, "--conditions", "cpu,gpu"])
    assert exit_info.value.code == 2
    assert list(tmp_path.iterdir()) == []
