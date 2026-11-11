import json
import os
import signal
import threading
import time
from types import SimpleNamespace
from typing import ClassVar
from urllib.error import URLError

import pytest
from agent_test_doubles import (
    SOLVE_MEAN,
    FakeLiveServer,
    FakePolicySource,
    ScriptedModel,
    chat_response,
    policy_version,
)
from agentwatch_agent import cli
from agentwatch_agent.config import RunConfiguration, RunLimits
from agentwatch_agent.sweep_protocol import (
    INJECTOR_CONCURRENCY,
    INJECTOR_PROMPT_TOKENS,
    SAMPLING_INTERVAL_SECONDS,
    Batch,
)
from agentwatch_worker.nvml import GpuIdentity, NvmlUnavailable


def test_tasks_command_lists_pinned_identities(capsys):
    assert cli.main(["tasks"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[1] for line in lines] == ["fix-mean-empty-r1", "fix-window-count-r1"]


def test_unknown_task_is_a_usage_error(capsys):
    assert cli.main(["run", "--task", "nope"]) == 2
    assert "Unknown task" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--task", "fix-mean-empty", "--api-url", "ftp://x"],
        ["run", "--task", "fix-mean-empty", "--run-timeout", "-5"],
        ["run", "--task", "fix-mean-empty", "--run-timeout", "inf"],
    ],
)
def test_invalid_arguments_exit_with_usage_errors(argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2


def test_unreachable_api_exits_without_running(tmp_path, capsys):
    code = cli.main(
        [
            "run",
            "--task",
            "fix-mean-empty",
            "--api-url",
            "http://127.0.0.1:9",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == cli.EXIT_LIVE_UNAVAILABLE
    assert "nothing ran" in capsys.readouterr().err


def configuration(**limits):
    return RunConfiguration(
        task_id="t-r1",
        task_content_sha256="0" * 64,
        model="m",
        system_prompt="p",
        tools=(),
        sampled_metrics=("cpu.psi.some.avg10.v1",),
        limits=RunLimits(**limits),
    )


def test_configuration_digest_is_stable_and_sensitive():
    assert configuration().digest() == configuration().digest()
    assert configuration().digest() != configuration(max_model_calls=9).digest()
    assert len(configuration().digest()) == 64


def test_sampling_budget_must_fit_the_live_record_limit():
    with pytest.raises(ValueError, match="sample budget"):
        configuration(run_timeout_seconds=24 * 3600)
    configuration(run_timeout_seconds=900)


@pytest.mark.parametrize(("field", "value"), [("model_backend", "tgi"), ("model_transport", "x")])
def test_unknown_backends_and_transports_are_rejected(field, value):
    settings = {
        "task_id": "t-r1",
        "task_content_sha256": "0" * 64,
        "model": "m",
        "system_prompt": "p",
        "tools": (),
        "sampled_metrics": ("cpu.psi.some.avg10.v1",),
    }
    with pytest.raises(ValueError, match=field):
        RunConfiguration(**settings, **{field: value})


def test_the_backend_selects_its_adapter_and_default_model(monkeypatch):
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    monkeypatch.delenv("AGENTWATCH_LLM_MODEL", raising=False)
    vllm = cli.build_model(
        cli.parser().parse_args(["run", "--task", "x", "--model-backend", "vllm"])
    )
    assert (vllm.backend, vllm.name, vllm.endpoint) == (
        "vllm",
        cli.DEFAULT_VLLM_MODEL,
        cli.DEFAULT_VLLM_URL,
    )
    assert vllm.num_ctx == 4096
    ollama = cli.build_model(cli.parser().parse_args(["run", "--task", "x", "--model", "q"]))
    assert (ollama.backend, ollama.name) == ("ollama", "q")


def test_vllm_contention_only_targets_a_loopback_server(tmp_path, capsys):
    argv = ["experiment", "--task", "fix-mean-empty", "--output-dir", str(tmp_path)]
    argv += ["--model-backend", "vllm", "--vllm-url", "http://10.0.0.5:8000"]
    assert cli.main(argv) == cli.EXIT_USAGE
    assert "loopback" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "limits",
    [{"max_model_calls": 0}, {"max_tool_calls": True}, {"run_timeout_seconds": float("nan")}],
)
def test_invalid_limits_are_rejected(limits):
    with pytest.raises(ValueError):
        RunLimits(**limits)


def test_overlong_run_timeout_is_a_usage_error_that_creates_nothing(tmp_path, capsys):
    code = cli.main(
        ["run", "--task", "fix-mean-empty", "--run-timeout", "20000", "--output-dir", str(tmp_path)]
    )
    assert code == 2
    assert "sample budget" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_refused_records_exit_five_and_recoverable_gaps_exit_three(tmp_path, capsys):
    journal = tmp_path / "live-journal.jsonl"
    assert cli.delivery_exit(True, 2, journal) == cli.EXIT_INCOMPLETE_DELIVERY
    assert "resend" in capsys.readouterr().err
    assert cli.delivery_exit(False, 2, journal) == cli.EXIT_RECORDS_REFUSED
    assert "resending identical content cannot change that" in capsys.readouterr().err
    assert cli.delivery_exit(False, 0, journal) is None


def test_local_setup_failure_message_does_not_blame_the_api(tmp_path, capsys, monkeypatch):
    from agentwatch_agent.workspace import Workspace

    monkeypatch.setattr(Workspace, "create", staticmethod(lambda _task: 1 / 0))
    code = cli.main(["run", "--task", "fix-mean-empty", "--output-dir", str(tmp_path)])
    error = capsys.readouterr().err
    assert code == cli.EXIT_LIVE_UNAVAILABLE
    assert "local setup failed before contacting the live API" in error
    assert "make dev-worker" not in error


def test_overhead_command_exits_five_when_completion_is_refused(tmp_path, monkeypatch):
    from agent_test_doubles import FakeLiveServer

    server = FakeLiveServer()
    server.refuse_completion = True
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    original = cli.measure_overhead
    monkeypatch.setattr(
        cli,
        "measure_overhead",
        lambda *args, **kwargs: original(
            *args, **kwargs, repetitions=1, warmup=0, proc=tmp_path / "no-procfs"
        ),
    )
    assert cli.main(["overhead", "--output-dir", str(tmp_path)]) == cli.EXIT_RECORDS_REFUSED


def overhead_with(server, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "LiveClient", lambda *_args: server)
    original = cli.measure_overhead
    monkeypatch.setattr(
        cli,
        "measure_overhead",
        lambda *args, **kwargs: original(
            *args, **kwargs, repetitions=1, warmup=0, proc=tmp_path / "no-procfs"
        ),
    )
    return lambda *extra: cli.main(["overhead", "--output-dir", str(tmp_path), *extra])


def test_overhead_exit_codes_follow_the_documented_contract(tmp_path, monkeypatch):
    from agent_test_doubles import FakeLiveServer

    server = FakeLiveServer()
    overhead = overhead_with(server, tmp_path, monkeypatch)
    assert overhead("--patch", "rewrite-everything") == cli.EXIT_USAGE
    assert server.runs == {}
    server.down = True
    assert overhead() == cli.EXIT_LIVE_UNAVAILABLE
    server.down = False
    assert overhead() == cli.EXIT_PASSED


def test_overhead_failure_reports_whether_its_closure_was_delivered(tmp_path, monkeypatch):
    from agent_test_doubles import FakeLiveServer
    from agentwatch_agent import overhead as overhead_module

    real = overhead_module.run_project_tests
    calls = {"count": 0}

    def crash_in_instrumented_repetition(*args, **kwargs):
        # Repetition 0 is uninstrumented; crashing in repetition 1 leaves step records queued.
        calls["count"] += 1
        if calls["count"] % 2 == 0:
            raise RuntimeError("workload crashed")
        return real(*args, **kwargs)

    monkeypatch.setattr(overhead_module, "run_project_tests", crash_in_instrumented_repetition)
    server = FakeLiveServer()
    overhead = overhead_with(server, tmp_path, monkeypatch)
    assert overhead() == cli.EXIT_FAILED  # failed, and its records and closure were delivered
    server.unavailable_calls = {"ingest"}
    assert overhead() == cli.EXIT_INCOMPLETE_DELIVERY  # records undelivered: closure deferred


def test_unwritable_output_directory_exits_four(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    code = cli.main(["run", "--task", "fix-mean-empty", "--output-dir", str(blocker / "x")])
    assert code == cli.EXIT_LIVE_UNAVAILABLE
    assert "cannot write run evidence" in capsys.readouterr().err


def test_overhead_failure_with_refused_records_exits_five(tmp_path, monkeypatch):
    from agent_test_doubles import FakeLiveServer, http_error
    from agentwatch_agent import overhead as overhead_module

    real = overhead_module.run_project_tests
    calls = {"count": 0}

    def crash_in_instrumented_repetition(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] % 2 == 0:
            raise RuntimeError("workload crashed")
        return real(*args, **kwargs)

    monkeypatch.setattr(overhead_module, "run_project_tests", crash_in_instrumented_repetition)
    server = FakeLiveServer()
    original_ingest = server.ingest

    def refuse_one_step(run_id, batch):
        # Content-based, like a real refusal: every record of this step is refused, so the
        # per-record resend cannot get them accepted either.
        if any(getattr(record, "step_id", None) == "overhead-1" for record in batch.records):
            raise http_error(422, "refused")
        return original_ingest(run_id, batch)

    server.ingest = refuse_one_step
    overhead = overhead_with(server, tmp_path, monkeypatch)
    assert overhead() == cli.EXIT_RECORDS_REFUSED


def test_sampling_overhead_reads_one_dcgm_identity_for_every_request(tmp_path, monkeypatch):
    captured = {}

    def fake_measure(task, model, chat, output_dir, *, service_probes=None):
        captured["probes"] = service_probes
        output_dir.mkdir(parents=True)  # as the real measurement does
        return {"status": "failed", "error": "not measured in this test"}

    monkeypatch.setattr(cli, "measure_sampling_overhead", fake_measure)
    argv = ["sampling-overhead", "--model-backend", "vllm", "--output-dir", str(tmp_path)]
    argv += ["--dcgm-url", "http://127.0.0.1:9400/metrics"]
    assert cli.main(argv) == cli.EXIT_FAILED
    server = cli.ResourceRegistration(
        resource_id="model-server-1", service_id="vllm", resource_class="inference"
    )
    first, second = captured["probes"](server), captured["probes"](server)
    assert [p.resource.service_id for p in first] == ["vllm", "nvidia-dcgm"]
    assert first[1].resource.resource_id == second[1].resource.resource_id


@pytest.mark.parametrize("value", ["vLLM", "olama", ""])
def test_a_mistyped_backend_in_the_environment_is_refused(monkeypatch, value):
    # argparse checks `choices` only on the command line; the default must be checked too.
    monkeypatch.setenv("AGENTWATCH_MODEL_BACKEND", value)
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["run", "--task", "x"])


def test_a_valid_backend_in_the_environment_is_used(monkeypatch):
    monkeypatch.setenv("AGENTWATCH_MODEL_BACKEND", "vllm")
    assert cli.parser().parse_args(["run", "--task", "x"]).model_backend == "vllm"


class PolicyLiveServer(FakeLiveServer):
    """The fake live API plus the O02 policy read the runner makes (test double)."""

    def __init__(self, policy):
        super().__init__()
        self.policy_source = FakePolicySource(policy)
        self.base_url = "http://fake-live"

    def get_policy(self, target="coding-agent"):
        return self.policy_source.get_policy(target)


def fake_cli_services(monkeypatch, policy, scripts):
    server = PolicyLiveServer(policy)
    monkeypatch.setattr(cli, "LiveClient", lambda *_args, **_kwargs: server)
    models = iter([ScriptedModel(script) for script in scripts])
    monkeypatch.setattr(cli, "build_model", lambda _args: next(models))
    return server


def test_an_unreadable_policy_exits_four_and_names_the_way_out(tmp_path, capsys, monkeypatch):
    server = fake_cli_services(monkeypatch, policy_version(1), [SOLVE_MEAN])
    server.policy_source.failures = [URLError("policy service down")] * 4
    code = cli.main(["run", "--task", "fix-mean-empty", "--output-dir", str(tmp_path)])
    error = capsys.readouterr().err
    assert code == cli.EXIT_LIVE_UNAVAILABLE
    assert "concurrency policy" in error and "--no-policy" in error
    assert server.runs == {}


def test_no_policy_runs_without_reading_the_policy(tmp_path, monkeypatch):
    server = fake_cli_services(monkeypatch, policy_version(1), [SOLVE_MEAN])
    code = cli.main(
        ["run", "--task", "fix-mean-empty", "--output-dir", str(tmp_path), "--no-policy"]
    )
    assert code == cli.EXIT_PASSED
    assert server.policy_source.reads == 0
    (evidence_path,) = tmp_path.glob("*/agent-evidence.json")
    assert json.loads(evidence_path.read_text())["admission"]["status"] == "not_configured"


def test_a_signal_cancels_a_running_batch_while_the_main_thread_waits(
    tmp_path, capsys, monkeypatch
):
    """A real SIGINT reaches the batch's handler during its join, so Ctrl-C is prompt."""
    release = threading.Event()
    blocking = [ScriptedModel(SOLVE_MEAN, block=release, block_on=1) for _ in range(2)]
    server = PolicyLiveServer(policy_version(1, inference=2, tests=1))
    monkeypatch.setattr(cli, "LiveClient", lambda *_args, **_kwargs: server)
    models = iter(blocking)
    monkeypatch.setattr(cli, "build_model", lambda _args: next(models))

    def interrupt():
        assert all(model.blocked.wait(10) for model in blocking)
        os.kill(os.getpid(), signal.SIGINT)
        # The abandoned requests end afterwards, so their permits come back before the drain.
        threading.Timer(0.2, release.set).start()

    threading.Thread(target=interrupt, daemon=True).start()
    started = time.monotonic()
    code = cli.main(
        ["batch", "--task", "fix-mean-empty", "--repeat", "2", "--output-dir", str(tmp_path)]
    )
    assert time.monotonic() - started < 20
    assert code == cli.EXIT_FAILED
    (batch_file,) = tmp_path.glob("batch-*/batch.json")
    saved = json.loads(batch_file.read_text())
    assert [run["state"] for run in saved["runs"]] == ["cancelled"] * 2
    assert {run["reason"] for run in saved["runs"]} == {"Cancelled by operator signal SIGINT"}
    assert saved["permits_returned"] is True
    capsys.readouterr()


@pytest.mark.parametrize(
    "extra",
    [
        ["--test-max-queue", "-1"],
        ["--inference-max-wait", "0"],
        ["--policy-poll", "inf"],
    ],
)
def test_invalid_admission_arguments_are_usage_errors(extra):
    for command in ("run", "batch"):
        with pytest.raises(SystemExit) as exit_info:
            cli.main([command, "--task", "fix-mean-empty", *extra])
        assert exit_info.value.code == 2


@pytest.mark.parametrize("command", ["run", "batch"])
def test_admission_bounds_beyond_their_range_are_usage_errors(command, tmp_path, capsys):
    argv = [command, "--task", "fix-mean-empty", "--output-dir", str(tmp_path)]
    assert cli.main([*argv, "--inference-max-wait", "4000"]) == 2
    assert cli.main([*argv, "--test-max-queue", "2000"]) == 2
    assert "max_" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_an_oversized_batch_is_a_usage_error_that_creates_nothing(tmp_path, capsys):
    code = cli.main(
        ["batch", "--task", "fix-mean-empty", "--repeat", "33", "--output-dir", str(tmp_path)]
    )
    assert code == 2 and "at most 32 runs" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_a_batch_exits_with_its_most_severe_run(tmp_path, capsys, monkeypatch):
    wrong_patch = [
        chat_response(tool_calls=[("apply_patch", {"patch_id": "return-zero-for-empty"})]),
        chat_response("Applied the patch."),
    ]
    server = fake_cli_services(
        monkeypatch, policy_version(2, inference=1, tests=1), [SOLVE_MEAN, wrong_patch]
    )
    code = cli.main(
        [
            "batch",
            "--task",
            "fix-mean-empty",
            "--repeat",
            "2",
            "--output-dir",
            str(tmp_path),
            "--policy-poll",
            "0.05",
        ]
    )
    assert code == cli.EXIT_FAILED
    (batch_file,) = tmp_path.glob("batch-*/batch.json")
    saved = json.loads(batch_file.read_text())
    assert sorted(run["state"] for run in saved["runs"]) == ["completed", "failed"]
    assert saved["permits_returned"] is True
    assert saved["admission"]["policy_version"] == 2
    assert saved["policy_watcher"]["poll_failures"] == 0
    assert len(server.runs) == 2
    assert "Batch " in capsys.readouterr().out


def test_a_batch_with_an_unreadable_policy_exits_four(tmp_path, capsys, monkeypatch):
    server = fake_cli_services(monkeypatch, policy_version(1), [SOLVE_MEAN, SOLVE_MEAN])
    server.policy_source.failures = [URLError("down")] * 4
    code = cli.main(
        ["batch", "--task", "fix-mean-empty", "--repeat", "2", "--output-dir", str(tmp_path)]
    )
    assert code == cli.EXIT_LIVE_UNAVAILABLE
    assert "concurrency policy" in capsys.readouterr().err
    (batch_file,) = tmp_path.glob("batch-*/batch.json")
    assert {run["state"] for run in json.loads(batch_file.read_text())["runs"]} == {"not_started"}
    assert server.runs == {}


# capacity-sweep: the sweep driver, runners and batches are stubs; no model, GPU or server runs.


class StubSweep:
    """Test double for `CapacitySweep`: records its construction and returns a canned record."""

    record: ClassVar[dict] = {}
    error: ClassVar[Exception | None] = None
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cancels = []
        StubSweep.instances.append(self)

    def request_cancel_from_signal(self, source, reason):
        self.cancels.append((source, reason))

    def run(self):
        if self.error is not None:
            raise self.error
        return self.record


@pytest.fixture
def sweep_stubs(monkeypatch):
    """Two stub sources (one per task) and recording stand-ins for runners and batches."""
    StubSweep.instances = []
    StubSweep.error = None
    StubSweep.record = {"state": "completed", "ramp_end": "ceiling", "capacity_decision": None}
    tasks = {"a": "fix-mean-empty", "b": "fix-window-count"}
    monkeypatch.setattr(cli, "source_task", lambda directory: cli.load_task(tasks[directory.name]))
    monkeypatch.setattr(
        cli,
        "load_replay_plan",
        lambda directory, task: SimpleNamespace(task_id=task.task_id, source=directory.name),
    )
    monkeypatch.setattr(
        cli,
        "ReplayRunner",
        lambda **kwargs: SimpleNamespace(kind="replay", **({"admission": None} | kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "AgentRunner",
        lambda **kwargs: SimpleNamespace(kind="live", **({"admission": None} | kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "AgentBatch",
        lambda runners, controller, path: SimpleNamespace(
            runners=list(runners), controller=controller, path=path
        ),
    )
    monkeypatch.setattr(cli, "CapacitySweep", StubSweep)
    monkeypatch.setattr(cli, "host_record", lambda: {"gpu": {"status": "unavailable"}})
    monkeypatch.setattr(cli, "code_revision", lambda: {"status": "recorded", "commit": "c0ffee"})
    return StubSweep


def sweep_argv(tmp_path, *extra, configuration="cfg-default"):
    return [
        "capacity-sweep",
        "--configuration",
        configuration,
        "--source",
        str(tmp_path / "a"),
        "--source",
        str(tmp_path / "b"),
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def assert_refused_before_running(code, tmp_path, sweep_stubs, capsys, message):
    assert code == cli.EXIT_USAGE
    assert message in capsys.readouterr().err
    assert sweep_stubs.instances == []
    assert not (tmp_path / "out").exists()


def test_capacity_sweep_needs_two_sources_of_different_tasks(
    tmp_path, sweep_stubs, capsys, monkeypatch
):
    monkeypatch.setattr(cli, "source_task", lambda _directory: cli.load_task("fix-mean-empty"))
    code = cli.main(sweep_argv(tmp_path))
    assert_refused_before_running(code, tmp_path, sweep_stubs, capsys, "different tasks")
    one_source = [
        "capacity-sweep",
        "--configuration",
        "cfg-default",
        "--source",
        str(tmp_path / "a"),
    ]
    code = cli.main(one_source + ["--output-dir", str(tmp_path / "out")])
    assert_refused_before_running(code, tmp_path, sweep_stubs, capsys, "exactly 2 sources")


def test_capacity_sweep_refuses_an_unknown_configuration(tmp_path):
    argv = sweep_argv(tmp_path, configuration="cfg-huge")
    with pytest.raises(SystemExit) as refused:
        cli.main(argv)
    assert refused.value.code == cli.EXIT_USAGE


def test_capacity_sweep_refuses_an_unreplayable_source(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    code = cli.main(sweep_argv(tmp_path))
    assert code == cli.EXIT_USAGE
    assert "cannot replay" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_capacity_sweep_refuses_the_ollama_backend(tmp_path, sweep_stubs, capsys):
    code = cli.main(sweep_argv(tmp_path, "--model-backend", "ollama"))
    assert_refused_before_running(code, tmp_path, sweep_stubs, capsys, "vllm")


def test_capacity_sweep_refuses_a_model_other_than_the_declared_one(tmp_path, sweep_stubs, capsys):
    code = cli.main(sweep_argv(tmp_path, "--model", "other/model@abc"))
    assert_refused_before_running(code, tmp_path, sweep_stubs, capsys, "declared model")


def test_capacity_sweep_only_floods_a_loopback_server(tmp_path, sweep_stubs, capsys):
    code = cli.main(sweep_argv(tmp_path, "--vllm-url", "http://10.0.0.5:8000"))
    assert_refused_before_running(code, tmp_path, sweep_stubs, capsys, "loopback")


@pytest.mark.parametrize(
    ("configuration", "attempts"),
    [
        ("cfg-hardware", ["0.8:failed: CUDA out of memory"]),  # the last must have started
        ("cfg-hardware", ["0.5:started"]),  # not a declared utilization
        ("cfg-default", ["0.8:started"]),  # not this configuration's utilization
        ("cfg-hardware", ["0.8:started", "0.75:started"]),  # only the last one starts
        ("cfg-hardware", ["high:started"]),
        ("cfg-hardware", ["0.8"]),
        ("cfg-hardware", ["0.8:  "]),
    ],
)
def test_start_attempts_must_end_with_the_reported_utilization(
    tmp_path, sweep_stubs, capsys, configuration, attempts
):
    extra = [flag for attempt in attempts for flag in ("--start-attempt", attempt)]
    code = cli.main(sweep_argv(tmp_path, *extra, configuration=configuration))
    assert_refused_before_running(code, tmp_path, sweep_stubs, capsys, "start attempt")


def test_start_attempts_are_recorded_as_given(tmp_path, sweep_stubs):
    extra = ["--start-attempt", "0.8:failed: CUDA out of memory", "--start-attempt", "0.75:started"]
    assert cli.main(sweep_argv(tmp_path, *extra, configuration="cfg-hardware")) == 0
    (sweep,) = sweep_stubs.instances
    assert sweep.kwargs["start_attempts"] == [
        {"gpu_memory_utilization": 0.8, "outcome": "failed: CUDA out of memory"},
        {"gpu_memory_utilization": 0.75, "outcome": "started"},
    ]
    assert sweep.kwargs["configuration"].name == "cfg-hardware"


def test_no_start_attempts_are_recorded_when_none_are_given(tmp_path, sweep_stubs):
    assert cli.main(sweep_argv(tmp_path)) == 0
    assert sweep_stubs.instances[0].kwargs["start_attempts"] is None


@pytest.mark.parametrize(
    ("state", "code"),
    [
        ("completed", 0),
        ("refused_configuration", 2),
        ("stopped_resend_needed", 3),
        ("stopped_vllm_restart", 1),
        ("stopped_injector_cleanup_unconfirmed", 1),
        ("stopped_run_not_started", 1),
        ("stopped_warmup_failed", 1),
        ("stopped_server_busy_after_idle_wait", 1),
        ("stopped_server_state_unavailable", 1),
        ("cancelled", 1),
        ("failed", 1),
    ],
)
def test_capacity_sweep_exit_codes_follow_the_sweep_state(
    tmp_path, sweep_stubs, capsys, state, code
):
    sweep_stubs.record = {
        "state": state,
        "ramp_end": "hard_stop",
        "stop_reason": "why it stopped",
        "capacity_decision": {"capacity": 4, "result": "measured"},
        "label_validation": {"overall": {"verdict": "validated"}},
    }
    assert cli.main(sweep_argv(tmp_path)) == code
    output = capsys.readouterr()
    assert state in output.out
    assert "capacity 4 (measured)" in output.out
    assert "cfg-default-" in output.out  # the sweep folder is printed


def test_an_unexpected_sweep_error_propagates_and_restores_the_signal_handlers(
    tmp_path, sweep_stubs
):
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    sweep_stubs.error = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        cli.main(sweep_argv(tmp_path))
    assert {sig: signal.getsignal(sig) for sig in before} == before


def test_a_signal_cancels_the_sweep(tmp_path, sweep_stubs, monkeypatch):
    def run_and_interrupt(self):
        signal.raise_signal(signal.SIGTERM)
        return self.record | {"state": "cancelled"}

    monkeypatch.setattr(StubSweep, "run", run_and_interrupt)
    assert cli.main(sweep_argv(tmp_path)) == cli.EXIT_FAILED
    assert sweep_stubs.instances[0].cancels == [("signal", "Cancelled by operator signal SIGTERM")]


def test_the_sweep_is_built_under_a_new_folder_with_the_requested_settings(tmp_path, sweep_stubs):
    cli.main(sweep_argv(tmp_path, "--run-timeout", "300"))
    (sweep,) = sweep_stubs.instances
    folder = sweep.kwargs["output_dir"]
    assert folder.parent == tmp_path / "out"
    assert folder.name.startswith("cfg-default-")
    assert sweep.kwargs["run_timeout_seconds"] == 300
    assert [plan.source for plan in sweep.kwargs["sources"]] == ["a", "b"]
    assert sweep.kwargs["host"] == {"gpu": {"status": "unavailable"}}


def test_replay_batches_use_each_runs_source_one_clock_and_no_policy(tmp_path, sweep_stubs):
    cli.main(sweep_argv(tmp_path))
    kwargs = sweep_stubs.instances[0].kwargs
    clock = cli.MonotonicClock()
    batch = Batch("ramp", 3, 1, (1, 0, 1), False)
    built = kwargs["make_batch"](batch, tmp_path / "batch", clock)
    assert built.controller is None and built.path == tmp_path / "batch"
    assert [runner.task.name for runner in built.runners] == [
        "fix-window-count",
        "fix-mean-empty",
        "fix-window-count",
    ]
    assert [runner.plan.source for runner in built.runners] == ["b", "a", "b"]
    assert [runner.output_dir.name for runner in built.runners] == [
        "01-fix-window-count",
        "02-fix-mean-empty",
        "03-fix-window-count",
    ]
    assert all(runner.output_dir.parent == tmp_path / "batch" / "runs" for runner in built.runners)
    assert all(runner.clock is clock and runner.admission is None for runner in built.runners)
    # The protocol's declared sampling interval, passed explicitly.
    assert {runner.sampling_interval_seconds for runner in built.runners} == {
        SAMPLING_INTERVAL_SECONDS
    }
    assert kwargs["code_revision"] == {"status": "recorded", "commit": "c0ffee"}
    models = [runner.model for runner in built.runners]
    assert len({id(model) for model in models}) == 3  # one model client per run


def test_live_batches_alternate_the_two_tasks(tmp_path, sweep_stubs):
    cli.main(sweep_argv(tmp_path))
    built = sweep_stubs.instances[0].kwargs["make_live_batch"](
        3, tmp_path / "live", cli.MonotonicClock()
    )
    assert [(runner.kind, runner.task.name) for runner in built.runners] == [
        ("live", "fix-mean-empty"),
        ("live", "fix-window-count"),
        ("live", "fix-mean-empty"),
    ]
    assert all(runner.admission is None for runner in built.runners)


def test_the_injector_is_a_prefill_flood_on_the_served_model(tmp_path, sweep_stubs):
    cli.main(sweep_argv(tmp_path))
    kwargs = sweep_stubs.instances[0].kwargs
    (injector,) = kwargs["make_injectors"](cli.MonotonicClock(), 932.0)
    assert type(injector).__name__ == "PrefillFloodInjector"
    assert injector.url == cli.DEFAULT_VLLM_URL
    assert injector.model == cli.DEFAULT_VLLM_MODEL
    assert injector.concurrency == INJECTOR_CONCURRENCY
    assert injector.prompt_tokens == INJECTOR_PROMPT_TOKENS
    assert injector.max_duration_seconds == 932.0


def test_host_record_keeps_an_unavailable_gpu_unavailable(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise NvmlUnavailable("NVML library was not found on this host", unsupported=True)

    monkeypatch.setattr(cli.NvmlDevice, "open", unavailable)
    host = cli.host_record()
    assert host["gpu"]["status"] == "unavailable" and "NVML" in host["gpu"]["reason"]
    assert host["cpu_count"] is None or host["cpu_count"] >= 1
    assert host["kernel"]


def test_host_record_names_the_gpu_when_nvml_opens(monkeypatch):
    identity = GpuIdentity(0, 1, "Test GPU", "1.2.3", 8 * 1024**3)
    closed = []
    device = SimpleNamespace(identity=identity, close=lambda: closed.append(True))
    monkeypatch.setattr(cli.NvmlDevice, "open", lambda *_a, **_k: device)
    gpu = cli.host_record()["gpu"]
    assert gpu["status"] == "available"
    assert (gpu["name"], gpu["driver_version"], gpu["memory_total_bytes"]) == (
        "Test GPU",
        "1.2.3",
        8 * 1024**3,
    )
    assert closed == [True]


def test_sweep_analyse_writes_the_offline_analysis(tmp_path, monkeypatch, capsys):
    calls = []

    def analyse(directory):
        calls.append(directory)
        return {"capacity_decision": {"capacity": 2, "result": "incomplete"}}

    monkeypatch.setattr(cli, "analyse_sweep_folder", analyse)
    assert cli.main(["sweep-analyse", str(tmp_path)]) == cli.EXIT_PASSED
    assert calls == [tmp_path]
    output = capsys.readouterr().out
    assert "capacity 2 (incomplete)" in output and "analysis.json" in output


def test_sweep_analyse_refuses_a_folder_without_a_sweep_record(tmp_path, capsys):
    assert cli.main(["sweep-analyse", str(tmp_path)]) == cli.EXIT_USAGE
    assert "sweep.json" in capsys.readouterr().err
    assert not (tmp_path / "analysis.json").exists()


def test_a_completed_sweep_whose_analysis_failed_does_not_exit_0(tmp_path, sweep_stubs, capsys):
    sweep_stubs.record = {
        "state": "completed",
        "ramp_end": "ceiling",
        "capacity_decision": None,
        "analysis_errors": [{"stage": "label_validation", "error": "RuntimeError: x"}],
    }
    assert cli.main(sweep_argv(tmp_path)) == cli.EXIT_FAILED
    assert "sweep-analyse" in capsys.readouterr().err


def test_code_revision_records_the_commit_and_local_changes(monkeypatch):
    outputs = {"rev-parse": "abc123\n", "status": " M services/agent/x.py\n"}

    def run(command, **kwargs):
        assert kwargs["timeout"] > 0 and kwargs["check"] is True
        return SimpleNamespace(stdout=outputs[command[1]])

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.code_revision() == {"status": "recorded", "commit": "abc123", "dirty": True}


def test_code_revision_without_git_is_unavailable(monkeypatch):
    def run(command, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(cli.subprocess, "run", run)
    revision = cli.code_revision()
    assert revision["status"] == "unavailable" and "FileNotFoundError" in revision["reason"]
