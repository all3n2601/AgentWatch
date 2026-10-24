import pytest
from agentwatch_agent import cli
from agentwatch_agent.config import RunConfiguration, RunLimits


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
