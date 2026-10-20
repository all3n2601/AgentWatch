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
