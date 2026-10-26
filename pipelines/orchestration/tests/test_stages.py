import json
import subprocess
from pathlib import Path

import pytest
from agentwatch_orchestration import stages


class FakeRun:
    """Stands in for subprocess.run: records each command and answers from a script."""

    def __init__(self, responses: list[tuple[int, str, str]] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), **kwargs})
        code, stdout, stderr = self.responses.pop(0) if self.responses else (0, "", "")
        return subprocess.CompletedProcess(argv, code, stdout, stderr)

    def argvs(self) -> list[list[str]]:
        return [call["argv"] for call in self.calls]


@pytest.fixture
def config(tmp_path: Path) -> stages.PipelineConfig:
    (tmp_path / "Makefile").write_text("")
    return stages.PipelineConfig(root=tmp_path, timeout_seconds=5)


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(stages.subprocess, "run", fake)
    return fake


def test_acquire_without_run_dirs_runs_nothing(config, fake_run):
    assert stages.acquire(config, []) == 0
    assert fake_run.calls == []


def test_acquire_adds_every_run_dir_to_the_store_after_the_options(config, fake_run):
    stages.acquire(config, [".data/coding-demo", ".data/cpu-demo"])
    (argv,) = fake_run.argvs()
    assert argv[-6:] == [
        "agentwatch_data.evidence",
        "--store",
        str(config.root / "data/evidence"),
        "--",
        ".data/coding-demo",
        ".data/cpu-demo",
    ]


def test_acquire_counts_only_runs_newly_added(config, fake_run):
    fake_run.responses = [
        (0, "added: data/evidence/coding-a\nalready stored: data/evidence/coding-b\n", "")
    ]
    assert stages.acquire(config, [".data/a", ".data/b"]) == 1


@pytest.mark.parametrize("option", ["--store=/tmp/elsewhere", "-h"])
def test_acquire_refuses_run_dirs_that_look_like_options(config, fake_run, option):
    with pytest.raises(stages.StageError, match="look like options"):
        stages.acquire(config, [".data/coding-demo", option])
    assert fake_run.calls == []


def write_report(config, rows=357, runs=51, rejections=()):
    path = config.root / "data/datasets/builds/report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows, "runs": runs, "rejections": list(rejections)}))
    return path


def test_build_returns_the_release_and_report_counts(config, fake_run):
    report = write_report(config, rejections=[{"run": "r1", "reason": "bad"}])
    fake_run.responses = [(0, f"release: data/datasets/steps-v1.2+abc\nreport: {report}\n", "")]
    assert stages.build(config) == stages.BuildOutcome(
        release="data/datasets/steps-v1.2+abc",
        report=str(report),
        rows=357,
        runs=51,
        runs_rejected=1,
    )
    (argv,) = fake_run.argvs()
    assert argv[argv.index("--source") + 1] == f"local:{config.root / 'data/evidence'}"
    assert "api:" not in " ".join(argv)


def test_build_adds_the_api_source_when_configured(tmp_path, fake_run):
    (tmp_path / "Makefile").write_text("")
    config = stages.PipelineConfig(root=tmp_path, api_url="http://127.0.0.1:8090")
    fake_run.responses = [(0, f"release: r\nreport: {write_report(config)}\n", "")]
    stages.build(config)
    assert "api:http://127.0.0.1:8090" in fake_run.argvs()[0]


def test_build_reads_a_relative_report_path_from_the_root(config, fake_run):
    write_report(config, rows=7)
    fake_run.responses = [(0, "release: r\nreport: data/datasets/builds/report.json\n", "")]
    assert stages.build(config).rows == 7


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "unreadable"),
        ("{not json", "unreadable"),
        ('{"rows": 1, "runs": 1}', "unreadable"),
        ('{"rows": -1, "runs": 1, "rejections": []}', "invalid counts"),
        ('{"rows": true, "runs": 1, "rejections": []}', "invalid counts"),
    ],
)
def test_build_with_a_bad_report_fails(config, fake_run, content, message):
    report = config.root / "report.json"
    if content is not None:
        report.write_text(content)
    fake_run.responses = [(0, f"release: r\nreport: {report}\n", "")]
    with pytest.raises(stages.StageError, match=message):
        stages.build(config)


def test_build_without_a_release_line_fails(config, fake_run):
    fake_run.responses = [(0, "report: r.json\n", "")]
    with pytest.raises(stages.StageError, match="did not report a 'release:' line"):
        stages.build(config)


def test_a_failed_command_raises_with_its_exit_code_and_stderr_tail(config, fake_run):
    stderr = "\n".join(f"line {number}" for number in range(30))
    fake_run.responses = [(1, "", stderr)]
    with pytest.raises(stages.StageError) as raised:
        stages.validate(config, "data/datasets/r")
    message = str(raised.value)
    assert "validate failed with exit code 1" in message
    assert "line 29" in message
    assert "line 9\n" not in message  # only the tail is kept


def gate_result(config, drift: list[dict[str, str]]) -> str:
    """Write a gate result with the given drift; returns the gate's printout naming it."""
    path = config.root / "data/datasets/validations/r/attempt/result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "passed", "drift": drift}))
    relative = path.relative_to(config.root)
    return f"passed: r ({relative})\nresult: {relative}\n"


def test_validate_checks_the_given_release_not_the_newest(config, fake_run):
    fake_run.responses = [(0, gate_result(config, []), "")]
    stages.validate(config, "data/datasets/mine")
    (argv,) = fake_run.argvs()
    assert argv[argv.index("--release") + 1] == "data/datasets/mine"
    assert argv[argv.index("--project") + 1] == str(config.root / "pipelines/validation")


def test_validate_returns_the_drift_recorded_in_the_gates_result(config, fake_run):
    drift = [
        {"feature": "condition", "description": "L-infinity distance 0.4 exceeds 0.25"},
        {"feature": "kind", "description": "L-infinity distance 0.3 exceeds 0.25"},
    ]
    # The printout's own drift lines are ignored; only the durable result is read.
    printout = gate_result(config, drift) + "  drift other: printed only\n"
    fake_run.responses = [(0, printout, "")]
    assert stages.validate(config, "data/datasets/r") == [
        "condition: L-infinity distance 0.4 exceeds 0.25",
        "kind: L-infinity distance 0.3 exceeds 0.25",
    ]


def test_validate_without_drift_returns_no_findings(config, fake_run):
    fake_run.responses = [(0, gate_result(config, []), "")]
    assert stages.validate(config, "data/datasets/r") == []


def test_slices_reports_the_release_with_its_split_outside_the_stores(config, fake_run):
    report = config.root / ".data/reports/steps-v1.2+abc/slices.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"flags": [{"kind": "absent_from_split"}, {"kind": "x"}]}))
    printed = (
        f"slices: {report}\n"
        "summary: .data/reports/steps-v1.2+abc/slices.md\n"
        "  flag absent_from_split: kind=cpu: no rows in the test split\n"
    )
    fake_run.responses = [(0, printed, "")]
    outcome = stages.slices(config, "data/datasets/steps-v1.2+abc", "s.json")
    # The count comes from the report itself, not from the flag lines printed.
    assert outcome == stages.SliceOutcome(report=str(report), flags=2)
    (argv,) = fake_run.argvs()
    assert argv[argv.index("--release") + 1] == "data/datasets/steps-v1.2+abc"
    assert argv[argv.index("--split") + 1] == "s.json"
    assert argv[argv.index("--output") + 1] == str(config.root / ".data/reports/steps-v1.2+abc")


def test_slices_without_a_report_line_fails(config, fake_run):
    fake_run.responses = [(0, "summary: s.md\n", "")]
    with pytest.raises(stages.StageError, match="did not report a 'slices:' line"):
        stages.slices(config, "data/datasets/r", "s.json")


@pytest.mark.parametrize("content", ["not json", "{}", '{"flags": 3}'])
def test_slices_with_an_unreadable_report_fails(config, fake_run, content):
    (config.root / "slices.json").write_text(content)
    fake_run.responses = [(0, "slices: slices.json\n", "")]
    with pytest.raises(
        stages.StageError, match="slice report .* (is unreadable|has invalid flags)"
    ):
        stages.slices(config, "data/datasets/r", "s.json")


def test_validate_without_a_result_line_fails_rather_than_reporting_no_drift(config, fake_run):
    fake_run.responses = [(0, "passed: r (result.json)\n", "")]
    with pytest.raises(stages.StageError, match="did not report a 'result:' line"):
        stages.validate(config, "data/datasets/r")


@pytest.mark.parametrize("content", ["not json", "{}", '{"drift": [{"feature": "kind"}]}'])
def test_validate_with_an_unreadable_result_fails(config, fake_run, content):
    path = config.root / "result.json"
    path.write_text(content)
    fake_run.responses = [(0, "passed: r (result.json)\nresult: result.json\n", "")]
    with pytest.raises(stages.StageError, match="gate result .* is unreadable"):
        stages.validate(config, "data/datasets/r")


def test_a_timed_out_command_raises(config, monkeypatch):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(stages.subprocess, "run", timeout)
    with pytest.raises(stages.StageError, match="build timed out after 5 s"):
        stages.build(config)


def test_a_missing_tool_raises(config, monkeypatch):
    def missing(argv, **kwargs):
        raise FileNotFoundError(2, "No such file", argv[0])

    monkeypatch.setattr(stages.subprocess, "run", missing)
    with pytest.raises(stages.StageError, match="could not start uv"):
        stages.build(config)


def test_commands_run_from_the_root_without_airflows_virtual_environment(
    config, fake_run, monkeypatch
):
    monkeypatch.setenv("VIRTUAL_ENV", "/airflow/.venv")
    fake_run.responses = [(0, gate_result(config, []), "")]
    stages.validate(config, "r")
    (call,) = fake_run.calls
    assert call["cwd"] == config.root
    assert call["timeout"] == 5
    assert "VIRTUAL_ENV" not in call["env"]


@pytest.mark.parametrize(
    ("printed", "path", "reused"),
    [
        ("split: s/r/abc.json", "s/r/abc.json", False),
        ("split: s/r/abc.json (reused)", "s/r/abc.json", True),
    ],
)
def test_split_reports_its_file_and_leakage(config, fake_run, printed, path, reused):
    fake_run.responses = [
        (0, f"{printed}\n  train: 0 rows\nleakage: blocked (no training rows)\n", "")
    ]
    outcome = stages.split(config, "data/datasets/r")
    assert outcome == stages.SplitOutcome(
        path=path, reused=reused, leakage="blocked (no training rows)"
    )


def test_version_checks_integrity_before_snapshotting(config, fake_run):
    fake_run.responses = [(0, '{"unchanged": []}', ""), (0, "", ""), (0, "", "")]
    stages.version(config)
    status, integrity, add = fake_run.argvs()
    assert status[-5:] == ["dvc", "data", "status", "--granular", "--unchanged", "--json"][-5:]
    status_file = config.root / ".dvc/tmp/version-dataset-status.json"
    assert status_file.read_text() == '{"unchanged": []}'
    assert integrity[integrity.index("--dvc-status") + 1] == str(status_file)
    assert add[-3:] == [
        "add",
        str(config.root / "data/evidence"),
        str(config.root / "data/datasets"),
    ]


def test_version_snapshots_nothing_when_integrity_fails(config, fake_run):
    fake_run.responses = [(0, "{}", ""), (1, "", "release r does not match its manifest")]
    with pytest.raises(stages.StageError, match="does not match its manifest"):
        stages.version(config)
    assert len(fake_run.calls) == 2


def test_a_default_remote_is_configured(config, fake_run):
    fake_run.responses = [(0, "storage\n", "")]
    assert stages.remote_configured(config) is True


def test_no_default_remote_is_unconfigured(config, fake_run):
    fake_run.responses = [(1, "", "No default remote set\n")]
    assert stages.remote_configured(config) is False


@pytest.mark.parametrize(
    ("code", "stderr"),
    [
        (1, "ERROR: configuration error - config file error"),
        (251, "ERROR: you are not inside of a DVC repository"),
        (2, "error: Failed to spawn: `dvc`"),
    ],
)
def test_other_remote_failures_fail_the_push(config, fake_run, code, stderr):
    fake_run.responses = [(code, "", stderr)]
    with pytest.raises(stages.StageError, match=f"exit code {code}"):
        stages.remote_configured(config)


def test_config_from_env_reads_root_api_and_timeout(config):
    env = {
        "AGENTWATCH_ROOT": str(config.root),
        "AGENTWATCH_DATASET_API_URL": "http://api:8090",
        "AGENTWATCH_STAGE_TIMEOUT_SECONDS": "90",
    }
    loaded = stages.PipelineConfig.from_env(env)
    assert (loaded.root, loaded.api_url, loaded.timeout_seconds) == (
        config.root,
        "http://api:8090",
        90,
    )


def test_config_defaults_to_this_checkout():
    loaded = stages.PipelineConfig.from_env({})
    assert (loaded.root / "pipelines/orchestration").is_dir()
    assert loaded.api_url is None
    assert loaded.timeout_seconds == stages.DEFAULT_STAGE_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    ("timeout", "message"), [("soon", "whole number of seconds"), ("0", "must be positive")]
)
def test_config_rejects_an_invalid_timeout(config, timeout, message):
    env = {"AGENTWATCH_ROOT": str(config.root), "AGENTWATCH_STAGE_TIMEOUT_SECONDS": timeout}
    with pytest.raises(stages.StageError, match=message):
        stages.PipelineConfig.from_env(env)


def test_config_rejects_a_root_that_is_not_a_checkout(tmp_path):
    with pytest.raises(stages.StageError, match="not an AgentWatch checkout"):
        stages.PipelineConfig.from_env({"AGENTWATCH_ROOT": str(tmp_path)})


def test_absolute_store_paths_are_kept(config):
    assert config.path(Path("/srv/evidence")) == Path("/srv/evidence")
    assert config.path(Path("data/evidence")) == config.root / "data/evidence"
