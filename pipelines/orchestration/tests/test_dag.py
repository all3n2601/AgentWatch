from itertools import pairwise
from pathlib import Path

import pytest
from airflow.dag_processing.dagbag import DagBag

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    return bag.dags["agentwatch_dataset"]  # no metadata database needed


def test_tasks_run_in_pipeline_order(dag):
    order = ["acquire", "build", "validate", "split", "version", "push", "summarize", "alert"]
    for upstream, downstream in pairwise(order):
        assert downstream in dag.get_task(upstream).downstream_task_ids


def test_only_push_and_the_slack_alert_are_retried(dag):
    retries = {task.task_id: task.retries for task in dag.tasks}
    assert retries.pop("push") > 0
    assert retries.pop("alert") > 0
    assert set(retries.values()) == {0}


def test_every_task_alerts_slack_when_it_fails(dag):
    for task in dag.tasks:
        callbacks = [callback.__name__ for callback in task.on_failure_callback]
        assert callbacks == ["alert_on_failure"], task.task_id


def test_findings_are_alerted_after_the_summary(dag):
    assert dag.get_task("summarize").downstream_task_ids == {"alert"}


def test_one_run_at_a_time_and_triggered_by_hand(dag):
    assert dag.max_active_runs == 1
    assert dag.schedule is None
    assert dag.catchup is False


def test_summary_still_runs_when_push_is_skipped(dag):
    assert dag.get_task("summarize").trigger_rule == "none_failed"


@pytest.fixture
def stubbed_stages(monkeypatch, tmp_path):
    """Replace every stage command with a recorder, so a DAG run exercises only the wiring."""
    from agentwatch_orchestration import stages

    (tmp_path / "Makefile").write_text("")
    monkeypatch.setenv("AGENTWATCH_ROOT", str(tmp_path))
    calls: list[tuple[str, object]] = []
    built = stages.BuildOutcome(
        release="data/datasets/steps-v1.2+abc", report="b.json", rows=7, runs=2, runs_rejected=1
    )

    def record(name, result=None):
        def stage(_config, *args):
            calls.append((name, args))
            return result

        return stage

    monkeypatch.setattr(stages, "acquire", record("acquire", 1))
    monkeypatch.setattr(stages, "build", record("build", built))
    monkeypatch.setattr(stages, "validate", record("validate", ["kind: distance 0.3 exceeds 0.25"]))
    split = stages.SplitOutcome(path="s.json", reused=False, leakage="passed")
    monkeypatch.setattr(stages, "split", record("split", split))
    monkeypatch.setattr(stages, "version", record("version"))
    monkeypatch.setattr(stages, "remote_configured", record("remote_configured", False))
    monkeypatch.setattr(stages, "push", record("push"))
    return calls


@pytest.fixture(scope="module")
def metadata_database():
    from airflow.utils.db import resetdb

    resetdb()  # the temporary AIRFLOW_HOME from conftest, never a contributor's database


@pytest.fixture
def slack(monkeypatch):
    """Record Slack messages instead of sending them; tests never reach the network."""
    from agentwatch_orchestration import alerts

    monkeypatch.setenv(alerts.SLACK_WEBHOOK_ENV, "https://hooks.slack.invalid/T000/B000/x")
    sent: list[str] = []
    monkeypatch.setattr(alerts, "send_slack", lambda _url, text: sent.append(text))
    return sent


def test_a_run_passes_the_built_release_and_summarizes_after_a_skipped_push(
    dag, metadata_database, stubbed_stages, slack
):
    run = dag.test(run_conf={"run_dirs": [".data/coding-demo"]})
    assert run.state == "success"
    states = {ti.task_id: ti.state for ti in run.get_task_instances()}
    assert states == {
        "acquire": "success",
        "build": "success",
        "validate": "success",
        "split": "success",
        "version": "success",
        "push": "skipped",
        "summarize": "success",
        "alert": "success",
    }
    assert ("acquire", ([".data/coding-demo"],)) in stubbed_stages
    assert ("validate", ("data/datasets/steps-v1.2+abc",)) in stubbed_stages
    assert ("split", ("data/datasets/steps-v1.2+abc",)) in stubbed_stages
    assert all(name != "push" for name, _ in stubbed_stages)
    summary = run.get_task_instance("summarize").xcom_pull(task_ids="summarize")
    assert summary == {
        "release": "data/datasets/steps-v1.2+abc",
        "build_report": "b.json",
        "rows": 7,
        "runs": 2,
        "runs_rejected": 1,
        "evidence_added": 1,
        "split": "s.json",
        "leakage": "passed",
        "drift": ["kind: distance 0.3 exceeds 0.25"],
        "versioned": True,
        "pushed": False,
    }
    (message,) = slack
    assert "Release: data/datasets/steps-v1.2+abc" in message
    assert "rejected 1 of 3 runs" in message
    assert "Drift against the previous passing release: kind: distance 0.3" in message


def test_findings_are_skipped_not_sent_without_a_webhook(
    dag, metadata_database, stubbed_stages, monkeypatch
):
    from agentwatch_orchestration import alerts

    monkeypatch.delenv(alerts.SLACK_WEBHOOK_ENV, raising=False)
    run = dag.test(run_conf={"run_dirs": []})
    assert run.get_task_instance("alert").state == "skipped"


def test_a_failed_task_alerts_slack_and_stops_the_run(
    dag, metadata_database, stubbed_stages, slack, monkeypatch
):
    from agentwatch_orchestration import stages

    def gate_fails(_config, _release):
        raise stages.StageError("validate failed with exit code 1:\n  anomaly rows: missing")

    monkeypatch.setattr(stages, "validate", gate_fails)
    run = dag.test(run_conf={"run_dirs": []})
    assert run.state == "failed"
    assert run.get_task_instance("validate").state == "failed"
    assert run.get_task_instance("split").state == "upstream_failed"
    (message,) = slack
    assert "task `validate` failed" in message
    assert "anomaly rows: missing" in message


def test_the_webhook_can_come_from_the_checkouts_env_file(
    dag, metadata_database, stubbed_stages, monkeypatch, tmp_path
):
    from agentwatch_orchestration import alerts

    monkeypatch.delenv(alerts.SLACK_WEBHOOK_ENV, raising=False)
    # stubbed_stages makes tmp_path the checkout (AGENTWATCH_ROOT).
    (tmp_path / ".env").write_text(f"{alerts.SLACK_WEBHOOK_ENV}=https://hooks.slack.invalid/x\n")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(alerts, "send_slack", lambda url, text: sent.append((url, text)))
    run = dag.test(run_conf={"run_dirs": []})
    assert run.get_task_instance("alert").state == "success"
    ((url, _text),) = sent
    assert url == "https://hooks.slack.invalid/x"
