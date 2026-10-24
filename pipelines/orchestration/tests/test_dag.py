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
    order = ["acquire", "build", "validate", "split", "version", "push", "summarize"]
    for upstream, downstream in pairwise(order):
        assert downstream in dag.get_task(upstream).downstream_task_ids


def test_only_push_is_retried(dag):
    retries = {task.task_id: task.retries for task in dag.tasks}
    assert retries.pop("push") > 0
    assert set(retries.values()) == {0}


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
    monkeypatch.setattr(stages, "validate", record("validate"))
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


def test_a_run_passes_the_built_release_and_summarizes_after_a_skipped_push(
    dag, metadata_database, stubbed_stages
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
        "versioned": True,
        "pushed": False,
    }
