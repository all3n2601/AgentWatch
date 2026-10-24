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
