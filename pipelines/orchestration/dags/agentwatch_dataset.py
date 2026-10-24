"""The AgentWatch dataset pipeline as an Airflow DAG.

    acquire -> build -> validate -> split -> version -> push -> summarize

Each task runs one existing dataset command in its own environment through
agentwatch_orchestration.stages; the DAG owns ordering, retries, and task logs only. The DAG is
triggered by hand (or by a later collection schedule), with the recorded run directories to
add as the run_dirs parameter; with none, it rebuilds from the evidence already stored.

Only pushing to the DVC remote is retried: every other stage is deterministic, so a retry
would repeat the same failure, and the gate's anomalies must stop the run rather than be
retried away. One run at a time, because versioning snapshots the whole stores.
"""

import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from agentwatch_orchestration import stages
from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowSkipException

logger = logging.getLogger(__name__)

PUSH_RETRIES = 3
PUSH_RETRY_DELAY = timedelta(seconds=30)
PUSH_MAX_RETRY_DELAY = timedelta(minutes=5)


def config() -> stages.PipelineConfig:
    return stages.PipelineConfig.from_env(os.environ)


@dag(
    dag_id="agentwatch_dataset",
    description="Curate evidence, build, gate, split, and version an AgentWatch dataset release",
    schedule=None,
    start_date=datetime(2026, 10, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    params={
        "run_dirs": Param(
            [],
            type="array",
            items={"type": "string"},
            description="Recorded run directories to add to the evidence store, relative to the repository root",
        ),
    },
    tags=["agentwatch", "data"],
)
def agentwatch_dataset():
    @task
    def acquire(params: dict[str, object] | None = None) -> int:
        run_dirs = (params or {}).get("run_dirs") or []
        if not isinstance(run_dirs, list) or not all(isinstance(item, str) for item in run_dirs):
            raise stages.StageError("run_dirs must be a list of directory paths")
        added = stages.acquire(config(), run_dirs)
        logger.info("Added %d new runs to the evidence store", added)
        return added

    @task
    def build(_acquired: int) -> dict[str, object]:
        outcome = stages.build(config())
        if outcome.runs_rejected:
            logger.warning(
                "Build rejected %d runs; reasons are in %s", outcome.runs_rejected, outcome.report
            )
        return asdict(outcome)

    @task
    def validate(built: dict[str, object]) -> str:
        release = str(built["release"])
        stages.validate(config(), release)
        return release

    @task
    def split(release: str) -> dict[str, object]:
        outcome = stages.split(config(), release)
        if outcome.leakage != "passed":
            logger.warning("Leakage check %s: evaluation must not use this split", outcome.leakage)
        return {"path": outcome.path, "reused": outcome.reused, "leakage": outcome.leakage}

    @task
    def version(_split: dict[str, object]) -> bool:
        stages.version(config())
        return True

    @task(
        retries=PUSH_RETRIES,
        retry_delay=PUSH_RETRY_DELAY,
        retry_exponential_backoff=True,
        max_retry_delay=PUSH_MAX_RETRY_DELAY,
    )
    def push(_versioned: bool) -> bool:
        pipeline = config()
        if not stages.remote_configured(pipeline):
            raise AirflowSkipException("No DVC remote is configured; the version stays local")
        stages.push(pipeline)
        return True

    @task(trigger_rule="none_failed")
    def summarize(
        added: int,
        built: dict[str, object],
        release: str,
        split_outcome: dict[str, object],
        versioned: bool,
    ) -> dict[str, object]:
        from airflow.sdk import get_current_context

        pushed = get_current_context()["ti"].xcom_pull(task_ids="push") is True
        summary = stages.RunSummary(
            release=release,
            build_report=str(built["report"]),
            rows=int(built["rows"]),
            runs=int(built["runs"]),
            runs_rejected=int(built["runs_rejected"]),
            evidence_added=added,
            split=str(split_outcome["path"]),
            leakage=str(split_outcome["leakage"]),
            versioned=versioned,
            pushed=pushed,
        )
        logger.info("Run summary: %s", summary.as_dict())
        return summary.as_dict()

    added = acquire()
    built = build(added)
    validated = validate(built)
    split_outcome = split(validated)
    versioned = version(split_outcome)
    pushed = push(versioned)
    summary = summarize(added, built, validated, split_outcome, versioned)
    pushed >> summary


agentwatch_dataset()
