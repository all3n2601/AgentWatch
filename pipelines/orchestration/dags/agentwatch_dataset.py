"""The AgentWatch dataset pipeline as an Airflow DAG.

    acquire -> build -> validate -> split -> version -> push -> summarize -> alert

Each task runs one existing dataset command in its own environment through
agentwatch_orchestration.stages; the DAG owns ordering, retries, and task logs only. The DAG is
triggered by hand (or by a later collection schedule), with the recorded run directories to
add as the run_dirs parameter; with none, it rebuilds from the evidence already stored.

Only pushing to the DVC remote is retried: every other stage is deterministic, so a retry
would repeat the same failure, and the gate's anomalies must stop the run rather than be
retried away. One run at a time, because versioning snapshots the whole stores.

Alerts go to Slack (agentwatch_orchestration.alerts): every task's final failure, and, after
the summary, the run's summary with any findings of a run that otherwise succeeded (drift, too
many rejected runs, a leakage check that did not pass, a version that was not pushed). Posting an alert is retried; Slack is a network service.
"""

import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from agentwatch_orchestration import alerts, stages
from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowSkipException

logger = logging.getLogger(__name__)

PUSH_RETRIES = 3
PUSH_RETRY_DELAY = timedelta(seconds=30)
PUSH_MAX_RETRY_DELAY = timedelta(minutes=5)
ALERT_RETRIES = 3
ALERT_RETRY_DELAY = timedelta(seconds=30)


def config() -> stages.PipelineConfig:
    return stages.PipelineConfig.from_env(os.environ)


def alert_config() -> alerts.AlertConfig:
    """Alert settings from the environment, falling back to the checkout's .env file."""
    root = stages.PipelineConfig.from_env(os.environ).root
    return alerts.AlertConfig.from_env(alerts.alert_environment(root, os.environ))


def alert_on_failure(context) -> None:
    """Tell Slack that a task failed for good (after its retries), with its error and log link.

    A callback cannot fail the task any further, so an alert that cannot be sent is logged as
    an error in the task log instead.
    """
    ti = context["ti"]
    try:
        settings = alert_config()
        if settings.webhook_url is None:
            logger.warning(
                "Task %s failed; no Slack alert sent because %s is not set",
                ti.task_id,
                alerts.SLACK_WEBHOOK_ENV,
            )
            return
        message = alerts.failure_message(
            ti.dag_id, ti.run_id, ti.task_id, context.get("exception"), ti.log_url
        )
        alerts.send_slack(settings.webhook_url, message)
    except (alerts.AlertError, stages.StageError) as error:
        logger.error("Slack alert for failed task %s was not sent: %s", ti.task_id, error)


@dag(
    dag_id="agentwatch_dataset",
    description="Curate evidence, build, gate, split, and version an AgentWatch dataset release",
    schedule=None,
    start_date=datetime(2026, 10, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0, "on_failure_callback": alert_on_failure},
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
    def validate(built: dict[str, object]) -> dict[str, object]:
        release = str(built["release"])
        drift = stages.validate(config(), release)
        for line in drift:
            logger.warning("Drift against the previous passing release: %s", line)
        return {"release": release, "drift": drift}

    @task
    def split(validated: dict[str, object]) -> dict[str, object]:
        outcome = stages.split(config(), str(validated["release"]))
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
        validated: dict[str, object],
        split_outcome: dict[str, object],
        versioned: bool,
    ) -> dict[str, object]:
        from airflow.sdk import get_current_context

        pushed = get_current_context()["ti"].xcom_pull(task_ids="push") is True
        summary = stages.RunSummary(
            release=str(validated["release"]),
            build_report=str(built["report"]),
            rows=int(built["rows"]),
            runs=int(built["runs"]),
            runs_rejected=int(built["runs_rejected"]),
            evidence_added=added,
            split=str(split_outcome["path"]),
            leakage=str(split_outcome["leakage"]),
            drift=[str(line) for line in validated["drift"]],
            versioned=versioned,
            pushed=pushed,
        )
        logger.info("Run summary: %s", summary.as_dict())
        return summary.as_dict()

    @task(retries=ALERT_RETRIES, retry_delay=ALERT_RETRY_DELAY)
    def alert(summary: dict[str, object]) -> list[str]:
        settings = alert_config()
        drift = summary["drift"]
        if not isinstance(drift, list):
            raise alerts.AlertError(f"run summary drift must be a list, got {drift!r}")
        found = alerts.findings(summary, [str(line) for line in drift], settings.max_rejected_share)
        for line in found:
            logger.warning("Finding: %s", line)
        if settings.webhook_url is None:
            if not found:
                logger.info(
                    "No findings; summary not sent because %s is not set", alerts.SLACK_WEBHOOK_ENV
                )
                return found
            raise AirflowSkipException(
                f"{len(found)} finding(s) not sent to Slack because {alerts.SLACK_WEBHOOK_ENV} "
                "is not set; they are listed in this task's log"
            )
        alerts.send_slack(settings.webhook_url, alerts.run_message(summary, found))
        return found

    added = acquire()
    built = build(added)
    validated = validate(built)
    split_outcome = split(validated)
    versioned = version(split_outcome)
    pushed = push(versioned)
    summary = summarize(added, built, validated, split_outcome, versioned)
    pushed >> summary
    alert(summary)


agentwatch_dataset()
