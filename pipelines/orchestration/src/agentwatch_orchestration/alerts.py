"""Slack alerts for the dataset pipeline: task failures and findings that end a run cleanly.

Alerts go to a Slack incoming webhook. Its URL is a secret: it is read from the environment,
or from the checkout's git-ignored .env file, and never logged or committed.
Without it, alerts are not sent and the pipeline logs that they were not, so a missing
configuration is visible rather than silently treated as "nothing to report".

A run can succeed and still need a person's attention: drift against the previous passing
release, too many runs rejected by the build, a leakage check that did not pass, or a version
that was not pushed to a DVC remote. These are findings, reported with the run summary; a run
without findings still posts its summary, so a quiet channel never stands in for a clean run. Thresholds are declared here, before data is
collected; they must not be tuned to silence an alert.

This module does not import Airflow, so it is tested without it.
"""

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_ENV = "AGENTWATCH_ALERT_SLACK_WEBHOOK_URL"
MAX_REJECTED_SHARE_ENV = "AGENTWATCH_ALERT_MAX_REJECTED_SHARE"
ALERT_ENV_KEYS = (SLACK_WEBHOOK_ENV, MAX_REJECTED_SHARE_ENV)
ENV_FILE = Path(".env")  # relative to the repository root; git-ignored
# Declared 2026-10-25, before collection: more than 1 in 10 runs rejected means the evidence or
# the recorder needs a look, even though the build itself succeeded.
DEFAULT_MAX_REJECTED_SHARE = 0.10
SLACK_TIMEOUT_SECONDS = 10
ERROR_TAIL_CHARACTERS = 1500  # Slack messages stay readable; the full error is in the task log
LEAKAGE_PASSED = "passed"


class AlertError(RuntimeError):
    """Alerts are misconfigured, or Slack did not accept a message."""


@dataclass(frozen=True)
class AlertConfig:
    webhook_url: str | None
    max_rejected_share: float = DEFAULT_MAX_REJECTED_SHARE

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AlertConfig":
        webhook_url = env.get(SLACK_WEBHOOK_ENV) or None
        if webhook_url is not None:
            parts = urlsplit(webhook_url)
            if parts.scheme != "https" or not parts.netloc:
                # The URL itself is a secret, so the error names the variable, not its value.
                raise AlertError(f"{SLACK_WEBHOOK_ENV} must be an https:// Slack webhook URL")
        share_text = env.get(MAX_REJECTED_SHARE_ENV)
        max_rejected_share = DEFAULT_MAX_REJECTED_SHARE
        if share_text:
            try:
                max_rejected_share = float(share_text)
            except ValueError as error:
                raise AlertError(
                    f"{MAX_REJECTED_SHARE_ENV} must be a number from 0 to 1, got {share_text!r}"
                ) from error
            if not 0 <= max_rejected_share <= 1:
                raise AlertError(
                    f"{MAX_REJECTED_SHARE_ENV} must be from 0 to 1, got {max_rejected_share}"
                )
        return cls(webhook_url=webhook_url, max_rejected_share=max_rejected_share)


def read_env_file(path: Path, keys: Sequence[str]) -> dict[str, str]:
    """The given keys' values from a KEY=VALUE file such as .env; a missing file has none.

    Only the alert keys are read, so the rest of .env (database passwords, model settings)
    never reaches Airflow's environment through this. Values may be wrapped in one pair of
    single or double quotes; comments and blank lines are skipped.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise AlertError(f"{path} could not be read: {error}") from error
    values = {}
    for line in text.splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in keys:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[name] = value
    return values


def alert_environment(root: Path, env: Mapping[str, str]) -> dict[str, str]:
    """The alert settings from root/.env, overridden by any set in the environment itself."""
    values = read_env_file(root / ENV_FILE, ALERT_ENV_KEYS)
    values.update({key: env[key] for key in ALERT_ENV_KEYS if env.get(key)})
    return values


def findings(
    summary: Mapping[str, object], drift: Sequence[str], max_rejected_share: float
) -> list[str]:
    """What in a successful run needs a person's attention, one line per finding."""
    found = []
    runs = summary["runs"]
    rejected = summary["runs_rejected"]
    if not (type(runs) is int and type(rejected) is int):
        raise AlertError(f"run summary has invalid run counts: {runs!r}, {rejected!r}")
    considered = runs + rejected
    if considered and rejected / considered > max_rejected_share:
        found.append(
            f"The build rejected {rejected} of {considered} runs, more than the declared "
            f"{max_rejected_share:.0%}; reasons are in {summary['build_report']}"
        )
    if summary["leakage"] != LEAKAGE_PASSED:
        found.append(
            f"Leakage check did not pass ({summary['leakage']}); evaluation must not use "
            f"split {summary['split']}"
        )
    found.extend(f"Drift against the previous passing release: {line}" for line in drift)
    # A failed push fails its task (and alerts as a failure), so here it was skipped: DVC has
    # no default remote, and nobody else can pull this version.
    if summary["pushed"] is not True:
        found.append(
            "The version was not pushed: no DVC remote is configured, so it exists only in "
            "this checkout's DVC cache"
        )
    return found


def run_message(summary: Mapping[str, object], found: Sequence[str]) -> str:
    """The run summary, headed by its outcome, with one bullet per finding."""
    if found:
        heading = f":warning: AgentWatch dataset run finished with {len(found)} finding(s)"
    else:
        heading = ":white_check_mark: AgentWatch dataset run finished cleanly"
    lines = [
        heading,
        f"Release: {summary['release']}",
        (
            f"Rows: {summary['rows']}, runs: {summary['runs']}, "
            f"rejected: {summary['runs_rejected']}, new evidence: {summary['evidence_added']}"
        ),
        f"Split: {summary['split']} (leakage check: {summary['leakage']})",
        f"Versioned: {_yes_no(summary['versioned'])}, pushed: {_yes_no(summary['pushed'])}",
        *(f"• {line}" for line in found),
    ]
    return "\n".join(lines)


def _yes_no(value: object) -> str:
    return "yes" if value is True else "no"


def failure_message(
    dag_id: str, run_id: str, task_id: str, error: BaseException | str | None, log_url: str | None
) -> str:
    detail = str(error) if error is not None else "no error was recorded"
    if len(detail) > ERROR_TAIL_CHARACTERS:
        detail = "…" + detail[-ERROR_TAIL_CHARACTERS:]
    lines = [
        f":rotating_light: AgentWatch dataset task `{task_id}` failed",
        f"DAG: {dag_id}, run: {run_id}",
        f"```{detail}```",
    ]
    if log_url:
        lines.append(f"Log: {log_url}")
    return "\n".join(lines)


def send_slack(webhook_url: str, text: str, timeout_seconds: float = SLACK_TIMEOUT_SECONDS) -> None:
    """Post one message to a Slack incoming webhook; raises AlertError unless Slack accepts it."""
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            body = response.read(200).decode(errors="replace")
    except urllib.error.HTTPError as error:
        # Slack explains a rejected message in the body, for example invalid_token.
        reason = error.read(200).decode(errors="replace")
        raise AlertError(f"Slack rejected the alert with HTTP {error.code}: {reason}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise AlertError(f"Slack could not be reached: {error}") from error
    if status != 200:
        raise AlertError(f"Slack answered HTTP {status}: {body}")
