import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest
from agentwatch_orchestration import alerts


def summary(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "release": "data/datasets/steps-v1.2+abc",
        "build_report": "data/datasets/reports/b.json",
        "runs": 9,
        "runs_rejected": 1,
        "rows": 63,
        "evidence_added": 2,
        "split": "s.json",
        "leakage": "passed",
        "versioned": True,
        "pushed": True,
    }
    return base | overrides


def test_alerts_are_off_without_a_webhook():
    config = alerts.AlertConfig.from_env({})
    assert config.webhook_url is None
    assert config.max_rejected_share == alerts.DEFAULT_MAX_REJECTED_SHARE


def test_a_webhook_that_is_not_https_is_refused_without_echoing_it():
    with pytest.raises(alerts.AlertConfigError) as raised:
        alerts.AlertConfig.from_env({alerts.SLACK_WEBHOOK_ENV: "http://hooks.example/secret"})
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize("share", ["lots", "-0.1", "1.5", "nan"])
def test_an_invalid_rejected_share_is_refused(share):
    with pytest.raises(alerts.AlertConfigError, match=alerts.MAX_REJECTED_SHARE_ENV):
        alerts.AlertConfig.from_env({alerts.MAX_REJECTED_SHARE_ENV: share})


def test_a_clean_run_has_no_findings():
    assert alerts.findings(summary(), [], max_rejected_share=0.10) == []


def test_rejections_at_the_declared_share_are_not_a_finding():
    # 1 of 10 runs considered is exactly 10%: only more than the declared share alerts.
    assert alerts.findings(summary(runs=9, runs_rejected=1), [], 0.10) == []


def test_rejections_over_the_declared_share_are_a_finding():
    (finding,) = alerts.findings(summary(runs=8, runs_rejected=2), [], 0.10)
    assert "rejected 2 of 10 runs" in finding
    assert "data/datasets/reports/b.json" in finding


def test_an_empty_build_is_not_a_rejection_finding():
    assert alerts.findings(summary(runs=0, runs_rejected=0), [], 0.0) == []


def test_invalid_run_counts_are_refused():
    with pytest.raises(alerts.AlertError, match="invalid run counts"):
        alerts.findings(summary(runs="9"), [], 0.10)


def test_a_leakage_check_that_did_not_pass_is_a_finding():
    (finding,) = alerts.findings(summary(leakage="blocked: test split is empty"), [], 0.10)
    assert "blocked: test split is empty" in finding
    assert "s.json" in finding


def test_each_drift_line_is_a_finding():
    found = alerts.findings(summary(), ["kind: 0.3 > 0.25", "condition: 0.4 > 0.25"], 0.10)
    assert found == [
        "Drift against the previous passing release: kind: 0.3 > 0.25",
        "Drift against the previous passing release: condition: 0.4 > 0.25",
    ]


def test_a_version_that_was_not_pushed_is_a_finding():
    (finding,) = alerts.findings(summary(pushed=False), [], 0.10)
    assert "not pushed" in finding


def test_the_run_message_lists_the_summary_and_every_finding():
    slices = {"report": ".data/reports/r/slices.json", "flags": 4}
    message = alerts.run_message(summary(pushed=False), ["first", "second"], slices)
    assert message.startswith(":warning:")
    assert "2 finding(s)" in message
    assert "Release: data/datasets/steps-v1.2+abc" in message
    assert "Rows: 63, runs: 9, rejected: 1, new evidence: 2" in message
    assert "Split: s.json (leakage check: passed)" in message
    assert "Versioned: yes, pushed: no" in message
    assert "Bias slices: 4 flag(s), report .data/reports/r/slices.json" in message
    assert "• first" in message and "• second" in message


def test_slice_flags_are_listed_but_do_not_make_a_run_a_finding():
    message = alerts.run_message(summary(), [], {"report": "slices.json", "flags": 17})
    assert message.startswith(":white_check_mark:")
    assert "Bias slices: 17 flag(s), report slices.json" in message


def test_a_run_message_without_a_slice_report_says_so():
    message = alerts.run_message(summary(), [], None)
    assert "Bias slices: no report; the slices task did not finish" in message


def test_a_clean_run_message_says_so_without_bullets():
    message = alerts.run_message(summary(), [])
    assert message.startswith(":white_check_mark:")
    assert "finished cleanly" in message
    assert "•" not in message


def test_a_failure_message_keeps_only_the_errors_tail_and_links_the_log():
    error = RuntimeError("start " + "x" * 5000 + " end")
    message = alerts.failure_message("dag", "run-1", "validate", error, "http://airflow/log")
    assert "task `validate` failed" in message
    assert "end" in message and "start" not in message
    assert message.endswith("Log: http://airflow/log")


def test_slack_control_characters_are_escaped_in_both_messages():
    error = RuntimeError('File "<module>", line 1 & more')
    failure = alerts.failure_message("dag", "run-1", "validate", error, None)
    assert "&lt;module&gt;" in failure and "&amp; more" in failure and "<module>" not in failure
    found = ["Drift against the previous passing release: kind: <b> & c"]
    message = alerts.run_message(summary(release="data/datasets/<r>"), found)
    assert "Release: data/datasets/&lt;r&gt;" in message
    assert "kind: &lt;b&gt; &amp; c" in message


def test_a_failure_without_a_recorded_error_says_so():
    message = alerts.failure_message("dag", "run-1", "build", None, None)
    assert "no error was recorded" in message
    assert "Log:" not in message


class Webhook(BaseHTTPRequestHandler):
    """A local stand-in for Slack's incoming webhook; it answers with the class's status."""

    status: ClassVar[int] = 200
    received: ClassVar[list[dict[str, object]]] = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        type(self).received.append(json.loads(self.rfile.read(length)))
        body = b"ok" if self.status == 200 else b"invalid_token"
        self.send_response(self.status)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def webhook():
    Webhook.status = 200
    Webhook.received = []
    server = HTTPServer(("127.0.0.1", 0), Webhook)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/services/x"
    server.shutdown()
    server.server_close()


def test_send_slack_posts_the_text_as_json(webhook):
    alerts.send_slack(webhook, "hello")
    assert Webhook.received == [{"text": "hello"}]


def test_send_slack_raises_with_slacks_reason_when_rejected(webhook):
    Webhook.status = 403
    with pytest.raises(alerts.AlertError, match="HTTP 403: invalid_token"):
        alerts.send_slack(webhook, "hello")


def test_send_slack_raises_when_slack_cannot_be_reached():
    # Port 9 on loopback has no listener, so the connection is refused immediately.
    with pytest.raises(alerts.AlertError, match="could not be reached"):
        alerts.send_slack("http://127.0.0.1:9/services/x", "hello", timeout_seconds=2)


def test_the_env_file_supplies_only_the_alert_keys(tmp_path):
    (tmp_path / ".env").write_text(
        "# local settings\n"
        "AGENTWATCH_LLM_MODEL=qwen3:1.7b\n"
        f"{alerts.SLACK_WEBHOOK_ENV}='https://hooks.slack.invalid/T000/B000/x'\n"
        f'export {alerts.MAX_REJECTED_SHARE_ENV}="0.2"\n'
        "\n"
        "not a setting\n"
    )
    assert alerts.read_env_file(tmp_path / ".env", alerts.ALERT_ENV_KEYS) == {
        alerts.SLACK_WEBHOOK_ENV: "https://hooks.slack.invalid/T000/B000/x",
        alerts.MAX_REJECTED_SHARE_ENV: "0.2",
    }


def test_an_unquoted_values_trailing_comment_is_dropped(tmp_path):
    (tmp_path / ".env").write_text(
        f"{alerts.SLACK_WEBHOOK_ENV}=https://hooks.slack.invalid/x#frag # prod channel\n"
        f"{alerts.MAX_REJECTED_SHARE_ENV}='0.2 # kept, inside quotes' # dropped\n"
    )
    assert alerts.read_env_file(tmp_path / ".env", alerts.ALERT_ENV_KEYS) == {
        alerts.SLACK_WEBHOOK_ENV: "https://hooks.slack.invalid/x#frag",
        alerts.MAX_REJECTED_SHARE_ENV: "0.2 # kept, inside quotes",
    }


def test_a_missing_env_file_has_no_settings(tmp_path):
    assert alerts.read_env_file(tmp_path / ".env", alerts.ALERT_ENV_KEYS) == {}


def test_an_unreadable_env_file_is_an_error(tmp_path):
    (tmp_path / ".env").mkdir()  # a directory where the file should be
    with pytest.raises(alerts.AlertConfigError, match="could not be read"):
        alerts.read_env_file(tmp_path / ".env", alerts.ALERT_ENV_KEYS)


def test_the_environment_overrides_the_env_file(tmp_path):
    (tmp_path / ".env").write_text(
        f"{alerts.SLACK_WEBHOOK_ENV}=https://from-file.invalid/x\n"
        f"{alerts.MAX_REJECTED_SHARE_ENV}=0.2\n"
    )
    env = {
        alerts.SLACK_WEBHOOK_ENV: "https://from-env.invalid/x",
        alerts.MAX_REJECTED_SHARE_ENV: "",
    }
    assert alerts.alert_environment(tmp_path, env) == {
        alerts.SLACK_WEBHOOK_ENV: "https://from-env.invalid/x",
        alerts.MAX_REJECTED_SHARE_ENV: "0.2",  # an empty variable does not override the file
    }
