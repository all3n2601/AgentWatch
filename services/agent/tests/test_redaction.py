"""Trace redaction over synthetic strings. Every credential below is a made-up fixture."""

import pytest
from agentwatch_agent.redaction import redact_reason, redact_text

# Built from parts so no complete credential-shaped literal sits in the source.
GITHUB_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
ANTHROPIC_KEY = "sk-ant-" + "api03-" + "x" * 40
AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"
SLACK_TOKEN = "xoxb-" + "1234567890-abcdefghij"
# Dummy values for the labeled-secret and URL rules. Neutral names and plain values keep
# secret scanners from reading the fixtures as real credentials.
DUMMY_LONG = "example" + "value123"
DUMMY_QUOTED = "example" + "value"
DUMMY_SHORT = "dummy" + "42"
DUMMY_KEYLIKE = "abc123" + "xyz"
KEY_LABEL = "RSA PRIVATE" + " KEY"
PRIVATE_KEY = f"-----BEGIN {KEY_LABEL}-----\nnot-a-real-key\n-----END {KEY_LABEL}-----"


@pytest.mark.parametrize(
    ("raw", "expected", "kind"),
    [
        (f"push failed: {GITHUB_TOKEN} rejected", "push failed: <credential> rejected", "token"),
        (f"key {ANTHROPIC_KEY}", "key <credential>", "token"),
        (f"aws {AWS_KEY} denied", "aws <credential> denied", "token"),
        (f"slack {SLACK_TOKEN}", "slack <credential>", "token"),
        (f"loaded {PRIVATE_KEY} ok", "loaded <private-key> ok", "private_key"),
        (
            f"GET https://alice:{DUMMY_SHORT}@example.test/v1 failed",
            "GET https://<credentials>@example.test/v1 failed",
            "url_credentials",
        ),
        (
            f"Authorization: Bearer {DUMMY_LONG}",
            "Authorization: Bearer <credential>",
            "bearer",
        ),
        (f"API_KEY={DUMMY_KEYLIKE} rejected", "API_KEY=<credential> rejected", "secret_assignment"),
        (f'password: "{DUMMY_QUOTED}"', 'password: "<credential>"', "secret_assignment"),
        (f"db_password = {DUMMY_SHORT};", "db_password = <credential>;", "secret_assignment"),
        ("contact maintainer@example.test today", "contact <email> today", "email"),
        ("in /Users/alice/project/stats.py", "in <path>", "path"),
        ("in /home/bob/.venv/lib/x.py", "in <path>", "path"),
        ("C:\\Users\\bob\\file.txt failed", "<path> failed", "path"),
        ("tmp /private/var/folders/ab/T/pytest-1/x.py", "tmp <path>", "path"),
        ("cwd /tmp/pytest-of-alice/run", "cwd <path>", "path"),
        ("container /app/src/main.py", "container <path>", "path"),
    ],
)
def test_sensitive_values_are_replaced_and_counted(raw, expected, kind):
    text, counts = redact_text(raw)
    assert text == expected
    assert counts == {kind: 1}


@pytest.mark.parametrize(
    "safe",
    [
        "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "run 97fb140d-be6f-42f9-b31c-0d1ab305aabf",
        "max_tokens: 512, eval_count=20",
        "num_predict=512 tokens=20",
        "http://127.0.0.1:8000/metrics",
        # Regression: API routes recorded in evidence are not host paths.
        "/api/ps",
        "POST /v1/chat/completions returned 200",
        "probe /readyz then /metrics",
        "relative tests/test_stats.py and ratio 3/4",
        "<project>/tests/test_stats.py::test_mean FAILED",
        "def mean(values):\n    return sum(values) / len(values)",
        "a token is a unit of text",
    ],
)
def test_ordinary_text_is_unchanged(safe):
    assert redact_text(safe) == (safe, {})


def test_several_kinds_are_counted_together():
    text, counts = redact_text(f"{GITHUB_TOKEN} by a@example.test in /Users/alice/x/y.py")
    assert text == "<credential> by <email> in <path>"
    assert counts == {"token": 1, "email": 1, "path": 1}


def test_reasons_use_the_same_redaction_and_stay_bounded():
    reason = redact_reason(f"refused {GITHUB_TOKEN}\nfor alice@example.test")
    assert reason == "refused <credential> for <email>"
    assert len(redact_reason("token=" + "x" * 2000)) <= 512


def test_urls_in_reasons_are_kept_rather_than_treated_as_paths():
    # Regression: the path rule used to turn `http://host/x` into `http:/<path>`.
    reason = "connection to http://127.0.0.1:8090/v1/runs refused"
    assert redact_reason(reason) == reason
