"""The agent's admission overhead measurement is an attempt the overhead verdict can judge.

The measurement (`services/agent`) and the verdict (`pipelines/data`) declare
`admission-overhead-v1` separately, because neither package imports the other. This test saves
a measurement made with test doubles and reads it back through the verdict's attempt reader.
"""

import json

from agent_test_doubles import FakePolicySource, chat_response, policy_version
from agentwatch_agent.admission_overhead import measure_admission_overhead
from agentwatch_agent.delivery import RetryPolicy
from agentwatch_agent.tasks import load_task
from agentwatch_data.overhead_verdict import read_attempt


def test_a_saved_measurement_reads_as_paired_admission_attempt(tmp_path):
    report = measure_admission_overhead(
        load_task("fix-mean-empty"),
        "test-model",
        lambda payload: chat_response("done"),
        FakePolicySource(policy_version(0)),
        inference_service_id="ollama",
        repetitions=4,
        warmup=1,
        policy_poll_seconds=0.01,
        retry=RetryPolicy(attempts=1, initial_backoff_seconds=0, max_backoff_seconds=0),
    )
    path = tmp_path / "attempt-1" / "admission-overhead.json"
    path.parent.mkdir()
    path.write_text(json.dumps(report))
    attempt = read_attempt(path)
    assert attempt.status == "completed" and len(attempt.pairs) == 4
    assert attempt.unpaired_repetitions == 0
    assert attempt.settings["protocol"] == "admission-overhead-v1"
    for field in ("task_id", "model", "request_sha256", "runner_version", "policy_poll_seconds"):
        assert attempt.settings[field] == report[field]
    # The admission settings the measurement ran under are part of the group key.
    assert attempt.settings["admission_inference_service"] == report["policy"]["inference_service"]
    assert attempt.settings["admission_inference_enforced"] == (
        "enforced" if report["policy"]["enforced"] else "not_enforced"
    )
