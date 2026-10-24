"""Sampling-overhead protocol with a scripted chat function (a test double, never evidence)."""

import json

import pytest
from agent_test_doubles import FakeGpu
from agentwatch_agent.sampling_overhead import (
    PROTOCOL,
    first_request,
    measure_sampling_overhead,
    reply_digest,
)
from agentwatch_agent.tools import tool_definitions

# Illustrative Ollama durations in nanoseconds.
RESPONSE = {
    "message": {"role": "assistant", "content": "", "tool_calls": []},
    "total_duration": 20_000_000,
    "prompt_eval_duration": 5_000_000,
    "eval_duration": 10_000_000,
    "prompt_eval_count": 300,
    "eval_count": 20,
}


def measure(mean_task, tmp_path, chat, **settings):
    options = {
        "repetitions": 3,
        "warmup": 1,
        "sampling_interval_seconds": 0.01,
        "open_gpu": FakeGpu,
        "proc": tmp_path / "no-procfs",
    } | settings
    return measure_sampling_overhead(mean_task, "test-model", chat, tmp_path / "out", **options)


def test_workload_is_the_runners_first_request(mean_task):
    payload = first_request(mean_task, "test-model")
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["tools"] == list(tool_definitions(mean_task))
    assert (payload["stream"], payload["think"]) == (False, False)


def test_conditions_alternate_and_sampling_is_journaled(mean_task, tmp_path):
    seen = []

    def chat(payload):
        seen.append(payload)
        return RESPONSE

    report = measure(mean_task, tmp_path, chat)
    assert report["status"] == "completed"
    assert report["protocol"] == PROTOCOL
    assert len(seen) == 1 + 2 * 3
    assert [m["condition"] for m in report["measurements"]] == ["unsampled", "sampled"] * 3
    for condition in ("unsampled", "sampled"):
        summary = report["conditions"][condition]
        assert summary["wall_seconds"]["count"] == 3
        assert summary["total_duration_seconds"]["median"] == pytest.approx(0.02)
        assert summary["eval_counts"] == [20]
    sampling = report["sampling"]
    assert sampling["rounds"] >= 3
    assert sampling["availability_counts"]["gpu.nvml.utilization.gpu.v1"] == {
        "measured": sampling["rounds"]
    }
    journal = (tmp_path / "out/sampling-journal.jsonl").read_text().splitlines()
    assert len(journal) == 7 * sampling["rounds"]
    assert {json.loads(line)["kind"] for line in journal} == {"record"}


def test_failed_measurement_keeps_what_was_measured(mean_task, tmp_path):
    calls = []

    def chat(_payload):
        calls.append(1)
        if len(calls) == 4:
            raise OSError("connection reset")
        return RESPONSE

    report = measure(mean_task, tmp_path, chat)
    assert report["status"] == "failed"
    assert report["error"] == "OSError: connection reset"
    # One warm-up, then unsampled and sampled requests before the failing one.
    assert len(report["measurements"]) == 2
    assert report["conditions"]["unsampled"]["wall_seconds"]["count"] == 1
    assert report["median_wall_difference_seconds"] is not None


def test_failure_before_both_conditions_has_no_difference(mean_task, tmp_path):
    def chat(_payload):
        raise OSError("refused")

    report = measure(mean_task, tmp_path, chat)
    assert report["status"] == "failed"
    assert report["conditions"] == {"unsampled": None, "sampled": None}
    assert report["median_wall_difference_fraction"] is None


def tool_reply(call_ids, query="mean() contract"):
    calls = [
        {
            "id": call_id,
            "function": {"index": 0, "name": "search_docs", "arguments": {"query": query}},
        }
        for call_id in call_ids
    ]
    return {"message": {"role": "assistant", "content": "", "tool_calls": calls}}


def test_reply_digest_ignores_ollamas_random_tool_call_ids():
    # Ollama 0.35.1 returned the same call with ids call_jq4oxp6v and call_bvno9r3v.
    assert reply_digest(tool_reply(["call_jq4oxp6v"])) == reply_digest(
        tool_reply(["call_bvno9r3v"])
    )
    assert reply_digest(tool_reply(["call_a"])) != reply_digest(tool_reply(["call_a"], "median"))
    assert reply_digest({"message": {"content": "done"}}) != reply_digest(tool_reply(["call_a"]))
