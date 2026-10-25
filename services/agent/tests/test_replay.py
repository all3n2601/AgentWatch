"""Replay plans and the replay runner, against a scripted model and the in-memory live server.

Both are test doubles; nothing here is recorded-run evidence.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from agent_test_doubles import SOLVE_MEAN, ScriptedModel, chat_response
from agentwatch_agent.delivery import RetryPolicy
from agentwatch_agent.replay import (
    ReplayPlanError,
    ReplayRunner,
    load_replay_plan,
    reply_signature,
    tool_signature,
)
from agentwatch_agent.runner import EVIDENCE_FILE, AgentRunner
from agentwatch_agent.tasks import load_task

NO_SLEEP = RetryPolicy(attempts=2, initial_backoff_seconds=0, max_backoff_seconds=0)


def runner_settings(task, server, model, output_dir: Path) -> dict:
    return {
        "task": task,
        "model": model,
        "transport": server,
        "output_dir": output_dir,
        "sampling_interval_seconds": 0.06,
        "cancel_poll_seconds": 0.05,
        "flush_interval_seconds": 0.05,
        "retry": NO_SLEEP,
        "proc": output_dir.parent / "no-procfs",
    }


def record_source(task, server, tmp_path, script=SOLVE_MEAN) -> tuple[Path, ScriptedModel]:
    model = ScriptedModel(script)
    source = tmp_path / "source"
    AgentRunner(**runner_settings(task, server, model, source)).run()
    return source, model


def replay(task, server, plan, tmp_path, name, script=SOLVE_MEAN, **overrides):
    model = ScriptedModel(script)
    settings = runner_settings(task, server, model, tmp_path / name) | overrides
    return ReplayRunner(plan=plan, **settings), model


def test_plan_reproduces_each_model_input_and_requested_call(mean_task, server, tmp_path):
    source, source_model = record_source(mean_task, server, tmp_path)
    plan = load_replay_plan(source, mean_task)
    assert len(plan.turns) == 5 and plan.tool_call_count == 4
    for turn, payload in zip(plan.turns, source_model.payloads, strict=True):
        assert list(turn.messages) == payload["messages"]
    assert [call.name for turn in plan.turns for call in turn.calls] == [
        "search_docs",
        "read_file",
        "apply_patch",
        "run_tests",
    ]
    assert plan.search_queries() == ["mean of an empty input"]
    assert plan.model == "scripted-test-double"
    assert plan.chat_options["temperature"] == 0
    assert plan.task_id == mean_task.task_id


def test_replay_sends_the_recorded_inputs_and_keeps_step_ids(mean_task, server, tmp_path):
    source, source_model = record_source(mean_task, server, tmp_path)
    plan = load_replay_plan(source, mean_task)
    runner, model = replay(mean_task, server, plan, tmp_path, "replay")
    outcome = runner.run()
    assert (outcome.state, outcome.task_passed) == ("completed", True)
    assert [p["messages"] for p in model.payloads] == [p["messages"] for p in source_model.payloads]
    source_run, replay_run = (server.runs[run] for run in server.runs)
    assert sorted(replay_run.steps) == sorted(source_run.steps)
    saved = json.loads(outcome.evidence_path.read_text())
    record = saved["replay"]
    assert record["source_run_id"] == plan.source_run_id
    assert record["plan_sha256"] == plan.sha256()
    assert all(o["matches_source"] for o in record["model_outputs"])
    assert all(t["matches_source"] for t in record["tool_results"])
    assert saved["configuration"]["replay"] == {
        "method": "agent-replay-v1",
        "plan_sha256": plan.sha256(),
    }
    assert "messages" not in saved  # the conversation is the source's, identified by digest


def test_replays_of_one_plan_share_a_digest_distinct_from_the_source(mean_task, server, tmp_path):
    source, _ = record_source(mean_task, server, tmp_path)
    plan = load_replay_plan(source, mean_task)
    for name in ("first", "second"):
        replay(mean_task, server, plan, tmp_path, name)[0].run()
    digests = [run.registration.configuration_digest for run in server.runs.values()]
    assert digests[1] == digests[2] != digests[0]


def test_replay_executes_recorded_calls_whatever_the_model_replies(mean_task, server, tmp_path):
    source, _ = record_source(mean_task, server, tmp_path)
    plan = load_replay_plan(source, mean_task)
    different = [chat_response("I would rather not.")] * 5
    runner, _ = replay(mean_task, server, plan, tmp_path, "replay", script=different)
    outcome = runner.run()
    assert (outcome.state, outcome.task_passed) == ("completed", True)
    record = json.loads(outcome.evidence_path.read_text())["replay"]
    assert [o["matches_source"] for o in record["model_outputs"]] == [False] * 5
    assert all(t["matches_source"] for t in record["tool_results"])
    assert "tool-3-apply_patch" in server.runs[outcome.run_id].steps


def test_a_rejected_source_call_is_replayed_as_rejected(mean_task, server, tmp_path):
    script = [
        chat_response(tool_calls=[("delete_everything", {})]),
        *SOLVE_MEAN,
    ]
    source, _ = record_source(mean_task, server, tmp_path, script)
    plan = load_replay_plan(source, mean_task)
    outcome = replay(mean_task, server, plan, tmp_path, "replay", script=[*script])[0].run()
    assert outcome.state == "completed"
    steps = server.runs[outcome.run_id].steps
    assert steps["tool-1-rejected"].state == "failed"


def test_tool_definitions_that_changed_fail_the_replay(mean_task, server, tmp_path):
    source, _ = record_source(mean_task, server, tmp_path)
    plan = load_replay_plan(source, mean_task)
    changed = dataclasses.replace(plan, tools=plan.tools[:-1])
    outcome = replay(mean_task, server, changed, tmp_path, "replay")[0].run()
    assert outcome.state == "failed"
    assert outcome.reason == "Replay tool definitions differ from the source run"


def test_a_replay_must_use_the_source_model_and_options(mean_task, server, tmp_path):
    source, _ = record_source(mean_task, server, tmp_path)
    plan = load_replay_plan(source, mean_task)
    with pytest.raises(ValueError, match="recorded with model"):
        replay(mean_task, server, dataclasses.replace(plan, model="other"), tmp_path, "r")
    with pytest.raises(ValueError, match="chat options"):
        replay(mean_task, server, plan, tmp_path, "r", chat_options={"temperature": 1})


def test_a_plan_beyond_the_run_limits_is_refused(mean_task, server, tmp_path):
    from agentwatch_agent.config import RunLimits

    source, _ = record_source(mean_task, server, tmp_path)
    plan = load_replay_plan(source, mean_task)
    with pytest.raises(ValueError, match="limits"):
        replay(mean_task, server, plan, tmp_path, "r", limits=RunLimits(max_model_calls=4))


def test_a_failed_source_run_cannot_be_replayed(mean_task, server, tmp_path):
    wrong = [
        chat_response(tool_calls=[("apply_patch", {"patch_id": "return-zero-for-empty"})]),
        chat_response("Done."),
    ]
    source, _ = record_source(mean_task, server, tmp_path, wrong)
    with pytest.raises(ReplayPlanError, match="verification passed"):
        load_replay_plan(source, mean_task)


def edit_evidence(source: Path, change) -> None:
    path = source / EVIDENCE_FILE
    evidence = json.loads(path.read_text())
    change(evidence)
    path.write_text(json.dumps(evidence))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e["task"].update(content_sha256="0" * 64), "current content"),
        (lambda e: e.pop("messages"), "conversation"),
        (lambda e: e["messages"].pop(), "end with the model's answer"),
        (lambda e: e["messages"].pop(3), "one tool result per call"),
        (lambda e: e["model_calls"].pop(), "do not match"),
        (lambda e: e["model_calls"][0].update(requested_tool_calls=2), "do not match"),
        (lambda e: e["messages"][2].update(content=7), "malformed"),
        (lambda e: e["configuration"].pop("tools"), "tool definitions"),
        (lambda e: e.pop("run_id"), "run ID"),
        (lambda e: e.update(status="cancelled"), "verification passed"),
    ],
    ids=[
        "task-content",
        "no-conversation",
        "no-answer",
        "missing-result",
        "call-count",
        "requested-calls",
        "malformed-reply",
        "no-tools",
        "no-run-id",
        "not-completed",
    ],
)
def test_source_evidence_that_cannot_be_replayed_faithfully_is_refused(
    mean_task, server, tmp_path, change, message
):
    source, _ = record_source(mean_task, server, tmp_path)
    edit_evidence(source, change)
    with pytest.raises(ReplayPlanError, match=message):
        load_replay_plan(source, mean_task)


def test_missing_or_unreadable_evidence_is_refused(mean_task, tmp_path):
    with pytest.raises(ReplayPlanError, match="Cannot read"):
        load_replay_plan(tmp_path, mean_task)
    (tmp_path / EVIDENCE_FILE).write_text("[1, 2]")
    with pytest.raises(ReplayPlanError, match="not a JSON object"):
        load_replay_plan(tmp_path, mean_task)


def test_a_source_of_another_task_is_refused(mean_task, server, tmp_path):
    source, _ = record_source(mean_task, server, tmp_path)
    other = load_task("fix-window-count")
    with pytest.raises(ReplayPlanError, match=other.task_id):
        load_replay_plan(source, other)


def test_signatures_ignore_test_timings_but_not_outcomes():
    def result(outcome, output):
        content = {"outcome": outcome, "tests_run": 5, "output": output}
        return {"role": "tool", "tool_name": "run_tests", "content": json.dumps(content)}

    assert tool_signature(result("passed", "in 0.01s")) == tool_signature(
        result("passed", "in 0.02s")
    )
    assert tool_signature(result("passed", "x")) != tool_signature(result("failed", "x"))
    read = {"role": "tool", "tool_name": "read_file", "content": json.dumps({"content": "a"})}
    changed = read | {"content": json.dumps({"content": "b"})}
    assert tool_signature(read) != tool_signature(changed)
    reply = {"role": "assistant", "content": "x", "tool_calls": [{"function": {"name": "a"}}]}
    assert reply_signature(reply) == reply_signature(json.loads(json.dumps(reply)))
    assert reply_signature(reply) != reply_signature(reply | {"content": "y"})
