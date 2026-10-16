"""Runner behaviour against the in-memory live server and a scripted model (test doubles)."""

import json
import threading
from pathlib import Path
from urllib.error import URLError

import pytest
from agentwatch_agent.config import RunLimits
from agentwatch_agent.delivery import RetryPolicy, replay_journal
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.runner import AgentRunner
from conftest import SOLVE_MEAN, ScriptedModel, chat_response

NO_SLEEP = RetryPolicy(attempts=2, initial_backoff_seconds=0, max_backoff_seconds=0)


def make_runner(task, server, model, tmp_path, **overrides):
    settings = {
        "task": task,
        "model": model,
        "transport": server,
        "output_dir": tmp_path / "run",
        "sampling_interval_seconds": 0.05,
        "cancel_poll_seconds": 0.05,
        "flush_interval_seconds": 0.05,
        "retry": NO_SLEEP,
        "proc": tmp_path / "no-procfs",
    } | overrides
    return AgentRunner(**settings)


def evidence(outcome):
    return json.loads(Path(outcome.evidence_path).read_text())


def step_states(server):
    return {step_id: event.state for step_id, event in server.only_run().steps.items()}


def test_successful_task_produces_one_correlated_run(mean_task, server, tmp_path):
    outcome = make_runner(mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path).run()
    run = server.only_run()
    assert (outcome.state, outcome.task_passed, outcome.delivery_complete) == (
        "completed",
        True,
        True,
    )
    assert run.state == "completed"
    assert step_states(server) == {
        "agent-run": "completed",
        **{f"model-call-{i}": "completed" for i in range(1, 6)},
        "tool-1-search_docs": "completed",
        "tool-2-read_file": "completed",
        "tool-3-apply_patch": "completed",
        "tool-4-run_tests": "completed",
        "verify-tests": "completed",
    }
    events = list(run.steps.values())
    assert len({e.trace_id for e in events}) == 1
    assert {e.parent_step_id for e in events if e.step_id != "agent-run"} == {"agent-run"}
    types = {e.step_id: e.step_type for e in events}
    assert types["tool-1-search_docs"] == "retrieval"
    assert types["model-call-1"] == "model_call"
    resources = {r.resource_id: r.resource_class for r in run.registration.resources}
    assert resources[run.steps["model-call-1"].resource_id] == "inference"
    assert resources[run.steps["tool-1-search_docs"].resource_id] == "retrieval"
    assert resources[run.steps["verify-tests"].resource_id] == "cpu_sandbox"
    assert run.steps["verify-tests"].cpu_scope == "process_and_children"
    assert run.steps["model-call-1"].cpu_seconds is None

    saved = evidence(outcome)
    assert saved["configuration_digest"] == run.registration.configuration_digest
    assert saved["task"]["task_id"] == run.registration.task_id == "fix-mean-empty-r1"
    assert saved["cleanup"]["workspace_removed"] is True
    assert saved["verification"]["outcome"] == "passed"
    span_by_step = {e.step_id: e.span_id for e in events}
    for call in saved["model_calls"]:
        assert call["span_id"] == span_by_step[call["step_id"]]
        assert call["metrics"]["eval_count"] == 7
    samples = [r for r in run.records.values() if r["record_type"] == "resource_sample"]
    assert samples and all(s["availability"] == "unsupported" for s in samples)
    assert saved["resource_sampling"]["sample_rounds"] * 2 == len(samples)


def test_model_receives_tool_results_and_allowlisted_definitions(mean_task, server, tmp_path):
    model = ScriptedModel(SOLVE_MEAN)
    make_runner(mean_task, server, model, tmp_path).run()
    last = model.payloads[-1]["messages"]
    tool_results = [json.loads(m["content"]) for m in last if m["role"] == "tool"]
    assert tool_results[0]["results"][0]["section"] == "Mean of an empty input"
    assert tool_results[3]["outcome"] == "passed"
    assert {t["function"]["name"] for t in model.payloads[0]["tools"]} == {
        "search_docs",
        "read_file",
        "apply_patch",
        "run_tests",
    }
    assert model.payloads[0]["options"]["temperature"] == 0


def test_wrong_patch_fails_the_run_through_independent_verification(mean_task, server, tmp_path):
    script = [
        chat_response(tool_calls=[("apply_patch", {"patch_id": "return-zero-for-empty"})]),
        chat_response("Applied the patch; tests pass."),  # The model's claim is not trusted.
    ]
    outcome = make_runner(mean_task, server, ScriptedModel(script), tmp_path).run()
    assert (outcome.state, outcome.task_passed) == ("failed", False)
    assert server.only_run().reason == (
        "Agent finished, but the runner's verification tests did not pass"
    )
    assert step_states(server)["agent-run"] == "completed"
    assert evidence(outcome)["verification"]["outcome"] == "failed"


def test_rejected_call_is_never_executed_and_the_model_can_correct(mean_task, server, tmp_path):
    script = [
        chat_response(tool_calls=[("run_shell", {"command": "rm -rf /"})]),
        *SOLVE_MEAN,
    ]
    outcome = make_runner(mean_task, server, ScriptedModel(script), tmp_path).run()
    rejected = server.only_run().steps["tool-1-rejected"]
    assert (rejected.state, rejected.sequence, rejected.started_tick_ns) == ("failed", 1, None)
    assert rejected.reason.startswith("Rejected tool call: Unknown tool")
    assert outcome.state == "completed"
    record = evidence(outcome)["tool_calls"][0]
    assert (record["status"], record["requested_name"]) == ("rejected", "run_shell")


def test_failed_tool_is_returned_to_the_model_and_recorded(mean_task, server, tmp_path):
    script = [
        chat_response(tool_calls=[("apply_patch", {"patch_id": "return-none-for-empty"})]),
        chat_response(tool_calls=[("apply_patch", {"patch_id": "return-zero-for-empty"})]),
        chat_response("Done."),
    ]
    outcome = make_runner(mean_task, server, ScriptedModel(script), tmp_path).run()
    assert step_states(server)["tool-2-apply_patch"] == "failed"
    assert outcome.state == "completed"  # The first patch was right; verification decides.


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (URLError("connection refused"), "Model request failed (URLError)"),
        (RuntimeError("Invalid Ollama response"), "Model request failed (RuntimeError)"),
        ({"message": "not an object"}, "Model server returned a malformed chat response"),
        (chat_response(""), "Model replied with neither a tool call nor an answer"),
    ],
)
def test_model_failures_fail_the_run_with_a_shared_reason(
    mean_task, server, tmp_path, response, reason
):
    outcome = make_runner(mean_task, server, ScriptedModel([response]), tmp_path).run()
    run = server.only_run()
    assert (outcome.state, run.state, run.reason) == ("failed", "failed", reason)
    assert all(state in {"completed", "failed"} for state in step_states(server).values())
    assert evidence(outcome)["cleanup"]["workspace_removed"] is True


def test_model_call_limit_is_enforced(mean_task, server, tmp_path):
    script = [chat_response(tool_calls=[("read_file", {"path": "stats.py"})])] * 3
    runner = make_runner(
        mean_task, server, ScriptedModel(script), tmp_path, limits=RunLimits(max_model_calls=3)
    )
    outcome = runner.run()
    assert outcome.state == "failed"
    assert server.only_run().reason == "Model-call limit (3) reached without an answer"


def test_tool_call_limit_is_enforced(mean_task, server, tmp_path):
    calls = [("read_file", {"path": "stats.py"})] * 3
    runner = make_runner(
        mean_task,
        server,
        ScriptedModel([chat_response(tool_calls=calls)]),
        tmp_path,
        limits=RunLimits(max_tool_calls=2),
    )
    assert runner.run().state == "failed"
    assert server.only_run().reason == "Tool-call limit (2) reached"


def run_until_blocked(runner, model, action, delivered_step=None, server=None):
    """Run while a helper performs `action` once the model is blocked on its request
    (and, if given, once `delivered_step` has reached the server)."""

    def helper():
        assert model.blocked.wait(10), "model request never started"
        if delivered_step is not None:
            assert server.wait_for_step(delivered_step), f"{delivered_step} never arrived"
        action()

    thread = threading.Thread(target=helper)
    thread.start()
    try:
        return runner.run()
    finally:
        model.block.set()
        thread.join()


def test_remote_cancellation_stops_work_and_records_cleanup(mean_task, server, tmp_path):
    model = ScriptedModel(SOLVE_MEAN, block=threading.Event(), block_on=2)
    runner = make_runner(mean_task, server, model, tmp_path)
    outcome = run_until_blocked(
        runner,
        model,
        lambda: server.cancel(next(iter(server.runs)), "Stopped by reviewer"),
        delivered_step="model-call-2",
        server=server,
    )
    run = server.only_run()
    assert (outcome.state, run.state, run.reason) == (
        "cancelled",
        "cancelled",
        "Stopped by reviewer",
    )
    states = step_states(server)
    assert states["model-call-2"] == "cancelled" and states["agent-run"] == "cancelled"
    assert run.steps["model-call-2"].sequence == 2
    saved = evidence(outcome)
    assert saved["cancellation"]["source"] == "live_api"
    assert saved["cancellation"]["cleanup_confirmed_by_service"] is None
    assert saved["cleanup"] == saved["cleanup"] | {
        "workspace_removed": True,
        "abandoned_model_request": True,
    }
    assert outcome.delivery_complete


def test_local_signal_cancels_the_run_itself(mean_task, server, tmp_path):
    model = ScriptedModel(SOLVE_MEAN, block=threading.Event(), block_on=1)
    runner = make_runner(mean_task, server, model, tmp_path)
    outcome = run_until_blocked(
        runner,
        model,
        lambda: runner.request_cancel("signal", "Cancelled by operator signal SIGINT"),
    )
    run = server.only_run()
    assert (run.state, run.reason) == ("cancelled", "Cancelled by operator signal SIGINT")
    assert set(step_states(server).values()) == {"cancelled"}
    assert outcome.state == "cancelled"


def test_run_time_limit_abandons_a_slow_model(mean_task, server, tmp_path):
    model = ScriptedModel(SOLVE_MEAN, block=threading.Event(), block_on=1)
    runner = make_runner(
        mean_task, server, model, tmp_path, limits=RunLimits(run_timeout_seconds=0.3)
    )
    try:
        outcome = runner.run()
    finally:
        model.block.set()
    assert outcome.state == "failed"
    assert server.only_run().reason == "Run time limit reached while waiting for a model response"


def test_api_outage_keeps_a_replayable_journal(mean_task, server, tmp_path):
    model = ScriptedModel(SOLVE_MEAN, block=threading.Event(), block_on=3)
    runner = make_runner(mean_task, server, model, tmp_path)

    def outage():
        server.down = True
        model.block.set()

    outcome = run_until_blocked(runner, model, outage)
    assert outcome.task_passed
    assert (outcome.state, outcome.delivery_complete) == ("completed_locally_undelivered", False)
    assert server.only_run().state == "running"
    saved = evidence(outcome)
    assert saved["delivery"]["pending"] > 0
    server.down = False
    report = replay_journal(outcome.journal_path, server, NO_SLEEP)
    run = server.only_run()
    assert report.run_id == outcome.run_id
    assert run.state == "completed"
    assert set(step_states(server).values()) == {"completed"}


def test_unreachable_api_at_registration_runs_nothing(mean_task, server, tmp_path):
    server.down = True
    model = ScriptedModel(SOLVE_MEAN)
    with pytest.raises(RunNotStarted, match="URLError"):
        make_runner(mean_task, server, model, tmp_path).run()
    saved = json.loads((tmp_path / "run/agent-evidence.json").read_text())
    assert saved["status"] == "not_started"
    assert saved["cleanup"]["workspace_removed"] is True
    assert model.payloads == []
