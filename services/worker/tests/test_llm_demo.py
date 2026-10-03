from contextlib import nullcontext

import pytest
from agentwatch_worker.llm_demo import agent_loop


class Span:
    def set_attributes(self, attributes):
        pass


class Tracer:
    def start_as_current_span(self, name):
        return nullcontext(Span())


def fixture_result():
    return {
        "run_id": "test-run",
        "checks": {"passed": True},
        "clean": {"queue_seconds": 0.001, "execution_seconds": 0.2, "resource": "none"},
        "contended": {"queue_seconds": 1.0, "execution_seconds": 0.2, "resource": "cpu_sandbox"},
    }


def tool_call(name="run_tests", arguments=None):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": name,
                        "arguments": arguments if arguments is not None else {},
                    }
                },
            ],
        }
    }


def test_result_is_fed_back_before_final_answer():
    evidence = {}
    payloads = []
    calls = []

    def chat(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return tool_call()
        assert payload["messages"][-1]["role"] == "tool"
        assert '"queue_seconds": 1.0' in payload["messages"][-1]["content"]
        return {"message": {"role": "assistant", "content": "Tests passed; queue detected."}}

    def execute():
        calls.append(True)
        return fixture_result()

    result = agent_loop(chat, execute, Tracer(), "test-model", evidence)
    assert len(calls) == 1
    assert result["checks"]["passed"]
    assert all(evidence["checks"].values())
    assert len(evidence["inference"]) == 2


@pytest.mark.parametrize(
    "response",
    [
        tool_call("shell"),
        tool_call(arguments={"command": "anything"}),
        {"message": {"role": "assistant", "content": "I ran the tests."}},
    ],
)
def test_invalid_or_missing_tool_is_rejected_without_execution(response):
    def execute():
        pytest.fail("Invalid tool must never execute")

    with pytest.raises(ValueError):
        agent_loop(lambda _: response, execute, Tracer(), "test-model", {})


def test_repeated_tool_call_does_not_run_twice():
    calls = []

    def execute():
        calls.append(True)
        return fixture_result()

    with pytest.raises(ValueError, match="exactly one"):
        agent_loop(lambda _: tool_call(), execute, Tracer(), "test-model", {})
    assert len(calls) == 1


def test_invalid_arguments_can_be_corrected_without_executing_them():
    responses = iter(
        [
            tool_call(arguments={"test_cases": ["invented"]}),
            tool_call(),
            {"message": {"role": "assistant", "content": "Checks passed."}},
        ]
    )
    executions = []
    evidence = {}

    def chat(payload):
        if evidence.get("rejected_calls") and not executions:
            assert "No tool executed" in payload["messages"][-1]["content"]
        return next(responses)

    def execute():
        executions.append(True)
        return fixture_result()

    agent_loop(chat, execute, Tracer(), "test-model", evidence)
    assert len(executions) == 1
    assert len(evidence["rejected_calls"]) == 1
    assert all(evidence["checks"].values())
