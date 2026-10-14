import http.client
import io
import json
import sys
import urllib.error
from contextlib import nullcontext

import pytest
from agentwatch_worker import llm_demo
from agentwatch_worker.llm_demo import agent_loop, model_placement, observe_model_placement


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


def loaded(size=1000, size_vram=1000, name="qwen3:1.7b"):
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "size": size,
                "size_vram": size_vram,
                "digest": "8f68893c685c",
                "details": {"quantization_level": "Q4_K_M"},
                "context_length": 4096,
            }
        ]
    }


@pytest.mark.parametrize(
    ("size_vram", "fraction"),
    [(1000, 1.0), (0, 0.0), (250, 0.25)],
)
def test_model_placement_reports_ollama_memory_split(size_vram, fraction):
    placement = model_placement(loaded(size_vram=size_vram), "qwen3:1.7b")
    assert placement["status"] == "measured"
    assert placement["gpu_memory_fraction"] == fraction
    assert placement["size_vram_bytes"] == size_vram
    assert placement["digest"] == "8f68893c685c"


def test_unloaded_model_is_unavailable_not_cpu():
    placement = model_placement({"models": []}, "qwen3:1.7b")
    assert placement["status"] == "unavailable"
    assert "gpu_memory_fraction" not in placement


@pytest.mark.parametrize(
    "response",
    [
        {},
        [],
        loaded(size=0, size_vram=0),
        loaded(size_vram=2000),
        loaded(size_vram=-1),
        loaded(size_vram=True),
        loaded(size="1000"),
    ],
)
def test_inconsistent_placement_is_malformed(response):
    placement = model_placement(response, "qwen3:1.7b")
    assert placement["status"] == "malformed"
    assert "gpu_memory_fraction" not in placement


@pytest.mark.parametrize(
    ("failure", "reported"),
    [
        (urllib.error.URLError("connection refused"), "connection refused"),
        (ConnectionResetError("connection reset"), "connection reset"),
        (http.client.IncompleteRead(b"partial"), "IncompleteRead"),
    ],
)
def test_placement_request_failure_is_recorded_as_unavailable(failure, reported):
    def unreachable():
        raise failure

    placement = observe_model_placement(unreachable, "qwen3:1.7b")
    assert placement["status"] == "unavailable"
    assert placement["reason"].startswith("placement request failed:")
    assert reported in placement["reason"]
    assert "gpu_memory_fraction" not in placement
    assert isinstance(placement["observed_at_unix_ns"], int)


def respond_with(body: bytes):
    def urlopen(*_args, **_kwargs):
        return io.BytesIO(body)

    return urlopen


@pytest.mark.parametrize("body", [b"\xff", b"not json", b'{"models": '])
def test_undecodable_placement_response_is_malformed(monkeypatch, body):
    monkeypatch.setattr(llm_demo.urllib.request, "urlopen", respond_with(body))
    placement = observe_model_placement(
        lambda: llm_demo.ollama_loaded_models("http://ollama.test"), "qwen3:1.7b"
    )
    assert placement["status"] == "malformed"
    assert placement["reason"].startswith("placement response is not valid UTF-8 JSON:")
    assert "gpu_memory_fraction" not in placement


def test_placement_programming_errors_are_not_contained():
    def broken():
        raise KeyError("models")

    with pytest.raises(KeyError):
        observe_model_placement(broken, "qwen3:1.7b")


def test_failed_placement_read_preserves_completed_run(monkeypatch, tmp_path):
    responses = iter([tool_call(), {"message": {"role": "assistant", "content": "Checks passed."}}])
    monkeypatch.setattr(llm_demo, "ollama_chat", lambda _url, _payload: next(responses))
    monkeypatch.setattr(
        llm_demo, "run_demo", lambda _output, workload: fixture_result() | {"limitations": []}
    )
    monkeypatch.setattr(llm_demo.urllib.request, "urlopen", respond_with(b"\xff"))
    monkeypatch.setattr(
        sys, "argv", ["llm_demo", "--url", "http://ollama.test", "--output", str(tmp_path)]
    )

    llm_demo.main()

    agent = json.loads((tmp_path / "agent.json").read_text())
    results = json.loads((tmp_path / "results.json").read_text())
    assert agent["status"] == "passed"
    assert agent["model_placement"]["status"] == "malformed"
    assert results["checks"]["passed"]
    assert results["agent"]["model_placement"]["status"] == "malformed"
