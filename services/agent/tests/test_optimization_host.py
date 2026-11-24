"""Live wiring for the optimization runner (O06): waits before a batch, the served model's
identity, the injectors a manifest names, and the two commands' refusals.

The model servers are test doubles (patched readers and HTTP responses); nothing here talks to
a real server or records evidence.
"""

import io
import json
import threading
from types import SimpleNamespace
from urllib.error import URLError

import pytest
from agentwatch_agent import cli, optimization_host
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.optimization_experiment import ExperimentError
from agentwatch_agent.optimization_host import (
    injector_factory,
    ollama_model_identity,
    pacing_waiter,
    vllm_idle_waiter,
    vllm_model_identity,
)
from agentwatch_agent.vllm_server import ServerStateUnavailable
from agentwatch_injectors.cpu import CPU_METHOD, CpuSpinInjector
from agentwatch_injectors.prefill import PREFILL_METHOD, PrefillFloodInjector

PINNED = "Qwen/Qwen3-1.7B@70d244cc86cc"


class Pauses:
    """Records each pause instead of waiting; cancels when told to."""

    def __init__(self, cancel_on=None):
        self.seconds = []
        self.cancel_on = cancel_on

    def __call__(self, cancel, seconds):
        self.seconds.append(seconds)
        if self.cancel_on is not None and len(self.seconds) == self.cancel_on:
            cancel.set()
        return cancel.is_set()


def server_states(monkeypatch, *states):
    """Each read returns the next (running, waiting), or raises it when it is an exception."""
    remaining = list(states)

    def read(_url):
        state = remaining.pop(0)
        if isinstance(state, Exception):
            raise state
        return SimpleNamespace(running=state[0], waiting=state[1])

    monkeypatch.setattr(optimization_host, "read_server_state", read)


def ticking(step):
    now = [0.0]

    def monotonic():
        now[0] += step
        return now[0]

    return monotonic


def test_the_vllm_wait_polls_until_idle_then_paces(monkeypatch):
    server_states(monkeypatch, (1, 0), (0, 2), (0, 0))
    pauses = Pauses()
    wait = vllm_idle_waiter("http://vllm/metrics", pacing_seconds=10, pause=pauses)
    wait(threading.Event())
    assert pauses.seconds == [0.5, 0.5, 10]


def test_the_vllm_wait_fails_when_vllm_stays_busy(monkeypatch):
    server_states(monkeypatch, *[(2, 1)] * 5)
    wait = vllm_idle_waiter(
        "http://vllm/metrics", timeout_seconds=1.0, pause=Pauses(), monotonic=ticking(0.6)
    )
    with pytest.raises(RuntimeError, match="still had 2 running and 1 waiting after 1 s"):
        wait(threading.Event())


def test_the_vllm_wait_fails_when_its_metrics_are_unreadable(monkeypatch):
    server_states(monkeypatch, ServerStateUnavailable("connection refused"))
    with pytest.raises(RuntimeError, match="vLLM metrics unreadable: connection refused"):
        vllm_idle_waiter("http://vllm/metrics", pause=Pauses())(threading.Event())


def test_a_cancel_ends_the_wait_without_pacing(monkeypatch):
    server_states(monkeypatch, (1, 0), (1, 0))
    pauses = Pauses(cancel_on=1)
    vllm_idle_waiter("http://vllm/metrics", pacing_seconds=10, pause=pauses)(threading.Event())
    assert pauses.seconds == [0.5]  # woken in the idle poll; the pacing pause never starts


def test_the_pacing_wait_pauses_on_the_cancel_event():
    pauses = Pauses()
    cancel = threading.Event()
    pacing_waiter(7.0, pause=pauses)(cancel)
    assert pauses.seconds == [7.0]
    cancel.set()
    # The default pause is the event's own wait, so a set event returns at once.
    pacing_waiter(60.0)(cancel)


def test_the_vllm_model_identity_is_its_pinned_revision(monkeypatch):
    listing = {"data": [{"id": PINNED, "root": "Qwen/Qwen3-1.7B"}, {"id": "Qwen/Qwen3-1.7B"}]}
    monkeypatch.setattr(optimization_host, "vllm_models", lambda _url: listing)
    assert vllm_model_identity("http://vllm", PINNED) == {
        "backend": "vllm",
        "name": PINNED,
        "digest": "70d244cc86cc",
    }
    with pytest.raises(RuntimeError, match="revision-pinned model name"):
        vllm_model_identity("http://vllm", "Qwen/Qwen3-1.7B")
    with pytest.raises(RuntimeError, match="does not serve"):
        vllm_model_identity("http://vllm", "Qwen/Qwen3-4B@0000")


def ollama_answers(monkeypatch, answer):
    def urlopen(_url, timeout):
        assert timeout == optimization_host.HTTP_TIMEOUT_SECONDS
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(json.dumps(answer).encode())

    monkeypatch.setattr(optimization_host.urllib.request, "urlopen", urlopen)


def test_the_ollama_model_identity_is_its_listed_digest(monkeypatch):
    ollama_answers(monkeypatch, {"models": [{"name": "qwen3:1.7b", "digest": "8f68893c685c"}]})
    assert ollama_model_identity("http://ollama", "qwen3:1.7b") == {
        "backend": "ollama",
        "name": "qwen3:1.7b",
        "digest": "8f68893c685c",
    }


@pytest.mark.parametrize(
    "answer",
    [
        {"models": [{"name": "qwen3:1.7b", "digest": ""}]},
        {"models": [{"name": "qwen3:1.7b"}]},
        {"models": [{"name": "llama3:8b", "digest": "abc"}]},
        {"unexpected": True},
    ],
)
def test_an_ollama_model_without_a_listed_digest_is_refused(monkeypatch, answer):
    ollama_answers(monkeypatch, answer)
    with pytest.raises(RuntimeError, match="does not list qwen3:1.7b with a digest"):
        ollama_model_identity("http://ollama", "qwen3:1.7b")


def test_an_unreachable_ollama_is_an_error_not_an_identity(monkeypatch):
    # The runner turns this into "host state unreadable" and attempts nothing.
    ollama_answers(monkeypatch, URLError("connection refused"))
    with pytest.raises(URLError):
        ollama_model_identity("http://ollama", "qwen3:1.7b")


def test_the_injector_factory_builds_each_supported_method():
    make = injector_factory(endpoint="http://127.0.0.1:8000/v1", served_model_name=PINNED)
    clock = MonotonicClock()
    prefill = make(
        {
            "kind": "inference",
            "method": PREFILL_METHOD,
            "intensity": {"concurrency": 2, "prompt_tokens": 2048},
        },
        clock,
        60.0,
    )
    assert isinstance(prefill, PrefillFloodInjector)
    assert (prefill.model, prefill.concurrency, prefill.prompt_tokens) == (PINNED, 2, 2048)
    cpu = make({"kind": "cpu", "method": CPU_METHOD, "intensity": {"workers": 2}}, clock, 60.0)
    assert isinstance(cpu, CpuSpinInjector)
    with pytest.raises(ExperimentError, match="No injector for method 'qdrant-flood'"):
        make({"kind": "retrieval", "method": "qdrant-flood", "intensity": {}}, clock, 60.0)


def test_the_plan_refuses_a_run_without_the_policy(capsys):
    code = cli.main(
        ["optimization-plan", "--task", "fix-mean-empty", "--host-id", "test-host", "--no-policy"]
    )
    assert code == cli.EXIT_USAGE
    assert "always reads the policy" in capsys.readouterr().err


def test_the_plan_refuses_an_unknown_task(capsys):
    code = cli.main(["optimization-plan", "--task", "no-such-task", "--host-id", "test-host"])
    assert code == cli.EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_the_run_refuses_a_directory_without_an_experiment(tmp_path, capsys):
    code = cli.main(["optimization-run", str(tmp_path), "--host-id", "test-host"])
    assert code == cli.EXIT_USAGE
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        # Queue bounds are preregistered: the run takes them from the manifest only.
        ["--host-id", "test-host", "--inference-max-queue", "8"],
        # A host label is a neutral lowercase name, never a user or host name with capitals.
        ["--host-id", "Benoy_Laptop"],
    ],
)
def test_the_run_refuses_arguments_it_does_not_take(tmp_path, arguments):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["optimization-run", str(tmp_path), *arguments])
    assert stopped.value.code == 2
