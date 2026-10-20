"""Model access: the Ollama adapter, strict reply parsing, and a cancellable wait."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from agentwatch_worker.ollama import (
    OLLAMA_METRIC_KEYS,
    observe_model_placement,
    ollama_chat,
    ollama_loaded_models,
)

from agentwatch_agent.errors import RunCancelled, RunFailure


class ChatModel(Protocol):
    name: str
    endpoint: str

    def chat(self, payload: dict) -> dict: ...
    def observe_placement(self) -> dict | None: ...


class OllamaChatModel:
    def __init__(self, endpoint: str, name: str):
        self.endpoint = endpoint
        self.name = name

    def chat(self, payload: dict) -> dict:
        return ollama_chat(self.endpoint, payload)

    def observe_placement(self) -> dict | None:
        """Ollama's reported memory split for this model; not utilization or saturation."""
        return observe_model_placement(lambda: ollama_loaded_models(self.endpoint), self.name)


class ModelResponseError(ValueError):
    """The model server's reply does not have the expected chat shape."""


@dataclass(frozen=True)
class RequestedCall:
    # Untrusted and unvalidated: the dispatcher decides whether this call may run.
    name: object
    arguments: object


@dataclass(frozen=True)
class ModelReply:
    history_message: dict
    content: str
    calls: tuple[RequestedCall, ...]
    metrics: dict


def parse_reply(response: object) -> ModelReply:
    if not isinstance(response, dict) or not isinstance(response.get("message"), dict):
        raise ModelResponseError("Chat response has no message object")
    message = response["message"]
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ModelResponseError("Chat message content is not a string")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ModelResponseError("Chat message tool_calls is not a list")
    calls = []
    for raw in raw_calls:
        function = raw.get("function") if isinstance(raw, dict) else None
        if isinstance(function, dict):
            calls.append(RequestedCall(function.get("name"), function.get("arguments")))
        else:
            calls.append(RequestedCall(None, None))
    history = {"role": "assistant", "content": content}
    if raw_calls:
        history["tool_calls"] = raw_calls
    metrics = {key: response[key] for key in OLLAMA_METRIC_KEYS if key in response}
    return ModelReply(history, content, tuple(calls), metrics)


def wait_interruptibly[Result](
    request: Callable[[], Result],
    *,
    cancel: threading.Event,
    deadline_monotonic: float,
    activity: str = "a model response",
    poll_seconds: float = 0.1,
) -> Result:
    """Run a blocking request while staying responsive to cancellation and the deadline.

    A synchronous HTTP request cannot be interrupted, so on cancellation or timeout the
    request thread is abandoned (daemon) and may finish in the background; for chat, Ollama
    stops generating within the request's num_predict bound. Callers record this as evidence.
    """
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["response"] = request()
        # Re-raised in the waiting thread below, so nothing is swallowed.
        except BaseException as error:  # noqa: BLE001
            outcome["error"] = error

    if cancel.is_set():
        raise RunCancelled(f"Cancelled before waiting for {activity}")
    if time.monotonic() >= deadline_monotonic:
        raise RunFailure(f"Run time limit reached before waiting for {activity}")
    thread = threading.Thread(target=target, name="interruptible-request", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(poll_seconds)
        if not thread.is_alive():
            break
        if cancel.is_set():
            raise RunCancelled(f"Cancelled while waiting for {activity}")
        if time.monotonic() >= deadline_monotonic:
            raise RunFailure(f"Run time limit reached while waiting for {activity}")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["response"]
