"""Model access: Ollama and vLLM adapters, stream timing, strict reply parsing, cancellable wait.

Both adapters take the runner's canonical (Ollama-shaped) payload and return the canonical
response, so recording, replay and digests do not depend on the backend. Streaming is a
transport choice: when the runner observes a call, the adapter streams it and reports each
chunk, so the runner times the reply's first output on its own clock.
"""

import http.client
import json
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

from agentwatch_worker.ollama import (
    CHAT_TIMEOUT_SECONDS,
    OLLAMA_METRIC_KEYS,
    observe_model_placement,
    ollama_chat,
    ollama_chat_stream,
    ollama_loaded_models,
    request_timing,
)

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.errors import RunCancelled, RunFailure
from agentwatch_agent.stats import p90
from agentwatch_agent.vllm import (
    REQUEST_TIMING_METHOD as VLLM_TIMING_METHOD,
)
from agentwatch_agent.vllm import (
    VLLM_SERVICE_ID,
    served_model,
    vllm_chat,
    vllm_models,
)

STREAM_TIMING_METHOD = "model-stream-timing-v1"
# num_predict (512 by default) bounds output chunks; the cap only guards evidence size.
MAX_RECORDED_OUTPUT_TICKS = 2048
VLLM_QUEUE_PENDING_REASON = "Waiting for vLLM's request span (joined after the run)"


class ChatModel(Protocol):
    name: str
    endpoint: str
    # "ollama" or "vllm": part of run configuration, so the two never share a workload digest.
    backend: str
    service_id: str

    def chat(
        self,
        payload: dict,
        *,
        trace_context: str | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> dict: ...
    def observe_placement(self) -> dict | None: ...
    def request_timing(self, metrics: dict, client_wall_seconds: float) -> dict: ...


class OllamaChatModel:
    backend = "ollama"
    service_id = "ollama"

    def __init__(self, endpoint: str, name: str):
        self.endpoint = endpoint
        self.name = name

    def chat(
        self,
        payload: dict,
        *,
        trace_context: str | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> dict:
        # Ollama does not accept trace context; its request timing comes from its reply.
        if on_event is None:
            return ollama_chat(self.endpoint, payload)
        return ollama_chat_stream(self.endpoint, payload, on_event)

    def observe_placement(self) -> dict | None:
        """Ollama's reported memory split for this model; not utilization or saturation."""
        return observe_model_placement(lambda: ollama_loaded_models(self.endpoint), self.name)

    def request_timing(self, metrics: dict, client_wall_seconds: float) -> dict:
        return request_timing(metrics, client_wall_seconds)


class VllmChatModel:
    """vLLM's OpenAI-compatible server; its own per-request timing arrives as a trace span."""

    backend = "vllm"
    service_id = VLLM_SERVICE_ID

    def __init__(self, endpoint: str, name: str, *, num_ctx: int | None = None):
        self.endpoint = endpoint
        self.name = name
        self.num_ctx = num_ctx

    def chat(
        self,
        payload: dict,
        *,
        trace_context: str | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> dict:
        return vllm_chat(
            self.endpoint,
            payload,
            timeout_seconds=CHAT_TIMEOUT_SECONDS,
            trace_context=trace_context,
            on_event=on_event,
        )

    def observe_placement(self) -> dict | None:
        """The served model and its fixed context length; vLLM reports no memory split."""
        observed_at = time.monotonic_ns()
        try:
            placement = served_model(vllm_models(self.endpoint), self.name)
        except (OSError, http.client.HTTPException) as error:
            placement = {
                "source": "vllm /v1/models",
                "status": "unavailable",
                "reason": f"models request failed: {type(error).__name__}",
            }
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            placement = {
                "source": "vllm /v1/models",
                "status": "malformed",
                "reason": f"models response is not valid UTF-8 JSON: {error}",
            }
        if placement.get("status") == "measured" and self.num_ctx is not None:
            # The run declares num_ctx, but vLLM fixes the context at startup. An unreported
            # length is unknown, not a mismatch.
            served = placement.get("max_model_len")
            placement["context_matches_num_ctx"] = (
                None if served is None else served == self.num_ctx
            )
        return placement | {"read_seconds": (time.monotonic_ns() - observed_at) / 1e9}

    def request_timing(self, metrics: dict, client_wall_seconds: float) -> dict:
        """The runner's side now; vLLM's measured breakdown is joined from its span later."""
        return {
            "method": VLLM_TIMING_METHOD,
            "client_wall_seconds": client_wall_seconds,
            "completion_id": (metrics.get("vllm") or {}).get("id"),
            "observed_queue_seconds": None,
            "observed_queue_availability": "unavailable",
            "observed_queue_reason": VLLM_QUEUE_PENDING_REASON,
            "status": "pending_trace",
        }


def joined_vllm_timing(call_timing: dict, trace: dict) -> dict:
    """Combine a call's runner-side timing with its joined vLLM span (`vllm_traces`)."""
    measured = trace.get("status") == "measured"
    queue = trace.get("queue_seconds") if measured else None
    total = trace.get("e2e_seconds") if measured else None
    wall = call_timing.get("client_wall_seconds")
    outside = None
    if total is not None and isinstance(wall, float | int):
        outside = wall - total if wall - total >= -1e-6 else None
    return call_timing | {
        "server_trace": trace,
        "server_total_seconds": total,
        "observed_queue_seconds": queue,
        "observed_queue_availability": "measured" if queue is not None else trace.get("status"),
        "observed_queue_reason": None
        if queue is not None
        else "; ".join(trace.get("problems") or ["no usable vLLM span"]),
        "client_outside_server_seconds": outside,
        "derived_differences": "estimates across runner and vLLM clocks",
        "status": "reported" if measured else trace.get("status"),
    }


class StreamTimer:
    """Runner-clock ticks of one streamed reply: request sent, headers, chunks and output.

    Called from the request thread; a request abandoned on cancellation may still report, so
    reads and writes share a lock. The ticks are the run's own clock domain, comparable with
    step and sample ticks. Time to first output includes the server's queue and prefill and
    the network; it is the runner's observation, not a server-measured phase.
    """

    def __init__(self, clock: MonotonicClock):
        self.clock = clock
        self._lock = threading.Lock()
        self.request_tick: int | None = None
        self.response_tick: int | None = None
        self.first_chunk_tick: int | None = None
        self.last_chunk_tick: int | None = None
        self.chunks = 0
        self.output_ticks: list[int] = []
        self.output_chunks = 0

    def __call__(self, event: str) -> None:
        tick = self.clock.tick()
        with self._lock:
            if event == "request":
                self.request_tick = tick
            elif event == "response":
                self.response_tick = tick
            elif event == "chunk":
                self.chunks += 1
                if self.first_chunk_tick is None:
                    self.first_chunk_tick = tick
                self.last_chunk_tick = tick
            elif event == "output":
                self.output_chunks += 1
                if len(self.output_ticks) < MAX_RECORDED_OUTPUT_TICKS:
                    self.output_ticks.append(tick)

    def summary(self) -> dict:
        with self._lock:
            sent, outputs = self.request_tick, list(self.output_ticks)
            response, chunks, output_chunks = self.response_tick, self.chunks, self.output_chunks
            first_chunk, last_chunk = self.first_chunk_tick, self.last_chunk_tick

        def since_sent(tick: int | None) -> float | None:
            return None if sent is None or tick is None else (tick - sent) / 1e9

        gaps = [(later - earlier) / 1e9 for earlier, later in pairwise(outputs)]
        return {
            "method": STREAM_TIMING_METHOD,
            "clock_id": self.clock.clock_id,
            "request_tick_ns": sent,
            "response_tick_ns": response,
            "first_output_tick_ns": outputs[0] if outputs else None,
            "last_output_tick_ns": outputs[-1] if outputs else None,
            "chunks": chunks,
            "output_chunks": output_chunks,
            "time_to_response_seconds": since_sent(response),
            "time_to_first_chunk_seconds": since_sent(first_chunk),
            "time_to_first_output_seconds": since_sent(outputs[0] if outputs else None),
            "time_to_last_chunk_seconds": since_sent(last_chunk),
            "output_span_seconds": (outputs[-1] - outputs[0]) / 1e9 if outputs else None,
            "output_gap_median_seconds": statistics.median(gaps) if gaps else None,
            "output_gap_p90_seconds": p90(gaps),
            "output_gap_max_seconds": max(gaps) if gaps else None,
            "output_offsets_seconds": [since_sent(tick) for tick in outputs],
            "output_offsets_truncated": output_chunks > len(outputs),
        }


def chat_payload(model: str, messages: list[dict], tools: list[dict], options: dict) -> dict:
    """The runner's canonical chat request; thinking is off so replies stay bounded.

    `stream` stays False here: whether a call streams is the adapter's transport choice
    (`model_transport`), so recorded payloads and replays do not depend on it.
    """
    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "think": False,
        "options": dict(options),
    }


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
    if isinstance(response.get("vllm"), dict):
        metrics["vllm"] = response["vllm"]
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

    if cancel.is_set() or time.monotonic() >= deadline_monotonic:
        interrupted = (
            RunCancelled(f"Cancelled before waiting for {activity}")
            if cancel.is_set()
            else RunFailure(f"Run time limit reached before waiting for {activity}")
        )
        interrupted.request_started = False
        raise interrupted
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
