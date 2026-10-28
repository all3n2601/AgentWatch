"""vLLM's OpenAI-compatible chat API, behind the runner's canonical (Ollama-shaped) messages.

The runner records, replays and digests one message format. This module translates it at the
boundary: assistant tool calls get deterministic IDs that the following tool messages answer,
so a replay of the same messages sends byte-identical requests, and vLLM's reply (whose tool
arguments are a JSON string) comes back in the canonical shape. Model output stays untrusted:
arguments that are not a JSON object are passed through unparsed for the dispatcher to reject.

Each request can carry the step's W3C trace context. vLLM then records its own measured queue,
prefill and decode time for that request in a span whose parent is the step (`vllm_traces`).
"""

import json
import urllib.parse
import urllib.request
from collections.abc import Callable

VLLM_SERVICE_ID = "vllm"
CHAT_PATH = "/v1/chat/completions"
REQUEST_TIMING_METHOD = "vllm-request-trace-v1"
PLACEMENT_TIMEOUT_SECONDS = 10
# A streamed reply is assembled in memory; num_predict bounds generation far below this.
MAX_STREAM_BYTES = 16 * 1024 * 1024
# Canonical options with an OpenAI equivalent. num_ctx has none: the context is fixed by the
# server's --max-model-len, which the placement reading records and the runner checks.
OPTION_NAMES = {"temperature": "temperature", "num_predict": "max_tokens"}
# Server error text kept in a failure reason; the reply itself is untrusted.
MAX_ERROR_TEXT = 500
STREAM_EVENTS = ("response", "chunk", "output")


class VllmResponseError(ValueError):
    """vLLM answered, but not with a usable chat completion."""


def traceparent(trace_id: str, span_id: str) -> str:
    """W3C trace context naming the step's span as the parent of vLLM's request span."""
    return f"00-{trace_id}-{span_id}-01"


def _call_id(message_index: int, call_index: int) -> str:
    return f"call-{message_index}-{call_index}"


def openai_messages(messages: list[dict]) -> list[dict]:
    """Translate canonical messages; tool results answer the preceding assistant's calls."""
    translated: list[dict] = []
    pending: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            if pending:
                raise ValueError("an assistant message precedes the results of earlier calls")
            calls = message.get("tool_calls") or []
            entry: dict = {"role": "assistant", "content": message.get("content") or None}
            if calls:
                entry["tool_calls"] = []
                for call_index, call in enumerate(calls):
                    function = call.get("function") if isinstance(call, dict) else None
                    function = function if isinstance(function, dict) else {}
                    arguments = function.get("arguments")
                    call_id = _call_id(index, call_index)
                    pending.append(call_id)
                    entry["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": function.get("name"),
                                "arguments": arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments, allow_nan=False),
                            },
                        }
                    )
            translated.append(entry)
        elif role == "tool":
            if not pending:
                raise ValueError("a tool result does not answer any requested call")
            translated.append(
                {"role": "tool", "tool_call_id": pending.pop(0), "content": message["content"]}
            )
        else:
            if pending:
                raise ValueError("a message interrupts the results of requested calls")
            translated.append({"role": role, "content": message.get("content")})
    return translated


def chat_request(payload: dict, *, stream: bool) -> dict:
    """The OpenAI-compatible body for one canonical chat payload."""
    options = payload.get("options") or {}
    body: dict = {
        "model": payload["model"],
        "messages": openai_messages(payload["messages"]),
        "stream": stream,
        # Qwen3's chat template: the canonical `think: false` keeps replies bounded.
        "chat_template_kwargs": {"enable_thinking": bool(payload.get("think", False))},
    }
    if payload.get("tools"):
        body["tools"] = payload["tools"]
    for canonical, name in OPTION_NAMES.items():
        if canonical in options:
            body[name] = options[canonical]
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _arguments(raw: object) -> object:
    """Parse JSON-string arguments; text that is not JSON is passed through unparsed.

    Either way the dispatcher decides: anything but a JSON object is rejected there.
    """
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def canonical_response(
    content: str, calls: list[dict], finish_reason: object, details: dict
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"function": {"name": call.get("name"), "arguments": _arguments(call.get("arguments"))}}
            for call in calls
        ]
    return {"message": message, "vllm": details | {"finish_reason": finish_reason}}


def _first_choice(document: dict) -> dict:
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise VllmResponseError("completion has no choices")
    return choices[0]


def _details(document: dict) -> dict:
    return {
        "id": document.get("id"),
        "model": document.get("model"),
        "system_fingerprint": document.get("system_fingerprint"),
        "usage": document.get("usage") if isinstance(document.get("usage"), dict) else None,
    }


def parse_completion(document: object) -> dict:
    """A non-streamed completion in the canonical response shape."""
    if not isinstance(document, dict):
        raise VllmResponseError("completion is not a JSON object")
    choice = _first_choice(document)
    message = choice.get("message")
    if not isinstance(message, dict):
        raise VllmResponseError("completion choice has no message")
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        calls.append(function if isinstance(function, dict) else {})
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise VllmResponseError("completion content is not a string")
    return canonical_response(content, calls, choice.get("finish_reason"), _details(document))


class StreamAssembler:
    """Rebuild one reply from streamed chunks; tool-call fragments are joined by index."""

    def __init__(self):
        self.content: list[str] = []
        self.calls: dict[int, dict] = {}
        self.finish_reason: object = None
        self.document: dict = {}
        self.usage: dict | None = None

    def add(self, chunk: object) -> bool:
        """Apply one chunk; return whether it carried model output (text or a tool call)."""
        if not isinstance(chunk, dict):
            raise VllmResponseError("stream chunk is not a JSON object")
        # vLLM reports a failure after the 200 headers as an error chunk, then still sends
        # [DONE]: the reply so far is not the model's answer.
        if "error" in chunk or chunk.get("object") == "error":
            detail = chunk.get("error") if "error" in chunk else chunk.get("message")
            raise VllmResponseError(f"vLLM reported an error mid-stream: {detail}"[:MAX_ERROR_TEXT])
        if not self.document:
            self.document = chunk
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason") is not None:
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        output = False
        text = delta.get("content")
        if isinstance(text, str) and text:
            self.content.append(text)
            output = True
        for fragment in delta.get("tool_calls") or []:
            if not isinstance(fragment, dict) or not isinstance(fragment.get("index"), int):
                raise VllmResponseError("streamed tool call has no index")
            call = self.calls.setdefault(fragment["index"], {"name": None, "arguments": ""})
            function = (
                fragment.get("function") if isinstance(fragment.get("function"), dict) else {}
            )
            if isinstance(function.get("name"), str):
                call["name"] = function["name"]
            if isinstance(function.get("arguments"), str):
                call["arguments"] += function["arguments"]
            output = True
        return output

    def response(self) -> dict:
        if not self.document:
            raise VllmResponseError("stream ended without any chunk")
        # vLLM sets finish_reason on the last choice chunk of every completed reply.
        if self.finish_reason is None:
            raise VllmResponseError("stream ended without a finish reason")
        calls = [self.calls[index] for index in sorted(self.calls)]
        details = _details(self.document) | {"usage": self.usage}
        return canonical_response("".join(self.content), calls, self.finish_reason, details)


def _post(url: str, body: dict, headers: dict, timeout_seconds: float):
    request = urllib.request.Request(
        url.rstrip("/") + CHAT_PATH,
        data=json.dumps(body, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"} | headers,
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout_seconds)


def vllm_chat(
    url: str,
    payload: dict,
    *,
    timeout_seconds: float,
    trace_context: str | None = None,
    on_event: Callable[[str], None] | None = None,
) -> dict:
    """Send one canonical chat payload; stream it when `on_event` observes the chunks.

    `on_event` receives "response" when the reply's headers arrive, "chunk" for every data
    chunk and "output" when a chunk carries text or a tool call, so the caller can time them on
    its own clock. HTTP errors propagate (urllib raises them as OSError subclasses).
    """
    headers = {"traceparent": trace_context} if trace_context else {}
    stream = on_event is not None
    with _post(url, chat_request(payload, stream=stream), headers, timeout_seconds) as response:
        if not stream:
            return parse_completion(json.load(response))
        on_event("response")
        assembler, received, done = StreamAssembler(), 0, False
        while raw := response.readline(MAX_STREAM_BYTES - received + 1):
            # The read itself is bounded, so one endless line cannot exhaust memory.
            received += len(raw)
            if received > MAX_STREAM_BYTES:
                raise VllmResponseError(f"streamed reply exceeded {MAX_STREAM_BYTES} bytes")
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                done = True
                break
            output = assembler.add(json.loads(data))
            on_event("chunk")
            if output:
                on_event("output")
    if not done:
        raise VllmResponseError("stream ended before [DONE]")
    return assembler.response()


def vllm_models(url: str) -> dict:
    with urllib.request.urlopen(
        url.rstrip("/") + "/v1/models", timeout=PLACEMENT_TIMEOUT_SECONDS
    ) as response:
        return json.load(response)


def served_model(models: object, name: str) -> dict:
    """What vLLM reports for the served model: its source and fixed context length."""
    source = {"source": "vllm /v1/models"}
    data = models.get("data") if isinstance(models, dict) else None
    if not isinstance(data, list):
        return source | {"status": "malformed", "reason": "response has no data list"}
    entry = next((m for m in data if isinstance(m, dict) and m.get("id") == name), None)
    if entry is None:
        return source | {"status": "unavailable", "reason": "model is not served"}
    return source | {
        "status": "measured",
        "served_model": entry.get("id"),
        "root": entry.get("root"),
        "max_model_len": entry.get("max_model_len"),
    }


def vllm_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"expected an http(s) URL such as http://127.0.0.1:8000, got {value!r}")
    return value.rstrip("/")
