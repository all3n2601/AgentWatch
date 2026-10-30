"""A loopback vLLM test double: chat completions, /v1/models, /metrics and request spans.

Everything it returns is synthetic and never recorded-run evidence. When given a traces file it
appends one OTLP JSON `llm_request` span per completed request, parented by the request's
`traceparent`, as the Compose collector would for the real server.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = "test/model@0000"
METRICS_TEXT = f"""\
vllm:num_requests_running{{engine="0",model_name="{MODEL}"}} 1.0
vllm:num_requests_waiting{{engine="0",model_name="{MODEL}"}} 0.0
vllm:kv_cache_usage_perc{{engine="0",model_name="{MODEL}"}} 0.1
vllm:request_queue_time_seconds_sum{{engine="0",model_name="{MODEL}"}} 0.001
vllm:request_queue_time_seconds_count{{engine="0",model_name="{MODEL}"}} 3.0
"""


def tool_reply(name: str, arguments: str) -> dict:
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def text_reply(content: str) -> dict:
    return {"content": content}


def span_line(trace_id: str, parent: str, request_id: str, queue: float = 0.002) -> str:
    def double(key, value):
        return {"key": key, "value": {"doubleValue": value}}

    span = {
        "traceId": trace_id,
        "spanId": "ab" * 8,
        "parentSpanId": parent,
        "name": "llm_request",
        "startTimeUnixNano": "1793142726788721408",
        "endTimeUnixNano": "1793142727286720031",
        "attributes": [
            double("gen_ai.latency.time_in_queue", queue),
            double("gen_ai.latency.time_in_model_prefill", 0.01),
            double("gen_ai.latency.time_in_model_decode", 0.02),
            double("gen_ai.latency.time_in_model_inference", 0.03),
            double("gen_ai.latency.time_to_first_token", 0.012),
            double("gen_ai.latency.e2e", 0.033),
            {"key": "gen_ai.usage.prompt_tokens", "value": {"intValue": "40"}},
            {"key": "gen_ai.usage.completion_tokens", "value": {"intValue": "5"}},
            {"key": "gen_ai.request.id", "value": {"stringValue": request_id}},
        ],
    }
    document = {"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}
    return json.dumps(document) + "\n"


class FakeVllm:
    def __init__(self, replies: list[dict], traces_file: Path | None = None):
        self.replies = list(replies)
        self.traces_file = traces_file
        self.requests: list[dict] = []
        self.traceparents: list[str | None] = []
        self.metrics_text = METRICS_TEXT
        self.omit_done = False  # end streams without their [DONE] marker
        self.error_mid_stream = False  # an error chunk replaces the finish chunk; [DONE] follows
        self.omit_finish = False  # never send a finish reason
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, body: bytes, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/metrics":
                    self._send(200, owner.metrics_text.encode(), "text/plain")
                elif self.path == "/v1/models":
                    data = [{"id": MODEL, "root": "test/model", "max_model_len": 4096}]
                    self._send(200, json.dumps({"data": data}).encode())
                else:
                    self._send(404, b"{}")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with owner._lock:
                    owner.requests.append(body)
                    owner.traceparents.append(self.headers.get("traceparent"))
                    reply = owner.replies.pop(0)
                    request_id = f"chatcmpl-{len(owner.requests)}"
                if body.get("stream"):
                    self._stream(reply, request_id)
                else:
                    self._send(200, json.dumps(owner.completion(reply, request_id)).encode())
                owner.record_span(self.headers.get("traceparent"), request_id)

            def _stream(self, reply, request_id):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in owner.chunks(reply, request_id):
                    finishing = any(c.get("finish_reason") for c in chunk.get("choices", []))
                    if finishing and owner.error_mid_stream:
                        chunk = {"error": {"message": "engine died", "code": 500}}
                    elif finishing and owner.omit_finish:
                        continue
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                if not owner.omit_done:
                    self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def completion(reply: dict, request_id: str) -> dict:
        calls = [
            {"id": f"x{i}", "type": "function", "function": call}
            for i, call in enumerate(reply.get("tool_calls", []))
        ]
        message = {"role": "assistant", "content": reply.get("content"), "tool_calls": calls}
        return {
            "id": request_id,
            "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 5},
        }

    @staticmethod
    def chunks(reply: dict, request_id: str):
        def chunk(delta, finish=None):
            choice = {"index": 0, "delta": delta, "finish_reason": finish}
            return {"id": request_id, "model": MODEL, "choices": [choice]}

        yield chunk({"role": "assistant", "content": ""})
        content = reply.get("content")
        if content:
            for word in content.split(" "):
                yield chunk({"content": word + " "})
        for index, call in enumerate(reply.get("tool_calls", [])):
            yield chunk({"tool_calls": [{"index": index, "function": {"name": call["name"]}}]})
            arguments = call["arguments"]
            for piece in (arguments[: len(arguments) // 2], arguments[len(arguments) // 2 :]):
                yield chunk({"tool_calls": [{"index": index, "function": {"arguments": piece}}]})
        yield chunk({}, finish="stop")
        yield {"id": request_id, "choices": [], "usage": {"prompt_tokens": 40, "total": 45}}

    def record_span(self, traceparent: str | None, request_id: str) -> None:
        if self.traces_file is None or not traceparent:
            return
        _version, trace_id, parent, _flags = traceparent.split("-")
        with self._lock, self.traces_file.open("a", encoding="utf-8") as stream:
            stream.write(span_line(trace_id, parent, request_id))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
