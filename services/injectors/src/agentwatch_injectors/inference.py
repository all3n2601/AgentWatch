"""Inference contention: competing generation requests to the run's own model server.

A child process keeps `concurrency` non-streaming requests in flight against the same model
server and model the run uses, so they compete for the server's request slots and its device:
Ollama's `/api/generate` (`competing-ollama-generate-v1`) or vLLM's OpenAI-compatible
`/v1/chat/completions` (`competing-openai-chat-v1`). Only loopback endpoints are accepted: the injector loads designated local
capacity, never a shared or remote server. Each request is bounded by `num_predict` and a socket
timeout no longer than the time left. The child is ready once every worker has sent its first
request, so an unreachable server fails the start. Stopping ends the child, which closes its
connections;
requests still in flight are counted as abandoned, and Ollama stops generating for a closed
connection.

The competing requests use the run's `num_ctx` (or, like the run, none): a different context size
makes Ollama reload the model, which would measure model loading instead of contention. vLLM
fixes its context when it starts, so a vLLM injector takes no `num_ctx`.
"""

import http.client
import ipaddress
import json
import threading
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event, Semaphore

from agentwatch_injectors.base import (
    BACKSTOP_GRACE_SECONDS,
    BoundedInjector,
    InjectionClock,
    require_count,
    require_seconds,
)
from agentwatch_injectors.processes import (
    ChildGroup,
    finished,
    halt_children,
    new_group,
    start_children,
)

INFERENCE_METHOD = "competing-ollama-generate-v1"
VLLM_INFERENCE_METHOD = "competing-openai-chat-v1"
# server -> (request path, method name)
SERVERS = {
    "ollama": ("/api/generate", INFERENCE_METHOD),
    "vllm": ("/v1/chat/completions", VLLM_INFERENCE_METHOD),
}
MAX_CONCURRENCY = 4
MAX_NUM_PREDICT = 256
MAX_REQUEST_TIMEOUT_SECONDS = 120.0
MAX_NUM_CTX = 32768
# After a failed request a worker waits this long, so a refused endpoint is not hammered.
FAILURE_BACKOFF_SECONDS = 0.5
STATUS_POLL_SECONDS = 0.1
# Fixed, non-sensitive text that keeps the model generating until num_predict.
COMPETING_PROMPT = (
    "List the integers from 1 to 1000 in words, one per line, without any other text."
)
LOOPBACK_NAMES = ("localhost",)


def loopback_url(value: str) -> str:
    """Accept only an http(s) URL whose host is a loopback address or `localhost`."""
    parsed = urllib.parse.urlsplit(value)
    host = parsed.hostname
    if parsed.scheme not in ("http", "https") or not host:
        raise ValueError(f"expected an http(s) model server URL, got {value!r}")
    if host not in LOOPBACK_NAMES:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError(
                "the inference injector only loads a loopback model server (designated local "
                f"capacity); refusing host {host!r}"
            )
    return value.rstrip("/")


class GenerateRejected(ValueError):
    """The model server answered, but not with a completed generation."""


def generated_tokens(server: str, result: object) -> int | None:
    """Tokens one completed request generated, or raise if it did not complete."""
    if not isinstance(result, dict) or result.get("error"):
        raise GenerateRejected("model server returned an unusable generate response")
    if server == "ollama":
        if result.get("done") is not True:
            raise GenerateRejected("model server returned an unusable generate response")
        count = result.get("eval_count")
    else:
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise GenerateRejected("model server returned a completion without choices")
        if choices[0].get("finish_reason") is None:
            raise GenerateRejected("model server returned an unfinished completion")
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        count = usage.get("completion_tokens")
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def _generate(
    url: str, path: str, payload: dict, timeout_seconds: float, on_sent: Callable[[], None]
) -> dict:
    """Send one generate request, call `on_sent` once it is on the wire, and await the reply."""
    parsed = urllib.parse.urlsplit(url)
    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout_seconds)
    try:
        connection.request(
            "POST",
            parsed.path.rstrip("/") + path,
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        on_sent()
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise GenerateRejected(f"model server answered HTTP {response.status}")
    return json.loads(body)


def _compete(
    stop_event: Event,
    ready: Semaphore,
    reports: Connection,
    backstop_seconds: float,
    url: str,
    server: str,
    payload: dict,
    concurrency: int,
    request_timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + backstop_seconds
    lock = threading.Lock()
    counts: Counter = Counter()
    failures: Counter = Counter()
    eval_counts: list[int] = []
    workers_sent: set[int] = set()

    def sent(index: int) -> None:
        with lock:
            counts["sent"] += 1
            first = index not in workers_sent
            workers_sent.add(index)
            # Ready (once) when every worker has a request on the wire: load is running.
            if first and len(workers_sent) == concurrency:
                ready.release()

    def worker(index: int) -> None:
        while not finished(stop_event, deadline):
            timeout = min(request_timeout_seconds, deadline - time.monotonic())
            if timeout <= 0:
                break
            with lock:
                counts["started"] += 1
                counts["in_flight"] += 1
            try:
                result = _generate(url, SERVERS[server][0], payload, timeout, lambda: sent(index))
                tokens = generated_tokens(server, result)
            except (OSError, http.client.HTTPException, ValueError) as error:
                with lock:
                    counts["failed"] += 1
                    failures[type(error).__name__] += 1
                    counts["in_flight"] -= 1
                stop_event.wait(FAILURE_BACKOFF_SECONDS)
                continue
            with lock:
                counts["completed"] += 1
                counts["in_flight"] -= 1
                if tokens is not None:
                    eval_counts.append(tokens)

    for index in range(concurrency):
        # Daemon threads: a request blocked on its socket must not keep the child alive.
        threading.Thread(
            target=worker, args=(index,), name=f"competing-request-{index}", daemon=True
        ).start()
    while not stop_event.wait(STATUS_POLL_SECONDS):
        if time.monotonic() >= deadline:
            break
    with lock:
        reports.send(
            {
                "requests_started": counts["started"],
                "requests_sent": counts["sent"],
                "requests_completed": counts["completed"],
                "requests_failed": counts["failed"],
                "failures_by_type": dict(failures),
                "requests_abandoned_in_flight": counts["in_flight"],
                "generated_tokens": sum(eval_counts),
                "stopped_by_backstop": not stop_event.is_set(),
            }
        )
    reports.close()


class CompetingInferenceInjector(BoundedInjector):
    kind = "inference"
    method = INFERENCE_METHOD
    resource_class = "inference"

    def __init__(
        self,
        clock: InjectionClock,
        max_duration_seconds: float,
        *,
        url: str,
        model: str,
        num_ctx: int | None,
        concurrency: int = 2,
        num_predict: int = 128,
        request_timeout_seconds: float = 60.0,
        server: str = "ollama",
    ):
        super().__init__(clock, max_duration_seconds)
        if server not in SERVERS:
            raise ValueError(f"server must be one of {', '.join(SERVERS)}")
        if server == "vllm" and num_ctx is not None:
            raise ValueError("num_ctx does not apply to vLLM, whose context is fixed at startup")
        self.server = server
        self.method = SERVERS[server][1]
        self.url = loopback_url(url)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonblank model name")
        self.model = model
        self.num_ctx = None if num_ctx is None else require_count("num_ctx", num_ctx, MAX_NUM_CTX)
        self.concurrency = require_count("concurrency", concurrency, MAX_CONCURRENCY)
        self.num_predict = require_count("num_predict", num_predict, MAX_NUM_PREDICT)
        self.request_timeout_seconds = require_seconds(
            "request_timeout_seconds", request_timeout_seconds, MAX_REQUEST_TIMEOUT_SECONDS
        )
        self._group: ChildGroup | None = None

    def bounds(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "request_timeout_seconds": self.request_timeout_seconds,
            "backstop_seconds": self._backstop_seconds(),
        }

    def target(self) -> dict:
        # Only the host and port are recorded, never credentials or paths from the URL.
        return {
            "scope": "loopback model server",
            "server": self.server,
            "model": self.model,
            "endpoint_host": urllib.parse.urlsplit(self.url).hostname,
            "endpoint_port": urllib.parse.urlsplit(self.url).port,
        }

    def _backstop_seconds(self) -> float:
        return self.max_duration_seconds + BACKSTOP_GRACE_SECONDS

    def payload(self) -> dict:
        if self.server == "vllm":
            return {
                "model": self.model,
                "messages": [{"role": "user", "content": COMPETING_PROMPT}],
                "max_tokens": self.num_predict,
                "temperature": 0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        options = {"temperature": 0, "num_predict": self.num_predict}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return {
            "model": self.model,
            "prompt": COMPETING_PROMPT,
            "stream": False,
            "think": False,
            "options": options,
        }

    def _launch(self) -> None:
        # Owned before the child starts, so a failed start halts and reports it.
        self._group = new_group()
        start_children(
            self._group,
            _compete,
            [
                (
                    self.url,
                    self.server,
                    self.payload(),
                    self.concurrency,
                    self.request_timeout_seconds,
                )
            ],
            self._backstop_seconds(),
            "agentwatch-inference-injector",
        )

    def _halt(self) -> tuple[dict, dict]:
        if self._group is None:
            return {"requests_started": 0}, {"confirmed": True, "children": []}
        reports, cleanup = halt_children(self._group)
        report = reports[0] if reports else None
        if report is None:
            # The child was stopped before it reported: its counts are unknown, not zero.
            return {"report": "missing"}, cleanup
        return {"report": "received"} | report, cleanup
