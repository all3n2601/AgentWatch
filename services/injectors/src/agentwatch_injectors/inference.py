"""Inference contention: competing generation requests to the run's own model server.

A child process keeps `concurrency` non-streaming `/api/generate` requests in flight against
the same Ollama endpoint and model the run uses, so they compete for the server's request slots
and its device. Only loopback endpoints are accepted: the injector loads designated local
capacity, never a shared or remote server. Each request is bounded by `num_predict` and a socket
timeout no longer than the time left. Stopping ends the child, which closes its connections;
requests still in flight are counted as abandoned, and Ollama stops generating for a closed
connection.

The competing requests use the run's `num_ctx` (or, like the run, none): a different context size
makes Ollama reload the model, which would measure model loading instead of contention.
"""

import http.client
import ipaddress
import json
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event, Semaphore

from agentwatch_injectors.base import (
    BACKSTOP_GRACE_SECONDS,
    BoundedInjector,
    InjectionClock,
    require_count,
    require_seconds,
)
from agentwatch_injectors.processes import ChildGroup, finished, halt_children, spawn_children

INFERENCE_METHOD = "competing-ollama-generate-v1"
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


def _generate(url: str, payload: dict, timeout_seconds: float) -> dict:
    request = urllib.request.Request(
        url + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("error") or result.get("done") is not True:
        raise ValueError("model server returned an unusable generate response")
    return result


def _compete(
    stop_event: Event,
    ready: Semaphore,
    reports: Connection,
    backstop_seconds: float,
    url: str,
    payload: dict,
    concurrency: int,
    request_timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + backstop_seconds
    lock = threading.Lock()
    counts: Counter = Counter()
    failures: Counter = Counter()
    eval_counts: list[int] = []

    def worker() -> None:
        while not finished(stop_event, deadline):
            timeout = min(request_timeout_seconds, deadline - time.monotonic())
            if timeout <= 0:
                break
            with lock:
                counts["started"] += 1
                counts["in_flight"] += 1
            try:
                result = _generate(url, payload, timeout)
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
                if isinstance(result.get("eval_count"), int):
                    eval_counts.append(result["eval_count"])

    for index in range(concurrency):
        # Daemon threads: a request blocked on its socket must not keep the child alive.
        threading.Thread(target=worker, name=f"competing-request-{index}", daemon=True).start()
    ready.release()
    while not stop_event.wait(STATUS_POLL_SECONDS):
        if time.monotonic() >= deadline:
            break
    with lock:
        reports.send(
            {
                "requests_started": counts["started"],
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
    ):
        super().__init__(clock, max_duration_seconds)
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
            "server": "ollama",
            "model": self.model,
            "endpoint_host": urllib.parse.urlsplit(self.url).hostname,
            "endpoint_port": urllib.parse.urlsplit(self.url).port,
        }

    def _backstop_seconds(self) -> float:
        return self.max_duration_seconds + BACKSTOP_GRACE_SECONDS

    def payload(self) -> dict:
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
        self._group = spawn_children(
            _compete,
            [(self.url, self.payload(), self.concurrency, self.request_timeout_seconds)],
            self._backstop_seconds(),
            "agentwatch-inference-injector",
        )

    def _halt(self) -> tuple[dict, dict]:
        if self._group is None:
            return {"requests_started": 0}, {"confirmed": True, "children": []}
        reports, cleanup = halt_children(self._group)
        report = reports[0]
        if report is None:
            # The child was stopped before it reported: its counts are unknown, not zero.
            return {"report": "missing"}, cleanup
        return {"report": "received"} | report, cleanup
