"""Injector test doubles: a test clock, child targets and a fake model server.

Kept out of conftest.py so spawned child processes can import them by a unique module name.
Everything here is synthetic and never experiment evidence.
"""

import json
import signal
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class TestClock:
    """A monotonic clock domain for tests, shaped like the agent runner's."""

    __test__ = False  # not a pytest test class

    def __init__(self, clock_id: str = "test-clock"):
        self.clock_id = clock_id
        self._origin = time.monotonic_ns()

    def tick(self) -> int:
        return time.monotonic_ns() - self._origin

    def now(self) -> datetime:
        return datetime.now(UTC)


def ignore_stop(stop_event, ready, reports, backstop_seconds) -> None:
    """A misbehaving child: ignores the stop event and SIGTERM, never reports.

    It is ready once SIGTERM is ignored. Its own 60 s limit only keeps a failed test from
    leaking it; the test kills it first.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.release()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        time.sleep(0.05)


def never_ready(stop_event, ready, reports, backstop_seconds) -> None:
    """A child that exits without ever applying load."""


def wait_until(predicate, timeout_seconds: float = 15.0) -> bool:
    """Poll a condition the code under test brings about; never a fixed sleep."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeOllama:
    """Test double for a loopback model server (Ollama's /api/generate by default, or any
    reply given). Synthetic, never evidence."""

    def __init__(
        self, status: int = 200, hold: threading.Event | None = None, reply: dict | None = None
    ):
        self.status = status
        self.hold = hold  # when set, requests block until it is released
        self.reply = reply if reply is not None else {"done": True, "eval_count": 5}
        self.payloads: list[dict] = []
        self.paths: list[str] = []
        self.held = threading.Semaphore(0)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.payloads.append(json.loads(body))
                owner.paths.append(self.path)
                if owner.hold is not None:
                    owner.held.release()
                    owner.hold.wait(30)
                reply = owner.reply if owner.status == 200 else {"error": "x"}
                data = json.dumps(reply).encode()
                try:
                    self.send_response(owner.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # the injector closed the connection: an abandoned request

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        if self.hold is not None:
            self.hold.set()
        self.server.shutdown()
        self.server.server_close()
