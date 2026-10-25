"""Retrieval contention: a bounded flood of searches against the runner's in-process index.

The agent's retrieval is a synchronous in-process BM25 index, not a networked service, so the
only executor a competing query can wait for is the runner process's interpreter. Flood threads
in the runner's own process search an identical index over the same documents; the runner's
own search then waits for the interpreter between their searches, which lengthens its measured
service time. That also slows every other thread of the runner (sampling, delivery, dispatch),
so this contention is not retrieval-exclusive, and the log says so. A vector database in its own
container would give retrieval contention of its own; none exists yet.
"""

import threading
import time
from collections.abc import Callable, Sequence

from agentwatch_injectors.base import (
    BACKSTOP_GRACE_SECONDS,
    STOP_GRACE_SECONDS,
    BoundedInjector,
    InjectionClock,
    require_count,
)

RETRIEVAL_METHOD = "in-process-index-query-flood-v1"
MAX_THREADS = 4
MAX_QUERIES = 64
MAX_QUERY_CHARS = 200


class IndexQueryFloodInjector(BoundedInjector):
    kind = "retrieval"
    method = RETRIEVAL_METHOD
    resource_class = "retrieval"

    def __init__(
        self,
        clock: InjectionClock,
        max_duration_seconds: float,
        *,
        search: Callable[[str], object],
        queries: Sequence[str],
        threads: int = 2,
    ):
        super().__init__(clock, max_duration_seconds)
        queries = tuple(queries)
        if not 1 <= len(queries) <= MAX_QUERIES or not all(
            isinstance(query, str) and query.strip() and len(query) <= MAX_QUERY_CHARS
            for query in queries
        ):
            raise ValueError(
                f"queries must be 1 to {MAX_QUERIES} nonblank strings of at most "
                f"{MAX_QUERY_CHARS} characters"
            )
        self.search = search
        self.queries = queries
        self.threads = require_count("threads", threads, MAX_THREADS)
        self._stop = threading.Event()
        self._workers: list[threading.Thread] = []
        # Separate from the lifecycle lock, which stop() holds while halting the threads.
        self._counts_lock = threading.Lock()
        self._searches = 0
        self._errors: list[str] = []

    def bounds(self) -> dict:
        return {"threads": self.threads, "queries": len(self.queries)}

    def target(self) -> dict:
        return {
            "scope": "runner process interpreter",
            "index": "identical in-process index over the task's documents",
            "detail": "slows every runner thread, not only retrieval",
        }

    def _flood(self, offset: int) -> None:
        # A backstop only: the watchdog stops the flood at its bound first, so a thread that
        # ended before the stop request really did stop contending early.
        deadline = time.monotonic() + self.max_duration_seconds + BACKSTOP_GRACE_SECONDS
        position = offset
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                self.search(self.queries[position % len(self.queries)])
            # Recorded as the injector's error; the thread stops instead of spinning on it.
            except Exception as error:  # noqa: BLE001
                with self._counts_lock:
                    self._errors.append(f"{type(error).__name__}: {error}"[:500])
                return
            position += 1
            with self._counts_lock:
                self._searches += 1

    def _launch(self) -> None:
        for index in range(self.threads):
            worker = threading.Thread(
                target=self._flood, args=(index,), name=f"retrieval-flood-{index}", daemon=True
            )
            self._workers.append(worker)
            worker.start()

    def _halt(self) -> tuple[dict, dict]:
        # A thread that ended before the stop request (a search error, or its own deadline)
        # stopped contending early; consumers must not treat the injection as sustained.
        running_at_stop = sum(1 for worker in self._workers if worker.is_alive())
        self._stop.set()
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        for worker in self._workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        running = sum(1 for worker in self._workers if worker.is_alive())
        with self._counts_lock:
            activity = {
                "threads_started": len(self._workers),
                "threads_running_at_stop": running_at_stop,
                "searches": self._searches,
                "search_errors": list(self._errors),
            }
        # A thread cannot be killed: one still running is reported, never assumed stopped.
        return activity, {"confirmed": running == 0, "threads_still_running": running}
