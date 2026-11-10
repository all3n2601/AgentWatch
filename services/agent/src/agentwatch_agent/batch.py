"""Run several agent tasks at once in one runner process, under one admission controller.

A batch is the unit the optimization contract compares (opt-contract-v1): tasks submitted
together to one runner. Each task is an ordinary live run with its own evidence; they share
this process's clock domain and admission controller, so the concurrency policy's limits
apply across them. Limits still count this process only (ADR 0010).

`batch.json` records each run's outcome and the process's admission state. Its wall time
covers the batch from the first run's setup to the last run's closure, so it includes setup
and closure work; the per-task times compared by an evaluation come from each run's steps.
"""

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agentwatch_agent.admission import AdmissionController, PolicyWatcher
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.runner import AgentRunner, RunOutcome, write_json_atomic

BATCH_METHOD = "agent-batch-v1"
BATCH_FILE = "batch.json"
MAX_BATCH_RUNS = 32
# After the runs close, permits held by abandoned model requests are given this long to come
# back; any still held are reported, never assumed returned.
DEFAULT_DRAIN_SECONDS = 30.0


@dataclass
class BatchRun:
    index: int
    runner: AgentRunner
    outcome: RunOutcome | None = None
    not_started: RunNotStarted | None = None
    error: BaseException | None = None

    def state(self) -> str | None:
        if self.outcome is not None:
            return self.outcome.state
        # A runner error leaves no outcome; its run state is then in its own evidence.
        return "not_started" if self.not_started is not None else None

    def evidence(self) -> dict:
        outcome = self.outcome
        return {
            "index": self.index,
            "task_id": self.runner.task.task_id,
            "output_dir": str(self.runner.output_dir),
            "run_id": str(outcome.run_id) if outcome else self.runner.evidence.get("run_id"),
            "state": self.state(),
            "reason": outcome.reason if outcome else None,
            "task_passed": outcome.task_passed if outcome else False,
            "delivery_complete": outcome.delivery_complete if outcome else None,
            "resend_needed": outcome.resend_needed if outcome else None,
            "records_refused": outcome.records_refused if outcome else None,
            "completion_refused": outcome.completion_refused if outcome else None,
            "not_started": (
                {"stage": self.not_started.stage, "error": str(self.not_started)[:500]}
                if self.not_started
                else None
            ),
            "error": f"{type(self.error).__name__}: {self.error}"[:500] if self.error else None,
        }


class AgentBatch:
    """Runs its runners concurrently and records the batch, whatever each run's outcome."""

    def __init__(
        self,
        runners: Sequence[AgentRunner],
        controller: AdmissionController | None,
        output_dir: Path,
        *,
        watcher: PolicyWatcher | None = None,
        drain_seconds: float = DEFAULT_DRAIN_SECONDS,
    ):
        if not 1 <= len(runners) <= MAX_BATCH_RUNS:
            raise ValueError(f"A batch has from 1 to {MAX_BATCH_RUNS} runs")
        clocks = {id(runner.clock) for runner in runners}
        if len(clocks) != 1:
            raise ValueError("A batch's runs share one clock domain: this process's")
        controllers = {
            id(runner.admission.controller if runner.admission is not None else None)
            for runner in runners
        }
        if controllers != {id(controller)}:
            raise ValueError("A batch's runs share one admission controller, or none")
        self.batch_id = str(uuid4())
        self.runs = [BatchRun(index, runner) for index, runner in enumerate(runners, 1)]
        self.controller = controller
        self.watcher = watcher
        self.output_dir = output_dir
        self.drain_seconds = drain_seconds

    @property
    def path(self) -> Path:
        return self.output_dir / BATCH_FILE

    def request_cancel_from_signal(self, source: str, reason: str) -> None:
        """Safe in a signal handler: each runner's cancellation is signal-safe."""
        for run in self.runs:
            run.runner.request_cancel_from_signal(source, reason)

    def run(self) -> list[BatchRun]:
        """Run every task; raise RunNotStarted (stage "policy") if the policy is unreadable.

        The policy is read once for the whole batch before any run starts, so an unavailable
        policy service fails the batch at once instead of once per run.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(UTC).isoformat()
        admission = self.runs[0].runner.admission
        if admission is not None:
            try:
                admission.ensure_policy()
            # Re-raised as RunNotStarted: nothing ran, and the batch record says why.
            except Exception as error:
                not_started = RunNotStarted(f"{type(error).__name__}: {error}", "policy")
                for run in self.runs:
                    run.not_started = not_started
                self._write(started_at, wall_seconds=0.0, permits_returned=None)
                raise not_started from error
        started = time.monotonic()
        threads = [
            threading.Thread(target=self._run_one, args=(run,), name=f"batch-run-{run.index}")
            for run in self.runs
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall_seconds = time.monotonic() - started
        returned = (
            self.controller.wait_until_idle(self.drain_seconds)
            if self.controller is not None
            else None
        )
        self._write(started_at, wall_seconds=wall_seconds, permits_returned=returned)
        return self.runs

    def _write(self, started_at: str, *, wall_seconds: float, permits_returned: bool | None):
        write_json_atomic(
            self.path,
            {
                "method": BATCH_METHOD,
                "batch_id": self.batch_id,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "batch_wall_seconds": wall_seconds,
                "clock_id": self.runs[0].runner.clock.clock_id,
                "runs": [run.evidence() for run in self.runs],
                "admission": (
                    self.controller.summary()
                    if self.controller is not None
                    else {"status": "not_configured"}
                ),
                "policy_watcher": self.watcher.summary() if self.watcher is not None else None,
                # True when every permit was returned once the runs closed; None without
                # admission or when nothing ran. A held permit is work still running.
                "permits_returned": permits_returned,
            },
        )

    @staticmethod
    def _run_one(run: BatchRun) -> None:
        try:
            run.outcome = run.runner.run()
        except RunNotStarted as not_started:
            run.not_started = not_started
        # Kept for the batch record; the caller re-raises it once every run is recorded.
        except Exception as error:  # noqa: BLE001
            run.error = error
