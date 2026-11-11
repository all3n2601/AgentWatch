"""Drive one `natural-overload-sweep-v1` capacity sweep of one vLLM configuration.

Phases, in order: the configuration check, a warm-up batch (recorded, never analysed), the
ramp until a hard stop or the ceiling, the capacity decision, live confirmation at capacity
and the injected condition at `injected_level(capacity)`. Each batch is one `AgentBatch` with
its own `MonotonicClock`, shared by its runners and, when injected, its injection set. vLLM's
state is read before and after every batch, each reading retried `STATE_READ_ATTEMPTS`
times; after each batch the driver waits until vLLM is idle and then pauses `PACING_SECONDS`.

A hard stop in the ramp ends the ramp only; the later phases still run when capacity is at
least 1. A hard stop in the warm-up, a vLLM restart, unconfirmed injector cleanup, an injector
that failed to start, a live-API failure (runs never started, a run needing `resend`, which
outranks the others, or refused records), cancellation, a server whose state cannot be read,
or a server that is busy outside the ramp stop the whole sweep. Neither a live-API failure nor
an unreadable server is a capacity hard stop: the batch it affects is excluded from level
metrics, counted. A stopped sweep still analyses the ramp batches it recorded
(`analysis_scope: partial_ramp`). `sweep.json` is rewritten after every batch, so a stopped or
crashed sweep keeps what it recorded. There is no resume.
"""

import http.client
import itertools
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agentwatch_injectors.base import BoundedInjector, InjectorCleanupError, InjectorError
from agentwatch_injectors.injection_set import InjectionCancelled, InjectionSet

from agentwatch_agent.batch import AgentBatch, BatchRun
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.experiment import injection_active
from agentwatch_agent.replay import ReplayPlan, canonical_sha256
from agentwatch_agent.runner import write_json_atomic
from agentwatch_agent.session import CancellationState
from agentwatch_agent.sweep_analysis import (
    LIVE_API_FAILURE,
    batch_metrics_or_error,
    classify_run,
    live_api_problem,
    ramp_analysis,
    server_changes,
)
from agentwatch_agent.sweep_labels import ramp_label_results
from agentwatch_agent.sweep_protocol import (
    IDLE_POLL_SECONDS,
    IDLE_WAIT_SECONDS,
    INJECTOR_BOUND_MARGIN_SECONDS,
    INJECTOR_RAMP_SECONDS,
    PACING_SECONDS,
    RAMP_LEVELS,
    SOURCE_COUNT,
    STATE_READ_ATTEMPTS,
    STATE_READ_RETRY_PAUSE_SECONDS,
    WARMUP_BATCH,
    Batch,
    VllmConfiguration,
    injected_batches,
    injected_level,
    live_batches,
    protocol,
    protocol_sha256,
    ramp_batches,
)
from agentwatch_agent.vllm_server import (
    NOT_INSPECTED,
    ServerState,
    ServerStateUnavailable,
    verify_configuration,
)

SWEEP_FILE = "sweep.json"
SWEEP_PROCESS_SCOPE = (
    "time.process_time and time.monotonic around the batch's run: the whole sweep process "
    "(every runner thread, sampler and the driver), not one run"
)
SWEEP_METHOD = "capacity-sweep-v1"
MAX_ERROR_TEXT = 1000

# Capacity hard-stop codes. A live-API problem is never one (owner decision 2026-11-11), nor
# is a server whose state cannot be read (Ruling 18): both stop the sweep instead.
INFRASTRUCTURE_FAILURE = "infrastructure_failure"
SERVER_BUSY = "server_busy_after_idle_wait"
VLLM_RESTART = "vllm_restart"
INJECTOR_CLEANUP_UNCONFIRMED = "injector_cleanup_unconfirmed"
# The server's own state: a busy server stops the sweep in every phase after the ramp.
SERVER_EVENTS = (SERVER_BUSY,)
# vLLM's /metrics could not be read within the declared retries: a stop cause, not a hard stop.
SERVER_STATE_UNAVAILABLE = "server_state_unavailable"

# Sweep states. Exit codes (mapped by the CLI): completed 0, refused_configuration 2,
# stopped_resend_needed 3, every other state 1.
COMPLETED = "completed"
REFUSED_CONFIGURATION = "refused_configuration"
STOPPED_VLLM_RESTART = "stopped_vllm_restart"
STOPPED_INJECTOR_CLEANUP = "stopped_injector_cleanup_unconfirmed"
STOPPED_RUN_NOT_STARTED = "stopped_run_not_started"
STOPPED_RESEND_NEEDED = "stopped_resend_needed"
# Records or completion refused, or the API unreachable at a run's start.
STOPPED_LIVE_API_FAILURE = "stopped_live_api_failure"
STOPPED_INJECTOR_START_FAILED = "stopped_injector_start_failed"
# A hard stop in the warm-up: the ramp never started, so nothing about capacity was measured.
STOPPED_WARMUP_FAILED = "stopped_warmup_failed"
# A busy server outside the ramp, or an unreadable one anywhere, stops the sweep: the next
# batch would start on a server whose state is unknown or still loaded.
STOPPED_SERVER_BUSY = f"stopped_{SERVER_BUSY}"
STOPPED_SERVER_STATE_UNAVAILABLE = f"stopped_{SERVER_STATE_UNAVAILABLE}"
CANCELLED = "cancelled"
FAILED = "failed"
# Stops caused by the operator, the live API or an unreadable server, not by the server under
# test: a batch they affect is excluded from level metrics (counted) instead of read as a slow
# or failing level.
NOT_HARD_STOP_STATES = (
    CANCELLED,
    STOPPED_RUN_NOT_STARTED,
    STOPPED_RESEND_NEEDED,
    STOPPED_LIVE_API_FAILURE,
    STOPPED_SERVER_STATE_UNAVAILABLE,
)
# The exclusion cause of a batch with a `live_api_failure` run.
LIVE_API_EXCLUSION = "live_api"

# How the ramp ended, recorded separately from the sweep state.
RAMP_HARD_STOP = "hard_stop"
RAMP_CEILING = "ceiling"
RAMP_STOPPED = "stopped"
RAMP_CANCELLED = "cancelled"

FULL_RAMP = "full_ramp"
PARTIAL_RAMP = "partial_ramp"
NO_ANALYSIS = "none"

CAPACITY_DECISION = "capacity_decision"
LABEL_VALIDATION = "label_validation"
ANALYSIS_FAILED = "analysis_error"  # the capacity_decision phase status when it raised

PHASES = ("configuration_check", "warmup", "ramp", "capacity_decision", "live", "injected")
RAMP_PHASES = ("warmup", "ramp")
# Runs in these states were cut short by the sweep's own stop, so they are no hard stop.
STOPPED_RUN_STATES = (CANCELLED, "not_started")

BatchFactory = Callable[[Batch, Path, MonotonicClock], AgentBatch]
InjectorFactory = Callable[[MonotonicClock, float], list[BoundedInjector]]


def analysis_scope(record: dict) -> str:
    """What an analysis of this `sweep.json` record covers; offline uses the same mapping.

    `none` when no batch ran, `full_ramp` when the ramp ended at a hard stop or the ceiling,
    otherwise `partial_ramp` (the sweep stopped during the warm-up or ramp).
    """
    if record["phases"]["warmup"]["status"] == "not_run":
        return NO_ANALYSIS
    return FULL_RAMP if record.get("ramp_end") in (RAMP_CEILING, RAMP_HARD_STOP) else PARTIAL_RAMP


class _SweepStop(Exception):
    """Ends the sweep with `state`; raised once the evidence that caused it is recorded."""

    def __init__(self, state: str, reason: str):
        super().__init__(reason)
        self.state = state
        self.reason = reason


@dataclass
class _Executed:
    runs: list[BatchRun] = field(default_factory=list)
    not_started: RunNotStarted | None = None
    cleanup_error: InjectorCleanupError | None = None
    # The flood failed to start (its own cleanup confirmed); the batch never ran.
    start_error: InjectorError | None = None
    cancelled: bool = False


def _text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:MAX_ERROR_TEXT]


class CapacitySweep:
    def __init__(
        self,
        *,
        configuration: VllmConfiguration,
        sources: list[ReplayPlan],
        output_dir: Path,
        run_timeout_seconds: float,
        make_batch: BatchFactory,
        make_live_batch: BatchFactory,
        make_injectors: InjectorFactory,
        read_state: Callable[[], ServerState],
        models: Callable[[], dict],
        host: dict,
        read_launch: Callable[[], dict] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        process_time: Callable[[], float] = time.process_time,
        start_attempts: list[dict] | None = None,
        code_revision: dict | None = None,
    ):
        if len(sources) != SOURCE_COUNT:
            raise ValueError(f"A sweep replays exactly {SOURCE_COUNT} sources")
        self.configuration = configuration
        self.sources = list(sources)
        # Absolute once, so every runner records an absolute folder: a relative one would only
        # resolve from the working directory the sweep was started in.
        self.output_dir = output_dir.resolve()
        self.run_timeout_seconds = run_timeout_seconds
        self.make_batch = make_batch
        self.make_live_batch = make_live_batch
        self.make_injectors = make_injectors
        self.read_state = read_state
        self.models = models
        # The vLLM container's launch arguments (`vllm_launch`); never raises.
        self.read_launch = read_launch
        self.host = host
        self.sleep = sleep
        self.monotonic = monotonic
        self.process_time = process_time
        self.start_attempts = start_attempts
        self.code_revision = code_revision
        self.cancellation = CancellationState()
        self.record: dict = {}
        self._current: AgentBatch | None = None
        self._phase = PHASES[0]
        self._sequence = 0
        self._start_time_seconds: float | None = None
        self._idle_confirmed = False

    @property
    def path(self) -> Path:
        return self.output_dir / SWEEP_FILE

    def request_cancel_from_signal(self, source: str, reason: str) -> None:
        """Signal-safe: cancel the current batch and any injector, and run nothing further."""
        self.cancellation.cancel_from_signal(source, reason)
        batch = self._current
        if batch is not None:
            batch.request_cancel_from_signal(source, reason)

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record = self._initial_record()
        self._save()
        try:
            try:
                self._sweep()
                self.record["state"] = COMPLETED
            except _SweepStop as stop:
                self._record_stop(stop)
            self._analyse()
        except BaseException as error:
            batch = self._current
            if batch is not None:
                batch.request_cancel_from_signal("sweep", f"Sweep failed: {type(error).__name__}")
            self.record |= {"state": FAILED, "error": _text(error)}
            self._end_phase(FAILED)
            # The failure may have left evidence half-written: analyse the folder offline.
            if self.record["label_validation"] is None:
                done = "the capacity decision ran; " if self.record["capacity_decision"] else ""
                self.record |= {
                    "analysis_scope": self.record["analysis_scope"] or NO_ANALYSIS,
                    "analysis_note": f"{done}label validation not run: the sweep failed "
                    f"({_text(error)})",
                }
            raise
        finally:
            self.record["finished_at"] = datetime.now(UTC).isoformat()
            self._save()
        return self.record

    def _initial_record(self) -> dict:
        document = protocol(self.configuration, self.run_timeout_seconds)
        return {
            "method": SWEEP_METHOD,
            "sweep_id": str(uuid4()),
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
            "protocol": document,
            "protocol_sha256": protocol_sha256(document),
            # Recorded, not hashed: which code ran, and the exact replay plans it ran.
            "code_revision": self.code_revision or {"status": "not_recorded"},
            "workload_sha256": canonical_sha256([plan.sha256() for plan in self.sources]),
            # Held-out test evidence: never for training, calibration or threshold tuning.
            "held_out": True,
            "configuration": {
                "declared": document["configuration"],
                "verified": None,
                "launch": None,
            },
            "start_attempts": self.start_attempts,
            "host": self.host,
            "sources": [
                {
                    "index": index,
                    "task_id": plan.task_id,
                    "source_run_id": plan.source_run_id,
                    "source_evidence_sha256": plan.source_evidence_sha256,
                    "plan_sha256": plan.sha256(),
                }
                for index, plan in enumerate(self.sources)
            ],
            "admission": {"status": "not_configured"},
            # Readings that failed within their declared retries: {reading, phase, level,
            # batch, attempts, error}.
            "server": {"initial_state": None, "read_failures": []},
            "phases": {phase: {"status": "not_run"} for phase in PHASES},
            "batches": [],
            "injection_logs": [],
            "hard_stops": [],
            "level_summaries": None,
            # Ramp batches left out of level metrics, each with its level and causes.
            "excluded_batches": [],
            "capacity_decision": None,
            "label_validation": None,
            "max_running_check": None,
            "state": "running",
            "ramp_end": None,
            "stop_reason": None,
            # Every cause found when the sweep stopped; `state` is the one that ranks first.
            "stop_causes": [],
            # full_ramp, partial_ramp (the sweep stopped during the ramp) or none.
            "analysis_scope": None,
            "analysis_note": None,
            # Analysis stages that raised: {stage, error}. The sweep's state is never replaced.
            "analysis_errors": [],
            "cancellation": None,
            "error": None,
        }

    def _save(self) -> None:
        write_json_atomic(self.path, self.record)

    def _begin(self, phase: str, **details: object) -> None:
        self._phase = phase
        self.record["phases"][phase] = {"status": "running", **details}

    def _complete(self, phase: str) -> None:
        self.record["phases"][phase]["status"] = COMPLETED

    def _end_phase(self, status: str) -> None:
        current = self.record["phases"][self._phase]
        if current["status"] == "running":
            current["status"] = status
        if self._phase in RAMP_PHASES and self.record["ramp_end"] is None:
            self.record["ramp_end"] = RAMP_CANCELLED if status == CANCELLED else RAMP_STOPPED

    def _record_stop(self, stop: _SweepStop) -> None:
        self.record |= {"state": stop.state, "stop_reason": stop.reason}
        if not self.record["stop_causes"]:
            self.record["stop_causes"] = [{"state": stop.state, "reason": stop.reason}]
        request = self.cancellation.request
        if stop.state == CANCELLED and request is not None:
            self.record["cancellation"] = {"source": request.source, "reason": request.reason}
        if stop.state == REFUSED_CONFIGURATION:
            self.record["phases"]["configuration_check"]["status"] = "refused"
        else:
            self._end_phase(CANCELLED if stop.state == CANCELLED else "stopped")

    # Phases

    def _sweep(self) -> None:
        self._check_configuration()
        self._ramp()
        decided = self._try_analysis(
            CAPACITY_DECISION, lambda: self._decide_capacity(analysis_scope(self.record))
        )
        decision = self.record["capacity_decision"] if decided else None
        capacity = decision["capacity"] if decision is not None else None
        if capacity is None or capacity < 1:
            reason = (
                f"capacity below 1 or not measured (result {decision['result']})"
                if decision is not None
                else "the capacity decision failed (see analysis_errors)"
            )
            for phase in ("live", "injected"):
                self.record["phases"][phase] = {"status": "skipped", "reason": reason}
            return
        self._begin("live", level=capacity)
        self._ensure_idle("live", capacity)
        for batch in live_batches(capacity):
            self._post_ramp_batch(batch, self._factory(self.make_live_batch, batch))
        self._complete("live")
        level = injected_level(capacity)
        self._begin("injected", level=level)
        self._ensure_idle("injected", level)
        for batch in injected_batches(level):
            self._post_ramp_batch(batch, self._replay_factory(batch))
        self._complete("injected")

    def _check_configuration(self) -> None:
        self._begin("configuration_check")
        try:
            state = self._read_state("configuration_check", "configuration_check", None, None)
        except ServerStateUnavailable as error:
            raise _SweepStop(STOPPED_SERVER_STATE_UNAVAILABLE, str(error)) from error
        try:
            models = self.models()
        # The /v1/models read is bounded like /metrics: URLError and timeouts are OSErrors and
        # a body that is not JSON is a ValueError.
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise _SweepStop(
                STOPPED_SERVER_STATE_UNAVAILABLE, f"vLLM /v1/models unavailable: {_text(error)}"
            ) from error
        self._start_time_seconds = state.start_time_seconds
        launch = self.read_launch() if self.read_launch is not None else NOT_INSPECTED
        self.record["configuration"]["launch"] = launch
        verified = verify_configuration(self.configuration, models, state, launch)
        self.record["server"]["initial_state"] = asdict(state)
        self.record["configuration"]["verified"] = verified
        problems = []
        if verified["contradicted"]:
            problems.append(f"vLLM contradicts the declared {', '.join(verified['contradicted'])}")
        if state.start_time_seconds is None:
            problems.append(
                "vLLM reported no process_start_time_seconds, so a restart could not be detected"
            )
        problems.extend(self._start_attempt_problems(verified))
        if problems:
            raise _SweepStop(REFUSED_CONFIGURATION, "; ".join(problems))
        self._complete("configuration_check")

    def _start_attempt_problems(self, verified: dict) -> list[str]:
        """The recorded start attempts must describe the vLLM that is running.

        The attempt that started must be the utilization vLLM reports, and a utilization other
        than the first declared one (a fallback) needs the attempts that led to it.
        """
        reported = verified["settings"]["gpu_memory_utilization"]["reported"]
        first = self.configuration.gpu_memory_utilization[0]
        problem = None
        if self.start_attempts:
            started = self.start_attempts[-1]["gpu_memory_utilization"]
            if reported is None:
                problem = (
                    f"the start attempt that started used {started:g}, but vLLM reported no "
                    "gpu_memory_utilization to check it against"
                )
            elif not math.isclose(started, reported):
                problem = (
                    f"the start attempt that started used {started:g}, but vLLM reports "
                    f"gpu_memory_utilization {reported:g}"
                )
        elif reported is not None and not math.isclose(reported, first):
            problem = (
                f"vLLM runs at gpu_memory_utilization {reported:g}, not the first declared "
                f"{first:g}: record the start attempts that led to it (--start-attempt)"
            )
        return [] if problem is None else [problem]

    def _ramp(self) -> None:
        self._begin("warmup")
        # `_run_batch` records its hard-stop events itself, before the idle wait can stop.
        events = self._run_batch(WARMUP_BATCH, self._replay_factory(WARMUP_BATCH))
        if events:
            codes = ", ".join(event["code"] for event in events)
            raise _SweepStop(STOPPED_WARMUP_FAILED, f"hard stop in the warm-up batch: {codes}")
        self._complete("warmup")
        self._begin("ramp")
        for batch in itertools.chain.from_iterable(ramp_batches(n) for n in RAMP_LEVELS):
            events = self._run_batch(batch, self._replay_factory(batch))
            if events:
                self.record["ramp_end"] = RAMP_HARD_STOP
                break
        else:
            self.record["ramp_end"] = RAMP_CEILING
        self._complete("ramp")

    def _analyse(self) -> None:
        """Analyse the recorded ramp on every outcome but a failure; mark a partial ramp.

        A sweep stopped or cancelled during the warm-up or ramp still gets level summaries,
        the capacity decision and label validation over the ramp batches it recorded. An
        analysis error is recorded in `analysis_errors` and never replaces the sweep's state.
        """
        if self.record["phases"]["warmup"]["status"] == "not_run":
            self.record |= {
                "analysis_scope": NO_ANALYSIS,
                "analysis_note": f"no batch ran ({self.record['state']})",
            }
            return
        if self.record["phases"][CAPACITY_DECISION]["status"] == "not_run":
            self.record["analysis_note"] = (
                f"ramp incomplete, {self.record['state']}: {self.record['stop_reason']}"
            )
            self._try_analysis(
                CAPACITY_DECISION, lambda: self._decide_capacity(analysis_scope(self.record))
            )
        self._try_analysis(LABEL_VALIDATION, self._validate_labels)
        self._save()

    def _try_analysis(self, stage: str, step: Callable[[], object]) -> bool:
        """Run one analysis stage; record its error instead of losing the sweep's state.

        The evidence on disk is untouched, so `sweep-analyse` can analyse the folder again.
        """
        try:
            step()
        # Any analysis bug: recorded in analysis_errors and the exit code, never dropped.
        except Exception as error:  # noqa: BLE001
            self.record["analysis_errors"].append({"stage": stage, "error": _text(error)})
            if stage == CAPACITY_DECISION:
                self.record["phases"][CAPACITY_DECISION]["status"] = ANALYSIS_FAILED
            note = self.record["analysis_note"]
            self.record["analysis_note"] = (
                f"{note + '; ' if note else ''}{stage} failed ({_text(error)}); analyse the "
                "folder again with `sweep-analyse`"
            )
            self._save()
            return False
        return True

    def _decide_capacity(self, scope: str) -> dict:
        self._begin(CAPACITY_DECISION, scope=scope)
        self.record["analysis_scope"] = scope
        analysis = ramp_analysis(
            self.record["batches"],
            self.record["hard_stops"],
            self.run_timeout_seconds,
            lambda entry: entry["metrics"],
        )
        self.record |= analysis
        self._complete(CAPACITY_DECISION)
        self._save()
        return analysis["capacity_decision"]

    def _validate_labels(self) -> None:
        self.record |= ramp_label_results(
            self.output_dir, self.record["batches"], self.configuration.max_num_seqs
        )

    # Batches

    def _replay_factory(self, batch: Batch) -> Callable[[Path, MonotonicClock], AgentBatch]:
        return self._factory(self.make_batch, batch)

    @staticmethod
    def _factory(make: BatchFactory, batch: Batch) -> Callable[[Path, MonotonicClock], AgentBatch]:
        return lambda path, clock: make(batch, path, clock)

    def _post_ramp_batch(
        self, batch: Batch, make: Callable[[Path, MonotonicClock], AgentBatch]
    ) -> None:
        """Run a live, injected or control batch; a server event stops the sweep.

        Infrastructure failures there are evidence of the condition, recorded on the batch;
        `_run_batch` records a busy server as a hard stop.
        """
        events = self._run_batch(batch, make)
        for event in events:
            if event["code"] in SERVER_EVENTS:
                raise _SweepStop(f"stopped_{event['code']}", event["detail"])

    def _ensure_idle(self, phase: str, level: int) -> None:
        """Before a phase after the ramp: the server must be idle and paced, as between batches."""
        if self._idle_confirmed:
            return
        result = self._wait_until_idle(phase, level, None)
        if result is not None:
            code, detail = result
            if code == SERVER_STATE_UNAVAILABLE:
                raise _SweepStop(STOPPED_SERVER_STATE_UNAVAILABLE, detail)
            event = self._event(code, phase, level, None, detail)
            self.record["hard_stops"].append(event)
            raise _SweepStop(
                STOPPED_VLLM_RESTART if code == VLLM_RESTART else f"stopped_{code}", detail
            )
        self._idle_confirmed = True
        self.sleep(PACING_SECONDS)

    def _event(self, code: str, phase: str, level: int, sequence: int | None, detail: str) -> dict:
        return {"code": code, "phase": phase, "level": level, "batch": sequence, "detail": detail}

    def _read_failed(
        self,
        reading: str,
        phase: str,
        level: int | None,
        sequence: int | None,
        attempts: int,
        error: str,
    ) -> None:
        self.record["server"]["read_failures"].append(
            {
                "reading": reading,
                "phase": phase,
                "level": level,
                "batch": sequence,
                "attempts": attempts,
                "error": error[:MAX_ERROR_TEXT],
            }
        )

    def _read_state(
        self, reading: str, phase: str, level: int | None, sequence: int | None
    ) -> ServerState:
        """One reading, tried `STATE_READ_ATTEMPTS` times with the declared pause between.

        Raises the last `ServerStateUnavailable` once every attempt failed, recorded in
        `server.read_failures`. Bounded: each read has the `/metrics` timeout.
        """
        for attempt in range(1, STATE_READ_ATTEMPTS + 1):
            try:
                return self.read_state()
            except ServerStateUnavailable as error:
                last = error
                if attempt < STATE_READ_ATTEMPTS:
                    self.sleep(STATE_READ_RETRY_PAUSE_SECONDS)
        self._read_failed(reading, phase, level, sequence, STATE_READ_ATTEMPTS, str(last))
        raise last

    def _record_hard_stops(self, batch: Batch, events: list[dict]) -> None:
        """Add a batch's hard-stop events to `hard_stops`, once.

        In the warm-up and ramp every event is a hard stop. After the ramp, a run's
        infrastructure failure is evidence of the condition (kept on the batch only); a busy
        server stops the sweep and is recorded.
        """
        self.record["hard_stops"].extend(
            event
            for event in events
            if batch.phase in RAMP_PHASES or event["code"] in SERVER_EVENTS
        )

    def _stop_with(self, state: str, event: dict, reason: str) -> None:
        self.record["hard_stops"].append(event)
        raise _SweepStop(state, reason)

    def _check_cancelled(self) -> None:
        if self.cancellation.event.is_set():
            request = self.cancellation.request
            raise _SweepStop(CANCELLED, request.reason if request else "Cancelled")

    def _restart_detail(self, state: ServerState) -> str | None:
        """Why `state` shows a restarted vLLM, or None. An unreported start time cannot."""
        first, now = self._start_time_seconds, state.start_time_seconds
        if first is None or now is None or now == first:
            return None
        return f"vLLM process_start_time_seconds changed from {first} to {now}"

    def _run_batch(
        self, batch: Batch, make: Callable[[Path, MonotonicClock], AgentBatch]
    ) -> list[dict]:
        """Run one batch between server readings; return its hard-stop events.

        Raises `_SweepStop` for a condition that stops the sweep, and re-raises a runner's
        unexpected error, each once the batch is recorded in `sweep.json`. A reading that fails
        every retry stops the sweep and is never a hard stop: before the batch, nothing ran;
        after it, the batch is excluded from level metrics.
        """
        self._check_cancelled()
        try:
            before = self._read_state("before", batch.phase, batch.level, None)
        except ServerStateUnavailable as error:
            raise _SweepStop(
                STOPPED_SERVER_STATE_UNAVAILABLE,
                f"vLLM state unreadable before a {batch.phase} batch at N = {batch.level}: {error}",
            ) from error
        restart = self._restart_detail(before)
        if restart is not None:
            event = self._event(VLLM_RESTART, batch.phase, batch.level, None, restart)
            self._stop_with(STOPPED_VLLM_RESTART, event, restart)

        sequence = self._sequence
        self._sequence += 1
        name = f"{sequence:02d}-{batch.phase}-n{batch.level}-r{batch.repeat}"
        batch_dir = self.output_dir / "batches" / name
        clock = MonotonicClock()
        agent_batch = make(batch_dir, clock)
        self._current = agent_batch
        # A cancellation that arrived after the check above found no batch to forward to; from
        # here on `request_cancel_from_signal` reaches this batch itself.
        if self.cancellation.event.is_set():
            self._current = None
            self._check_cancelled()
        entry = self._batch_entry(batch, sequence, name, agent_batch, clock)
        self.record["batches"].append(entry)
        self._save()
        cpu_before, wall_before = self.process_time(), self.monotonic()
        try:
            executed = self._execute(agent_batch, batch, clock, entry, sequence)
        finally:
            entry["sweep_process"] = {
                "cpu_seconds": self.process_time() - cpu_before,
                "wall_seconds": self.monotonic() - wall_before,
                "scope": SWEEP_PROCESS_SCOPE,
            }
        self._idle_confirmed = False
        try:
            after: ServerState | None = self._read_state(
                "after", batch.phase, batch.level, sequence
            )
            after_error = None
        except ServerStateUnavailable as error:
            after, after_error = None, str(error)
        self._current = None
        runs = entry["runs"]
        entry["server"] = {
            "before": asdict(before),
            "after": asdict(after) if after is not None else None,
            "after_error": after_error,
            "changes": server_changes(asdict(before), asdict(after)) if after else None,
        }
        metrics = batch_metrics_or_error(batch_dir, self.run_timeout_seconds)
        entry["wall_seconds"] = metrics.get("wall_seconds")
        if entry["analysed"]:
            entry["metrics"] = metrics
        if executed.not_started is not None:
            entry["not_started"] = {
                "stage": executed.not_started.stage,
                "error": str(executed.not_started)[:MAX_ERROR_TEXT],
            }
        causes = self._stop_causes(batch, sequence, executed, runs, after, after_error)
        # Genuine infrastructure failures are hard stops whatever else stopped the sweep; the
        # causes' own events (cleanup, restart) are recorded in every phase. Both are recorded
        # before a runner error is re-raised, so unconfirmed cleanup is never lost, and before
        # the idle wait, which can stop the sweep without returning.
        events = self._run_events(batch, sequence, runs)
        cause_events = [cause["event"] for cause in causes if cause["event"] is not None]
        entry["hard_stop_events"] = [event["code"] for event in cause_events + events]
        self.record["hard_stops"].extend(cause_events)
        self._record_hard_stops(batch, events)
        self._save()

        errors = [run.error for run in executed.runs if run.error is not None]
        live_api = [run.get("index") for run in runs if run["classification"] == LIVE_API_FAILURE]
        # A batch with a live-API failure, or cut short by a stop that is not a hard stop, says
        # nothing about its level.
        excluding = [{"state": FAILED, "reason": _text(error)} for error in errors[:1]]
        if live_api:
            reason = f"runs {live_api} failed on the live API"
            excluding.append({"state": LIVE_API_EXCLUSION, "reason": reason})
        excluding += [
            {"state": cause["state"], "reason": cause["reason"]}
            for cause in causes
            if cause["state"] in NOT_HARD_STOP_STATES
        ]
        if excluding:
            entry["metrics_exclusion"] = {"causes": excluding}
            self._save()
        if errors:
            raise errors[0]
        if causes:
            # `resend` outranks every other cause: its exit code tells the operator to resend.
            chosen = next((c for c in causes if c["state"] == STOPPED_RESEND_NEEDED), causes[0])
            self.record["stop_causes"] = [
                {"state": cause["state"], "reason": cause["reason"]} for cause in causes
            ]
            raise _SweepStop(chosen["state"], chosen["reason"])
        idle = self._wait_until_idle(batch.phase, batch.level, sequence)
        if idle is not None:
            code, detail = idle
            if code == SERVER_STATE_UNAVAILABLE:
                # The batch was read before and after it, so it stays evidence.
                raise _SweepStop(STOPPED_SERVER_STATE_UNAVAILABLE, detail)
            event = self._event(code, batch.phase, batch.level, sequence, detail)
            entry["hard_stop_events"].append(code)
            if code == VLLM_RESTART:
                self._stop_with(STOPPED_VLLM_RESTART, event, detail)
            events.append(event)
            self._record_hard_stops(batch, [event])
            self._save()
        if idle is None:
            self._idle_confirmed = True
            self.sleep(PACING_SECONDS)
        return events

    def _batch_entry(
        self,
        batch: Batch,
        sequence: int,
        name: str,
        agent_batch: AgentBatch,
        clock: MonotonicClock,
    ) -> dict:
        return {
            "sequence": sequence,
            "phase": batch.phase,
            "level": batch.level,
            "repeat": batch.repeat,
            "injected": batch.injected,
            # Per run, the source it replays; a live run is a free agent on that source's task.
            "sources": [
                {"index": index, "task_id": self.sources[index].task_id} for index in batch.sources
            ],
            "folder": f"batches/{name}",
            "batch_id": agent_batch.batch_id,
            "clock_id": clock.clock_id,
            "analysed": batch.phase != "warmup",
            "injection_log": None,
            "runs": [],
        }

    def _execute(
        self,
        agent_batch: AgentBatch,
        batch: Batch,
        clock: MonotonicClock,
        entry: dict,
        sequence: int,
    ) -> _Executed:
        """Run the batch, inside the prefill flood when it is an injected batch."""
        executed = _Executed()
        invoked = False
        injection = None
        if batch.injected:
            bound_seconds = self.run_timeout_seconds + INJECTOR_BOUND_MARGIN_SECONDS
            injection = InjectionSet(
                clock,
                self.make_injectors(clock, bound_seconds),
                ramp_seconds=INJECTOR_RAMP_SECONDS,
                cancel=self.cancellation.event,
            )
        try:
            with injection if injection is not None else nullcontext():
                invoked = True
                executed.runs = agent_batch.run()
        except InjectionCancelled:
            executed.cancelled = True
        except RunNotStarted as error:
            executed.not_started = error
        except InjectorCleanupError as error:
            # Contention may still be running: the caller stops the sweep once this is saved.
            executed.cleanup_error = error
            entry["injection_error"] = str(error)[:MAX_ERROR_TEXT]
        except InjectorError as error:
            # Only a flood that failed to start (its cleanup confirmed, else the error above);
            # anything else from inside the batch is unexpected.
            if invoked:
                raise
            executed.start_error = error
            entry["injection_error"] = str(error)[:MAX_ERROR_TEXT]
        finally:
            if invoked and not executed.runs:
                # run() raised: the batch's own runs keep whatever evidence they have.
                executed.runs = list(agent_batch.runs)
            entry["runs"] = [
                run.evidence() | {"classification": classify_run(run.evidence())}
                for run in executed.runs
            ]
            if injection is not None:
                log_path = f"injections/{sequence:02d}.json"
                (self.output_dir / "injections").mkdir(parents=True, exist_ok=True)
                log = injection.log()
                write_json_atomic(self.output_dir / log_path, log)
                entry |= {
                    "injection_log": log_path,
                    "injection_cleanup_confirmed": log["cleanup_confirmed"],
                    "injection_active": injection_active(log),
                }
                self.record["injection_logs"].append(log_path)
        return executed

    def _stop_causes(
        self,
        batch: Batch,
        sequence: int,
        executed: _Executed,
        runs: list[dict],
        after: ServerState | None,
        after_error: str | None,
    ) -> list[dict]:
        """Every condition of this batch that stops the sweep, in rank order after resend.

        Each cause has the sweep `state`, a `reason` and the hard-stop `event` it records (or
        None). `after_error` is why the reading after the batch failed, or None.
        """

        def cause(state: str, reason: str, code: str | None = None) -> dict:
            event = self._event(code, batch.phase, batch.level, sequence, reason) if code else None
            return {"state": state, "reason": reason, "event": event}

        causes = []
        if executed.cleanup_error is not None:
            detail = str(executed.cleanup_error)[:MAX_ERROR_TEXT]
            causes.append(cause(STOPPED_INJECTOR_CLEANUP, detail, INJECTOR_CLEANUP_UNCONFIRMED))
        restart = self._restart_detail(after) if after is not None else None
        if restart is not None:
            causes.append(cause(STOPPED_VLLM_RESTART, restart, VLLM_RESTART))
        if executed.start_error is not None:
            detail = f"the injector failed to start: {_text(executed.start_error)}"
            causes.append(cause(STOPPED_INJECTOR_START_FAILED, detail))
        if (
            executed.cancelled
            or self.cancellation.event.is_set()
            or any(run.get("state") == CANCELLED for run in runs)
        ):
            request = self.cancellation.request
            causes.append(
                cause(CANCELLED, request.reason if request else f"batch {sequence:02d} cancelled")
            )
        if after_error is not None:
            detail = f"vLLM state unreadable after batch {sequence:02d}: {after_error}"
            causes.append(cause(STOPPED_SERVER_STATE_UNAVAILABLE, detail))
        not_started = [run["index"] for run in runs if run.get("state") == "not_started"]
        if executed.not_started is not None or not_started:
            stage = f" ({executed.not_started.stage})" if executed.not_started else ""
            causes.append(
                cause(
                    STOPPED_RUN_NOT_STARTED,
                    f"batch {sequence:02d}: runs not started{stage} {not_started}",
                )
            )
        resend = [run["index"] for run in runs if run.get("resend_needed") is True]
        if resend:
            detail = f"batch {sequence:02d}: runs {resend} need resend"
            causes.append(cause(STOPPED_RESEND_NEEDED, detail))
        # Every other live-API failure: refused records or completion, an API unreachable at
        # the run's start, or delivery incomplete without a pending resend.
        other = {
            run["index"]: live_api_problem(run)
            for run in runs
            if run["classification"] == LIVE_API_FAILURE
            and run["index"] not in resend
            and run["index"] not in not_started
        }
        if other:
            problems = ", ".join(f"run {index}: {problem}" for index, problem in other.items())
            detail = f"batch {sequence:02d}: live API failures ({problems})"
            causes.append(cause(STOPPED_LIVE_API_FAILURE, detail))
        return causes

    def _run_events(self, batch: Batch, sequence: int, runs: list[dict]) -> list[dict]:
        """The batch's genuine infrastructure failures as one event, or none.

        A run cancelled or never started by the sweep's own stop, or one whose problem is the
        live API (classified `live_api_failure`), is never a hard stop.
        """
        failed = [
            run.get("index")
            for run in runs
            if run["classification"] == "infrastructure_failure"
            and run.get("state") not in STOPPED_RUN_STATES
        ]
        if not failed:
            return []
        return [
            self._event(
                INFRASTRUCTURE_FAILURE, batch.phase, batch.level, sequence, f"runs {failed}"
            )
        ]

    def _wait_until_idle(
        self, phase: str, level: int, sequence: int | None
    ) -> tuple[str, str] | None:
        """Poll until vLLM reports 0 running and 0 waiting, at most `IDLE_WAIT_SECONDS`.

        Returns None when idle, else a code and detail: a restart, a server still busy at the
        deadline, or `SERVER_STATE_UNAVAILABLE` when the last poll before the deadline could
        not be read (recorded in `server.read_failures`). A failed read or an unreported count
        is unknown, never idle, and is polled again until the deadline.
        """
        deadline = self.monotonic() + IDLE_WAIT_SECONDS
        failed_polls = 0
        while True:
            problem = None
            try:
                state = self.read_state()
            except ServerStateUnavailable as error:
                state, problem = None, str(error)
            if state is not None:
                restart = self._restart_detail(state)
                if restart is not None:
                    return VLLM_RESTART, restart
                if state.running is None or state.waiting is None:
                    problem = "vLLM did not report running and waiting requests"
                elif state.running == 0 and state.waiting == 0:
                    return None
            failed_polls += problem is not None
            if self.monotonic() >= deadline:
                if problem is not None:
                    self._read_failed("idle_wait", phase, level, sequence, failed_polls, problem)
                    return SERVER_STATE_UNAVAILABLE, (
                        f"vLLM state unreadable at the end of the {IDLE_WAIT_SECONDS:g} s idle "
                        f"wait: {problem}"
                    )
                return SERVER_BUSY, (
                    f"{state.running:g} running and {state.waiting:g} waiting requests after "
                    f"{IDLE_WAIT_SECONDS:g} s"
                )
            self.sleep(IDLE_POLL_SECONDS)
            self._check_cancelled()
