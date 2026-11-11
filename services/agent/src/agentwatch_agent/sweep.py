"""Drive one `natural-overload-sweep-v1` capacity sweep of one vLLM configuration.

Phases, in order: the configuration check, a warm-up batch (recorded, never analysed), the
ramp until a hard stop or the ceiling, the capacity decision, live confirmation at capacity
and the injected condition at `injected_level(capacity)`. Each batch is one `AgentBatch` with
its own `MonotonicClock`, shared by its runners and, when injected, its injection set. vLLM's
state is read before and after every batch; after each batch the driver waits until vLLM is
idle and then pauses `PACING_SECONDS`.

A hard stop in the ramp ends the ramp only; the later phases still run when capacity is at
least 1. A hard stop in the warm-up, a vLLM restart, unconfirmed injector cleanup, runs that
never started, a run needing `resend` (which outranks the others), cancellation, or a server
that is busy or unreadable outside the ramp stop the whole sweep. A stopped sweep still
analyses the ramp batches it recorded (`analysis_scope: partial_ramp`). `sweep.json` is
rewritten after every batch, so a stopped or crashed sweep keeps what it recorded. There is
no resume.
"""

import http.client
import itertools
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agentwatch_injectors.base import BoundedInjector, InjectorCleanupError
from agentwatch_injectors.injection_set import InjectionCancelled, InjectionSet

from agentwatch_agent.batch import AgentBatch, BatchRun
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.experiment import injection_active
from agentwatch_agent.replay import ReplayPlan
from agentwatch_agent.runner import JOURNAL_FILE, write_json_atomic
from agentwatch_agent.session import CancellationState
from agentwatch_agent.sweep_analysis import (
    SM_CLOCK,
    _resolve_run_dir,
    batch_metrics,
    capacity_decision,
    classify_run,
    level_summary,
    server_changes,
)
from agentwatch_agent.sweep_labels import baselines, label_validation, max_running_check
from agentwatch_agent.sweep_protocol import (
    IDLE_WAIT_SECONDS,
    INJECTOR_BOUND_MARGIN_SECONDS,
    INJECTOR_RAMP_SECONDS,
    PACING_SECONDS,
    RAMP_LEVELS,
    SOURCE_COUNT,
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
from agentwatch_agent.vllm_server import ServerState, ServerStateUnavailable, verify_configuration

SWEEP_FILE = "sweep.json"
SWEEP_METHOD = "capacity-sweep-v1"
IDLE_POLL_SECONDS = 1.0  # between idle-wait readings; the wait itself is IDLE_WAIT_SECONDS
MAX_ERROR_TEXT = 1000

# Hard-stop codes.
INFRASTRUCTURE_FAILURE = "infrastructure_failure"
DELIVERY_INCOMPLETE = "delivery_incomplete"
SERVER_BUSY = "server_busy_after_idle_wait"
VLLM_RESTART = "vllm_restart"
SERVER_STATE_UNAVAILABLE = "server_state_unavailable"
INJECTOR_CLEANUP_UNCONFIRMED = "injector_cleanup_unconfirmed"

# Sweep states. Exit codes (mapped by the CLI): completed 0, refused_configuration 2,
# stopped_resend_needed 3, every other state 1.
COMPLETED = "completed"
REFUSED_CONFIGURATION = "refused_configuration"
STOPPED_VLLM_RESTART = "stopped_vllm_restart"
STOPPED_INJECTOR_CLEANUP = "stopped_injector_cleanup_unconfirmed"
STOPPED_RUN_NOT_STARTED = "stopped_run_not_started"
STOPPED_RESEND_NEEDED = "stopped_resend_needed"
# A hard stop in the warm-up: the ramp never started, so nothing about capacity was measured.
STOPPED_WARMUP_FAILED = "stopped_warmup_failed"
# Outside the ramp a busy or unreadable server stops the sweep: the next batch would start on
# a server whose state is unknown or still loaded.
STOPPED_SERVER_BUSY = f"stopped_{SERVER_BUSY}"
STOPPED_SERVER_STATE_UNAVAILABLE = f"stopped_{SERVER_STATE_UNAVAILABLE}"
CANCELLED = "cancelled"
FAILED = "failed"

# How the ramp ended, recorded separately from the sweep state.
RAMP_HARD_STOP = "hard_stop"
RAMP_CEILING = "ceiling"
RAMP_STOPPED = "stopped"
RAMP_CANCELLED = "cancelled"

FULL_RAMP = "full_ramp"
PARTIAL_RAMP = "partial_ramp"
NO_ANALYSIS = "none"

PHASES = ("configuration_check", "warmup", "ramp", "capacity_decision", "live", "injected")
RAMP_PHASES = ("warmup", "ramp")

BatchFactory = Callable[[Batch, Path, MonotonicClock], AgentBatch]
LiveBatchFactory = Callable[[int, Path, MonotonicClock], AgentBatch]
InjectorFactory = Callable[[MonotonicClock, float], list[BoundedInjector]]


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
        make_live_batch: LiveBatchFactory,
        make_injectors: InjectorFactory,
        read_state: Callable[[], ServerState],
        models: Callable[[], dict],
        host: dict,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        start_attempts: list[dict] | None = None,
    ):
        if len(sources) != SOURCE_COUNT:
            raise ValueError(f"A sweep replays exactly {SOURCE_COUNT} sources")
        self.configuration = configuration
        self.sources = list(sources)
        self.output_dir = output_dir
        self.run_timeout_seconds = run_timeout_seconds
        self.make_batch = make_batch
        self.make_live_batch = make_live_batch
        self.make_injectors = make_injectors
        self.read_state = read_state
        self.models = models
        self.host = host
        self.sleep = sleep
        self.monotonic = monotonic
        self.start_attempts = start_attempts
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
            # Held-out test evidence: never for training, calibration or threshold tuning.
            "held_out": True,
            "configuration": {"declared": document["configuration"], "verified": None},
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
            "server": {"initial_state": None},
            "phases": {phase: {"status": "not_run"} for phase in PHASES},
            "batches": [],
            "injection_logs": [],
            "hard_stops": [],
            "level_summaries": None,
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
        decision = self._decide_capacity(FULL_RAMP)
        capacity = decision["capacity"]
        if capacity is None or capacity < 1:
            reason = f"capacity below 1 or not measured (result {decision['result']})"
            for phase in ("live", "injected"):
                self.record["phases"][phase] = {"status": "skipped", "reason": reason}
            return
        self._begin("live", level=capacity)
        self._ensure_idle("live", capacity)
        for batch in live_batches(capacity):
            self._post_ramp_batch(
                batch, lambda path, clock: self.make_live_batch(capacity, path, clock)
            )
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
            state = self.read_state()
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
        verified = verify_configuration(self.configuration, models, state)
        self.record["server"]["initial_state"] = asdict(state)
        self.record["configuration"]["verified"] = verified
        problems = []
        if verified["contradicted"]:
            problems.append(f"vLLM contradicts the declared {', '.join(verified['contradicted'])}")
        if state.start_time_seconds is None:
            problems.append(
                "vLLM reported no process_start_time_seconds, so a restart could not be detected"
            )
        if problems:
            raise _SweepStop(REFUSED_CONFIGURATION, "; ".join(problems))
        self._complete("configuration_check")

    def _ramp(self) -> None:
        self._begin("warmup")
        events = self._run_batch(WARMUP_BATCH, self._replay_factory(WARMUP_BATCH))
        if events:
            self.record["hard_stops"].extend(events)
            codes = ", ".join(event["code"] for event in events)
            raise _SweepStop(STOPPED_WARMUP_FAILED, f"hard stop in the warm-up batch: {codes}")
        self._complete("warmup")
        self._begin("ramp")
        for batch in itertools.chain.from_iterable(ramp_batches(n) for n in RAMP_LEVELS):
            events = self._run_batch(batch, self._replay_factory(batch))
            if events:
                self.record["hard_stops"].extend(events)
                self.record["ramp_end"] = RAMP_HARD_STOP
                break
        else:
            self.record["ramp_end"] = RAMP_CEILING
        self._complete("ramp")

    def _analyse(self) -> None:
        """Analyse the recorded ramp on every outcome but a failure; mark a partial ramp.

        A sweep stopped or cancelled during the warm-up or ramp still gets level summaries,
        the capacity decision and label validation over the ramp batches it recorded.
        """
        if self.record["phases"]["warmup"]["status"] == "not_run":
            self.record |= {
                "analysis_scope": NO_ANALYSIS,
                "analysis_note": f"no batch ran ({self.record['state']})",
            }
            return
        if self.record["capacity_decision"] is None:
            self._decide_capacity(PARTIAL_RAMP)
            self.record["analysis_note"] = (
                f"ramp incomplete, {self.record['state']}: {self.record['stop_reason']}"
            )
        self._validate_labels()
        self._save()

    def _decide_capacity(self, scope: str) -> dict:
        self._begin("capacity_decision", scope=scope)
        self.record["analysis_scope"] = scope
        by_level: dict[int, list[dict]] = {}
        for entry in self.record["batches"]:
            if entry["phase"] == "ramp" and "metrics" in entry:
                changes = entry["server"]["changes"]
                server = {"server": changes} if changes is not None else {}
                by_level.setdefault(entry["level"], []).append(entry["metrics"] | server)
        summaries: dict[int, dict] = {}
        baseline_clock = None
        for level in sorted(by_level):
            summaries[level] = level_summary(by_level[level], baseline_clock if level > 1 else None)
            if level == RAMP_LEVELS[0]:
                baseline_clock = summaries[level]["device"]["metrics"][SM_CLOCK]["mean"]
        hard_stops: dict[int, list[str]] = {}
        for event in self.record["hard_stops"]:
            if event["phase"] == "ramp":
                hard_stops.setdefault(event["level"], []).append(event["code"])
        decision = capacity_decision(summaries, hard_stops, self.run_timeout_seconds)
        self.record |= {"level_summaries": summaries, "capacity_decision": decision}
        self._complete("capacity_decision")
        self._save()
        return decision

    def _validate_labels(self) -> None:
        """Labels of N >= 2 ramp runs against the N = 1 baselines, and the running-count check.

        Level keys stay ints so N = 1 is never validated against itself. Unreadable evidence is
        recorded as such; the folder can be analysed again without the GPU.
        """
        runs, without_evidence = self._ramp_run_dirs()
        try:
            reference = baselines(runs.get(1, []))
            validation = label_validation(
                {level: paths for level, paths in runs.items() if level >= 2}, reference
            )
            self.record["label_validation"] = validation | {
                "runs_without_evidence": without_evidence
            }
        # CoverageError and json.JSONDecodeError are ValueErrors.
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.record["label_validation"] = {"status": "unreadable", "error": _text(error)}
        try:
            self.record["max_running_check"] = max_running_check(
                [path for paths in runs.values() for path in paths],
                self.configuration.max_num_seqs,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.record["max_running_check"] = {"status": "unreadable", "error": _text(error)}

    def _ramp_run_dirs(self) -> tuple[dict[int, list[Path]], int]:
        runs: dict[int, list[Path]] = {}
        without_evidence = 0
        for entry in self.record["batches"]:
            if entry["phase"] != "ramp":
                continue
            batch_dir = self.output_dir / entry["folder"]
            for run in entry["runs"]:
                run_dir, _found_by = _resolve_run_dir(batch_dir, run.get("output_dir"))
                files = ("agent-evidence.json", JOURNAL_FILE)
                if run_dir is None or not all((run_dir / name).is_file() for name in files):
                    without_evidence += 1
                    continue
                runs.setdefault(entry["level"], []).append(run_dir)
        return runs, without_evidence

    # Batches

    def _replay_factory(self, batch: Batch) -> Callable[[Path, MonotonicClock], AgentBatch]:
        return lambda path, clock: self.make_batch(batch, path, clock)

    def _post_ramp_batch(
        self, batch: Batch, make: Callable[[Path, MonotonicClock], AgentBatch]
    ) -> None:
        """Run a live, injected or control batch; a server event stops the sweep.

        Infrastructure failures there are evidence of the condition, recorded on the batch.
        """
        events = self._run_batch(batch, make)
        for event in events:
            if event["code"] in (SERVER_BUSY, SERVER_STATE_UNAVAILABLE):
                self.record["hard_stops"].append(event)
                raise _SweepStop(f"stopped_{event['code']}", event["detail"])

    def _ensure_idle(self, phase: str, level: int) -> None:
        """Before a phase after the ramp: the server must be idle and paced, as between batches."""
        if self._idle_confirmed:
            return
        result = self._wait_until_idle()
        if result is not None:
            code, detail = result
            event = self._event(code, phase, level, None, detail)
            self.record["hard_stops"].append(event)
            raise _SweepStop(
                STOPPED_VLLM_RESTART if code == VLLM_RESTART else f"stopped_{code}", detail
            )
        self._idle_confirmed = True
        self.sleep(PACING_SECONDS)

    def _event(self, code: str, phase: str, level: int, sequence: int | None, detail: str) -> dict:
        return {"code": code, "phase": phase, "level": level, "batch": sequence, "detail": detail}

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
        unexpected error, each once the batch is recorded in `sweep.json`.
        """
        self._check_cancelled()
        try:
            before = self.read_state()
        except ServerStateUnavailable as error:
            # Idle must be confirmed again before anything else runs on this server.
            self._idle_confirmed = False
            return [
                self._event(SERVER_STATE_UNAVAILABLE, batch.phase, batch.level, None, str(error))
            ]
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
        executed = self._execute(agent_batch, batch, clock, entry, sequence)
        self._idle_confirmed = False
        try:
            after: ServerState | None = self.read_state()
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
        metrics = batch_metrics(batch_dir, self.run_timeout_seconds)
        entry["wall_seconds"] = metrics.get("wall_seconds")
        if entry["analysed"]:
            entry["metrics"] = metrics
        if executed.not_started is not None:
            entry["not_started"] = {
                "stage": executed.not_started.stage,
                "error": str(executed.not_started)[:MAX_ERROR_TEXT],
            }
        causes = self._stop_causes(batch, sequence, executed, runs, after)
        # Recorded before a runner error is re-raised, so unconfirmed cleanup is never lost.
        for cause in causes:
            if cause["event"] is not None:
                self.record["hard_stops"].append(cause["event"])
        self._save()

        errors = [run.error for run in executed.runs if run.error is not None]
        if errors:
            raise errors[0]
        if causes:
            # `resend` outranks every other cause: its exit code tells the operator to resend.
            chosen = next((c for c in causes if c["state"] == STOPPED_RESEND_NEEDED), causes[0])
            self.record["stop_causes"] = [
                {"state": cause["state"], "reason": cause["reason"]} for cause in causes
            ]
            raise _SweepStop(chosen["state"], chosen["reason"])
        events = self._run_events(batch, sequence, runs)
        if after is None:
            events.append(
                self._event(
                    SERVER_STATE_UNAVAILABLE, batch.phase, batch.level, sequence, after_error
                )
            )
        idle = self._wait_until_idle()
        if idle is not None:
            code, detail = idle
            event = self._event(code, batch.phase, batch.level, sequence, detail)
            if code == VLLM_RESTART:
                self._stop_with(STOPPED_VLLM_RESTART, event, detail)
            events.append(event)
        entry["hard_stop_events"] = [event["code"] for event in events]
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
        # Live runs are free agents chosen by the live factory, not replays of a source.
        replayed = batch.phase != "live"
        return {
            "sequence": sequence,
            "phase": batch.phase,
            "level": batch.level,
            "repeat": batch.repeat,
            "injected": batch.injected,
            "sources": [
                {"index": index, "task_id": self.sources[index].task_id} for index in batch.sources
            ]
            if replayed
            else None,
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
    ) -> list[dict]:
        """Every condition of this batch that stops the sweep, in rank order after resend.

        Each cause has the sweep `state`, a `reason` and the hard-stop `event` it records (or
        None).
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
        if (
            executed.cancelled
            or self.cancellation.event.is_set()
            or any(run.get("state") == CANCELLED for run in runs)
        ):
            request = self.cancellation.request
            causes.append(
                cause(CANCELLED, request.reason if request else f"batch {sequence:02d} cancelled")
            )
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
            causes.append(cause(STOPPED_RESEND_NEEDED, detail, DELIVERY_INCOMPLETE))
        return causes

    def _run_events(self, batch: Batch, sequence: int, runs: list[dict]) -> list[dict]:
        """One event per code for runs that failed for an infrastructure reason."""
        by_code: dict[str, list[object]] = {}
        for run in runs:
            if run["classification"] != "infrastructure_failure":
                continue
            code = (
                DELIVERY_INCOMPLETE
                if run.get("delivery_complete") is not True
                else (INFRASTRUCTURE_FAILURE)
            )
            by_code.setdefault(code, []).append(run.get("index"))
        return [
            self._event(code, batch.phase, batch.level, sequence, f"runs {indexes}")
            for code, indexes in by_code.items()
        ]

    def _wait_until_idle(self) -> tuple[str, str] | None:
        """Poll until vLLM reports 0 running and 0 waiting, at most `IDLE_WAIT_SECONDS`.

        Returns None when idle, else the hard-stop code and detail. An unreported count is
        unknown, never idle.
        """
        deadline = self.monotonic() + IDLE_WAIT_SECONDS
        while True:
            try:
                state = self.read_state()
            except ServerStateUnavailable as error:
                return SERVER_STATE_UNAVAILABLE, str(error)
            restart = self._restart_detail(state)
            if restart is not None:
                return VLLM_RESTART, restart
            if state.running is None or state.waiting is None:
                return SERVER_STATE_UNAVAILABLE, "vLLM did not report running and waiting requests"
            if state.running == 0 and state.waiting == 0:
                return None
            if self.monotonic() >= deadline:
                return SERVER_BUSY, (
                    f"{state.running:g} running and {state.waiting:g} waiting requests after "
                    f"{IDLE_WAIT_SECONDS:g} s"
                )
            self.sleep(IDLE_POLL_SECONDS)
            self._check_cancelled()
