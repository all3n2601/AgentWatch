"""Run a preregistered optimization experiment's batches and write their records (O06).

An experiment directory (`optimization_experiment`) holds `manifest.json` (`opt-experiment-manifest-v1`, written once
before any batch) and `schedule.json`, the batch order `make optimization-schedule` derives
from it. The data pipeline validates manifests, derives schedules and compares results
(`docs/optimization-evidence.md`); this runner re-derives neither. It checks that the schedule
names this manifest, runs each scheduled batch in order and writes `batches/<NNN>.json`
(`opt-batch-v1`), which `make compare-optimization` validates.

For each batch, after waiting for the model server to be idle:

1. read the host's identity (model, hardware, power, task fixtures). A host that no longer
   matches the manifest (for example, on battery) stops the experiment before the batch is
   attempted, so the slot stays runnable once the host is fixed;
2. set the batch's policy through the policy API, on the model server this runner calls, and
   read it back. From here on the batch is attempted and always gets a record;
3. start the condition's injectors, if any, and submit every frozen task at its preregistered
   offset, in one runner process sharing one admission controller;
4. stop the injectors, confirm their cleanup, and record each task's outcome, ticks, admission
   waits and run evidence.

Records are never overwritten: a later invocation resumes at the first unrecorded slot, so no
attempted batch is rerun to replace its result. Any batch that does not complete stops the
experiment, so a broken host or service does not consume the schedule. On every exit the
policy is restored to the baseline (no limits) and read back.
"""

import json
import math
import threading
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from agentwatch_injectors.base import BoundedInjector, InjectorCleanupError, InjectorError
from agentwatch_injectors.cpu import CPU_METHOD
from agentwatch_injectors.injection_set import InjectionCancelled, InjectionSet
from agentwatch_injectors.prefill import PREFILL_METHOD
from agentwatch_sdk.policy import PolicyChange, PolicyChangeResult, PolicyLimits, PolicyVersion

from agentwatch_agent.admission import (
    AdmissionController,
    AdmissionSettings,
    PolicyWatcher,
    RunnerAdmission,
)
from agentwatch_agent.batch import AgentBatch, BatchRun
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.coverage import CoverageError, read_journal
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.experiment import ROOT_STEP, injection_active
from agentwatch_agent.optimization_experiment import (
    BATCHES_DIR,
    CONTROLS,
    Experiment,
    ExperimentError,
    FrozenTask,
    Slot,
    recorded_slots,
)
from agentwatch_agent.runner import EVIDENCE_FILE, JOURNAL_FILE, AgentRunner, write_json_atomic
from agentwatch_agent.tasks import Task, load_task

RUNNER_METHOD = "opt-batch-runner-v1"
BATCH_FORMAT = "opt-batch-v1"
RUNS_DIR = "runs"
LOG_FILE = "runner-log.jsonl"
POLICY_OPERATOR = "opt-batch-runner"
# The runner's one wording for a run deadline (AgentRunner._run_limit_reason).
RUN_LIMIT_PREFIX = "Run exceeded its "
UNDELIVERED_SUFFIX = "_locally_undelivered"
# The manifest names an injector by the resource it contends; the injectors name their own
# kinds and resource classes differently, so each supported method maps to one manifest kind.
METHOD_KINDS = {PREFILL_METHOD: "inference", CPU_METHOD: "cpu"}
# Injectors outlive the batch's run deadline by this margin; the watchdog stops them after it.
INJECTOR_BOUND_MARGIN_SECONDS = 32.0
DEFAULT_RAMP_SECONDS = 2.0


class HostChanged(RuntimeError):
    """The host no longer matches the manifest, or could not be read; nothing was attempted."""


class PolicyApplyError(RuntimeError):
    """The policy service did not end up serving the limits a batch needs."""


class PolicyService(Protocol):
    def get_policy(self, target: str = "coding-agent") -> PolicyVersion: ...

    def change_policy(
        self, change: PolicyChange, target: str = "coding-agent"
    ) -> PolicyChangeResult: ...


@dataclass(frozen=True)
class Host:
    """What the experiment runner needs from this host and its services; tests pass doubles."""

    policy: PolicyService
    # The model server this runner calls ("ollama" or "vllm"); candidate limits name it.
    inference_service_id: str
    make_runner: Callable[[Task, Path, MonotonicClock, RunnerAdmission], AgentRunner]
    make_injector: Callable[[dict, MonotonicClock, float], BoundedInjector]
    # The manifest's model, hardware, power and state blocks, as measured now.
    observe_identity: Callable[[], dict]
    wait_until_idle: Callable[[], None]
    run_timeout_seconds: float
    policy_poll_seconds: float = 1.0
    drain_seconds: float = 30.0
    ramp_seconds: float = DEFAULT_RAMP_SECONDS


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class RunSummary:
    attempted: list[int] = field(default_factory=list)
    statuses: dict[int, str] = field(default_factory=dict)
    stopped: str | None = None
    restored: bool | None = None


class ExperimentRunner:
    def __init__(self, experiment: Experiment, host: Host):
        self.experiment = experiment
        self.host = host
        self._cancel = threading.Event()
        self._cancel_reason: str | None = None
        self._batch: AgentBatch | None = None

    # Cancellation ------------------------------------------------------------------------

    def request_cancel_from_signal(self, source: str, reason: str) -> None:
        """Safe in a signal handler: only the first signal touches the event's lock (as in
        `CancellationState`); the runners' own cancel is signal-safe."""
        if self._cancel_reason is None:
            self._cancel_reason = f"{reason} ({source})"
            self._cancel.set()  # interrupts an injector ramp
        batch = self._batch
        if batch is not None:
            batch.request_cancel_from_signal(source, reason)

    # Log -----------------------------------------------------------------------------------

    def _log(self, event: str, **details) -> None:
        line = json.dumps({"at": _now(), "event": event} | details, sort_keys=True)
        with (self.experiment.root / LOG_FILE).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    # Preflight -----------------------------------------------------------------------------

    def preflight(self) -> None:
        """Refuse an experiment this host cannot run faithfully, before any batch."""
        for frozen in self.experiment.tasks:
            task = load_task(frozen.name)
            if task.task_id != frozen.task_id or task.content_sha256 != frozen.content_sha256:
                raise ExperimentError(
                    f"{frozen.task_id}: this checkout has {task.task_id} with another content hash"
                )
            runner = self.host.make_runner(
                task,
                self.experiment.root / RUNS_DIR / "preflight",
                MonotonicClock(),
                self._admission()[2],
            )
            if runner.configuration_digest() != frozen.configuration_digest:
                raise ExperimentError(
                    f"{frozen.task_id}: this runner's configuration digest "
                    f"{runner.configuration_digest()} is not the preregistered one"
                )
        for condition, planned in self.experiment.injectors.items():
            for injector in planned:
                method = injector.get("method")
                if METHOD_KINDS.get(method) != injector.get("kind"):
                    raise ExperimentError(
                        f"{condition}: injector {method!r} of kind {injector.get('kind')!r} is "
                        f"not one this runner can apply ({METHOD_KINDS})"
                    )
                # Built once without starting, so bad intensities fail before any batch.
                self.host.make_injector(injector, MonotonicClock(), self._injector_bound())

    def _injector_bound(self) -> float:
        return self.host.run_timeout_seconds + INJECTOR_BOUND_MARGIN_SECONDS

    def _admission(self) -> tuple[AdmissionController, PolicyWatcher, RunnerAdmission]:
        """A fresh controller per batch, under the manifest's queue bounds: a watcher thread is
        single-use, and a controller never goes back to a lower policy version."""
        controller = AdmissionController(
            AdmissionSettings(**self.experiment.queue_bounds),
            inference_service_id=self.host.inference_service_id,
        )
        watcher = PolicyWatcher(self.host.policy, controller, self.host.policy_poll_seconds)
        return controller, watcher, RunnerAdmission(controller, self.host.policy, watcher)

    # Policy --------------------------------------------------------------------------------

    def apply_policy(self, limits: dict[str, int], reason: str) -> PolicyVersion:
        """Make `limits` the active policy, on this runner's model server, and read it back."""
        service = self.host.inference_service_id
        current = self.host.policy.get_policy()
        if not self._serves(current, limits):
            change = PolicyChange(
                request_id=uuid4(),
                expected_version=current.version,
                action="set" if limits else "disable",
                limits=PolicyLimits(**limits) if limits else None,
                inference_service=service if limits else None,
                operator=POLICY_OPERATOR,
                reason=reason[:512],
            )
            self.host.policy.change_policy(change)
        active = self.host.policy.get_policy()
        if not self._serves(active, limits):
            raise PolicyApplyError(
                f"Policy version {active.version} serves {active.limits.model_dump(exclude_none=True)} "
                f"on {active.controlled_services['inference_requests'].service_id}, not {limits}"
            )
        self._log(
            "policy",
            version=active.version,
            limits=limits,
            inference_service=active.controlled_services["inference_requests"].service_id,
            changed=active.version != current.version,
            reason=reason,
        )
        return active

    def _serves(self, policy: PolicyVersion, limits: dict[str, int]) -> bool:
        if policy.limits.model_dump(exclude_none=True) != limits:
            return False
        named = policy.controlled_services["inference_requests"].service_id
        # Without an inference limit, the service it would name limits nothing.
        return "inference_requests" not in limits or named == self.host.inference_service_id

    def restore_baseline(self) -> bool:
        try:
            self.apply_policy({}, f"{self.experiment.experiment_id}: restore the baseline")
            return True
        except Exception as error:  # noqa: BLE001 - reported, and the caller exits non-zero
            self._log("restore_failed", error=f"{type(error).__name__}: {error}"[:500])
            return False

    # Batches -------------------------------------------------------------------------------

    def run(self) -> RunSummary:
        summary = RunSummary()
        start = recorded_slots(self.experiment.root)
        self._log(
            "start",
            runner=RUNNER_METHOD,
            manifest_sha256=self.experiment.manifest_sha256,
            first_slot=start,
            slots=len(self.experiment.slots),
        )
        try:
            for slot in self.experiment.slots[start:]:
                if self._cancel_reason is not None:
                    summary.stopped = f"Cancelled before slot {slot.index}: {self._cancel_reason}"
                    break
                try:
                    record = self.run_slot(slot)
                except HostChanged as error:
                    summary.stopped = f"Slot {slot.index} not attempted: {error}"
                    break
                summary.attempted.append(slot.index)
                summary.statuses[slot.index] = record["status"]
                if record["status"] != "completed":
                    summary.stopped = f"Slot {slot.index} {record['status']}: {record['reason']}"
                    break
        finally:
            summary.restored = self.restore_baseline()
            self._log("stop", stopped=summary.stopped, restored=summary.restored)
        return summary

    def run_slot(self, slot: Slot) -> dict:
        path = self.experiment.root / BATCHES_DIR / f"{slot.index:03d}.json"
        if path.exists():
            raise ExperimentError(f"{path.name} already exists; records are never overwritten")
        try:
            self.host.wait_until_idle()
            identity = self.host.observe_identity()
        except Exception as error:
            raise HostChanged(f"host state unreadable: {type(error).__name__}: {error}") from error
        differing = [
            key for key, value in identity.items() if value != self.experiment.declared[key]
        ]
        if differing:
            raise HostChanged(
                f"{', '.join(differing)} differ from the manifest: "
                + "; ".join(f"{key} is {identity[key]}" for key in differing)
            )
        record = self._record_head(slot, identity)
        try:
            policy = self.apply_policy(
                slot.limits,
                f"{self.experiment.experiment_id} slot {slot.index}: {slot.configuration}",
            )
        except Exception as error:  # noqa: BLE001 - the batch is attempted and recorded
            return self._write(path, record | self._not_started(f"Policy not applied: {error}"))
        record["policy"]["version"] = policy.version
        return self._write(path, record | self._execute(slot, policy))

    def _record_head(self, slot: Slot, identity: dict) -> dict:
        return {
            "format": BATCH_FORMAT,
            "manifest_sha256": self.experiment.manifest_sha256,
            "batch_id": str(uuid4()),
            **slot.head(),
            "policy": {"sha256": slot.policy_sha256, "limits": slot.limits, "version": None},
            **identity,
        }

    def _not_started(self, reason: str) -> dict:
        return {
            "status": "not_started",
            "reason": reason[:1000],
            "started_at": None,
            "finished_at": None,
            "clock_id": None,
            "injection": {"active": None, "cleanup_confirmed": None, "injectors": None},
            "admission": {"starved_admissions": None, "leaked_permits": None},
            "tasks": [
                {
                    "task_id": task.task_id,
                    "run_id": None,
                    "state": "not_started",
                    "acceptance": None,
                    "cold": False,
                    "submitted_tick_ns": None,
                    "finished_tick_ns": None,
                    "admission_waits": dict.fromkeys(CONTROLS),
                    "evidence_dir": None,
                }
                for task in self.experiment.tasks
            ],
        }

    def _execute(self, slot: Slot, policy: PolicyVersion) -> dict:
        directory = self.experiment.root / RUNS_DIR / f"{slot.index:03d}"
        clock = MonotonicClock()
        controller, watcher, admission = self._admission()
        runners = [
            self.host.make_runner(
                load_task(frozen.name),
                directory / f"{position:02d}-{frozen.name}",
                clock,
                admission,
            )
            for position, frozen in enumerate(self.experiment.tasks, 1)
        ]
        batch = AgentBatch(
            runners,
            controller,
            directory,
            watcher=watcher,
            drain_seconds=self.host.drain_seconds,
            start_offsets_seconds=[task.offset_seconds for task in self.experiment.tasks],
        )
        planned = self.experiment.injectors[slot.condition]
        injection = (
            InjectionSet(
                clock,
                [self.host.make_injector(item, clock, self._injector_bound()) for item in planned],
                ramp_seconds=self.host.ramp_seconds,
                cancel=self._cancel,
            )
            if planned
            else None
        )
        problems: list[str] = []
        submitted = False
        started_at = _now()
        self._batch = batch
        try:
            with ExitStack() as stack:
                stack.enter_context(watcher)
                stack.enter_context(injection if injection is not None else nullcontext())
                submitted = True
                batch.run()
        except RunNotStarted as error:
            problems.append(f"Batch not started ({error.stage}): {error}")
        except InjectionCancelled as error:
            problems.append(f"Cancelled during the injector ramp: {error}")
        except InjectorCleanupError as error:
            problems.append(f"Injector cleanup not confirmed: {error}")
        except InjectorError as error:
            problems.append(f"Injector did not start: {error}")
        except Exception as error:
            # Starting an injector can fail with any error (a process that will not spawn); the
            # set has stopped the ones it started. After submission, errors are the batch's own.
            if submitted:
                raise
            problems.append(f"Injector did not start: {type(error).__name__}: {error}")
        finally:
            self._batch = None
        finished_at = _now()
        ran = submitted and any(run.submitted_tick_ns is not None for run in batch.runs)
        if not ran:
            return self._not_started("; ".join(problems) or "No task was submitted")
        tasks = [
            self._task(run, frozen, slot)
            for run, frozen in zip(batch.runs, self.experiment.tasks, strict=True)
        ]
        summary = controller.summary()
        log = injection.log() if injection is not None else None
        status, reason = self._status(slot, batch.runs, summary, log, problems, policy)
        if injection is not None:
            (directory / "injection.json").write_text(json.dumps(log, indent=2) + "\n")
        return {
            "status": status,
            "reason": reason,
            "started_at": started_at,
            "finished_at": finished_at,
            "clock_id": clock.clock_id,
            "injection": {
                "active": injection_active(log) if log is not None else None,
                "cleanup_confirmed": log["cleanup_confirmed"] if log is not None else None,
                "injectors": [dict(item) for item in planned]
                if log is not None or not planned
                else None,
            },
            "admission": {
                "starved_admissions": sum(
                    control["refused"].get("queue_full", 0)
                    + control["refused"].get("wait_exceeded", 0)
                    for control in summary["controls"].values()
                ),
                "leaked_permits": summary["open_permits"],
            },
            "tasks": tasks,
        }

    def _status(self, slot, runs, summary, log, problems, policy) -> tuple[str, str | None]:
        problems = list(problems)
        if self._cancel_reason is not None:
            problems.append(f"Cancelled: {self._cancel_reason}")
        if log is not None and not injection_active(log):
            problems.append("The injectors did not apply sustained, measured load")
        if summary["policy_version"] != policy.version or len(summary["policy_history"]) != 1:
            problems.append("The policy changed while the batch ran")
        for control in slot.limits:
            if summary["controls"][control]["enforced"] is not True:
                problems.append(
                    f"The {control} limit was not enforced: "
                    f"{summary['controls'][control]['not_enforced_reason']}"
                )
        for run in runs:
            outcome = run.outcome
            if outcome is not None and (
                outcome.resend_needed
                or not outcome.delivery_complete
                or outcome.records_refused
                or outcome.completion_refused
            ):
                problems.append(f"Run {run.index}: live delivery incomplete")
        if problems:
            return "aborted", "; ".join(problems)[:1000]
        failures = [
            f"Run {run.index}: {type(run.error).__name__}: {run.error}"
            if run.error is not None
            else f"Run {run.index} not started: {run.not_started}"
            for run in runs
            if run.error is not None or run.not_started is not None
        ]
        if failures:
            return "failed", "; ".join(failures)[:1000]
        return "completed", None

    def _task(self, run: BatchRun, frozen: FrozenTask, slot: Slot) -> dict:
        run_dir = run.runner.output_dir
        evidence = _evidence(run_dir)
        state = _task_state(run)
        started = state != "not_started"
        passed = evidence.get("task_passed")
        return {
            "task_id": frozen.task_id,
            "run_id": evidence.get("run_id")
            or (str(run.outcome.run_id) if run.outcome is not None else None),
            "state": state,
            "acceptance": {True: "passed", False: "failed"}.get(passed)
            if isinstance(passed, bool)
            else None,
            # Warm-up batches exist to absorb cold starts; pairs run on a warm server.
            "cold": slot.kind == "warmup",
            "submitted_tick_ns": run.submitted_tick_ns,
            "finished_tick_ns": _root_finished_tick(run_dir) if started else None,
            "admission_waits": _longest_waits(evidence),
            "evidence_dir": (
                run_dir.relative_to(self.experiment.root).as_posix()
                if started and (run_dir / EVIDENCE_FILE).exists()
                else None
            ),
        }

    def _write(self, path: Path, record: dict) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, record)
        self._log(
            "batch",
            index=record["index"],
            status=record["status"],
            reason=record["reason"],
            batch_id=record["batch_id"],
        )
        return record


def _evidence(run_dir: Path) -> dict:
    try:
        loaded = json.loads((run_dir / EVIDENCE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _task_state(run: BatchRun) -> str:
    if run.not_started is not None or run.submitted_tick_ns is None:
        return "not_started"
    if run.outcome is None:
        return "failed"  # A runner error; its own evidence says what happened.
    state = run.outcome.state.removesuffix(UNDELIVERED_SUFFIX)
    if state == "failed" and (run.outcome.reason or "").startswith(RUN_LIMIT_PREFIX):
        return "timed_out"
    return state if state in ("completed", "failed", "cancelled") else "failed"


def _root_finished_tick(run_dir: Path) -> int | None:
    """The root step's finish tick from the run's journal; None when it was not journaled."""
    try:
        _registration, records, _unusable = read_journal(run_dir / JOURNAL_FILE)
    except (OSError, CoverageError):
        return None
    events = [
        record
        for record in records
        if record.get("record_type") == "step_event" and record.get("step_id") == ROOT_STEP
    ]
    if not events:
        return None
    latest = max(events, key=lambda record: record["sequence"])
    tick = latest.get("finished_tick_ns")
    return tick if isinstance(tick, int) and not isinstance(tick, bool) else None


def _longest_waits(evidence: dict) -> dict[str, float | None]:
    """Each control's longest admission wait in the run; None when none was recorded."""
    waits: dict[str, float | None] = dict.fromkeys(CONTROLS)
    for record in evidence.get("admissions") or []:
        if not isinstance(record, dict) or record.get("control") not in waits:
            continue
        wait = record.get("admission_wait_seconds")
        if isinstance(wait, int | float) and not isinstance(wait, bool) and math.isfinite(wait):
            current = waits[record["control"]]
            waits[record["control"]] = wait if current is None else max(current, wait)
    return waits
