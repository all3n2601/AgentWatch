"""Run one fixed coding task with a real model and report it as a correlated live run.

Every model call, tool call and retrieval becomes a live step under one root step and trace.
Server-reported model metrics, the conversation, configuration and failure/cleanup details
cannot be carried by live-contracts-v1 StepEvents, so they are written to local evidence
(`agent-evidence.json`) keyed by the same step and span IDs.
"""

import http.client
import json
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from agentwatch_sdk.live import ResourceRegistration, RunCreate
from agentwatch_worker.nvml import NvmlDevice

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import (
    CONTRACT_VERSION,
    DEFAULT_CANCEL_POLL_SECONDS,
    DEFAULT_CHAT_OPTIONS,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    RunConfiguration,
    RunLimits,
    validate_sample_budget,
)
from agentwatch_agent.delivery import (
    COMPLETION_REFUSED,
    TRANSIENT_ERRORS,
    Journal,
    LiveDelivery,
    LiveTransport,
    PeriodicFlusher,
    RetryPolicy,
    is_retryable,
)
from agentwatch_agent.errors import RunCancelled, RunFailure, RunNotStarted
from agentwatch_agent.model import (
    ChatModel,
    ModelReply,
    ModelResponseError,
    RequestedCall,
    StreamTimer,
    chat_payload,
    joined_vllm_timing,
    parse_reply,
    wait_interruptibly,
)
from agentwatch_agent.prompts import SYSTEM_PROMPT, task_message
from agentwatch_agent.redaction import redact_reason
from agentwatch_agent.resources import (
    CPU_METRICS,
    GPU_METRICS,
    CpuProbe,
    GpuProbe,
    Probe,
    ResourceSampler,
)
from agentwatch_agent.retrieval import RETRIEVAL_METHOD, DocumentIndex
from agentwatch_agent.service_probes import (
    DCGM_METRICS,
    VLLM_METRICS,
    ExpositionChangeLog,
    dcgm_probe,
    vllm_probe,
)
from agentwatch_agent.session import (
    TERMINAL_RUN_STATES,
    CancellationState,
    CancellationWatcher,
    RunSession,
)
from agentwatch_agent.steps import Step, StepRecorder, new_trace_id
from agentwatch_agent.tasks import Task
from agentwatch_agent.tools import ToolDispatcher, ToolFailed, ToolRejected, tool_message
from agentwatch_agent.vllm import traceparent
from agentwatch_agent.vllm_traces import collect_request_timings
from agentwatch_agent.workspace import Workspace

EVIDENCE_FILE = "agent-evidence.json"
JOURNAL_FILE = "live-journal.jsonl"
# Every vLLM series each round saw, as changes; local evidence for later analysis only.
VLLM_CHANGE_LOG_FILE = "vllm-metrics-changes.jsonl"
# Bounded wait after the run for vLLM's request spans (exported within about 1 s).
DEFAULT_TRACE_WAIT_SECONDS = 10.0
MAX_EVIDENCE_TEXT = 2000
MODEL_QUEUE_LIMITATIONS = {
    "ollama": (
        "Model-call queue delay is runner-side dispatch only; waiting inside the model server "
        "is part of execution wall time, not a measured inference queue. Ollama reports no "
        "queue wait, so each call's timing breakdown records the observed queue as unavailable"
    ),
    "vllm": (
        "Model-call queue delay is runner-side dispatch only; vLLM's own queue wait for each "
        "request is its measured span attribute (vllm-request-trace-v1), joined after the run "
        "and unavailable when no span arrived. vLLM's /metrics counters are server-wide, "
        "covering every request finished in a window, not one step's request"
    ),
}
LIMITATIONS = (
    "Model-call CPU time is not measured: inference runs in the model server's process",
    (
        "CPU PSI is host-wide and the cgroup counter covers the runner's cgroup; neither is "
        "step-exclusive, and macOS reports both as unsupported"
    ),
    (
        "GPU samples are whole-device NVML gauges (every process on the device), not the model "
        "server's own use; NVML utilization averages over the driver's sample period and does "
        "not establish saturation"
    ),
    (
        "Retrieval is in-process BM25 over fixed task documents, not a vector database; its "
        "service time is the search alone, and the index has no request queue (unsupported)"
    ),
    (
        "Tests run as a trusted-fixture subprocess, not an isolation sandbox; the model can "
        "only select reviewed patches, so no model-written code is executed"
    ),
    "Temperature 0 does not guarantee repeatable model output or timing",
    (
        "Time to first output is observed by the runner on its own clock; it includes the "
        "server's queue, prefill and the local network, and is not a server-measured phase"
    ),
)
VERIFICATION_FAILED = "Agent finished, but the runner's verification tests did not pass"
# Raised by urllib/json/Ollama helpers for a failed or unusable model request.
MODEL_REQUEST_ERRORS = (OSError, http.client.HTTPException, RuntimeError, ValueError)


@dataclass(frozen=True)
class RunOutcome:
    run_id: UUID
    state: str
    reason: str | None
    task_passed: bool
    delivery_complete: bool
    evidence_path: Path
    journal_path: Path
    # Recoverable by `resend` (pending, overflowed or an undelivered closure).
    resend_needed: bool = False
    # Refused by the service: identical content can never be accepted, so not resendable.
    records_refused: int = 0
    # The service refused completion although everything was delivered; closed as failed.
    completion_refused: bool = False


@dataclass(frozen=True)
class _RunContext:
    """What every step of one run needs: its recorder, root step, resources and deadline."""

    steps: StepRecorder
    root: Step
    resources: dict[str, ResourceRegistration]
    dispatcher: ToolDispatcher
    deadline_monotonic: float


def _bounded(value: object) -> object:
    """A JSON-safe, size-bounded copy of untrusted model output for local evidence."""
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        text = json.dumps(repr(value))
    if len(text) > MAX_EVIDENCE_TEXT:
        return {"truncated_json": text[:MAX_EVIDENCE_TEXT]}
    return json.loads(text)


def write_json_atomic(path: Path, data: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class AgentRunner:
    def __init__(
        self,
        *,
        task: Task,
        model: ChatModel,
        transport: LiveTransport,
        output_dir: Path,
        limits: RunLimits | None = None,
        chat_options: dict | None = None,
        sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
        cancel_poll_seconds: float = DEFAULT_CANCEL_POLL_SECONDS,
        flush_interval_seconds: float = 0.5,
        retry: RetryPolicy | None = None,
        clock: MonotonicClock | None = None,
        proc: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        open_gpu: Callable[[], NvmlDevice] | None = None,
        dcgm_metrics_url: str | None = None,
        vllm_traces_file: Path | None = None,
        trace_wait_seconds: float = DEFAULT_TRACE_WAIT_SECONDS,
    ):
        # Without open_gpu no GPU resource is registered or sampled; without a DCGM exporter
        # URL, no DCGM resource. A vLLM model server is always sampled through its /metrics.
        self.open_gpu = open_gpu
        self.dcgm_metrics_url = dcgm_metrics_url
        self.model = model
        # Validated before anything is created, so invalid settings leave nothing behind.
        validate_sample_budget(
            (limits or RunLimits()).run_timeout_seconds,
            sampling_interval_seconds,
            len(self._sampled_metrics()),
        )
        self.task = task
        self.vllm_traces_file = vllm_traces_file
        self.trace_wait_seconds = trace_wait_seconds
        self.transport = transport
        self.output_dir = output_dir
        self.limits = limits or RunLimits()
        self.chat_options = dict(chat_options or DEFAULT_CHAT_OPTIONS)
        self.sampling_interval_seconds = sampling_interval_seconds
        self.cancel_poll_seconds = cancel_poll_seconds
        self.flush_interval_seconds = flush_interval_seconds
        self.retry = retry or RetryPolicy()
        self.clock = clock or MonotonicClock()
        self.proc = proc
        self.cgroup_root = cgroup_root
        self.cancellation = CancellationState()
        self.evidence: dict = {}
        self._tools: tuple[dict, ...] = ()
        self._background: tuple = ()

    def request_cancel(self, source: str, reason: str) -> None:
        """Thread-safe: stop at the next checkpoint and clean up."""
        self.cancellation.cancel(source, reason)

    def request_cancel_from_signal(self, source: str, reason: str) -> None:
        """Safe to call from a signal handler, including for repeated signals."""
        self.cancellation.cancel_from_signal(source, reason)

    # Run orchestration ---------------------------------------------------------------------

    def run(self) -> RunOutcome:
        """Register, execute and close one live run; evidence is written on every path."""
        evidence_path = self.output_dir / EVIDENCE_FILE
        journal_path = self.output_dir / JOURNAL_FILE
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            journal = Journal(journal_path)
        except OSError as error:
            # No evidence can be written here, so the error itself is the whole report.
            raise RunNotStarted(
                f"cannot write run evidence to {self.output_dir}: {type(error).__name__}: {error}",
                "setup",
            ) from error
        self.evidence = {
            "status": "starting",
            "task": {
                "task_id": self.task.task_id,
                "name": self.task.name,
                "revision": self.task.revision,
                "content_sha256": self.task.content_sha256,
            },
            "model_endpoint": self.model.endpoint,
            "clock_id": self.clock.clock_id,
            "model_backend": self.model.backend,
            "model_calls": [],
            "tool_calls": [],
            "limitations": [MODEL_QUEUE_LIMITATIONS[self.model.backend], *LIMITATIONS],
        }
        with closing(journal):
            workspace: Workspace | None = None
            stage = "setup"  # becomes "registration" once the live API is contacted
            try:
                workspace = Workspace.create(self.task)
                build_started = time.monotonic_ns()
                index = DocumentIndex.from_directory(self.task.docs_dir)
                self.evidence["retrieval_index"] = {
                    "method": RETRIEVAL_METHOD,
                    "passages": len(index.passages),
                    "documents": len({passage.document for passage in index.passages}),
                    # Built during setup, before the run is registered: no step covers it.
                    "build_wall_seconds": (time.monotonic_ns() - build_started) / 1e9,
                }
                dispatcher = ToolDispatcher(self.task, workspace, index, self.limits)
                self._tools = dispatcher.definitions()
                resources = self._resources()
                configuration = self._configuration()
                self.evidence |= {
                    "configuration": configuration.as_dict(),
                    "configuration_digest": configuration.digest(),
                    "resources": [r.model_dump() for r in resources.values()],
                }
                registration = RunCreate(
                    contract_version=CONTRACT_VERSION,
                    request_id=uuid4(),
                    task_id=self.task.task_id,
                    configuration_digest=configuration.digest(),
                    clock_ids=[self.clock.clock_id],
                    resources=list(resources.values()),
                )
                stage = "registration"
                session = RunSession.open(self.transport, registration, journal, self.retry)
            # Nothing ran: keep the attempt's evidence, leave no workspace, and re-raise.
            except Exception as error:
                described = f"{type(error).__name__}: {error}"
                # A transient registration failure has an unknown outcome; RunSession.open
                # then journaled a closure, so `resend` closes the run if it exists.
                unconfirmed = stage == "registration" and is_retryable(error)
                self.evidence |= {
                    "status": "not_started",
                    "error": f"Run {stage} failed: {described}",
                    "registration_outcome": "unknown" if unconfirmed else None,
                    "cleanup": {"workspace_removed": workspace.remove() if workspace else True},
                }
                write_json_atomic(evidence_path, self.evidence)
                raise RunNotStarted(described, stage, closure_journaled=unconfirmed) from error
            return self._run_registered(
                session, journal, workspace, dispatcher, resources, evidence_path, journal_path
            )

    def _run_registered(
        self,
        session: RunSession,
        journal: Journal,
        workspace: Workspace,
        dispatcher: ToolDispatcher,
        resources: dict[str, ResourceRegistration],
        evidence_path: Path,
        journal_path: Path,
    ) -> RunOutcome:
        self.evidence["run_id"] = str(session.run_id)
        self.evidence["trace_id"] = trace_id = new_trace_id()
        delivery = LiveDelivery(self.transport, session.run_id, journal, retry=self.retry)
        steps = StepRecorder(
            run_id=session.run_id,
            task_id=self.task.task_id,
            trace_id=trace_id,
            clock=self.clock,
            emit=delivery.emit,
        )
        state, reason, task_passed = "failed", None, False
        errors: list[BaseException] = []
        self._background = ()
        sampler: ResourceSampler | None = None
        service_sampler: ResourceSampler | None = None
        watcher: CancellationWatcher | None = None
        background = ExitStack()
        try:
            flusher = background.enter_context(
                PeriodicFlusher(delivery, self.flush_interval_seconds)
            )
            sampler = background.enter_context(
                ResourceSampler(
                    delivery.emit,
                    run_id=session.run_id,
                    task_id=self.task.task_id,
                    clock=self.clock,
                    probes=self._probes(resources),
                    interval_seconds=self.sampling_interval_seconds,
                )
            )
            watcher = background.enter_context(
                CancellationWatcher(session, self.cancellation, self.cancel_poll_seconds)
            )
            service_probes = self._service_probes(resources)
            service_sampler = (
                background.enter_context(
                    ResourceSampler(
                        delivery.emit,
                        run_id=session.run_id,
                        task_id=self.task.task_id,
                        clock=self.clock,
                        probes=service_probes,
                        interval_seconds=self.sampling_interval_seconds,
                    )
                )
                if service_probes
                else None
            )
            self._background = tuple(
                component
                for component in (flusher, sampler, service_sampler, watcher)
                if component is not None
            )
            deadline = time.monotonic() + self.limits.run_timeout_seconds
            try:
                started_state = session.change_state("running", None)
            except TRANSIENT_ERRORS as error:
                if not is_retryable(error):
                    raise
                # Recoverable: the closure is deferred and `resend` replays both changes.
                raise RunFailure(
                    "Live API unavailable when starting the run",
                    f"{type(error).__name__}: {error}",
                ) from error
            if started_state != "running":
                raise RunCancelled(session.reason or f"Run was {session.state} before it started")
            # Submitted only now, so the root's queue delay is not runner start-up overhead.
            root = steps.submit("agent-run", "other", parent=None, resource=None)
            steps.start(root)
            task_passed = self._execute(_RunContext(steps, root, resources, dispatcher, deadline))
            state = "completed" if task_passed else "failed"
            reason = None if task_passed else VERIFICATION_FAILED
        except RunCancelled as cancelled:
            request = self.cancellation.request
            reason = request.reason if request else cancelled.reason
            # Another actor may have *failed* the run; its steps then end as failed, too.
            remote_failure = request is not None and request.state == "failed"
            state = "failed" if remote_failure else "cancelled"
        except RunFailure as failure:
            state, reason = "failed", failure.reason
            self.evidence["failure_detail"] = failure.detail
        # Recorded and closed as a failed run, then re-raised below: runner bugs must surface.
        except Exception as error:  # noqa: BLE001
            errors.append(error)
            state, reason = "failed", f"Runner error ({type(error).__name__})"
            self.evidence["failure_detail"] = f"{type(error).__name__}: {error}"
        finally:
            # A background thread that died after the last checkpoint still decides the run.
            state, reason = self._absorb_background_errors(errors, state, reason)
            # End steps before stopping threads and removing the workspace, so their finish
            # ticks do not include shutdown or cleanup time.
            self._attempt(errors, lambda: self._end_unfinished(steps, state, reason))
            self._attempt(errors, background.close)
            state, reason = self._absorb_background_errors(errors, state, reason)
            final_state, reason = self._finish(session, delivery, workspace, state, reason, errors)
            # After the run closed, so waiting for spans never lengthens the run itself.
            self._attempt(errors, lambda: self._join_server_traces(trace_id))
            self._record_outcome(
                final_state, reason, task_passed, delivery, sampler, steps, watcher
            )
            # None when the run has no service probes or failed before sampling started.
            self.evidence["service_sampling"] = (
                service_sampler.summary() if service_sampler is not None else None
            )
            write_json_atomic(evidence_path, self.evidence)
        if errors:
            raise errors[0]
        return RunOutcome(
            run_id=session.run_id,
            state=final_state,
            reason=self.evidence["reason"],
            task_passed=task_passed,
            delivery_complete=self.evidence["delivery"]["complete"],
            evidence_path=evidence_path,
            journal_path=journal_path,
            resend_needed=self.evidence["delivery"]["resend_needed"],
            records_refused=self.evidence["delivery"]["rejected"],
            completion_refused=self.evidence.get("completion_refused", False),
        )

    def _absorb_background_errors(
        self, errors: list[BaseException], state: str, reason: str | None
    ) -> tuple[str, str | None]:
        """Record new flusher/sampler/watcher errors; a run cannot complete after one."""
        recorded = self.evidence.setdefault("background_errors", [])
        new = []
        for component in self._background:
            error = component.error
            if error is None:
                continue
            described = f"{type(component).__name__}: {type(error).__name__}: {error}"[:500]
            if described not in recorded:
                recorded.append(described)
            if error not in errors:
                errors.append(error)
                new.append(error)
        if not new:
            return state, reason
        if state == "completed":
            return "failed", f"Runner error ({type(new[0]).__name__})"
        return state, reason

    @staticmethod
    def _end_unfinished(steps: StepRecorder, state: str, reason: str | None) -> None:
        for step in steps.unfinished():
            if state == "cancelled":
                steps.cancel(step, reason or "Run cancelled")
            else:
                steps.fail(step, reason or "Run failed")

    def _configuration(self) -> RunConfiguration:
        return RunConfiguration(
            task_id=self.task.task_id,
            task_content_sha256=self.task.content_sha256,
            model=self.model.name,
            system_prompt=SYSTEM_PROMPT,
            tools=self._tools,
            sampled_metrics=self._sampled_metrics(),
            model_backend=self.model.backend,
            model_transport="stream",
            chat_options=self.chat_options,
            limits=self.limits,
            sampling_interval_seconds=self.sampling_interval_seconds,
            cancel_poll_seconds=self.cancel_poll_seconds,
        )

    def _attempt[Result](
        self,
        errors: list[BaseException],
        operation: Callable[[], Result],
        default: Result | None = None,
    ) -> Result | None:
        """Run one cleanup action so a failure cannot skip the actions after it."""
        try:
            return operation()
        # Recorded in evidence and re-raised by the caller once every action has run.
        except Exception as error:  # noqa: BLE001
            errors.append(error)
            self.evidence.setdefault("cleanup_errors", []).append(
                f"{type(error).__name__}: {error}"[:500]
            )
            return default

    def _finish(
        self,
        session: RunSession,
        delivery: LiveDelivery,
        workspace: Workspace,
        state: str,
        reason: str | None,
        errors: list[BaseException],
    ) -> tuple[str, str | None]:
        """Remove the workspace, deliver what is queued, and close (or defer) the run."""
        self.evidence["cleanup"] = {
            "workspace_removed": self._attempt(errors, workspace.remove, False),
            "test_process_groups": "killed by run_project_tests on every exit path",
            "abandoned_model_request": any(
                call.get("abandoned_request") for call in self.evidence["model_calls"]
            ),
        }
        delivered = self._attempt(errors, delivery.flush, False)
        closed = self._attempt(
            errors, lambda: self._close_run(session, state, reason, bool(delivered))
        )
        if closed is None:
            self.evidence["run_state_delivered"] = False
            return f"{state}_locally_undelivered", reason
        return closed

    def _record_outcome(
        self,
        final_state: str,
        reason: str | None,
        task_passed: bool,
        delivery: LiveDelivery,
        sampler: ResourceSampler | None,
        steps: StepRecorder,
        watcher: CancellationWatcher | None,
    ) -> None:
        stats = delivery.stats()
        self.evidence |= {
            "status": final_state,
            "reason": redact_reason(reason) if reason else None,
            "task_passed": task_passed,
            "cancellation": self._cancellation_evidence(watcher),
            # None when the run failed before sampling started, distinct from zero samples.
            "resource_sampling": sampler.summary() if sampler is not None else None,
            "delivery": {
                "complete": stats.complete and self.evidence.get("run_state_delivered", True),
                # Something `resend` can still deliver: queued, overflowed or the closure.
                "resend_needed": stats.pending > 0
                or stats.not_queued_overflow > 0
                or not self.evidence.get("run_state_delivered", True),
                "emitted": stats.emitted,
                "delivered": stats.delivered,
                "pending": stats.pending,
                "not_queued_overflow": stats.not_queued_overflow,
                "rejected": stats.rejected,
                "rejections": stats.rejections,
                "transient_failures": stats.transient_failures,
                "last_error": stats.last_error,
            },
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_type": s.step_type,
                    "span_id": s.span_id,
                    "parent_step_id": s.parent_step_id,
                    "resource_id": s.resource_id,
                    "state": s.state,
                }
                for s in steps.steps
            ],
        }

    def _resources(self) -> dict[str, ResourceRegistration]:
        # Opaque per-runner-lifetime IDs: a new runner process is a new measured resource.
        suffix = uuid4().hex[:12]
        return (
            {
                "tool_host": ResourceRegistration(
                    resource_id=f"tool-host-{suffix}",
                    service_id="agentwatch-agent-runner",
                    resource_class="cpu_sandbox",
                ),
                "inference": ResourceRegistration(
                    resource_id=f"model-server-{suffix}",
                    service_id=self.model.service_id,
                    resource_class="inference",
                ),
                "docs_index": ResourceRegistration(
                    resource_id=f"docs-index-{suffix}",
                    service_id="agentwatch-agent-docs-index",
                    resource_class="retrieval",
                ),
            }
            | (
                {
                    # The GPU device itself, sampled whole; model placement links it to the server.
                    "gpu": ResourceRegistration(
                        resource_id=f"gpu-{suffix}",
                        service_id="nvidia-nvml",
                        resource_class="inference",
                    )
                }
                if self.open_gpu is not None
                else {}
            )
            | (
                {
                    # The same device read through DCGM: its own resource, so the two services'
                    # readings are never merged into one series.
                    "gpu_dcgm": ResourceRegistration(
                        resource_id=f"gpu-dcgm-{suffix}",
                        service_id="nvidia-dcgm",
                        resource_class="inference",
                    )
                }
                if self.dcgm_metrics_url is not None
                else {}
            )
        )

    def _sampled_metrics(self) -> tuple[str, ...]:
        metrics = CPU_METRICS + (GPU_METRICS if self.open_gpu is not None else ())
        service = (VLLM_METRICS if self.model.backend == "vllm" else ()) + (
            DCGM_METRICS if self.dcgm_metrics_url is not None else ()
        )
        return tuple(metric for metric, _unit in metrics) + tuple(e.metric for e in service)

    def _probes(self, resources: dict[str, ResourceRegistration]) -> list[Probe]:
        probes: list[Probe] = [CpuProbe(resources["tool_host"], self.proc, self.cgroup_root)]
        if self.open_gpu is not None:
            probes.append(GpuProbe(resources["gpu"], self.open_gpu))
        return probes

    def _service_probes(self, resources: dict[str, ResourceRegistration]) -> list[Probe]:
        """HTTP metrics sources, on their own thread so a slow fetch cannot delay CPU/GPU."""
        probes: list[Probe] = []
        if self.model.backend == "vllm":
            log = ExpositionChangeLog(self.output_dir / VLLM_CHANGE_LOG_FILE, self.clock)
            probes.append(
                vllm_probe(resources["inference"], self.model.endpoint, self.model.name, log)
            )
        if self.dcgm_metrics_url is not None:
            probes.append(dcgm_probe(resources["gpu_dcgm"], self.dcgm_metrics_url))
        return probes

    def _join_server_traces(self, trace_id: str) -> None:
        """Attach vLLM's measured per-request timing to each model call that was sent."""
        if self.model.backend != "vllm":
            return
        calls = [c for c in self.evidence["model_calls"] if isinstance(c.get("timing"), dict)]
        if self.vllm_traces_file is None:
            self.evidence["server_traces"] = {"status": "not_configured"}
            unavailable = {"status": "unavailable", "problems": ["no vLLM traces file given"]}
            for call in calls:
                call["timing"] = joined_vllm_timing(call["timing"], unavailable)
            return
        collected = collect_request_timings(
            self.vllm_traces_file,
            trace_id,
            {call["span_id"]: call["timing"].get("completion_id") for call in calls},
            wait_seconds=self.trace_wait_seconds,
        )
        for call in calls:
            call["timing"] = joined_vllm_timing(call["timing"], collected["calls"][call["span_id"]])
        self.evidence["server_traces"] = {k: v for k, v in collected.items() if k != "calls"}

    def _close_run(
        self, session: RunSession, state: str, reason: str | None, delivered: bool
    ) -> tuple[str, str | None]:
        """Close the live run, or defer the closure until its evidence can be delivered.

        Returns the resulting state and the reason actually recorded. A failed or cancelled
        closure makes the service refuse steps it has not yet seen, so while any record is
        undelivered the closure is only journaled: `resend` replays the records first and then
        this closure, in order.
        """
        request = self.cancellation.request
        try:
            if request is not None and request.source == "live_api":
                # Closed remotely; adopt that state instead of sending a new change.
                adopted = session.adopt_remote_closure(request.state or state, request.reason)
                return adopted, request.reason
            if not delivered:
                session.defer_change(state, reason)
                self.evidence["run_state_delivered"] = False
                self.evidence["run_state_error"] = "Closure deferred until records are resent"
                return f"{state}_locally_undelivered", reason
            result = session.change_state(state, reason)
            if result in TERMINAL_RUN_STATES:
                return result, reason
            if state == "completed":
                # Every record was delivered, so the refusal is permanent: refused or expired
                # step evidence. Resending cannot help; close the run as failed, truthfully.
                self.evidence["completion_refused"] = True
                result = session.change_state("failed", COMPLETION_REFUSED)
                if result in TERMINAL_RUN_STATES:
                    return result, COMPLETION_REFUSED
            self.evidence["run_state_delivered"] = False
            self.evidence["run_state_error"] = "Closure refused by the live API"
            return f"{state}_locally_undelivered", reason
        except TRANSIENT_ERRORS as error:
            if not is_retryable(error):
                raise
            self.evidence["run_state_delivered"] = False
            self.evidence["run_state_error"] = f"{type(error).__name__}: {error}"[:500]
            return f"{state}_locally_undelivered", reason

    def _cancellation_evidence(self, watcher: CancellationWatcher | None) -> dict:
        request = self.cancellation.request
        return {
            "requested": request is not None,
            "source": request.source if request else None,
            "reason": redact_reason(request.reason) if request else None,
            "poll_failures": watcher.poll_failures if watcher else None,
            "last_poll_error": watcher.last_error if watcher else None,
            # The runner stopped its own work; this is not service-confirmed cleanup.
            "cleanup_confirmed_by_service": None,
        }

    # Agent loop ------------------------------------------------------------------------------

    def _raise_background_errors(self) -> None:
        for component in self._background:
            if component.error is not None:
                raise component.error

    def _checkpoint(self, deadline: float) -> None:
        self._raise_background_errors()
        if self.cancellation.event.is_set():
            request = self.cancellation.request
            raise RunCancelled(request.reason if request else "Cancellation requested")
        if time.monotonic() >= deadline:
            raise RunFailure(self._run_limit_reason())

    def _run_limit_reason(self) -> str:
        # One wording wherever the deadline is hit: reasons feed dataset labels and slices.
        return f"Run exceeded its {self.limits.run_timeout_seconds:g} s time limit"

    def _execute(self, context: _RunContext) -> bool:
        self._converse(context)
        passed = self._verify(context)
        # A background thread that died during verification must not let the run complete.
        self._raise_background_errors()
        if passed:
            context.steps.complete(context.root)
        else:
            # The root step carries the verification outcome, so it fails with that reason. A
            # runner error after the root completed is reported on the run and in evidence.
            context.steps.fail(context.root, VERIFICATION_FAILED)
        return passed

    def _converse(self, context: _RunContext) -> None:
        """The agent loop: model calls and the tool calls they request, until an answer."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_message(self.task)},
        ]
        self.evidence["messages"] = messages
        model_calls = tool_calls = 0
        while True:
            self._checkpoint(context.deadline_monotonic)
            if model_calls >= self.limits.max_model_calls:
                raise RunFailure(
                    f"Model-call limit ({self.limits.max_model_calls}) reached without an answer"
                )
            model_calls += 1
            reply = self._model_step(context, messages, model_calls)
            messages.append(reply.history_message)
            if not reply.calls:
                if not reply.content.strip():
                    raise RunFailure("Model replied with neither a tool call nor an answer")
                self.evidence["final_answer"] = reply.content
                break
            for call in reply.calls:
                self._checkpoint(context.deadline_monotonic)
                if tool_calls >= self.limits.max_tool_calls:
                    raise RunFailure(f"Tool-call limit ({self.limits.max_tool_calls}) reached")
                tool_calls += 1
                messages.append(self._tool_step(context, call, tool_calls))

    def _model_step(self, context: _RunContext, messages: list[dict], index: int) -> ModelReply:
        steps = context.steps
        step = steps.submit(
            f"model-call-{index}",
            "model_call",
            parent=context.root,
            resource=context.resources["inference"],
        )
        record = {"step_id": step.step_id, "span_id": step.span_id, "request_index": index}
        self.evidence["model_calls"].append(record)
        payload = chat_payload(self.model.name, messages, list(self._tools), self.chat_options)
        timer = StreamTimer(self.clock)
        trace_context = traceparent(steps.trace_id, step.span_id)

        def request() -> dict:
            timer("request")
            return self.model.chat(payload, trace_context=trace_context, on_event=timer)

        steps.start(step)
        started = time.monotonic()
        failure: RunFailure | None = None
        try:
            response = wait_interruptibly(
                request,
                cancel=self.cancellation.event,
                deadline_monotonic=context.deadline_monotonic,
            )
            reply = parse_reply(response)
        except RunCancelled as cancelled:
            record["abandoned_request"] = cancelled.request_started
            raise
        except RunFailure as interrupted:
            # Only the run deadline interrupts the wait with a RunFailure.
            record["abandoned_request"] = interrupted.request_started
            failure = RunFailure(self._run_limit_reason(), interrupted.reason)
        except ModelResponseError as error:
            failure = RunFailure("Model server returned a malformed chat response", str(error))
        except MODEL_REQUEST_ERRORS as error:
            record["error"] = f"{type(error).__name__}: {error}"[:MAX_EVIDENCE_TEXT]
            failure = RunFailure(f"Model request failed ({type(error).__name__})", record["error"])
        finally:
            # Measured before any placement read, so it covers only the model request.
            record["request_wall_seconds"] = time.monotonic() - started
            record["stream_timing"] = timer.summary()
        if failure is not None:
            steps.fail(step, failure.reason)
            if failure.reason != self._run_limit_reason():
                # Where the model was when its request failed helps diagnose the failure.
                record["model_placement"] = self._observe_placement(context.deadline_monotonic)
            raise failure
        steps.complete(step)
        record["metrics"] = reply.metrics
        record["timing"] = self.model.request_timing(reply.metrics, record["request_wall_seconds"])
        record["requested_tool_calls"] = len(reply.calls)
        # Read after the step closed so the extra request is not timed as inference.
        record["model_placement"] = self._observe_placement(context.deadline_monotonic)
        return reply

    def _observe_placement(self, deadline_monotonic: float) -> dict | None:
        """Read optional model placement without delaying cancellation or the run deadline.

        Placement is ancillary evidence: an interrupted or skipped reading is recorded as
        unavailable, and the next checkpoint ends the run for the actual reason.
        """
        try:
            return wait_interruptibly(
                self.model.observe_placement,
                cancel=self.cancellation.event,
                deadline_monotonic=deadline_monotonic,
                activity="a model placement reading",
            )
        except (RunCancelled, RunFailure) as interrupted:
            return {
                "source": "model placement",
                "status": "unavailable",
                "reason": f"Not read: {interrupted}",
            }

    def _tool_step(self, context: _RunContext, call: RequestedCall, index: int) -> dict:
        steps, root, dispatcher = context.steps, context.root, context.dispatcher
        record = {"requested_name": _bounded(call.name), "arguments": _bounded(call.arguments)}
        self.evidence["tool_calls"].append(record)
        try:
            prepared = dispatcher.prepare(call.name, call.arguments)
        except ToolRejected as rejected:
            # Never executed: queued then failed at revision 1, with no start or CPU time.
            step = steps.submit(f"tool-{index}-rejected", "tool", parent=root, resource=None)
            steps.fail(step, f"Rejected tool call: {rejected}")
            record |= {
                "step_id": step.step_id,
                "span_id": step.span_id,
                "status": "rejected",
                "reason": str(rejected),
            }
            name = call.name if isinstance(call.name, str) else "unknown"
            return tool_message(name[:64], {"error": f"Not executed. {rejected}"})
        step = steps.submit(
            f"tool-{index}-{prepared.name}",
            prepared.step_type,
            parent=root,
            resource=context.resources[prepared.resource],
        )
        record |= {"step_id": step.step_id, "span_id": step.span_id, "name": prepared.name}
        steps.start(step)
        try:
            outcome = dispatcher.execute(
                prepared,
                cancel=self.cancellation.event,
                deadline_monotonic=context.deadline_monotonic,
            )
        except ToolFailed as failed:
            if time.monotonic() >= context.deadline_monotonic:
                # The tool was cut short by the run limit, not by its own failure.
                steps.fail(step, self._run_limit_reason())
                record |= {"status": "failed", "reason": self._run_limit_reason()}
                raise RunFailure(self._run_limit_reason(), str(failed)) from failed
            steps.fail(step, f"{prepared.name} failed: {failed}")
            record |= {"status": "failed", "reason": str(failed), "evidence": failed.evidence}
            return tool_message(prepared.name, {"error": str(failed)})
        steps.complete(step, cpu_seconds=outcome.cpu_seconds, cpu_scope=outcome.cpu_scope)
        record |= {"status": "completed", "evidence": outcome.evidence}
        return tool_message(prepared.name, outcome.content)

    def _verify(self, context: _RunContext) -> bool:
        """Re-run the tests independently: the model's own claims are never the outcome."""
        self._checkpoint(context.deadline_monotonic)
        steps = context.steps
        step = steps.submit(
            "verify-tests", "tool", parent=context.root, resource=context.resources["tool_host"]
        )
        steps.start(step)
        try:
            outcome = context.dispatcher.run_tests(
                cancel=self.cancellation.event, deadline_monotonic=context.deadline_monotonic
            )
        except ToolFailed as failed:
            # The tests did not finish (timeout or no run time left): that is not a test
            # failure, so the run fails for the actual reason instead of VERIFICATION_FAILED.
            if time.monotonic() >= context.deadline_monotonic:
                reason = self._run_limit_reason()
            else:
                reason = f"Runner verification could not complete: {failed}"
            steps.fail(step, reason)
            self.evidence["verification"] = {
                "step_id": step.step_id,
                "error": str(failed),
            } | failed.evidence
            raise RunFailure(reason, str(failed)) from failed
        steps.complete(step, cpu_seconds=outcome.cpu_seconds, cpu_scope=outcome.cpu_scope)
        self.evidence["verification"] = {"step_id": step.step_id} | outcome.evidence
        return outcome.evidence["outcome"] == "passed"


def default_output_dir(base: Path, task: Task) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return base / f"{task.name}-{stamp}-{uuid4().hex[:6]}"
