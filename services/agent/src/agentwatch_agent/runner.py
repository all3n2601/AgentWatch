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
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from agentwatch_sdk.live import ResourceRegistration, RunCreate

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import (
    CONTRACT_VERSION,
    DEFAULT_CANCEL_POLL_SECONDS,
    DEFAULT_CHAT_OPTIONS,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    RunConfiguration,
    RunLimits,
)
from agentwatch_agent.delivery import (
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
    parse_reply,
    wait_for_model,
)
from agentwatch_agent.prompts import SYSTEM_PROMPT, task_message
from agentwatch_agent.redaction import redact_reason
from agentwatch_agent.resources import CpuResourceSampler
from agentwatch_agent.retrieval import DocumentIndex
from agentwatch_agent.session import (
    TERMINAL_RUN_STATES,
    CancellationState,
    CancellationWatcher,
    RunSession,
)
from agentwatch_agent.steps import Step, StepRecorder, new_trace_id
from agentwatch_agent.tasks import Task
from agentwatch_agent.tools import ToolDispatcher, ToolFailed, ToolRejected, tool_message
from agentwatch_agent.workspace import Workspace

EVIDENCE_FILE = "agent-evidence.json"
JOURNAL_FILE = "live-journal.jsonl"
MAX_EVIDENCE_TEXT = 2000
LIMITATIONS = (
    (
        "Model-call queue delay is runner-side dispatch only; waiting inside the model server "
        "is part of execution wall time, not a measured inference queue"
    ),
    "Model-call CPU time is not measured: inference runs in the model server's process",
    (
        "CPU PSI is host-wide and the cgroup counter covers the runner's cgroup; neither is "
        "step-exclusive, and macOS reports both as unsupported"
    ),
    "Retrieval is in-process BM25 over fixed task documents, not a vector database",
    (
        "Tests run as a trusted-fixture subprocess, not an isolation sandbox; the model can "
        "only select reviewed patches, so no model-written code is executed"
    ),
    "Temperature 0 does not guarantee repeatable model output or timing",
)
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
    ):
        self.task = task
        self.model = model
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

    def request_cancel(self, source: str, reason: str) -> None:
        """Thread- and signal-safe: stop at the next checkpoint and clean up."""
        self.cancellation.cancel(source, reason)

    # Run orchestration ---------------------------------------------------------------------

    def run(self) -> RunOutcome:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = self.output_dir / EVIDENCE_FILE
        journal_path = self.output_dir / JOURNAL_FILE
        journal = Journal(journal_path)
        workspace = Workspace.create(self.task)
        dispatcher = ToolDispatcher(
            self.task, workspace, DocumentIndex.from_directory(self.task.docs_dir), self.limits
        )
        configuration = RunConfiguration(
            task_id=self.task.task_id,
            task_content_sha256=self.task.content_sha256,
            model=self.model.name,
            system_prompt=SYSTEM_PROMPT,
            tools=dispatcher.definitions(),
            chat_options=self.chat_options,
            limits=self.limits,
            sampling_interval_seconds=self.sampling_interval_seconds,
            cancel_poll_seconds=self.cancel_poll_seconds,
        )
        resources = self._resources()
        self.evidence = {
            "status": "starting",
            "task": {
                "task_id": self.task.task_id,
                "name": self.task.name,
                "revision": self.task.revision,
                "content_sha256": self.task.content_sha256,
            },
            "configuration": configuration.as_dict(),
            "configuration_digest": configuration.digest(),
            "model_endpoint": self.model.endpoint,
            "clock_id": self.clock.clock_id,
            "resources": [r.model_dump() for r in resources.values()],
            "model_calls": [],
            "tool_calls": [],
            "limitations": list(LIMITATIONS),
        }
        registration = RunCreate(
            contract_version=CONTRACT_VERSION,
            request_id=uuid4(),
            task_id=self.task.task_id,
            configuration_digest=configuration.digest(),
            clock_ids=[self.clock.clock_id],
            resources=list(resources.values()),
        )
        try:
            session = RunSession.open(self.transport, registration, journal, self.retry)
        # Evidence of the failed attempt is written, then the error is re-raised.
        except Exception as error:
            # Nothing ran; keep the attempt's evidence and leave no workspace behind.
            self.evidence |= {
                "status": "not_started",
                "error": f"Live run registration failed: {type(error).__name__}: {error}",
                "cleanup": {"workspace_removed": workspace.remove()},
            }
            write_json_atomic(evidence_path, self.evidence)
            journal.close()
            raise RunNotStarted(f"{type(error).__name__}: {error}") from error
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
        root = steps.submit("agent-run", "other", parent=None, resource=None)
        sampler = CpuResourceSampler(
            delivery.emit,
            run_id=session.run_id,
            task_id=self.task.task_id,
            clock=self.clock,
            resource=resources["tool_host"],
            interval_seconds=self.sampling_interval_seconds,
            proc=self.proc,
            cgroup_root=self.cgroup_root,
        )
        watcher = CancellationWatcher(session, self.cancellation, self.cancel_poll_seconds)
        flusher = PeriodicFlusher(delivery, self.flush_interval_seconds)
        self._background = (flusher, sampler, watcher)
        deadline = time.monotonic() + self.limits.run_timeout_seconds

        self._tools = dispatcher.definitions()
        state, reason, task_passed, unexpected = "failed", None, False, None
        final_state = "failed"
        flusher.__enter__()
        sampler.__enter__()
        watcher.__enter__()
        try:
            if session.change_state("running", None) != "running":
                raise RunCancelled(session.reason or f"Run was {session.state} before it started")
            steps.start(root)
            context = _RunContext(steps, root, resources, dispatcher, deadline)
            task_passed = self._execute(context)
            state = "completed" if task_passed else "failed"
            if not task_passed:
                reason = "Agent finished, but the runner's verification tests did not pass"
        except RunCancelled as cancelled:
            request = self.cancellation.request
            state, reason = "cancelled", request.reason if request else cancelled.reason
        except RunFailure as failure:
            state, reason = "failed", failure.reason
            self.evidence["failure_detail"] = failure.detail
        # Recorded and closed as a failed run, then re-raised below: runner bugs must surface.
        except Exception as error:  # noqa: BLE001
            unexpected = error
            state, reason = "failed", f"Runner error ({type(error).__name__})"
            self.evidence["failure_detail"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                watcher.__exit__(None, None, None)
                sampler.__exit__(None, None, None)
                self.evidence["cleanup"] = {
                    "workspace_removed": workspace.remove(),
                    "test_process_groups": "killed by run_project_tests on every exit path",
                    "abandoned_model_request": any(
                        call.get("abandoned_request") for call in self.evidence["model_calls"]
                    ),
                }
                for step in steps.unfinished():
                    if state == "cancelled":
                        steps.cancel(step, reason or "Run cancelled")
                    else:
                        steps.fail(step, reason or "Run failed")
                flusher.__exit__(None, None, None)
                delivery.flush()
                final_state = self._close_run(session, state, reason)
            finally:
                self._record_outcome(
                    final_state, reason, task_passed, delivery, sampler, steps, watcher
                )
                write_json_atomic(evidence_path, self.evidence)
                journal.close()
        if unexpected is not None:
            raise unexpected
        for component in (flusher, sampler, watcher):
            if component.error is not None:
                raise component.error
        return RunOutcome(
            run_id=session.run_id,
            state=final_state,
            reason=self.evidence["reason"],
            task_passed=task_passed,
            delivery_complete=self.evidence["delivery"]["complete"],
            evidence_path=evidence_path,
            journal_path=journal_path,
        )

    def _record_outcome(
        self,
        final_state: str,
        reason: str | None,
        task_passed: bool,
        delivery: LiveDelivery,
        sampler: CpuResourceSampler,
        steps: StepRecorder,
        watcher: CancellationWatcher,
    ) -> None:
        stats = delivery.stats()
        self.evidence |= {
            "status": final_state,
            "reason": redact_reason(reason) if reason else None,
            "task_passed": task_passed,
            "cancellation": self._cancellation_evidence(watcher),
            "resource_sampling": sampler.summary(),
            "delivery": {
                "complete": stats.complete and self.evidence.get("run_state_delivered", True),
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
        return {
            "tool_host": ResourceRegistration(
                resource_id=f"tool-host-{suffix}",
                service_id="agentwatch-agent-runner",
                resource_class="cpu_sandbox",
            ),
            "inference": ResourceRegistration(
                resource_id=f"model-server-{suffix}",
                service_id="ollama",
                resource_class="inference",
            ),
            "docs_index": ResourceRegistration(
                resource_id=f"docs-index-{suffix}",
                service_id="agentwatch-agent-docs-index",
                resource_class="retrieval",
            ),
        }

    def _close_run(self, session: RunSession, state: str, reason: str | None) -> str:
        """Close the live run, or report why the service could not record the closure.

        The change is journaled even when it cannot be delivered, so a later `resend` replays
        the outstanding records and then this closure, in order.
        """
        request = self.cancellation.request
        try:
            if request is not None and request.source == "live_api":
                # Already closed remotely; adopt the server's state instead of a new change.
                session.refresh()
                return session.state
            result = session.change_state(state, reason)
            if result not in TERMINAL_RUN_STATES:
                # The service refused completion because some step evidence never arrived.
                self.evidence["run_state_delivered"] = False
                self.evidence["run_state_error"] = "Closure refused until evidence is resent"
                return f"{state}_locally_undelivered"
            return result
        except TRANSIENT_ERRORS as error:
            if not is_retryable(error):
                raise
            self.evidence["run_state_delivered"] = False
            self.evidence["run_state_error"] = f"{type(error).__name__}: {error}"[:500]
            return f"{state}_locally_undelivered"

    def _cancellation_evidence(self, watcher: CancellationWatcher) -> dict:
        request = self.cancellation.request
        return {
            "requested": request is not None,
            "source": request.source if request else None,
            "reason": redact_reason(request.reason) if request else None,
            "poll_failures": watcher.poll_failures,
            "last_poll_error": watcher.last_error,
            # The runner stopped its own work; this is not service-confirmed cleanup.
            "cleanup_confirmed_by_service": None,
        }

    # Agent loop ------------------------------------------------------------------------------

    def _checkpoint(self, deadline: float) -> None:
        for component in self._background:
            if component.error is not None:
                raise component.error
        if self.cancellation.event.is_set():
            request = self.cancellation.request
            raise RunCancelled(request.reason if request else "Cancellation requested")
        if time.monotonic() >= deadline:
            limit = self.limits.run_timeout_seconds
            raise RunFailure(f"Run exceeded its {limit:g} s time limit")

    def _execute(self, context: _RunContext) -> bool:
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
        passed = self._verify(context)
        context.steps.complete(context.root)
        return passed

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
        payload = {
            "model": self.model.name,
            "messages": messages,
            "tools": list(self._tools),
            "stream": False,
            "think": False,
            "options": dict(self.chat_options),
        }
        steps.start(step)
        started = time.monotonic()
        try:
            response = wait_for_model(
                lambda: self.model.chat(payload),
                cancel=self.cancellation.event,
                deadline_monotonic=context.deadline_monotonic,
            )
            reply = parse_reply(response)
        except RunCancelled:
            record["abandoned_request"] = True
            raise
        except RunFailure as failure:
            record["abandoned_request"] = True
            steps.fail(step, failure.reason)
            raise
        except ModelResponseError as error:
            steps.fail(step, "Model server returned a malformed chat response")
            raise RunFailure("Model server returned a malformed chat response", str(error))
        except MODEL_REQUEST_ERRORS as error:
            reason = f"Model request failed ({type(error).__name__})"
            record["error"] = f"{type(error).__name__}: {error}"[:MAX_EVIDENCE_TEXT]
            steps.fail(step, reason)
            raise RunFailure(reason, record["error"]) from error
        finally:
            record["request_wall_seconds"] = time.monotonic() - started
        steps.complete(step)
        record["metrics"] = reply.metrics
        record["requested_tool_calls"] = len(reply.calls)
        # Read after the step closed so the extra request is not timed as inference.
        record["model_placement"] = self.model.observe_placement()
        return reply

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
            steps.fail(step, f"Runner verification failed: {failed}")
            self.evidence["verification"] = {"step_id": step.step_id, "error": str(failed)}
            return False
        steps.complete(step, cpu_seconds=outcome.cpu_seconds, cpu_scope=outcome.cpu_scope)
        self.evidence["verification"] = {"step_id": step.step_id} | outcome.evidence
        return outcome.evidence["outcome"] == "passed"


def default_output_dir(base: Path, task: Task) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return base / f"{task.name}-{stamp}-{uuid4().hex[:6]}"
