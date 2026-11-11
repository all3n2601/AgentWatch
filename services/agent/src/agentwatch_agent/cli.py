"""Command line: `run` a task, list `tasks`, measure `overhead`, or `resend` a run's journal.

`run` and `batch` apply the active concurrency policy (O03): model requests and test jobs wait
for this process's admission permits. If the policy cannot be read, nothing runs (exit 4);
`--no-policy` runs without admission limits and records that. `batch` runs several tasks at
once in this process; it exits with the most severe of its runs' codes, in the order 3, 5, 4,
1, 0.

`sampling-overhead` measures resource sampling's effect on a real model request; it exits 1
if the measurement did not complete (its partial evidence is kept). `coverage` reports how a
recorded run's samples cover its steps; it exits 2 if the run directory cannot support one.
`experiment` replays one verified run cleanly and under bounded contention; it exits 1 when
it stopped early or a declared check failed (its evidence is kept), and 3 when a replay's live
evidence needs `resend`. `capacity-sweep` ramps concurrent replays on one vLLM configuration
(exit codes in the agent README); a completed sweep whose analysis raised exits 1.
`sweep-analyse` rebuilds a recorded sweep's analysis offline into `analysis.json`; it exits 2
when the folder has no usable `sweep.json`. `sweep-record` prints the capacity record's tables
from recorded sweep folders; it exits 2 when a folder's record or runs cannot support them.

Exit codes (all commands):
  0  success: the task passed (run), or all evidence was delivered and the run is closed
  1  the run failed or was cancelled; its evidence was delivered
  2  invalid usage: bad arguments, unknown task or patch, or an unusable journal
  3  live evidence is incomplete: run `resend` once the live API is reachable
  4  nothing ran: local setup failed, or the live API did not register the run
  5  the live API permanently refused some evidence; it stays in the journal and the
     evidence file, and resending identical content cannot change that
  6  resend only: the run is still open and its journal has no closure; if the runner that
     wrote it is gone, rerun with --fail-unclosed to close the run as failed
"""

import argparse
import functools
import http.client
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.parse
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from uuid import uuid4

from agentwatch_injectors.base import BoundedInjector, InjectorCleanupError
from agentwatch_injectors.inference import loopback_url
from agentwatch_injectors.prefill import PrefillFloodInjector
from agentwatch_sdk import LiveClient
from agentwatch_sdk.live import ResourceRegistration
from agentwatch_worker.nvml import NvmlDevice, NvmlUnavailable
from agentwatch_worker.ollama import ollama_base_url

from agentwatch_agent.admission import (
    DEFAULT_MAX_QUEUE_LENGTH,
    DEFAULT_MAX_WAIT_SECONDS,
    DEFAULT_POLICY_POLL_SECONDS,
    AdmissionController,
    AdmissionSettings,
    PolicyWatcher,
    QueueBounds,
    RunnerAdmission,
)
from agentwatch_agent.batch import MAX_BATCH_RUNS, AgentBatch, BatchRun
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import (
    DEFAULT_CHAT_OPTIONS,
    DEFAULT_MODEL,
    MODEL_BACKENDS,
    RunLimits,
    validate_sample_budget,
)
from agentwatch_agent.coverage import CoverageError, coverage_report
from agentwatch_agent.delivery import (
    LiveRejectedError,
    is_retryable,
    replay_journal,
)
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.experiment import (
    CONDITIONS,
    INJECTOR_KINDS,
    MANIFEST_FILE,
    ExperimentSettings,
    ReplayExperiment,
    build_injectors,
)
from agentwatch_agent.model import ChatModel, OllamaChatModel, VllmChatModel
from agentwatch_agent.overhead import OverheadFailed, measure_overhead
from agentwatch_agent.replay import (
    ReplayPlan,
    ReplayPlanError,
    ReplayRunner,
    load_replay_plan,
)
from agentwatch_agent.resources import Probe
from agentwatch_agent.runner import (
    EVIDENCE_FILE,
    JOURNAL_FILE,
    AgentRunner,
    default_output_dir,
    sampled_metric_names,
    write_json_atomic,
)
from agentwatch_agent.sampling_overhead import measure_sampling_overhead
from agentwatch_agent.service_probes import GpmFileProbe, dcgm_probe, vllm_probe
from agentwatch_agent.session import CancellationState
from agentwatch_agent.sweep import (
    COMPLETED,
    REFUSED_CONFIGURATION,
    STOPPED_RESEND_NEEDED,
    CapacitySweep,
)
from agentwatch_agent.sweep_offline import ANALYSIS_FILE, SweepFolderError, analyse_sweep_folder
from agentwatch_agent.sweep_protocol import (
    CONFIGURATIONS,
    INJECTOR_BOUND_MARGIN_SECONDS,
    INJECTOR_CONCURRENCY,
    INJECTOR_PROMPT_TOKENS,
    RUN_TIMEOUT_SECONDS,
    SAMPLING_INTERVAL_SECONDS,
    SOURCE_COUNT,
    TRACE_WAIT_SECONDS,
    Batch,
    VllmConfiguration,
    sweep_run_limits,
)
from agentwatch_agent.sweep_record import render_coverage_tables, render_sweep_tables
from agentwatch_agent.tasks import Task, TaskDefinitionError, load_task, task_names
from agentwatch_agent.vllm import vllm_models
from agentwatch_agent.vllm_launch import DEFAULT_VLLM_CONTAINER, read_launch_arguments
from agentwatch_agent.vllm_server import SERVED_MODEL_NAME, read_server_state

EXIT_PASSED, EXIT_FAILED, EXIT_INCOMPLETE_DELIVERY, EXIT_LIVE_UNAVAILABLE = 0, 1, 3, 4
EXIT_RECORDS_REFUSED = 5
EXIT_RUN_UNCLOSED = 6
EXIT_USAGE = 2
DEFAULT_API_URL = "http://127.0.0.1:8090"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"
# The Compose service's served name, which carries the weights' revision (compose.yaml).
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-1.7B@70d244cc86cc"
DEFAULT_VLLM_TRACES_FILE = ".data/otel/vllm-traces.jsonl"
DEFAULT_SWEEP_DIR = ".data/capacity-sweeps"
START_ATTEMPT_STARTED = "started"
LIVE_TIMEOUT_SECONDS = 10
GIT_TIMEOUT_SECONDS = 5


def http_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise argparse.ArgumentTypeError(f"expected an http(s) URL, got {value!r}")
    # urlsplit checks the port only when it is read; 0 is not a port a client can connect to.
    try:
        port = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid port in {value!r}: {error}") from None
    if port == 0:
        raise argparse.ArgumentTypeError(f"invalid port in {value!r}: 0")
    return value


def positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds, got {value!r}") from None
    if not 0 < seconds < float("inf"):
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return seconds


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def nonnegative_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds, got {value!r}") from None
    if not 0 <= seconds < float("inf"):
        raise argparse.ArgumentTypeError("seconds must be finite and nonnegative")
    return seconds


def choice_list(choices: tuple[str, ...]):
    """Parse a comma-separated subset of `choices`, keeping the given order."""

    def parse(value: str) -> tuple[str, ...]:
        items = tuple(item.strip() for item in value.split(",") if item.strip())
        if not items or not set(items) <= set(choices):
            raise argparse.ArgumentTypeError(
                f"expected a comma-separated list of {', '.join(choices)}; got {value!r}"
            )
        return items

    return parse


def model_backend(value: str) -> str:
    # argparse applies `type` to a string default but checks `choices` only on the
    # command line, so this also refuses a mistyped AGENTWATCH_MODEL_BACKEND.
    if value not in MODEL_BACKENDS:
        raise argparse.ArgumentTypeError(
            f"must be one of {', '.join(MODEL_BACKENDS)}, got {value!r}"
        )
    return value


def add_model_arguments(command: argparse.ArgumentParser) -> None:
    """The model server a command talks to; --model defaults per backend."""
    command.add_argument(
        "--model-backend",
        choices=MODEL_BACKENDS,
        type=model_backend,
        default=os.environ.get("AGENTWATCH_MODEL_BACKEND", "ollama"),
    )
    command.add_argument(
        "--model",
        help="Model name (default: AGENTWATCH_LLM_MODEL for Ollama, VLLM_MODEL for vLLM)",
    )
    command.add_argument(
        "--ollama-url",
        type=ollama_base_url,
        default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL),
    )
    command.add_argument(
        "--vllm-url", type=http_url, default=os.environ.get("VLLM_URL", DEFAULT_VLLM_URL)
    )


def add_gpu_device_arguments(command: argparse.ArgumentParser) -> None:
    """Optional whole-device GPU evidence: DCGM fields and the Windows GPM exporter's windows."""
    command.add_argument(
        "--dcgm-url",
        type=http_url,
        default=os.environ.get("AGENTWATCH_DCGM_URL") or None,
        help="dcgm-exporter /metrics URL; DCGM is not sampled without it",
    )
    gpm = os.environ.get("AGENTWATCH_GPM_FILE")
    command.add_argument(
        "--gpm-file",
        type=Path,
        default=Path(gpm) if gpm else None,
        help="GPM exporter text file written on Windows (ADR 0008); GPM is not sampled without it",
    )


def add_server_evidence_arguments(command: argparse.ArgumentParser) -> None:
    """Optional evidence beside the run's own samples: vLLM request spans and GPU device fields."""
    command.add_argument(
        "--vllm-traces-file",
        type=Path,
        default=Path(os.environ.get("AGENTWATCH_VLLM_TRACES_FILE", DEFAULT_VLLM_TRACES_FILE)),
        help="OTLP JSON lines the Compose collector writes (vLLM backend only)",
    )
    add_gpu_device_arguments(command)


def add_admission_arguments(command: argparse.ArgumentParser) -> None:
    """Concurrency policy enforcement (O03): queue bounds are declared before the run."""
    group = command.add_argument_group("admission")
    group.add_argument(
        "--no-policy",
        action="store_true",
        help="Do not read the concurrency policy: no admission limits apply, as evidence records",
    )
    group.add_argument(
        "--policy-poll",
        type=positive_seconds,
        default=DEFAULT_POLICY_POLL_SECONDS,
        help="Seconds between reads of the active policy; a new version applies to new admissions",
    )
    for flag, label in (("inference", "model requests"), ("test", "test jobs")):
        group.add_argument(
            f"--{flag}-max-queue",
            type=nonnegative_int,
            default=DEFAULT_MAX_QUEUE_LENGTH,
            help=f"Most {label} waiting for a permit; 0 refuses instead of queueing",
        )
        group.add_argument(
            f"--{flag}-max-wait",
            type=positive_seconds,
            default=DEFAULT_MAX_WAIT_SECONDS,
            help=f"Longest wait in seconds for a {label} permit before the run fails",
        )


def admission_controller(args: argparse.Namespace, model: ChatModel) -> AdmissionController | None:
    """The process's controller from the declared queue bounds; None with --no-policy."""
    if args.no_policy:
        return None
    settings = AdmissionSettings(
        inference_requests=QueueBounds(args.inference_max_queue, args.inference_max_wait),
        test_jobs=QueueBounds(args.test_max_queue, args.test_max_wait),
    )
    return AdmissionController(settings, inference_service_id=model.service_id)


def build_model(args: argparse.Namespace) -> ChatModel:
    if args.model_backend == "vllm":
        name = args.model or os.environ.get("VLLM_MODEL", DEFAULT_VLLM_MODEL)
        # vLLM fixes the context at startup; placement readings check it matches num_ctx.
        return VllmChatModel(args.vllm_url, name, num_ctx=DEFAULT_CHAT_OPTIONS.get("num_ctx"))
    name = args.model or os.environ.get("AGENTWATCH_LLM_MODEL", DEFAULT_MODEL)
    return OllamaChatModel(args.ollama_url, name)


def server_evidence(args: argparse.Namespace) -> dict:
    return {
        "dcgm_metrics_url": args.dcgm_url,
        "gpm_file": args.gpm_file,
        "vllm_traces_file": args.vllm_traces_file,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="agentwatch-agent", description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run one fixed task with a real model")
    run.add_argument("--task", required=True, help="Task name; see the `tasks` command")
    add_model_arguments(run)
    add_server_evidence_arguments(run)
    run.add_argument(
        "--api-url", type=http_url, default=os.environ.get("AGENTWATCH_API_URL", DEFAULT_API_URL)
    )
    run.add_argument("--output-dir", type=Path, default=Path(".data/agent-runs"))
    run.add_argument(
        "--run-timeout",
        type=positive_seconds,
        default=RunLimits().run_timeout_seconds,
        help="Total run time limit in seconds",
    )
    add_admission_arguments(run)

    batch = commands.add_parser(
        "batch", help="Run several tasks at once in this process, under one admission controller"
    )
    batch.add_argument(
        "--task",
        action="append",
        required=True,
        help="Task name; repeat for several tasks (see the `tasks` command)",
    )
    batch.add_argument(
        "--repeat",
        type=positive_int,
        default=1,
        help=f"Copies of each task in the batch (at most {MAX_BATCH_RUNS} runs in all)",
    )
    add_model_arguments(batch)
    add_server_evidence_arguments(batch)
    batch.add_argument(
        "--api-url", type=http_url, default=os.environ.get("AGENTWATCH_API_URL", DEFAULT_API_URL)
    )
    batch.add_argument("--output-dir", type=Path, default=Path(".data/agent-batches"))
    batch.add_argument(
        "--run-timeout",
        type=positive_seconds,
        default=RunLimits().run_timeout_seconds,
        help="Time limit of each run in seconds",
    )
    add_admission_arguments(batch)

    commands.add_parser("tasks", help="List runnable tasks and their identities")

    experiment = commands.add_parser(
        "experiment",
        help="Replay one verified run cleanly and under bounded CPU, inference, retrieval and "
        "mixed contention",
    )
    experiment.add_argument("--task", required=True, help="Task name; see the `tasks` command")
    experiment.add_argument(
        "--source",
        type=Path,
        help="Recorded run directory to replay; by default a new run is recorded first",
    )
    add_model_arguments(experiment)
    add_server_evidence_arguments(experiment)
    experiment.add_argument(
        "--api-url", type=http_url, default=os.environ.get("AGENTWATCH_API_URL", DEFAULT_API_URL)
    )
    experiment.add_argument("--output-dir", type=Path, default=Path(".data/agent-experiments"))
    experiment.add_argument(
        "--run-timeout",
        type=positive_seconds,
        default=RunLimits().run_timeout_seconds,
        help="Time limit of the source run and of each replay, in seconds",
    )
    defaults = ExperimentSettings()
    experiment.add_argument(
        "--baseline-replays", type=positive_int, default=defaults.baseline_replays
    )
    experiment.add_argument(
        "--warmup-replays",
        type=nonnegative_int,
        default=defaults.warmup_replays,
        help="Replays run first and excluded from the baseline",
    )
    experiment.add_argument(
        "--conditions",
        type=choice_list(CONDITIONS),
        default=defaults.conditions,
        help=f"Comma-separated contention conditions (default: {','.join(CONDITIONS)})",
    )
    experiment.add_argument(
        "--mixed",
        type=choice_list(INJECTOR_KINDS),
        default=defaults.mixed,
        help=f"Injectors a mixed replay runs together (default: {','.join(defaults.mixed)})",
    )
    experiment.add_argument(
        "--ramp-seconds", type=nonnegative_seconds, default=defaults.ramp_seconds
    )
    experiment.add_argument(
        "--cpu-workers",
        type=positive_int,
        default=defaults.cpu_workers,
        help="Busy CPU processes (at most the CPUs available; default: all of them)",
    )
    experiment.add_argument(
        "--inference-concurrency", type=positive_int, default=defaults.inference_concurrency
    )
    experiment.add_argument(
        "--inference-num-predict", type=positive_int, default=defaults.inference_num_predict
    )
    experiment.add_argument(
        "--retrieval-threads", type=positive_int, default=defaults.retrieval_threads
    )

    sweep = commands.add_parser(
        "capacity-sweep",
        help="Ramp concurrent replays on one vLLM configuration and record its capacity",
    )
    sweep.add_argument("--configuration", required=True, choices=tuple(CONFIGURATIONS))
    sweep.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Recorded run directory to replay; give exactly two, of different tasks",
    )
    add_model_arguments(sweep)
    # The sweep measures vLLM only: it never falls back to AGENTWATCH_MODEL_BACKEND.
    sweep.set_defaults(model_backend="vllm")
    add_server_evidence_arguments(sweep)
    sweep.add_argument(
        "--api-url", type=http_url, default=os.environ.get("AGENTWATCH_API_URL", DEFAULT_API_URL)
    )
    sweep.add_argument("--output-dir", type=Path, default=Path(DEFAULT_SWEEP_DIR))
    sweep.add_argument(
        "--run-timeout",
        type=positive_seconds,
        default=RUN_TIMEOUT_SECONDS,
        help="Time limit of each run, in seconds; the protocol declares "
        f"{RUN_TIMEOUT_SECONDS:g} s, and any other value gives a different protocol hash",
    )
    sweep.add_argument(
        "--start-attempt",
        action="append",
        default=[],
        metavar="UTIL:OUTCOME",
        help="An attempt to start vLLM, e.g. '0.8:failed: CUDA out of memory' then "
        "'0.75:started'; repeatable, the last must be 'started'",
    )
    sweep.add_argument(
        "--vllm-container",
        default=DEFAULT_VLLM_CONTAINER,
        help="Docker container whose launch arguments verify max_num_seqs and the memory "
        "utilization (not verified when Docker is unreachable)",
    )

    analyse = commands.add_parser(
        "sweep-analyse",
        help="Rebuild a recorded capacity sweep's analysis from its folder into analysis.json",
    )
    analyse.add_argument("sweep_dir", type=Path, help="A capacity-sweep folder with sweep.json")

    record = commands.add_parser(
        "sweep-record",
        help="Print the capacity record's Markdown tables from recorded capacity-sweep folders",
    )
    record.add_argument(
        "sweep_dirs", type=Path, nargs="+", help="Capacity-sweep folders with sweep.json"
    )
    record.add_argument(
        "--coverage-jsonl",
        type=Path,
        help="Also print the sampling coverage tables and write their raw reports to this file",
    )

    overhead = commands.add_parser(
        "overhead", help="Measure instrumentation overhead on a task's test workload"
    )
    overhead.add_argument("--task", default="fix-mean-empty")
    overhead.add_argument("--patch", default="return-none-for-empty", help="Correct patch ID")
    overhead.add_argument(
        "--api-url", type=http_url, default=os.environ.get("AGENTWATCH_API_URL", DEFAULT_API_URL)
    )
    overhead.add_argument("--output-dir", type=Path, default=Path(".data/agent-overhead"))

    sampling = commands.add_parser(
        "sampling-overhead",
        help="Measure resource sampling's effect on a real model request (no live API needed)",
    )
    sampling.add_argument("--task", default="fix-mean-empty")
    add_model_arguments(sampling)
    add_gpu_device_arguments(sampling)
    sampling.add_argument("--output-dir", type=Path, default=Path(".data/sampling-overhead"))

    coverage = commands.add_parser(
        "coverage", help="Report how a recorded run's resource samples cover its steps"
    )
    coverage.add_argument("run_dir", type=Path, help="A run's output directory")

    resend = commands.add_parser("resend", help="Resend a run journal with original identities")
    resend.add_argument("journal", type=Path)
    resend.add_argument(
        "--api-url", type=http_url, default=os.environ.get("AGENTWATCH_API_URL", DEFAULT_API_URL)
    )
    resend.add_argument(
        "--fail-unclosed",
        action="store_true",
        help="Close a run whose journal has no closure as failed (only if its runner is gone)",
    )
    return root


def run_command(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task)
    except TaskDefinitionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    transport = LiveClient(args.api_url, timeout_seconds=LIVE_TIMEOUT_SECONDS)
    try:
        model = build_model(args)
        controller = admission_controller(args, model)
        watcher = (
            PolicyWatcher(transport, controller, args.policy_poll)
            if controller is not None
            else None
        )
        runner = AgentRunner(
            task=task,
            model=model,
            transport=transport,
            output_dir=default_output_dir(args.output_dir, task),
            limits=RunLimits(run_timeout_seconds=args.run_timeout),
            # A host without NVML records every GPU metric as unsupported, never as zero.
            open_gpu=NvmlDevice.open,
            admission=(
                RunnerAdmission(controller, transport, watcher) if controller is not None else None
            ),
            **server_evidence(args),
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    def handle_signal(number: int, _frame: FrameType | None) -> None:
        # No printing here: writing to a stream inside a signal handler can deadlock on its
        # internal lock. The cancellation is reported when the run closes.
        name = signal.Signals(number).name
        runner.request_cancel_from_signal("signal", f"Cancelled by operator signal {name}")

    previous = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ExitStack() as policy:
            if watcher is not None:
                policy.enter_context(watcher)
            outcome = runner.run()
    except RunNotStarted as error:
        return not_started_exit(error, args.api_url, runner.output_dir / "live-journal.jsonl")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(
        f"Run {outcome.run_id}: {outcome.state}"
        + (f" ({outcome.reason})" if outcome.reason else "")
    )
    print(f"Evidence: {outcome.evidence_path}")
    code = delivery_exit(
        outcome.resend_needed,
        outcome.records_refused,
        outcome.journal_path,
        completion_refused=outcome.completion_refused,
    )
    if code is not None:
        return code
    return EXIT_PASSED if outcome.state == "completed" and outcome.task_passed else EXIT_FAILED


def batch_command(args: argparse.Namespace) -> int:
    try:
        tasks = [load_task(name) for name in args.task]
    except TaskDefinitionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    count = len(tasks) * args.repeat
    if count > MAX_BATCH_RUNS:
        print(f"error: a batch has at most {MAX_BATCH_RUNS} runs, not {count}", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    batch_dir = args.output_dir / f"batch-{stamp}-{uuid4().hex[:6]}"
    transport = LiveClient(args.api_url, timeout_seconds=LIVE_TIMEOUT_SECONDS)
    # One clock domain for this process, so every run's ticks (and admission waits) compare.
    clock = MonotonicClock()
    try:
        # One model client per run, so no request state is shared between threads.
        models = [build_model(args) for _ in range(count)]
        controller = admission_controller(args, models[0])
        watcher = (
            PolicyWatcher(transport, controller, args.policy_poll)
            if controller is not None
            else None
        )
        admission = (
            RunnerAdmission(controller, transport, watcher) if controller is not None else None
        )
        batch_tasks = [task for task in tasks for _ in range(args.repeat)]
        runners = [
            AgentRunner(
                task=task,
                model=model,
                transport=transport,
                output_dir=batch_dir / "runs" / f"{index:02d}-{task.name}",
                limits=RunLimits(run_timeout_seconds=args.run_timeout),
                clock=clock,
                open_gpu=NvmlDevice.open,
                admission=admission,
                **server_evidence(args),
            )
            for index, (task, model) in enumerate(zip(batch_tasks, models, strict=True), 1)
        ]
        batch = AgentBatch(runners, controller, batch_dir, watcher=watcher)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    def handle_signal(number: int, _frame: FrameType | None) -> None:
        name = signal.Signals(number).name
        batch.request_cancel_from_signal("signal", f"Cancelled by operator signal {name}")

    previous = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ExitStack() as policy:
            if watcher is not None:
                policy.enter_context(watcher)
            runs = batch.run()
    except RunNotStarted as error:
        print(f"Batch {batch.batch_id}: {batch.path}")
        return not_started_exit(error, args.api_url, batch.path)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(f"Batch {batch.batch_id}: {batch.path}")
    return batch_exit(runs, args.api_url)


def batch_exit(runs: list[BatchRun], api_url: str) -> int:
    """The most severe run's exit code; a runner error is re-raised once all are recorded."""
    errors = [run.error for run in runs if run.error is not None]
    if errors:
        raise errors[0]
    codes = []
    for run in runs:
        journal = run.runner.output_dir / "live-journal.jsonl"
        if run.not_started is not None:
            print(f"Run {run.index} ({run.runner.task.task_id}): not started")
            codes.append(not_started_exit(run.not_started, api_url, journal))
            continue
        outcome = run.outcome
        print(
            f"Run {run.index} {outcome.run_id}: {outcome.state}"
            + (f" ({outcome.reason})" if outcome.reason else "")
        )
        code = delivery_exit(
            outcome.resend_needed,
            outcome.records_refused,
            outcome.journal_path,
            completion_refused=outcome.completion_refused,
        )
        if code is None:
            passed = outcome.state == "completed" and outcome.task_passed
            code = EXIT_PASSED if passed else EXIT_FAILED
        codes.append(code)
    for code in (
        EXIT_INCOMPLETE_DELIVERY,
        EXIT_RECORDS_REFUSED,
        EXIT_LIVE_UNAVAILABLE,
        EXIT_FAILED,
    ):
        if code in codes:
            return code
    return EXIT_PASSED


def not_started_exit(error: RunNotStarted, api_url: str, journal: Path) -> int:
    if error.stage == "policy":
        cause = (
            f"the concurrency policy at {api_url} could not be read or applied ({error}). "
            "Use --no-policy to run without admission limits."
        )
    elif error.stage != "registration":
        cause = f"local setup failed before contacting the live API ({error})."
    elif error.closure_journaled:
        cause = (
            f"the live API at {api_url} did not confirm the registration ({error}). If it "
            "recorded the run, close it once the API is reachable with:\n"
            f"  uv run --package agentwatch-agent python -m agentwatch_agent resend {journal}"
        )
    else:
        cause = f"the live API at {api_url} refused the registration ({error})."
    print(f"error: nothing ran: {cause}", file=sys.stderr)
    return EXIT_LIVE_UNAVAILABLE


def delivery_exit(
    resend_needed: bool, records_refused: int, journal: Path, *, completion_refused: bool = False
) -> int | None:
    """Exit code for incomplete live evidence, or None when all of it was delivered."""
    if resend_needed:
        print(
            "Live evidence is incomplete. Resend it once the API is reachable with:\n"
            f"  uv run --package agentwatch-agent python -m agentwatch_agent resend {journal}",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE_DELIVERY
    if records_refused or completion_refused:
        refused = f"{records_refused} records" if records_refused else "completion"
        print(
            f"The live API permanently refused {refused}, so the run was closed with "
            f"incomplete evidence. Everything is kept in {journal} and the evidence file; "
            "resending identical content cannot change that.",
            file=sys.stderr,
        )
        return EXIT_RECORDS_REFUSED
    return None


def experiment_command(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task)
        settings = ExperimentSettings(
            baseline_replays=args.baseline_replays,
            warmup_replays=args.warmup_replays,
            conditions=args.conditions,
            mixed=args.mixed,
            ramp_seconds=args.ramp_seconds,
            cpu_workers=args.cpu_workers,
            inference_concurrency=args.inference_concurrency,
            inference_num_predict=args.inference_num_predict,
            retrieval_threads=args.retrieval_threads,
        )
        limits = RunLimits(run_timeout_seconds=args.run_timeout)
        settings.injection_bound_seconds(limits.run_timeout_seconds)
        model = build_model(args)
        if any("inference" in settings.kinds(c) for c in settings.conditions):
            loopback_url(model.endpoint)  # contention only ever targets local capacity
    except (TaskDefinitionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    output_dir = default_output_dir(args.output_dir, task)
    transport = LiveClient(args.api_url, timeout_seconds=LIVE_TIMEOUT_SECONDS)
    # What a signal cancels: the source run while it records, then the experiment.
    cancellable: list = []

    def handle_signal(number: int, _frame: FrameType | None) -> None:
        # Lock-free and print-free, as for `run`.
        if cancellable:
            reason = f"Cancelled by operator signal {signal.Signals(number).name}"
            cancellable[-1].request_cancel_from_signal("signal", reason)

    previous = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        source_dir = args.source
        if source_dir is None:
            source_dir = output_dir / "source"
            code = record_source(task, model, transport, source_dir, limits, args, cancellable)
            if code is not None:
                return code
        try:
            plan = load_replay_plan(source_dir, task)
        except ReplayPlanError as error:
            print(f"error: cannot replay {source_dir}: {error}", file=sys.stderr)
            return EXIT_USAGE

        def make_runner(replay_dir: Path, clock: MonotonicClock) -> ReplayRunner:
            return ReplayRunner(
                plan=plan,
                task=task,
                model=model,
                transport=transport,
                output_dir=replay_dir,
                limits=limits,
                clock=clock,
                open_gpu=NvmlDevice.open,
                **server_evidence(args),
            )

        try:
            make_runner(output_dir / "replays", MonotonicClock())  # validates; creates nothing
        except ValueError as error:
            print(f"error: cannot replay {source_dir}: {error}", file=sys.stderr)
            return EXIT_USAGE
        experiment = ReplayExperiment(
            plan=plan,
            settings=settings,
            output_dir=output_dir,
            run_timeout_seconds=limits.run_timeout_seconds,
            make_runner=make_runner,
            make_injectors=functools.partial(
                build_injectors,
                settings=settings,
                plan=plan,
                task=task,
                model_url=model.endpoint,
                model_backend=model.backend,
            ),
            source={
                "run_dir": str(source_dir),
                "run_id": plan.source_run_id,
                "evidence_sha256": plan.source_evidence_sha256,
                "recorded_by_experiment": args.source is None,
            },
        )
        cancellable.append(experiment)
        try:
            manifest = experiment.run()
        except InjectorCleanupError as error:
            print(
                f"error: an injector could not be confirmed stopped ({error}); check the host "
                f"for leftover load. Evidence: {output_dir / MANIFEST_FILE}",
                file=sys.stderr,
            )
            return EXIT_FAILED
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return report_experiment(manifest, output_dir)


def record_source(
    task: Task,
    model: ChatModel,
    transport: LiveClient,
    source_dir: Path,
    limits: RunLimits,
    args: argparse.Namespace,
    cancellable: list,
) -> int | None:
    """Record the run to replay; an exit code when it cannot be replayed, otherwise None."""
    runner = AgentRunner(
        task=task,
        model=model,
        transport=transport,
        output_dir=source_dir,
        limits=limits,
        open_gpu=NvmlDevice.open,
        **server_evidence(args),
    )
    cancellable.append(runner)
    try:
        outcome = runner.run()
    except RunNotStarted as error:
        return not_started_exit(error, args.api_url, source_dir / JOURNAL_FILE)
    print(f"Source run {outcome.run_id}: {outcome.state}; evidence: {outcome.evidence_path}")
    if outcome.state != "completed" or not outcome.task_passed:
        print(
            "error: the source run did not complete with passing verification, so it cannot be "
            "replayed; its evidence is kept",
            file=sys.stderr,
        )
        return EXIT_FAILED
    return None


def report_experiment(manifest: dict, output_dir: Path) -> int:
    cv = manifest.get("baseline_variability", {}).get("agent-run", {}).get("cv")
    print(
        f"Experiment {manifest['status']}: {len(manifest['replays'])} replays; baseline "
        f"agent-run CV {'unavailable' if cv is None else f'{cv:.1%}'}"
    )
    for name, passed in manifest.get("checks", {}).items():
        print(f"  {'pass' if passed else 'FAIL'}  {name}")
    print(f"Evidence: {output_dir / MANIFEST_FILE}")
    pending = [replay for replay in manifest["replays"] if replay.get("resend_needed")]
    for replay in pending:
        journal = output_dir / replay["evidence_dir"] / JOURNAL_FILE
        print(
            "Live evidence is incomplete; resend it with:\n"
            f"  uv run --package agentwatch-agent python -m agentwatch_agent resend {journal}",
            file=sys.stderr,
        )
    if pending:
        return EXIT_INCOMPLETE_DELIVERY
    return EXIT_PASSED if manifest.get("passed") else EXIT_FAILED


def source_task(source_dir: Path) -> Task:
    """The task a recorded run's evidence names."""
    path = source_dir / EVIDENCE_FILE
    try:
        name = json.loads(path.read_bytes())["task"]["name"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ReplayPlanError(f"Cannot read the task from source evidence {path}") from error
    if not isinstance(name, str):
        raise ReplayPlanError(f"Source evidence {path} names no task")
    return load_task(name)


def load_sweep_sources(source_dirs: list[Path]) -> list[tuple[Task, ReplayPlan]]:
    """Each source's task and replay plan; two sources must be of different tasks."""
    if len(source_dirs) != SOURCE_COUNT:
        raise ValueError(f"a sweep replays exactly {SOURCE_COUNT} sources, not {len(source_dirs)}")
    try:
        tasks = [source_task(directory) for directory in source_dirs]
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("the sources must be runs of different tasks, not one task twice")
        return [
            (task, load_replay_plan(directory, task))
            for directory, task in zip(source_dirs, tasks, strict=True)
        ]
    except ReplayPlanError as error:
        raise ValueError(f"cannot replay the sources: {error}") from error


def endpoint_port(url: str) -> int:
    """The port of an http(s) URL, explicit or the scheme's default."""
    split = urllib.parse.urlsplit(url)
    return split.port or (443 if split.scheme == "https" else 80)


def parse_start_attempts(
    attempts: list[str], configuration: VllmConfiguration
) -> list[dict] | None:
    """The operator's vLLM start attempts; the last must be a declared utilization that started."""
    if not attempts:
        return None
    parsed = []
    for text in attempts:
        utilization, separator, outcome = text.partition(":")
        try:
            value = float(utilization)
        except ValueError:
            value = float("nan")
        if not separator or not 0 < value <= 1 or not outcome.strip():
            raise ValueError(f"start attempt {text!r} is not UTILIZATION:OUTCOME, e.g. 0.8:started")
        parsed.append({"gpu_memory_utilization": value, "outcome": outcome.strip()})
    if any(attempt["outcome"] == START_ATTEMPT_STARTED for attempt in parsed[:-1]):
        raise ValueError("start attempts: only the last one may be 'started'")
    last = parsed[-1]
    if last["outcome"] != START_ATTEMPT_STARTED:
        raise ValueError("the last start attempt must be 'started', with the utilization vLLM uses")
    if last["gpu_memory_utilization"] not in configuration.gpu_memory_utilization:
        declared = ", ".join(str(value) for value in configuration.gpu_memory_utilization)
        raise ValueError(
            f"the start attempt that started used {last['gpu_memory_utilization']}, but "
            f"{configuration.name} declares {declared}"
        )
    return parsed


def host_record() -> dict:
    """Platform, CPU count and GPU identity; anything unavailable is recorded as unavailable."""
    try:
        device = NvmlDevice.open()
    except NvmlUnavailable as error:
        gpu = {"status": "unavailable", "reason": str(error)}
    else:
        try:
            identity = asdict(device.identity)
            gpu = {"status": "available"} | {
                key: identity[key]
                for key in ("name", "driver_version", "memory_total_bytes", "device_count")
            }
        finally:
            device.close()
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
    }


def code_revision() -> dict:
    """The git commit of this code and whether tracked files had local changes, or why not."""
    here = Path(__file__).resolve().parent

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=True,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        changes = git("status", "--porcelain", "--untracked-files=no")
    # CalledProcessError and TimeoutExpired are SubprocessErrors; a missing git is an OSError.
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"[:500]}
    return {"status": "recorded", "commit": commit, "dirty": bool(changes.strip())}


def capacity_sweep_command(args: argparse.Namespace) -> int:
    configuration = CONFIGURATIONS[args.configuration]
    try:
        if args.model_backend != "vllm":
            raise ValueError("the capacity sweep measures vLLM; use --model-backend vllm")
        start_attempts = parse_start_attempts(args.start_attempt, configuration)
        # The protocol's declared limits, which it hashes, with this sweep's run timeout.
        limits = sweep_run_limits(args.run_timeout)
        model = build_model(args)
        if model.name != SERVED_MODEL_NAME:
            raise ValueError(f"the sweep's declared model is {SERVED_MODEL_NAME}, not {model.name}")
        loopback_url(model.endpoint)  # contention only ever targets local capacity
        # Every sweep runner samples the GPU and vLLM, plus DCGM and GPM when given; a run
        # limit over the sample budget would make every runner refuse to start.
        validate_sample_budget(
            limits.run_timeout_seconds,
            SAMPLING_INTERVAL_SECONDS,
            len(
                sampled_metric_names(
                    gpu=True,
                    vllm=True,
                    dcgm=args.dcgm_url is not None,
                    gpm=args.gpm_file is not None,
                )
            ),
        )
        sources = load_sweep_sources(args.source)
        for _task, plan in sources:
            # A replay runs only on the backend and model it was recorded with (ReplayRunner).
            if plan.model_backend != "vllm":
                raise ValueError(f"source {plan.task_id} was recorded on {plan.model_backend}")
            if plan.model != SERVED_MODEL_NAME:
                raise ValueError(f"source {plan.task_id} was recorded with model {plan.model}")
    except (TaskDefinitionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    transport = LiveClient(args.api_url, timeout_seconds=LIVE_TIMEOUT_SECONDS)
    evidence = server_evidence(args)
    metrics_url = args.vllm_url.rstrip("/") + "/metrics"

    def runner_settings(directory: Path, clock: MonotonicClock) -> dict:
        # No admission: the sweep never reads the concurrency policy. One model client per run.
        return {
            "model": build_model(args),
            "transport": transport,
            "output_dir": directory,
            "limits": limits,
            "clock": clock,
            "open_gpu": NvmlDevice.open,
            # The protocol's declared values, passed explicitly.
            "sampling_interval_seconds": SAMPLING_INTERVAL_SECONDS,
            "trace_wait_seconds": TRACE_WAIT_SECONDS,
            **evidence,
        }

    def make_batch(batch: Batch, path: Path, clock: MonotonicClock) -> AgentBatch:
        runners = []
        for index, source in enumerate(batch.sources, 1):
            task, plan = sources[source]
            directory = path / "runs" / f"{index:02d}-{task.name}"
            runners.append(ReplayRunner(plan=plan, task=task, **runner_settings(directory, clock)))
        return AgentBatch(runners, None, path)

    def make_live_batch(batch: Batch, path: Path, clock: MonotonicClock) -> AgentBatch:
        # Free agents on the tasks of the batch's declared sources, alternating as declared.
        runners = []
        for index, source in enumerate(batch.sources, 1):
            task, _plan = sources[source]
            directory = path / "runs" / f"{index:02d}-{task.name}"
            runners.append(AgentRunner(task=task, **runner_settings(directory, clock)))
        return AgentBatch(runners, None, path)

    def make_injectors(clock: MonotonicClock, bound_seconds: float) -> list[BoundedInjector]:
        return [
            PrefillFloodInjector(
                clock,
                bound_seconds,
                url=model.endpoint,
                model=SERVED_MODEL_NAME,
                concurrency=INJECTOR_CONCURRENCY,
                prompt_tokens=INJECTOR_PROMPT_TOKENS,
            )
        ]

    try:
        # Validates the runners and the injector settings; nothing is created or started.
        every_source = tuple(range(len(sources)))
        make_batch(Batch("ramp", len(sources), 0, every_source, False), Path(), MonotonicClock())
        # With the bound the sweep gives the flood, so a too-long run limit is refused here.
        make_injectors(MonotonicClock(), limits.run_timeout_seconds + INJECTOR_BOUND_MARGIN_SECONDS)
    except ValueError as error:
        print(f"error: cannot run the sweep: {error}", file=sys.stderr)
        return EXIT_USAGE
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    sweep_dir = args.output_dir / f"{configuration.name}-{stamp}-{uuid4().hex[:6]}"
    sweep = CapacitySweep(
        configuration=configuration,
        sources=[plan for _task, plan in sources],
        output_dir=sweep_dir,
        run_timeout_seconds=limits.run_timeout_seconds,
        make_batch=make_batch,
        make_live_batch=make_live_batch,
        make_injectors=make_injectors,
        read_state=functools.partial(read_server_state, metrics_url),
        models=functools.partial(vllm_models, model.endpoint),
        read_launch=functools.partial(
            read_launch_arguments,
            args.vllm_container,
            endpoint_port=endpoint_port(model.endpoint),
        ),
        host=host_record(),
        start_attempts=start_attempts,
        code_revision=code_revision(),
    )

    def handle_signal(number: int, _frame: FrameType | None) -> None:
        # Lock-free and print-free, as for `run`.
        reason = f"Cancelled by operator signal {signal.Signals(number).name}"
        sweep.request_cancel_from_signal("signal", reason)

    previous = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        record = sweep.run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return report_sweep(record, sweep_dir)


def report_sweep(record: dict, sweep_dir: Path) -> int:
    """Print the sweep's outcome and map its state to the exit code."""
    state = record["state"]
    print(f"Sweep: {sweep_dir}")
    print(f"State: {state}; ramp ended: {record.get('ramp_end') or 'not reached'}")
    if record.get("stop_reason"):
        print(f"Stopped: {record['stop_reason']}")
    decision = record.get("capacity_decision")
    if decision:
        print(f"Capacity: capacity {decision['capacity']} ({decision['result']})")
    overall = (record.get("label_validation") or {}).get("overall")
    if overall:
        print(f"Label validation: {overall['verdict']}")
    analysis_errors = record.get("analysis_errors") or []
    for error in analysis_errors:
        print(f"error: the {error['stage']} analysis failed: {error['error']}", file=sys.stderr)
    if analysis_errors:
        print(
            f"The evidence is kept; analyse it again with `sweep-analyse {sweep_dir}`.",
            file=sys.stderr,
        )
    if state == STOPPED_RESEND_NEEDED:
        print(
            "Live evidence is incomplete for at least one run; resend each journal named in "
            f"{sweep_dir / 'sweep.json'} with the `resend` command.",
            file=sys.stderr,
        )
    if state == COMPLETED:
        # A sweep whose analysis raised is not a success, although its evidence is complete.
        return EXIT_FAILED if analysis_errors else EXIT_PASSED
    if state == REFUSED_CONFIGURATION:
        return EXIT_USAGE
    if state == STOPPED_RESEND_NEEDED:
        return EXIT_INCOMPLETE_DELIVERY
    return EXIT_FAILED


def sweep_analyse_command(args: argparse.Namespace) -> int:
    try:
        document = analyse_sweep_folder(args.sweep_dir)
    except SweepFolderError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    decision = document["capacity_decision"]
    print(f"Capacity: capacity {decision['capacity']} ({decision['result']})")
    print(f"Analysis: {args.sweep_dir / ANALYSIS_FILE}")
    return EXIT_PASSED


def sweep_record_command(args: argparse.Namespace) -> int:
    try:
        results = [render_sweep_tables(sweep_dir) for sweep_dir in args.sweep_dirs]
        coverage = (
            [render_coverage_tables(sweep_dir) for sweep_dir in args.sweep_dirs]
            if args.coverage_jsonl is not None
            else []
        )
        if coverage:
            reports = [report for _table, sweep_reports in coverage for report in sweep_reports]
            lines = [json.dumps(report, allow_nan=False) for report in reports]
            # newline="\n": the record's files use LF on every platform.
            with args.coverage_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(lines) + "\n")
    except (SweepFolderError, CoverageError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    print("\n".join(results), end="")
    for table, _reports in coverage:
        print("\n" + table, end="")
    return EXIT_PASSED


def tasks_command() -> int:
    for name in task_names():
        task = load_task(name)
        print(f"{task.name}\t{task.task_id}\t{task.content_sha256[:12]}\t{task.title}")
    return EXIT_PASSED


def overhead_command(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task)
    except TaskDefinitionError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    if task.patch(args.patch) is None:
        ids = ", ".join(patch.patch_id for patch in task.patches)
        print(
            f"error: task {task.name} has no patch {args.patch!r}; use one of: {ids}",
            file=sys.stderr,
        )
        return EXIT_USAGE
    output_dir = default_output_dir(args.output_dir, task)
    cancel = CancellationState()

    def handle_signal(number: int, _frame: FrameType | None) -> None:
        # Lock-free and print-free, as for `run`; the measurement stops at its next check.
        name = signal.Signals(number).name
        cancel.cancel_from_signal("signal", f"Cancelled by operator signal {name}")

    previous = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        report = measure_overhead(
            task,
            args.patch,
            LiveClient(args.api_url, LIVE_TIMEOUT_SECONDS),
            output_dir,
            cancel=cancel,
        )
    except RunNotStarted as error:
        return not_started_exit(error, args.api_url, output_dir / "overhead-journal.jsonl")
    except OverheadFailed as error:
        print(f"error: {error} ({error.__cause__!r}); no measurement was recorded", file=sys.stderr)
        code = delivery_exit(error.closure == "deferred", error.refused, error.journal)
        return EXIT_FAILED if code is None else code
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    path = output_dir / "overhead.json"
    write_json_atomic(path, report)
    walls = report["workload_wall_seconds"]
    print(
        f"Workload median {walls['uninstrumented']['median']:.4f} s uninstrumented, "
        f"{walls['instrumented']['median']:.4f} s instrumented "
        f"({report['median_difference_fraction']:+.1%}); evidence: {path}"
    )
    delivery = report["delivery"]
    code = delivery_exit(
        delivery["resend_needed"],
        delivery["refused"],
        Path(report["journal"]),
        completion_refused=delivery["completion_refused"],
    )
    return EXIT_PASSED if code is None else code


def resend_command(args: argparse.Namespace) -> int:
    try:
        report = replay_journal(
            args.journal,
            LiveClient(args.api_url, LIVE_TIMEOUT_SECONDS),
            fail_unclosed=args.fail_unclosed,
        )
    except LiveRejectedError as error:
        print(f"error: the live API refused the replay: {error}", file=sys.stderr)
        return EXIT_RECORDS_REFUSED
    except (OSError, ValueError, http.client.HTTPException) as error:
        if is_retryable(error):
            print(
                f"error: the live API at {args.api_url} is unreachable ({error}); nothing is "
                "lost. Resend again once it is reachable.",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE_DELIVERY
        print(f"error: cannot resend {args.journal}: {error}", file=sys.stderr)
        return EXIT_USAGE
    print(
        f"Run {report.run_id}: resent {report.records_sent} records and applied "
        f"{report.changes_applied} state changes; skipped {report.records_skipped_rejected} "
        f"previously refused records. Run state: {report.final_state} "
        f"(journaled: {report.expected_state})."
    )
    notes = {
        "truncated_final_line": "ignored an interrupted final journal line",
        "changes_superseded": "the run was closed by another actor before its journaled closure",
        "run_created_by_replay": "the live API had not recorded this run; resend registered "
        "and closed it",
        "completion_refused": "the live API refused completion, so the run was closed as failed",
        "closed_unclosed_run": "the run had no journaled closure and was closed as failed",
    }
    for field_name, note in notes.items():
        if getattr(report, field_name):
            print(f"note: {note}", file=sys.stderr)
    for rejection in report.rejections:
        print(
            f"refused record {rejection['record_id']} ({rejection['status']}): "
            f"{rejection['detail']}",
            file=sys.stderr,
        )
    for unresolved in report.unresolved_changes:
        print(
            f"run change to {unresolved['state']} was refused ({unresolved['status']}): "
            f"{unresolved['detail']}",
            file=sys.stderr,
        )
    if report.outcome == "unclosed":
        print(
            "The run is still open and its journal has no closure (the runner may still be "
            "running, or it stopped abruptly). If it is gone, rerun with --fail-unclosed.",
            file=sys.stderr,
        )
        return EXIT_RUN_UNCLOSED
    if report.outcome == "retry":
        return EXIT_INCOMPLETE_DELIVERY
    if report.outcome == "refused":
        if report.refused_records:
            delivery_exit(False, report.refused_records, args.journal)
        return EXIT_RECORDS_REFUSED
    return EXIT_PASSED


def sampling_overhead_command(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task)
    except TaskDefinitionError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    output_dir = default_output_dir(args.output_dir, task)
    model = build_model(args)
    # One identity for the whole measurement: every sampled request reads the same device.
    dcgm_gpu = ResourceRegistration(
        resource_id=f"gpu-dcgm-{uuid4().hex[:12]}",
        service_id="nvidia-dcgm",
        resource_class="inference",
    )

    gpm_gpu = ResourceRegistration(
        resource_id=f"gpu-gpm-{uuid4().hex[:12]}",
        service_id="nvidia-nvml-gpm",
        resource_class="inference",
    )

    def service_probes(server: ResourceRegistration) -> list[Probe]:
        probes: list[Probe] = []
        if model.backend == "vllm":
            probes.append(vllm_probe(server, model.endpoint, model.name))
        if args.dcgm_url is not None:
            probes.append(dcgm_probe(dcgm_gpu, args.dcgm_url))
        if args.gpm_file is not None:
            probes.append(GpmFileProbe(gpm_gpu, args.gpm_file))
        return probes

    with_services = (
        model.backend == "vllm" or args.dcgm_url is not None or args.gpm_file is not None
    )
    report = measure_sampling_overhead(
        task,
        model.name,
        model.chat,
        output_dir,
        service_probes=service_probes if with_services else None,
    )
    path = output_dir / "sampling-overhead.json"
    write_json_atomic(path, report)
    if report["status"] != "completed":
        print(
            f"error: measurement {report['status']}: {report['error']}; kept: {path}",
            file=sys.stderr,
        )
        return EXIT_FAILED
    conditions = report["conditions"]
    print(
        f"Request median {conditions['unsampled']['wall_seconds']['median']:.4f} s unsampled, "
        f"{conditions['sampled']['wall_seconds']['median']:.4f} s sampled "
        f"({report['median_wall_difference_fraction']:+.1%}); evidence: {path}"
    )
    return EXIT_PASSED


def coverage_command(args: argparse.Namespace) -> int:
    try:
        report = coverage_report(args.run_dir)
    except (CoverageError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    print(json.dumps(report, indent=2, allow_nan=False))
    return EXIT_PASSED


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "batch":
        return batch_command(args)
    if args.command == "tasks":
        return tasks_command()
    if args.command == "experiment":
        return experiment_command(args)
    if args.command == "capacity-sweep":
        return capacity_sweep_command(args)
    if args.command == "sweep-analyse":
        return sweep_analyse_command(args)
    if args.command == "sweep-record":
        return sweep_record_command(args)
    if args.command == "overhead":
        return overhead_command(args)
    if args.command == "sampling-overhead":
        return sampling_overhead_command(args)
    if args.command == "coverage":
        return coverage_command(args)
    return resend_command(args)
