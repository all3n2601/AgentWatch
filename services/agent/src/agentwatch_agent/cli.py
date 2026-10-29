"""Command line: `run` a task, list `tasks`, measure `overhead`, or `resend` a run's journal.

`sampling-overhead` measures resource sampling's effect on a real model request; it exits 1
if the measurement did not complete (its partial evidence is kept). `coverage` reports how a
recorded run's samples cover its steps; it exits 2 if the run directory cannot support one.
`experiment` replays one verified run cleanly and under bounded contention; it exits 1 when
it stopped early or a declared check failed (its evidence is kept), and 3 when a replay's live
evidence needs `resend`.

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
import signal
import sys
import urllib.parse
from pathlib import Path
from types import FrameType
from uuid import uuid4

from agentwatch_injectors.base import InjectorCleanupError
from agentwatch_injectors.inference import loopback_url
from agentwatch_sdk import LiveClient
from agentwatch_sdk.live import ResourceRegistration
from agentwatch_worker.nvml import NvmlDevice
from agentwatch_worker.ollama import ollama_base_url

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import (
    DEFAULT_CHAT_OPTIONS,
    DEFAULT_MODEL,
    MODEL_BACKENDS,
    RunLimits,
)
from agentwatch_agent.coverage import CoverageError, coverage_report
from agentwatch_agent.delivery import LiveRejectedError, is_retryable, replay_journal
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
from agentwatch_agent.replay import ReplayPlanError, ReplayRunner, load_replay_plan
from agentwatch_agent.resources import Probe
from agentwatch_agent.runner import (
    JOURNAL_FILE,
    AgentRunner,
    default_output_dir,
    write_json_atomic,
)
from agentwatch_agent.sampling_overhead import measure_sampling_overhead
from agentwatch_agent.service_probes import GpmFileProbe, dcgm_probe, vllm_probe
from agentwatch_agent.session import CancellationState
from agentwatch_agent.tasks import Task, TaskDefinitionError, load_task, task_names

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
LIVE_TIMEOUT_SECONDS = 10


def http_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise argparse.ArgumentTypeError(f"expected an http(s) URL, got {value!r}")
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


def add_model_arguments(command: argparse.ArgumentParser) -> None:
    """The model server a command talks to; --model defaults per backend."""
    command.add_argument(
        "--model-backend",
        choices=MODEL_BACKENDS,
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
    try:
        runner = AgentRunner(
            task=task,
            model=build_model(args),
            transport=LiveClient(args.api_url, timeout_seconds=LIVE_TIMEOUT_SECONDS),
            output_dir=default_output_dir(args.output_dir, task),
            limits=RunLimits(run_timeout_seconds=args.run_timeout),
            # A host without NVML records every GPU metric as unsupported, never as zero.
            open_gpu=NvmlDevice.open,
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


def not_started_exit(error: RunNotStarted, api_url: str, journal: Path) -> int:
    if error.stage != "registration":
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
    if args.command == "tasks":
        return tasks_command()
    if args.command == "experiment":
        return experiment_command(args)
    if args.command == "overhead":
        return overhead_command(args)
    if args.command == "sampling-overhead":
        return sampling_overhead_command(args)
    if args.command == "coverage":
        return coverage_command(args)
    return resend_command(args)
