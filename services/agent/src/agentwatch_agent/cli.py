"""Command line: `run` a task, list `tasks`, measure `overhead`, or `resend` a run's journal.

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
import http.client
import os
import signal
import sys
import urllib.parse
from pathlib import Path
from types import FrameType

from agentwatch_sdk import LiveClient
from agentwatch_worker.ollama import ollama_base_url

from agentwatch_agent.config import DEFAULT_MODEL, RunLimits
from agentwatch_agent.delivery import LiveRejectedError, is_retryable, replay_journal
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.model import OllamaChatModel
from agentwatch_agent.overhead import OverheadFailed, measure_overhead
from agentwatch_agent.runner import AgentRunner, default_output_dir, write_json_atomic
from agentwatch_agent.session import CancellationState
from agentwatch_agent.tasks import TaskDefinitionError, load_task, task_names

EXIT_PASSED, EXIT_FAILED, EXIT_INCOMPLETE_DELIVERY, EXIT_LIVE_UNAVAILABLE = 0, 1, 3, 4
EXIT_RECORDS_REFUSED = 5
EXIT_RUN_UNCLOSED = 6
EXIT_USAGE = 2
DEFAULT_API_URL = "http://127.0.0.1:8090"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="agentwatch-agent", description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run one fixed task with a real model")
    run.add_argument("--task", required=True, help="Task name; see the `tasks` command")
    run.add_argument("--model", default=os.environ.get("AGENTWATCH_LLM_MODEL", DEFAULT_MODEL))
    run.add_argument(
        "--ollama-url",
        type=ollama_base_url,
        default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL),
    )
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

    overhead = commands.add_parser(
        "overhead", help="Measure instrumentation overhead on a task's test workload"
    )
    overhead.add_argument("--task", default="fix-mean-empty")
    overhead.add_argument("--patch", default="return-none-for-empty", help="Correct patch ID")
    overhead.add_argument(
        "--api-url", type=http_url, default=os.environ.get("AGENTWATCH_API_URL", DEFAULT_API_URL)
    )
    overhead.add_argument("--output-dir", type=Path, default=Path(".data/agent-overhead"))

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
            model=OllamaChatModel(args.ollama_url, args.model),
            transport=LiveClient(args.api_url, timeout_seconds=LIVE_TIMEOUT_SECONDS),
            output_dir=default_output_dir(args.output_dir, task),
            limits=RunLimits(run_timeout_seconds=args.run_timeout),
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


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "tasks":
        return tasks_command()
    if args.command == "overhead":
        return overhead_command(args)
    return resend_command(args)
