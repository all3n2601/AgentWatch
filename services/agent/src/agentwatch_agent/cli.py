"""Command line: `run` a task, list `tasks`, or `resend` a run's journal.

Exit codes:
  0  task passed and all live evidence was delivered
  1  run failed or was cancelled; its evidence was delivered
  2  invalid command-line usage
  3  live evidence is incomplete: resend the journal once the API is reachable
  4  the live API did not register the run (unreachable or refused), so nothing ran
"""

import argparse
import os
import signal
import sys
import urllib.parse
from pathlib import Path
from types import FrameType

from agentwatch_sdk import LiveClient
from agentwatch_worker.ollama import ollama_base_url

from agentwatch_agent.config import DEFAULT_MODEL, RunLimits
from agentwatch_agent.delivery import replay_journal
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.model import OllamaChatModel
from agentwatch_agent.overhead import measure_overhead
from agentwatch_agent.runner import AgentRunner, default_output_dir, write_json_atomic
from agentwatch_agent.tasks import TaskDefinitionError, load_task, task_names

EXIT_PASSED, EXIT_FAILED, EXIT_INCOMPLETE_DELIVERY, EXIT_LIVE_UNAVAILABLE = 0, 1, 3, 4
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
        print(
            f"error: the live API at {args.api_url} did not register the run ({error}); "
            "nothing ran. Start it with `make dev-worker` and check the evidence file in "
            f"{runner.output_dir}.",
            file=sys.stderr,
        )
        return EXIT_LIVE_UNAVAILABLE
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(
        f"Run {outcome.run_id}: {outcome.state}"
        + (f" ({outcome.reason})" if outcome.reason else "")
    )
    print(f"Evidence: {outcome.evidence_path}")
    if not outcome.delivery_complete:
        print(
            f"Live evidence is incomplete. Resend it with:\n"
            f"  uv run --package agentwatch-agent python -m agentwatch_agent resend "
            f"{outcome.journal_path}",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE_DELIVERY
    return EXIT_PASSED if outcome.state == "completed" and outcome.task_passed else EXIT_FAILED


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
        return 2
    output_dir = default_output_dir(args.output_dir, task)
    report = measure_overhead(
        task, args.patch, LiveClient(args.api_url, LIVE_TIMEOUT_SECONDS), output_dir
    )
    path = output_dir / "overhead.json"
    write_json_atomic(path, report)
    walls = report["workload_wall_seconds"]
    print(
        f"Workload median {walls['uninstrumented']['median']:.4f} s uninstrumented, "
        f"{walls['instrumented']['median']:.4f} s instrumented "
        f"({report['median_difference_fraction']:+.1%}); evidence: {path}"
    )
    return EXIT_PASSED


def resend_command(args: argparse.Namespace) -> int:
    report = replay_journal(args.journal, LiveClient(args.api_url, LIVE_TIMEOUT_SECONDS))
    print(
        f"Run {report.run_id}: resent {report.records_sent} records and applied "
        f"{report.changes_applied} state changes; skipped {report.records_skipped_rejected} "
        f"previously rejected records. Run state: {report.final_state} "
        f"(journaled: {report.expected_state})."
    )
    if report.truncated_final_line:
        print("note: ignored an interrupted final journal line", file=sys.stderr)
    if report.changes_superseded:
        print("note: the run was closed by another actor before its journaled closure")
    for rejection in report.rejections:
        print(
            f"rejected record {rejection['record_id']} ({rejection['status']}): "
            f"{rejection['detail']}",
            file=sys.stderr,
        )
    for unresolved in report.unresolved_changes:
        print(
            f"run change to {unresolved['state']} was refused ({unresolved['status']}): "
            f"{unresolved['detail']}",
            file=sys.stderr,
        )
    return EXIT_PASSED if report.complete else EXIT_INCOMPLETE_DELIVERY


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "tasks":
        return tasks_command()
    if args.command == "overhead":
        return overhead_command(args)
    return resend_command(args)
