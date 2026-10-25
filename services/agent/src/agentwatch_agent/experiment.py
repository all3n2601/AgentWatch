"""Clean and contended replays of one recorded run, under a protocol declared before measuring.

Protocol `agent-replay-experiment-v1` (see `ExperimentSettings.protocol`): discarded warm-up
replays, then clean baseline replays, then each contention condition followed by a clean control
replay, so drift between conditions shows up in the controls. Every replay is a live run of the
same replay plan (`replay.py`). A condition's injectors start before the replay (and ramp up),
stop after it, and are bounded to the replay's run limit plus a margin.

Outputs, under one experiment directory:

- `experiment.json`: the declared protocol, the source run, one entry per replay (role,
  condition, live run ID, state, step timings), clean-baseline variability, descriptive
  per-step comparisons with the baseline, and checks. It is rewritten after every replay, so
  an interrupted experiment keeps what it measured.
- `replays/<NN>/`: each replay's `agent-evidence.json` and `live-journal.jsonl`. Directory
  names carry the order only, never the condition.
- `injections/<NN>.json`: the injection-set log of each contended replay.

Conditions and injection logs are labeling metadata. They are kept out of the replays' evidence
and live records, and must never become model features. The comparisons are observations, not
labels: labeling rules belong to the dataset pipeline, and a slowdown under an uncontrolled
host is not proof that the injector caused it.
"""

import functools
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agentwatch_injectors.base import (
    MAX_INJECTION_SECONDS,
    BoundedInjector,
    InjectorCleanupError,
    require_count,
)
from agentwatch_injectors.cpu import CpuSpinInjector, available_cpus
from agentwatch_injectors.inference import (
    MAX_CONCURRENCY,
    MAX_NUM_PREDICT,
    CompetingInferenceInjector,
)
from agentwatch_injectors.injection_set import InjectionCancelled, InjectionSet
from agentwatch_injectors.retrieval import (
    MAX_QUERIES,
    MAX_QUERY_CHARS,
    MAX_THREADS,
    IndexQueryFloodInjector,
)

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.coverage import CoverageError, read_journal
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.replay import ReplayPlan, ReplayRunner
from agentwatch_agent.retrieval import DocumentIndex
from agentwatch_agent.runner import JOURNAL_FILE, RunOutcome, write_json_atomic
from agentwatch_agent.session import CancellationState
from agentwatch_agent.tasks import Task
from agentwatch_agent.tools import SEARCH_RESULTS

PROTOCOL_VERSION = "agent-replay-experiment-v1"
MANIFEST_FILE = "experiment.json"
INJECTOR_KINDS = ("cpu", "inference", "retrieval")
CONDITIONS = (*INJECTOR_KINDS, "mixed")
MIN_BASELINE_REPLAYS = 3
MAX_REPLAYS_PER_ROLE = 20
# Declared before measuring: the implementation plan's 10-15% clean-replay variability target,
# taking its upper end, as the CPU demo's stable-baseline check does.
STABLE_CV_MAX = 0.15
ROOT_STEP = "agent-run"
ReplayRunnerFactory = Callable[[Path, MonotonicClock], ReplayRunner]
InjectorFactory = Callable[[Sequence[str], MonotonicClock, float], list[BoundedInjector]]


@dataclass(frozen=True)
class ExperimentSettings:
    baseline_replays: int = 5
    warmup_replays: int = 1
    conditions: tuple[str, ...] = CONDITIONS
    # The injectors a `mixed` replay runs together.
    mixed: tuple[str, ...] = ("cpu", "inference")
    ramp_seconds: float = 2.0
    # Each injector's hard bound is the replay's run limit plus ramp plus this margin.
    injection_margin_seconds: float = 30.0
    cpu_workers: int = field(default_factory=available_cpus)
    inference_concurrency: int = 2
    inference_num_predict: int = 128
    retrieval_threads: int = 2

    def __post_init__(self):
        if not MIN_BASELINE_REPLAYS <= self.baseline_replays <= MAX_REPLAYS_PER_ROLE:
            raise ValueError(
                f"baseline_replays must be from {MIN_BASELINE_REPLAYS} to {MAX_REPLAYS_PER_ROLE}"
            )
        if not 0 <= self.warmup_replays <= MAX_REPLAYS_PER_ROLE:
            raise ValueError(f"warmup_replays must be from 0 to {MAX_REPLAYS_PER_ROLE}")
        if len(set(self.conditions)) != len(self.conditions) or not set(self.conditions) <= set(
            CONDITIONS
        ):
            raise ValueError(f"conditions must be distinct values of {CONDITIONS}")
        if len(set(self.mixed)) < 2 or not set(self.mixed) <= set(INJECTOR_KINDS):
            raise ValueError(f"mixed must name at least two distinct injectors of {INJECTOR_KINDS}")
        # The injectors' own hard bounds, checked before anything runs.
        require_count("cpu_workers", self.cpu_workers, available_cpus())
        require_count("inference_concurrency", self.inference_concurrency, MAX_CONCURRENCY)
        require_count("inference_num_predict", self.inference_num_predict, MAX_NUM_PREDICT)
        require_count("retrieval_threads", self.retrieval_threads, MAX_THREADS)
        for name in ("ramp_seconds", "injection_margin_seconds"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and nonnegative")

    def kinds(self, condition: str) -> tuple[str, ...]:
        if condition == "clean":
            return ()
        return self.mixed if condition == "mixed" else (condition,)

    def schedule(self) -> list[tuple[str, str]]:
        """(role, condition) for every replay, in run order."""
        order = [("warmup", "clean")] * self.warmup_replays
        order += [("baseline", "clean")] * self.baseline_replays
        for condition in self.conditions:
            order += [("contended", condition), ("clean_control", "clean")]
        return order

    def injection_bound_seconds(self, run_timeout_seconds: float) -> float:
        bound = run_timeout_seconds + self.ramp_seconds + self.injection_margin_seconds
        if bound > MAX_INJECTION_SECONDS:
            raise ValueError(
                f"A {run_timeout_seconds:g} s replay needs a {bound:g} s injection bound, above "
                f"the {MAX_INJECTION_SECONDS:g} s cap; lower the run limit"
            )
        return bound

    def protocol(self, run_timeout_seconds: float) -> dict:
        return {
            "version": PROTOCOL_VERSION,
            "workload": "replay of one recorded, verified agent run (agent-replay-v1)",
            "warmup_replays": self.warmup_replays,
            "baseline_replays": self.baseline_replays,
            "conditions": list(self.conditions),
            "mixed": list(self.mixed),
            "order": (
                "warm-up (discarded), clean baselines, then each condition followed by a clean "
                "control"
            ),
            "ramp_seconds": self.ramp_seconds,
            "run_timeout_seconds": run_timeout_seconds,
            "injection_bound_seconds": self.injection_bound_seconds(run_timeout_seconds),
            "injectors": {
                "cpu": {"workers": self.cpu_workers},
                "inference": {
                    "concurrency": self.inference_concurrency,
                    "num_predict": self.inference_num_predict,
                },
                "retrieval": {"threads": self.retrieval_threads},
            },
            "variability": {
                "statistic": (
                    "coefficient of variation (sample stdev / mean) of each step's total duration "
                    "(submission to finish) across baseline replays"
                ),
                "gate": f"root step ({ROOT_STEP}) coefficient of variation <= {STABLE_CV_MAX}",
            },
            "comparison": "descriptive: each step's duration against its baseline mean and stdev",
            "threshold": None,
        }


def build_injectors(
    kinds: Sequence[str],
    clock: MonotonicClock,
    bound_seconds: float,
    *,
    settings: ExperimentSettings,
    plan: ReplayPlan,
    task: Task,
    ollama_url: str,
) -> list[BoundedInjector]:
    """The real injectors for one condition, all recording ticks in the replay's clock."""
    injectors: list[BoundedInjector] = []
    for kind in kinds:
        if kind == "cpu":
            injectors.append(CpuSpinInjector(clock, bound_seconds, settings.cpu_workers))
        elif kind == "inference":
            injectors.append(
                CompetingInferenceInjector(
                    clock,
                    bound_seconds,
                    url=ollama_url,
                    model=plan.model,
                    # The run's own context size, so Ollama never reloads the model for it.
                    num_ctx=plan.chat_options.get("num_ctx"),
                    concurrency=settings.inference_concurrency,
                    num_predict=settings.inference_num_predict,
                )
            )
        elif kind == "retrieval":
            index = DocumentIndex.from_directory(task.docs_dir)
            queries = plan.search_queries() or [passage.heading for passage in index.passages]
            queries = list(dict.fromkeys(q[:MAX_QUERY_CHARS] for q in queries if q.strip()))
            injectors.append(
                IndexQueryFloodInjector(
                    clock,
                    bound_seconds,
                    search=functools.partial(index.search, limit=SEARCH_RESULTS),
                    queries=queries[:MAX_QUERIES],
                    threads=settings.retrieval_threads,
                )
            )
        else:
            raise ValueError(f"Unknown injector {kind!r}")
    return injectors


def step_timings(run_dir: Path) -> dict[str, dict]:
    """Each journaled step's final state and its durations in seconds, from its own ticks."""
    _registration, records, _unusable = read_journal(run_dir / JOURNAL_FILE)
    latest: dict[str, dict] = {}
    for record in records:
        if record.get("record_type") != "step_event":
            continue
        current = latest.get(record["step_id"])
        if current is None or record["sequence"] > current["sequence"]:
            latest[record["step_id"]] = record
    timings = {}
    for step_id, event in latest.items():
        submitted, started, finished = (
            event["submitted_tick_ns"],
            event["started_tick_ns"],
            event["finished_tick_ns"],
        )
        timings[step_id] = {
            "step_type": event["step_type"],
            "state": event["state"],
            "total_seconds": (finished - submitted) / 1e9 if finished is not None else None,
            "queue_seconds": (started - submitted) / 1e9 if started is not None else None,
            "execution_seconds": (
                (finished - started) / 1e9 if None not in (started, finished) else None
            ),
        }
    return timings


def variability(durations: list[float]) -> dict:
    mean = statistics.fmean(durations)
    stdev = statistics.stdev(durations) if len(durations) > 1 else None
    return {
        "count": len(durations),
        "mean_seconds": mean,
        "stdev_seconds": stdev,
        "cv": stdev / mean if stdev is not None and mean > 0 else None,
        "min_seconds": min(durations),
        "max_seconds": max(durations),
    }


def baseline_variability(baselines: list[dict]) -> dict[str, dict]:
    """Per step completed in every baseline replay: the spread of its total duration."""
    usable = [entry["steps"] for entry in baselines if entry.get("steps")]
    if not usable:
        return {}
    common = set.intersection(
        *({step for step, t in steps.items() if t["state"] == "completed"} for steps in usable)
    )
    return {
        step: variability([steps[step]["total_seconds"] for steps in usable])
        for step in sorted(common)
    }


def compare_with_baseline(steps: dict[str, dict], baseline: dict[str, dict]) -> dict[str, dict]:
    """Descriptive per-step difference from the clean baseline; never a label."""
    comparison = {}
    for step, reference in baseline.items():
        timing = steps.get(step)
        if timing is None or timing["total_seconds"] is None:
            comparison[step] = {"status": "missing"}
            continue
        difference = timing["total_seconds"] - reference["mean_seconds"]
        stdev = reference["stdev_seconds"]
        comparison[step] = {
            "status": timing["state"],
            "total_seconds": timing["total_seconds"],
            "difference_seconds": difference,
            "ratio": timing["total_seconds"] / reference["mean_seconds"]
            if reference["mean_seconds"] > 0
            else None,
            "baseline_stdevs": difference / stdev if stdev else None,
        }
    return comparison


def injection_active(log: dict) -> bool:
    """Whether every injector of a set measurably did work while it ran."""
    measures = {
        "cpu": "hash_rounds",
        "inference": "requests_completed",
        "retrieval": "searches",
    }
    injections = log.get("injections") or []
    return bool(injections) and all(
        (injection.get("activity") or {}).get(measures[injection["injector"]], 0) > 0
        for injection in injections
    )


class ReplayExperiment:
    def __init__(
        self,
        *,
        plan: ReplayPlan,
        settings: ExperimentSettings,
        output_dir: Path,
        run_timeout_seconds: float,
        make_runner: ReplayRunnerFactory,
        make_injectors: InjectorFactory,
        source: dict,
    ):
        self.plan = plan
        self.settings = settings
        self.output_dir = output_dir
        self.run_timeout_seconds = run_timeout_seconds
        self.injection_bound_seconds = settings.injection_bound_seconds(run_timeout_seconds)
        self.make_runner = make_runner
        self.make_injectors = make_injectors
        self.source = source
        self.cancellation = CancellationState()
        self._runner: ReplayRunner | None = None
        self.manifest: dict = {}

    def request_cancel_from_signal(self, source: str, reason: str) -> None:
        """Signal-safe: stop the current replay and any ramp, and run nothing further."""
        self.cancellation.cancel_from_signal(source, reason)
        runner = self._runner
        if runner is not None:
            runner.request_cancel_from_signal(source, reason)

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "protocol": self.settings.protocol(self.run_timeout_seconds),
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "source": self.source,
            "plan": {
                "plan_sha256": self.plan.sha256(),
                "model": self.plan.model,
                "chat_options": self.plan.chat_options,
                "task_id": self.plan.task_id,
                "model_calls": len(self.plan.turns),
                "tool_calls": self.plan.tool_call_count,
            },
            "replays": [],
            "limitations": LIMITATIONS,
        }
        self._save()
        try:
            for index, (role, condition) in enumerate(self.settings.schedule()):
                if self.cancellation.event.is_set():
                    self.manifest["status"] = "cancelled"
                    break
                entry, cleanup_error = self._replay(index, role, condition)
                self.manifest["replays"].append(entry)
                self._summarize()
                self._save()
                if cleanup_error is not None:
                    raise cleanup_error
                if entry["state"] == "not_started":
                    self.manifest["status"] = "stopped_replay_not_started"
                    break
                if entry["state"] == "cancelled" or self.cancellation.event.is_set():
                    self.manifest["status"] = "cancelled"
                    break
            else:
                self.manifest["status"] = "completed"
        except InjectorCleanupError as error:
            # Contention may still be running: stop the experiment and say so loudly.
            self.manifest["status"] = "stopped_injector_cleanup_unconfirmed"
            self.manifest["error"] = f"{type(error).__name__}: {error}"[:1000]
            raise
        except BaseException as error:
            self.manifest["status"] = "failed"
            self.manifest["error"] = f"{type(error).__name__}: {error}"[:1000]
            raise
        finally:
            self.manifest["finished_at"] = datetime.now(UTC).isoformat()
            self._summarize()
            self._save()
        return self.manifest

    def _replay(
        self, index: int, role: str, condition: str
    ) -> tuple[dict, InjectorCleanupError | None]:
        """Run one replay; return its entry and any unconfirmed injector cleanup to raise."""
        name = f"{index:02d}"
        replay_dir = self.output_dir / "replays" / name
        clock = MonotonicClock()
        kinds = self.settings.kinds(condition)
        injection = InjectionSet(
            clock,
            self.make_injectors(kinds, clock, self.injection_bound_seconds),
            ramp_seconds=self.settings.ramp_seconds if kinds else 0.0,
            cancel=self.cancellation.event,
        )
        entry: dict = {
            "index": index,
            "role": role,
            "condition": condition,
            "evidence_dir": f"replays/{name}",
            "injection_log": f"injections/{name}.json" if kinds else None,
        }
        runner = self.make_runner(replay_dir, clock)
        self._runner = runner
        outcome: RunOutcome | None = None
        cleanup_error: InjectorCleanupError | None = None
        try:
            with injection:
                if self.cancellation.event.is_set():
                    raise InjectionCancelled("Cancelled before the replay started")
                outcome = runner.run()
        except InjectionCancelled:
            entry["state"] = "cancelled"
        except RunNotStarted as error:
            entry |= {"state": "not_started", "error": f"{error.stage}: {error}"[:500]}
        except InjectorCleanupError as error:
            # Raised by the caller once this replay's entry is saved.
            cleanup_error = error
            entry |= {"state": "injector_cleanup_unconfirmed", "error": str(error)[:500]}
        finally:
            self._runner = None
            if kinds:
                (self.output_dir / "injections").mkdir(parents=True, exist_ok=True)
                write_json_atomic(self.output_dir / "injections" / f"{name}.json", injection.log())
                entry["injection_cleanup_confirmed"] = injection.log()["cleanup_confirmed"]
                entry["injection_active"] = injection_active(injection.log())
        if outcome is None:
            return entry, cleanup_error
        entry |= {
            "run_id": str(outcome.run_id),
            "state": outcome.state,
            "task_passed": outcome.task_passed,
            "delivery_complete": outcome.delivery_complete,
            "resend_needed": outcome.resend_needed,
            "records_refused": outcome.records_refused,
        }
        try:
            entry["steps"] = step_timings(replay_dir)
        except (CoverageError, OSError) as error:
            entry["steps"] = None
            entry["steps_error"] = f"{type(error).__name__}: {error}"[:500]
        replay = runner.evidence.get("replay") or {}
        outputs = replay.get("model_outputs") or []
        entry["model_outputs_matching_source"] = sum(1 for o in outputs if o["matches_source"])
        entry["model_outputs"] = len(outputs)
        return entry, cleanup_error

    def _summarize(self) -> None:
        replays = self.manifest["replays"]
        baselines = [r for r in replays if r["role"] == "baseline"]
        reference = baseline_variability(baselines)
        self.manifest["baseline_variability"] = reference
        root = reference.get(ROOT_STEP, {})
        for entry in replays:
            if entry["role"] in ("contended", "clean_control") and entry.get("steps"):
                entry["comparison_with_baseline"] = compare_with_baseline(entry["steps"], reference)
        contended = [r for r in replays if r["role"] == "contended"]
        reference_steps = [s["steps"] for s in baselines if s.get("steps")]
        expected_steps = sorted(reference_steps[0]) if reference_steps else None
        executed = [r for r in replays if r.get("steps")]
        self.manifest["checks"] = {
            "baseline_replays_complete": len(baselines) == self.settings.baseline_replays
            and all(_succeeded(r) for r in baselines),
            "stable_baseline": root.get("cv") is not None and root["cv"] <= STABLE_CV_MAX,
            "contended_replays_complete": len(contended) == len(self.settings.conditions)
            and all(_succeeded(r) for r in contended),
            "same_steps_every_replay": expected_steps is not None
            and all(sorted(r["steps"]) == expected_steps for r in executed),
            "injectors_cleaned_up": all(
                r.get("injection_cleanup_confirmed") is True for r in contended
            ),
            "injectors_active": bool(contended)
            and all(r.get("injection_active") is True for r in contended),
            "evidence_delivered": all(
                r.get("delivery_complete") is True for r in replays if "run_id" in r
            ),
        }
        self.manifest["passed"] = self.manifest["status"] == "completed" and all(
            self.manifest["checks"].values()
        )

    def _save(self) -> None:
        write_json_atomic(self.output_dir / MANIFEST_FILE, self.manifest)


def _succeeded(entry: dict) -> bool:
    return entry.get("state") == "completed" and entry.get("task_passed") is True


LIMITATIONS = [
    "Comparisons are observations on one host; they are not labels or causal claims",
    (
        "CPU contention is host-wide: in WSL2 it loads the whole VM, including the model "
        "server's container, so it can also slow inference"
    ),
    (
        "Retrieval contention floods an identical in-process index in the runner's own process; "
        "it slows every runner thread, not only retrieval, and is not a networked retrieval "
        "service"
    ),
    (
        "Inference contention shares the model server's request slots and device; Ollama reports "
        "no queue wait, so waiting inside it is not directly observed"
    ),
    "Temperature 0 does not guarantee identical model output; mismatches are counted per replay",
]
