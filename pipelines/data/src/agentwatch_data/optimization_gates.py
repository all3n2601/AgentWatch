"""Apply the optimization contract's gates to an experiment's paired comparison (O06).

    python -m agentwatch_data.optimization_gates <experiment dir> --contract C \
        [--overhead-verdict overhead-verdict.json] [--output DIR]

The comparison record (O04, `optimization_comparison`) holds observed paired effects and says
`gates: not_evaluated`. This module recomputes that record from the experiment directory,
judges every gate of `docs/optimization-contract.md` for each (condition, candidate), and writes
`gates.json` next to `comparison.json`. Thresholds come from the contract's `[gates]` table,
read from the same file the comparison used (checked by hash); nothing is tuned here.

Each gate is `pass`, `fail`, `inconclusive` (the evidence cannot decide it) or `not_evaluated`
(no measurement exists for it). A candidate's outcome is:

- `fail` when a zero-tolerance gate fails (a new failure or timeout, a task newly failing its
  acceptance tests, a starved admission or a leaked permit): more evidence cannot undo one;
- otherwise `inconclusive` when any of its comparisons is insufficient (missing, rejected or
  unpaired evidence, too few pairs or tasks, a development experiment) or its baseline is
  unstable, because the contract keeps the baseline then;
- otherwise `fail` when any gate fails, `pass` when every gate passes, and `inconclusive`
  when any gate is inconclusive or not evaluated.

At most one passing candidate is promoted: the one with the lowest contended median batch time,
the smaller limit on a tie. Nothing is promoted while the contract is not `accepted`. A pass
describes the declared workload, model and host only, not a general causal improvement.

Exit codes: 0 when the report was written, whatever the outcomes; 1 when the manifest, the
contract or the overhead verdict cannot be used, in which case nothing is written.
"""

import argparse
import dataclasses
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.extract import canonical_sha256, parse_json
from agentwatch_data.files import (
    confined_path,
    json_text,
    read_bounded_text,
    write_atomic,
)
from agentwatch_data.optimization_batches import BATCHES_DIR
from agentwatch_data.optimization_comparison import (
    DEFAULT_REPORTS,
    MANIFEST_FILE,
    coefficient_of_variation,
    compare,
)
from agentwatch_data.optimization_manifest import (
    ManifestError,
    OptimizationContract,
    load_optimization_contract,
    read_manifest,
)

GATES_METHOD = "opt-gates-v1"
REPORT_FILE = "gates.json"
PASS, FAIL, INCONCLUSIVE, NOT_EVALUATED = "pass", "fail", "inconclusive", "not_evaluated"
OVERHEAD_VERDICTS = {
    "within_budget": PASS,
    "over_budget": FAIL,
    "inconclusive": INCONCLUSIVE,
    "not_evaluated": NOT_EVALUATED,
}
# The agent's root step spans a run from its submission to its end (services/agent).
ROOT_STEP = "agent-run"
JOURNAL_FILE = "live-journal.jsonl"
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_VERDICT_BYTES = 4 * 1024 * 1024
# The agent's admission overhead measurement (agentwatch_agent.admission_overhead).
ADMISSION_PROTOCOL = "admission-overhead-v1"
# Gates whose threshold is zero occurrences: one observed failure decides them on any sample.
ZERO_TOLERANCE_GATES = ("failures_and_timeouts", "quality", "starvation_and_leaks")


class GateInputError(ValueError):
    """An input the gates depend on cannot be used; nothing is written."""


@dataclass(frozen=True)
class Thresholds:
    """The contract's `[gates]` table."""

    improvement_conditions: tuple[str, ...]
    guardrail_conditions: tuple[str, ...]
    min_median_batch_time_reduction: float
    min_throughput_ratio: float
    max_p95_task_latency_ratio: float
    max_new_failures: int
    max_new_timeouts: int
    max_tasks_newly_failing_acceptance: int
    max_starved_admissions: int
    max_leaked_permits: int
    max_median_admission_overhead: float
    max_median_instrumentation_overhead: float
    max_baseline_root_step_cv: float


def load_thresholds(path: Path, contract: OptimizationContract) -> Thresholds:
    """The gate thresholds of the very contract file `contract` was loaded from."""
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise GateInputError(f"Optimization contract {path} cannot be read: {error}") from error
    if canonical_sha256(parsed) != contract.sha256:
        raise GateInputError("The contract changed after the comparison loaded it")
    gates = parsed.get("gates")
    if not isinstance(gates, dict):
        raise GateInputError("The contract has no [gates] table")
    values = {}
    for field in dataclasses.fields(Thresholds):
        name, kind, value = field.name, field.type, gates.get(field.name)
        if kind == tuple[str, ...]:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise GateInputError(f"gates.{name} must be a list of condition names")
            values[name] = tuple(value)
        elif kind is int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GateInputError(f"gates.{name} must be a non-negative integer")
            values[name] = value
        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise GateInputError(f"gates.{name} must be a finite number")
            values[name] = float(value)
    if gates.get("improvement_interval_excludes_zero") is not True:
        raise GateInputError("This evaluator implements only an interval that excludes zero")
    return Thresholds(**values)


def gate(name: str, status: str, reason: str, observed=None, threshold=None) -> dict:
    return {
        "gate": name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "reason": reason,
    }


# --- Per-comparison gates -----------------------------------------------------------------


def improvement_gates(effects: dict, limits: Thresholds) -> list[dict]:
    reduction = effects["median_batch_time_reduction"]
    interval = effects["median_batch_time_reduction_interval"]
    threshold = limits.min_median_batch_time_reduction
    improvement = (
        gate(
            "improvement",
            PASS,
            "median batch time fell by at least the threshold",
            reduction,
            threshold,
        )
        if reduction >= threshold
        else gate(
            "improvement",
            FAIL,
            "median batch time fell by less than the threshold",
            reduction,
            threshold,
        )
    )
    low, high = interval["low"], interval["high"]
    bounds = None if low is None else [low, high]
    if low is None:
        uncertainty = gate(
            "improvement_uncertainty", INCONCLUSIVE, "fewer than 2 pairs: no interval", bounds
        )
    elif low <= 0 <= high:
        uncertainty = gate(
            "improvement_uncertainty",
            INCONCLUSIVE,
            "the interval includes zero",
            bounds,
            "excludes 0",
        )
    elif high < 0:
        # Excluding zero means excluding it on the side of improvement: an interval wholly
        # below zero is a confident slowdown, not uncertainty about a gain.
        uncertainty = gate(
            "improvement_uncertainty",
            FAIL,
            "the interval lies below zero: the candidate was slower",
            bounds,
            "excludes 0",
        )
    else:
        uncertainty = gate(
            "improvement_uncertainty",
            PASS,
            "the interval lies above zero",
            bounds,
            "excludes 0",
        )
    return [improvement, uncertainty]


def ratio_gate(name: str, observed, threshold: float, *, at_least: bool) -> dict:
    if observed is None:
        return gate(name, INCONCLUSIVE, "not measured on both sides", None, threshold)
    passed = observed >= threshold if at_least else observed <= threshold
    word = "at least" if at_least else "at most"
    return gate(
        name,
        PASS if passed else FAIL,
        f"candidate/baseline is {'' if passed else 'not '}{word} the threshold",
        observed,
        threshold,
    )


def outcome_gates(changes: dict, limits: Thresholds) -> list[dict]:
    failures = {"new_failures": changes["new_failures"], "new_timeouts": changes["new_timeouts"]}
    allowed = {"new_failures": limits.max_new_failures, "new_timeouts": limits.max_new_timeouts}
    failed = any(failures[key] > allowed[key] for key in failures)
    quality = changes["newly_failing_acceptance"]
    unknown = changes["acceptance_unknown"]
    if quality > limits.max_tasks_newly_failing_acceptance:
        quality_gate = gate(
            "quality", FAIL, "a task passing at baseline failed with the candidate", quality, 0
        )
    elif unknown:
        quality_gate = gate(
            "quality", INCONCLUSIVE, f"{unknown} task outcomes have no recorded acceptance", quality
        )
    else:
        quality_gate = gate("quality", PASS, "no task newly fails its acceptance tests", quality, 0)
    return [
        gate(
            "failures_and_timeouts",
            FAIL if failed else PASS,
            "new failures or timeouts with the candidate" if failed else "none new",
            failures,
            allowed,
        ),
        quality_gate,
    ]


def starvation_gate(candidate: dict, limits: Thresholds) -> dict:
    starved, leaked = candidate["starved_admissions"], candidate["leaked_permits"]
    over = sum(wait["over_declared_max_wait"] for wait in candidate["admission_waits"].values())
    observed = {"starved_admissions": starved, "leaked_permits": leaked, "over_max_wait": over}
    threshold = {
        "starved_admissions": limits.max_starved_admissions,
        "leaked_permits": limits.max_leaked_permits,
    }
    if (starved is not None and starved > limits.max_starved_admissions) or over:
        return gate(
            "starvation_and_leaks", FAIL, "an admission waited past its bound", observed, threshold
        )
    if leaked is not None and leaked > limits.max_leaked_permits:
        return gate(
            "starvation_and_leaks", FAIL, "a permit was held after its batch", observed, threshold
        )
    if starved is None or leaked is None:
        return gate(
            "starvation_and_leaks",
            INCONCLUSIVE,
            "a candidate batch could not observe starvation or leaks",
            observed,
            threshold,
        )
    return gate(
        "starvation_and_leaks", PASS, "no starved admission or leaked permit", observed, threshold
    )


# --- Baseline stability -------------------------------------------------------------------


def root_step_seconds(root: Path, evidence_dir: str, run_id: str | None) -> float | None:
    """The root step's total duration from the run's own journal records (its `run_id`); None
    when not journaled whole. Another run's records in the file are never used."""
    if run_id is None:
        return None
    try:
        path = confined_path(root, evidence_dir, GateInputError) / JOURNAL_FILE
        text = read_bounded_text(path, MAX_JOURNAL_BYTES, GateInputError)
    except GateInputError:
        return None
    latest = None
    for line in text.splitlines():
        try:
            entry = parse_json(line)
        except ValueError:
            continue  # an unusable line; the step's other events still count
        if not isinstance(entry, dict):
            continue  # valid JSON but not a journal entry
        payload = entry.get("payload")
        if (
            entry.get("kind") != "record"
            or not isinstance(payload, dict)
            or payload.get("record_type") != "step_event"
            or payload.get("step_id") != ROOT_STEP
            or payload.get("run_id") != run_id
            or not isinstance(payload.get("sequence"), int)
        ):
            continue
        if latest is None or payload["sequence"] > latest["sequence"]:
            latest = payload
    if latest is None:
        return None
    submitted, finished = latest.get("submitted_tick_ns"), latest.get("finished_tick_ns")
    if not all(
        isinstance(tick, int) and not isinstance(tick, bool) for tick in (submitted, finished)
    ):
        return None
    return (finished - submitted) / 1e9 if finished >= submitted else None


def baseline_stability(root: Path, item: dict, files: dict[str, str], limits: Thresholds) -> dict:
    """Per frozen task, the CV of its root step's total duration over the baseline batches of
    complete pairs (completed, warm tasks); the largest must be within the threshold."""
    durations: dict[str, list[float]] = {}
    missing = 0
    for row in item["per_pair"]:
        record = read_record(root, files[row["baseline_batch_id"]])
        for task in record["tasks"]:
            if task["state"] != "completed" or task["cold"]:
                continue
            seconds = root_step_seconds(root, task["evidence_dir"], task.get("run_id"))
            if seconds is None:
                missing += 1
            else:
                durations.setdefault(task["task_id"], []).append(seconds)
    threshold = limits.max_baseline_root_step_cv
    cvs = {task: coefficient_of_variation(values) for task, values in sorted(durations.items())}
    if missing or not cvs or None in cvs.values():
        return gate(
            "baseline_stability",
            INCONCLUSIVE,
            f"root step durations: {missing} unreadable, or a task has fewer than 2",
            cvs,
            threshold,
        )
    worst = max(cvs.values())
    return gate(
        "baseline_stability",
        PASS if worst <= threshold else FAIL,
        f"largest per-task root step CV {worst:.3f}",
        cvs,
        threshold,
    )


def read_record(root: Path, file: str) -> dict:
    path = confined_path(root, f"{BATCHES_DIR}/{file}", GateInputError)
    record = parse_json(read_bounded_text(path, MAX_RECORD_BYTES, GateInputError))
    if not isinstance(record, dict) or not isinstance(record.get("tasks"), list):
        raise GateInputError(f"{file} has no task list")
    return record


# --- Experiment-wide gates ----------------------------------------------------------------


def overhead_gate(
    name: str,
    verdict: dict | None,
    model: str,
    host_id: str,
    threshold: float,
    *,
    admission: bool,
    contract_sha256: str,
) -> dict:
    """An overhead verdict's group for this experiment's model and host, when one was given.

    Overhead depends on the host, so only a group measured on the manifest's host
    (`hardware.host_id`, the verdict's `host_label`) counts. Admission groups
    (`admission-overhead-v1`) judge the admission gate and every other protocol the
    instrumentation gate, so one verdict file can never stand in for the other. A verdict
    judged under another contract, or against a budget other than this gate's threshold, is
    not used either. The gate records the verdict file's hash, its contract and the matched
    group's settings, task and budget.
    """
    if verdict is None:
        return gate(name, NOT_EVALUATED, "no overhead verdict was given", None, threshold)
    declared = verdict.get("contract") if isinstance(verdict.get("contract"), dict) else {}
    # Verdict reports name their contract by the first 8 digits of its hash.
    verdict_contract = declared.get("sha256")
    source = {
        "verdict_sha256": canonical_sha256(verdict),
        "contract_sha256": verdict_contract,
        "settings": None,
    }
    if (
        not isinstance(verdict_contract, str)
        or not verdict_contract
        or not (contract_sha256.startswith(verdict_contract))
    ):
        return gate(
            name,
            NOT_EVALUATED,
            f"the verdict was judged under contract {verdict_contract!r}, not this one "
            f"({contract_sha256[:8]})",
            None,
            threshold,
        ) | {"source": source}
    groups = [
        group
        for group in verdict.get("groups") or []
        if isinstance(group, dict)
        and isinstance(group.get("settings"), dict)
        and group["settings"].get("model") == model
        and group["settings"].get("host_label") == host_id
        and (group["settings"].get("protocol") == ADMISSION_PROTOCOL) == admission
    ]
    if len(groups) != 1:
        return gate(
            name,
            NOT_EVALUATED,
            f"the verdict has {len(groups)} matching groups for model {model} on host "
            f"{host_id}, not one",
            None,
            threshold,
        ) | {"source": source}
    (group,) = groups
    budget = verdict_budget(verdict, group, admission=admission)
    source |= {
        "settings": group["settings"],
        "task_id": group["settings"].get("task_id"),
        "budget": budget,
    }
    if budget != threshold:
        return gate(
            name,
            NOT_EVALUATED,
            f"the verdict was judged against a budget of {budget!r}, not this gate's {threshold}",
            None,
            threshold,
        ) | {"source": source}
    status = OVERHEAD_VERDICTS.get(group.get("verdict"))
    if status is None:
        raise GateInputError(f"Unknown overhead verdict {group.get('verdict')!r}")
    protocol = group["settings"].get("protocol")
    return gate(
        name,
        status,
        f"{protocol} verdict {group['verdict']} for {model} on {host_id}",
        group["verdict"],
        threshold,
    ) | {"source": source}


def verdict_budget(verdict: dict, group: dict, *, admission: bool):
    """The budget a verdict group was judged against: its own `budget` when it records one,
    otherwise the report's budget for that kind of overhead."""
    if "budget" in group:
        return group["budget"]
    budgets = verdict.get("budget") if isinstance(verdict.get("budget"), dict) else {}
    key = "max_median_admission_overhead" if admission else "max_median_instrumentation_overhead"
    return budgets.get(key)


# --- Outcomes -----------------------------------------------------------------------------


def judge_comparison(root: Path, item: dict, files: dict, limits: Thresholds) -> dict:
    condition, effects = item["condition"], item["effects"]
    gates = []
    if effects is None:
        gates.append(gate("effects", INCONCLUSIVE, "no complete pair to compare"))
    else:
        if condition in limits.improvement_conditions:
            gates.extend(improvement_gates(effects, limits))
        if condition in limits.guardrail_conditions:
            gates.append(
                ratio_gate(
                    "throughput",
                    effects["throughput_ratio"],
                    limits.min_throughput_ratio,
                    at_least=True,
                )
            )
            gates.append(
                ratio_gate(
                    "tail_latency",
                    effects["p95_warm_task_latency_ratio"],
                    limits.max_p95_task_latency_ratio,
                    at_least=False,
                )
            )
            gates.append(starvation_gate(effects["candidate"], limits))
        gates.append(baseline_stability(root, item, files, limits))
    if condition in limits.guardrail_conditions:
        gates.extend(outcome_gates(item["outcome_changes"], limits))
    return {
        "condition": condition,
        "sufficient": item["sufficient"],
        "insufficient_reasons": item["insufficient_reasons"],
        "gates": gates,
    }


def candidate_outcome(conditions: list[dict], shared: list[dict]) -> tuple[str, list[str]]:
    gates = [(entry["condition"], item) for entry in conditions for item in entry["gates"]]
    gates += [("experiment", item) for item in shared]
    decisive = [
        f"{where}: {item['gate']}"
        for where, item in gates
        if item["gate"] in ZERO_TOLERANCE_GATES and item["status"] == FAIL
    ]
    if decisive:
        return FAIL, decisive
    reasons = [
        f"{entry['condition']}: {reason}"
        for entry in conditions
        for reason in entry["insufficient_reasons"]
    ]
    if reasons:
        return INCONCLUSIVE, reasons
    unstable = [
        f"{where}: baseline {item['status']}"
        for where, item in gates
        if item["gate"] == "baseline_stability" and item["status"] != PASS
    ]
    if unstable:
        return INCONCLUSIVE, unstable
    failed = [f"{where}: {item['gate']}" for where, item in gates if item["status"] == FAIL]
    if failed:
        return FAIL, failed
    open_gates = [
        f"{where}: {item['gate']} {item['status']}"
        for where, item in gates
        if item["status"] != PASS
    ]
    return (INCONCLUSIVE, open_gates) if open_gates else (PASS, [])


def promotion(candidates: list[dict], contract: OptimizationContract) -> dict:
    passing = [candidate for candidate in candidates if candidate["outcome"] == PASS]
    tried = len(candidates)
    if contract.status != "accepted":
        return {
            "promoted": None,
            "candidates_tried": tried,
            "reason": f"the contract is {contract.status}, not accepted: nothing is promoted",
        }
    if not passing:
        return {
            "promoted": None,
            "candidates_tried": tried,
            "reason": "no candidate passed every gate",
        }
    best = min(
        passing,
        key=lambda candidate: (
            candidate["contended_median_batch_seconds"],
            sorted(candidate["limits"].values()),
        ),
    )
    return {
        "promoted": best["candidate"],
        "candidates_tried": tried,
        "reason": "lowest contended median batch time among candidates passing every gate",
    }


def read_verdict(path: Path | None) -> dict | None:
    if path is None:
        return None
    verdict = parse_json(read_bounded_text(path, MAX_VERDICT_BYTES, GateInputError))
    if not isinstance(verdict, dict):
        raise GateInputError(f"{path.name} is not a JSON object")
    return verdict


def evaluate(
    root: Path,
    contract_path: Path,
    overhead_verdict: Path | None = None,
    admission_verdict: Path | None = None,
) -> dict:
    contract = load_optimization_contract(contract_path)
    limits = load_thresholds(contract_path, contract)
    root = root.resolve()
    manifest = read_manifest(root / MANIFEST_FILE, contract)
    comparison = compare(root, contract)
    model = manifest.model["name"]
    host_id = manifest.hardware["host_id"]
    files = {
        batch["batch_id"]: batch["file"] for batch in comparison["batches"] if batch["accepted"]
    }
    shared = [
        overhead_gate(
            "admission_overhead",
            read_verdict(admission_verdict),
            model,
            host_id,
            limits.max_median_admission_overhead,
            admission=True,
            contract_sha256=contract.sha256,
        ),
        overhead_gate(
            "instrumentation_overhead",
            read_verdict(overhead_verdict),
            model,
            host_id,
            limits.max_median_instrumentation_overhead,
            admission=False,
            contract_sha256=contract.sha256,
        ),
    ]
    candidates = []
    for name in sorted({item["candidate"] for item in comparison["comparisons"]}):
        items = [item for item in comparison["comparisons"] if item["candidate"] == name]
        conditions = [judge_comparison(root, item, files, limits) for item in items]
        outcome, reasons = candidate_outcome(conditions, shared)
        contended = [
            item["effects"]["candidate"]["median_batch_seconds"]
            for item in items
            if item["condition"] in limits.improvement_conditions and item["effects"]
        ]
        candidates.append(
            {
                "candidate": name,
                "limits": items[0]["candidate_limits"],
                "outcome": outcome,
                "reasons": reasons,
                "contended_median_batch_seconds": min(contended) if contended else None,
                "conditions": conditions,
            }
        )
    return {
        "method": GATES_METHOD,
        "contract": {
            "version": contract.version,
            "status": contract.status,
            "sha256": contract.sha256[:8],
        },
        "manifest_sha256": comparison["manifest_sha256"],
        "experiment_id": comparison["experiment_id"],
        "purpose": manifest.purpose,
        # The comparison judged, so a later comparison of the same files can be matched to it.
        "comparison_sha256": canonical_sha256(comparison),
        "thresholds": dataclasses.asdict(limits),
        "experiment_gates": shared,
        "candidates": candidates,
        "promotion": promotion(candidates, contract),
        "limitations": [
            "Describes the declared workload, model and host; not a general causal improvement",
            "Without an admission overhead verdict, no candidate can pass every gate",
            "Starvation and leaks are judged on candidate batches; the baseline has no limits",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--overhead-verdict", type=Path, help="A07 overhead-verdict.json")
    parser.add_argument(
        "--admission-verdict", type=Path, help="overhead-verdict.json of admission-overhead-v1"
    )
    parser.add_argument("--output", type=Path, help=f"report directory (default {DEFAULT_REPORTS})")
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            args.experiment, args.contract, args.overhead_verdict, args.admission_verdict
        )
    except (ManifestError, GateInputError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output = args.output or DEFAULT_REPORTS / report["manifest_sha256"][:12]
    path = output / REPORT_FILE
    write_atomic(path, json_text(report))
    print(f"gates: {path}")
    for candidate in report["candidates"]:
        print(f"  {candidate['candidate']}: {candidate['outcome']}")
    print(f"  promotion: {report['promotion']['promoted']} ({report['promotion']['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
