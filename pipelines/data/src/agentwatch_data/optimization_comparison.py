"""Paired baseline/candidate comparison records for optimization experiments (O04).

    python -m agentwatch_data.optimization_comparison schedule --manifest M --contract C
    python -m agentwatch_data.optimization_comparison compare <experiment dir> --contract C

`schedule` prints the preregistered batch order for the experiment runner (O06). `compare`
validates every batch record against the manifest and writes one deterministic comparison
record per (condition, candidate) to .data/reports/optimization/<manifest hash>/comparison.json.

A pair counts only when both of its scheduled batches have a valid, completed record with a
measured batch time; every other pair is reported as unpaired with its reason, and every
record, accepted or rejected, is listed. Effects are observed paired differences on the
declared workload and hardware, with a seeded percentile bootstrap interval. They are not
gate outcomes: opt-contract-v1's gates are applied by the hardware evaluation (O06), so
`gates` is always `not_evaluated` here, and nothing in a record establishes a general causal
improvement.
"""

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.files import json_text, write_atomic
from agentwatch_data.optimization_batches import (
    BatchRecord,
    BatchRejection,
    TaskResult,
    read_batches,
)
from agentwatch_data.optimization_manifest import (
    BASELINE,
    CANDIDATE,
    PAIR,
    WARMUP,
    Manifest,
    ManifestError,
    OptimizationContract,
    Policy,
    ScheduledBatch,
    load_optimization_contract,
    read_manifest,
    schedule,
)

COMPARISON_METHOD = "opt-comparison-v1"
MANIFEST_FILE = "manifest.json"
REPORT_FILE = "comparison.json"
DEFAULT_REPORTS = Path(".data/reports/optimization")
INTERVAL_METHOD = "paired_bootstrap_percentile"
FAILED_STATES = ("failed", "cancelled")
P95 = 0.95
GATES_NOT_EVALUATED = {
    "status": "not_evaluated",
    "reason": (
        "O04 records observed paired effects only; opt-contract-v1 gates are applied by the "
        "hardware evaluation (O06) once the contract is accepted"
    ),
}


@dataclass(frozen=True)
class Pair:
    pair_id: str
    baseline: BatchRecord | None
    candidate: BatchRecord | None
    problem: str | None  # why the pair cannot be compared; None when it is complete

    @property
    def complete(self) -> bool:
        return self.problem is None


def side_problem(
    configuration: str, slot: ScheduledBatch, record: BatchRecord | None, rejected: dict
) -> str | None:
    if slot.index in rejected:
        return f"{configuration} batch {slot.index} was rejected ({rejected[slot.index]})"
    if record is None:
        return f"{configuration} batch {slot.index} has no record"
    if record.status != "completed":
        return f"{configuration} batch {slot.index} is {record.status}"
    if not record.batch_seconds:
        # Zero cannot be a real batch time, and a relative reduction needs a positive baseline.
        return f"{configuration} batch {slot.index} has no positive measured batch time"
    return None


def build_pairs(
    slots: tuple[ScheduledBatch, ...],
    records: list[BatchRecord],
    rejections: list[BatchRejection],
) -> dict[tuple[str, str], list[Pair]]:
    """Every scheduled pair by (condition, candidate), complete or with the reason it is not."""
    by_index = {record.slot.index: record for record in records}
    rejected = {}
    for rejection in rejections:
        stem = rejection.file.removesuffix(".json")
        if stem.isdigit():
            rejected[int(stem)] = rejection.code
    sides: dict[str, dict[str, ScheduledBatch]] = {}
    for slot in slots:
        if slot.kind == PAIR:
            sides.setdefault(slot.pair_id, {})[slot.configuration] = slot
    pairs: dict[tuple[str, str], list[Pair]] = {}
    for pair_id, both in sides.items():
        baseline_slot, candidate_slot = both[BASELINE], both[CANDIDATE]
        baseline = by_index.get(baseline_slot.index)
        candidate = by_index.get(candidate_slot.index)
        problems = [
            problem
            for problem in (
                side_problem(BASELINE, baseline_slot, baseline, rejected),
                side_problem(CANDIDATE, candidate_slot, candidate, rejected),
            )
            if problem
        ]
        key = (baseline_slot.condition, baseline_slot.candidate)
        pairs.setdefault(key, []).append(
            Pair(pair_id, baseline, candidate, "; ".join(problems) or None)
        )
    return pairs


# --- Statistics -------------------------------------------------------------------------------


def nearest_rank(values: list[float], quantile: float) -> float | None:
    """The nearest-rank percentile: the smallest value with at least `quantile` at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def bootstrap_median(
    values: list[float], resamples: int, level: float, seed: int
) -> tuple[float, float] | None:
    """Percentile interval of the median, resampling pairs with a recorded seed."""
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values))) for _ in range(resamples)
    )
    tail = (1 - level) / 2
    return medians[math.floor(tail * resamples)], medians[math.ceil((1 - tail) * resamples) - 1]


def coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2 or statistics.fmean(values) == 0:
        return None
    return statistics.stdev(values) / statistics.fmean(values)


# --- Effects ----------------------------------------------------------------------------------


def tasks_of(batches: Iterable[BatchRecord]) -> list[TaskResult]:
    return [task for batch in batches for task in batch.tasks]


def outcome_changes(pairs: list[Pair]) -> dict[str, int]:
    """Failures, timeouts and acceptance failures the candidate added on the same tasks."""
    changes = {
        "new_failures": 0,
        "new_timeouts": 0,
        "newly_failing_acceptance": 0,
        "acceptance_unknown": 0,
    }
    for pair in pairs:
        baseline = {task.task_id: task for task in pair.baseline.tasks}
        for task in pair.candidate.tasks:
            before = baseline[task.task_id]
            if before.state == "completed" and task.state in FAILED_STATES:
                changes["new_failures"] += 1
            if task.state == "timed_out" and before.state != "timed_out":
                changes["new_timeouts"] += 1
            if before.acceptance is None or task.acceptance is None:
                changes["acceptance_unknown"] += 1
            elif before.acceptance == "passed" and task.acceptance == "failed":
                changes["newly_failing_acceptance"] += 1
    return changes


def admission_total(batches: list[BatchRecord], field: str) -> int | None:
    """A sum over batches, or None when any batch could not observe it (never a partial sum)."""
    values = [getattr(batch, field) for batch in batches]
    return None if not values or None in values else sum(values)


def side_summary(batches: list[BatchRecord]) -> dict:
    tasks = tasks_of(batches)
    warm = [task.latency_seconds for task in tasks if not task.cold and task.state == "completed"]
    waits = [task.admission_wait_seconds for task in tasks]
    seconds = [batch.batch_seconds for batch in batches]
    completed = sum(task.state == "completed" for task in tasks)
    return {
        "batches": len(batches),
        "completed_tasks": completed,
        "cold_tasks": sum(task.cold for task in tasks),
        "median_batch_seconds": median_or_none(seconds),
        "throughput_tasks_per_second": ratio(completed, sum(seconds)),
        "p95_warm_task_latency_seconds": nearest_rank(
            [latency for latency in warm if latency is not None], P95
        ),
        "warm_tasks_without_latency": sum(latency is None for latency in warm),
        "median_admission_wait_seconds": median_or_none(
            [wait for wait in waits if wait is not None]
        ),
        "tasks_without_admission_wait": sum(wait is None for wait in waits),
        "starved_admissions": admission_total(batches, "starved_admissions"),
        "leaked_permits": admission_total(batches, "leaked_permits"),
        "policy_versions": sorted(
            {batch.policy_version for batch in batches if batch.policy_version is not None}
        ),
    }


def effects(
    pairs: list[Pair], manifest: Manifest, contract: OptimizationContract
) -> tuple[dict, list[dict]]:
    """Observed paired effects over complete pairs, and one row per complete pair."""
    rows = [
        {
            "pair_id": pair.pair_id,
            "baseline_batch_id": pair.baseline.batch_id,
            "candidate_batch_id": pair.candidate.batch_id,
            "baseline_seconds": pair.baseline.batch_seconds,
            "candidate_seconds": pair.candidate.batch_seconds,
            # Relative reduction of batch completion time; positive means the candidate was faster.
            "reduction": 1 - pair.candidate.batch_seconds / pair.baseline.batch_seconds,
        }
        for pair in pairs
    ]
    baseline = side_summary([pair.baseline for pair in pairs])
    candidate = side_summary([pair.candidate for pair in pairs])
    reductions = [row["reduction"] for row in rows]
    interval = bootstrap_median(
        reductions, contract.bootstrap_resamples, contract.interval_level, manifest.seed
    )
    summary = {
        "baseline": baseline,
        "candidate": candidate,
        "median_reduction": median_or_none(reductions),
        "reduction_interval": {
            "method": INTERVAL_METHOD,
            "level": contract.interval_level,
            "resamples": contract.bootstrap_resamples,
            "seed": manifest.seed,
            "low": None if interval is None else interval[0],
            "high": None if interval is None else interval[1],
        },
        "throughput_ratio": ratio(
            candidate["throughput_tasks_per_second"], baseline["throughput_tasks_per_second"]
        ),
        "p95_warm_task_latency_ratio": ratio(
            candidate["p95_warm_task_latency_seconds"], baseline["p95_warm_task_latency_seconds"]
        ),
        "baseline_batch_cv": coefficient_of_variation(
            [pair.baseline.batch_seconds for pair in pairs]
        ),
        **outcome_changes(pairs),
    }
    return summary, rows


def sufficiency(pairs: list[Pair], manifest: Manifest, contract: OptimizationContract) -> list[str]:
    """Why the complete pairs fall short of the contract's minimums; empty when they do not."""
    reasons = []
    if len(pairs) < contract.min_paired_batches:
        reasons.append(
            f"{len(pairs)} complete pairs, fewer than the contract's {contract.min_paired_batches}"
        )
    for side in (BASELINE, CANDIDATE):
        completed = sum(
            task.state == "completed"
            for pair in pairs
            for task in (pair.baseline if side == BASELINE else pair.candidate).tasks
        )
        if completed < contract.min_completed_tasks:
            reasons.append(
                f"{completed} completed {side} tasks, fewer than the contract's "
                f"{contract.min_completed_tasks}"
            )
    if manifest.purpose != "evaluation":
        reasons.append("development experiments choose settings and are never gated")
    return reasons


def comparison(
    condition: str,
    candidate: Policy,
    pairs: list[Pair],
    manifest: Manifest,
    contract: OptimizationContract,
) -> dict:
    complete = [pair for pair in pairs if pair.complete]
    observed, rows = effects(complete, manifest, contract) if complete else (None, [])
    shortfalls = sufficiency(complete, manifest, contract)
    return {
        "condition": condition,
        "candidate": candidate.name,
        "candidate_limits": candidate.limits,
        "candidate_policy_sha256": candidate.sha256,
        "baseline_policy_sha256": manifest.baseline.sha256,
        "pairs": {
            "scheduled": len(pairs),
            "complete": len(complete),
            "unpaired": [
                {"pair_id": pair.pair_id, "reason": pair.problem}
                for pair in sorted(pairs, key=lambda pair: pair.pair_id)
                if not pair.complete
            ],
        },
        "per_pair": sorted(rows, key=lambda row: row["pair_id"]),
        "sufficient": not shortfalls,
        "insufficient_reasons": shortfalls,
        "effects": observed,
        "gates": GATES_NOT_EVALUATED,
    }


def batch_entry(record: BatchRecord) -> dict:
    slot = record.slot
    return {
        "file": record.file,
        "sha256": record.file_sha256,
        "accepted": True,
        "batch_id": record.batch_id,
        "index": slot.index,
        "kind": slot.kind,
        "condition": slot.condition,
        "configuration": slot.configuration,
        "candidate": slot.candidate,
        "pair_index": slot.pair_index,
        "status": record.status,
        "reason": record.reason,
        "policy_version": record.policy_version,
        "injection_active": record.injection_active,
        "run_ids": sorted(task.run_id for task in record.tasks if task.run_id is not None),
    }


def rejection_entry(rejection: BatchRejection) -> dict:
    return {
        "file": rejection.file,
        "sha256": rejection.file_sha256,
        "accepted": False,
        "code": rejection.code,
        "detail": rejection.detail,
    }


def compare(root: Path, contract: OptimizationContract) -> dict:
    """The comparison record of one experiment directory; deterministic for the same files."""
    root = root.resolve()
    manifest = read_manifest(root / MANIFEST_FILE, contract)
    slots = schedule(manifest)
    records, rejections = read_batches(root, manifest, slots, contract)
    pairs = build_pairs(slots, records, rejections)
    recorded = {record.slot.index for record in records} | {
        int(rejection.file.removesuffix(".json"))
        for rejection in rejections
        if rejection.file.removesuffix(".json").isdigit()
    }
    return {
        "method": COMPARISON_METHOD,
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.sha256,
        "purpose": manifest.purpose,
        "contract": {
            "version": contract.version,
            "status": contract.status,
            "sha256": contract.sha256[:8],
        },
        "seed": manifest.seed,
        "candidates_tried": len(manifest.candidates),
        "scheduled_batches": len(slots),
        "missing_batches": sorted(slot.index for slot in slots if slot.index not in recorded),
        "warmup_batches": sum(record.slot.kind == WARMUP for record in records),
        "batches": sorted(
            [batch_entry(record) for record in records]
            + [rejection_entry(rejection) for rejection in rejections],
            key=lambda entry: entry["file"],
        ),
        "comparisons": [
            comparison(
                condition.name,
                candidate,
                pairs[(condition.name, candidate.name)],
                manifest,
                contract,
            )
            for condition in manifest.conditions
            for candidate in manifest.candidates
        ],
    }


def schedule_document(manifest: Manifest) -> dict:
    return {
        "manifest_sha256": manifest.sha256,
        "batches": [
            {
                "index": slot.index,
                "kind": slot.kind,
                "condition": slot.condition,
                "configuration": slot.configuration,
                "candidate": slot.candidate,
                "pair_index": slot.pair_index,
                "policy_sha256": slot.policy_sha256,
                "limits": manifest.policy(
                    slot.candidate if slot.configuration == CANDIDATE else None
                ).limits,
            }
            for slot in schedule(manifest)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("schedule", help="print the preregistered batch order")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--contract", type=Path, required=True)
    run = commands.add_parser("compare", help="validate batches and write the comparison")
    run.add_argument("experiment", type=Path)
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--output", type=Path, help=f"report directory (default {DEFAULT_REPORTS})")
    args = parser.parse_args(argv)
    try:
        contract = load_optimization_contract(args.contract)
        if args.command == "schedule":
            print(json.dumps(schedule_document(read_manifest(args.manifest, contract)), indent=2))
            return 0
        report = compare(args.experiment, contract)
    except ManifestError as error:
        print(f"Optimization comparison failed: {error}", file=sys.stderr)
        return 1
    output = args.output or DEFAULT_REPORTS / report["manifest_sha256"][:12]
    path = output / REPORT_FILE
    write_atomic(path, json_text(report))
    print(f"comparison: {path}")
    for entry in report["comparisons"]:
        pairs = entry["pairs"]
        state = "sufficient" if entry["sufficient"] else "insufficient"
        print(
            f"  {entry['condition']} {entry['candidate']}: "
            f"{pairs['complete']}/{pairs['scheduled']} pairs, {state}"
        )
    for entry in report["batches"]:
        if not entry["accepted"]:
            print(f"  rejected {entry['file']}: {entry['code']}: {entry['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
