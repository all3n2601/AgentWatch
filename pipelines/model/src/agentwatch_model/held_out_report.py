"""Held-out report: miss and false-alarm measures per scorer, truth and slice.

The measures were declared before any scoring (docs/superpowers/specs/
2026-11-11-a06-held-out-scoring-design.md, "Report measures"). Nothing here is a gate: the
report is a descriptive baseline. A rate whose denominator is zero is None, never zero, and a
truth that could not be judged for a step leaves that step out of that truth's measures.
"""

from collections import Counter

from agentwatch_data.replay_labels import task_group

from agentwatch_model.held_out_scorers import CLASSIFIER_SCORER, ScoredStep
from agentwatch_model.metrics import NONE, bootstrap_interval

INFERENCE = "inference"
ABSTAINED = "abstained"
GATE = "none (descriptive baseline)"
BATCH_KEY = "batch_id"
NO_PLAN_REASON = (
    "runs without a replay plan, so no step baseline "
    "(live-confirmation runs are not replays and have none)"
)
LIMITATIONS = (
    "one laptop, one sweep per configuration",
    "dataset v1's replays were recorded mostly on Ollama, A06 on vLLM",
    "no `steps-v1.3` feature describes GPU or vLLM state",
    (
        "one of the two A06 tasks (`fix-mean-empty`) was seen in training, "
        "so only `fix-median-even` tests an unseen task"
    ),
)
# Per-step counters, in the order they are summed for rates and bootstrap draws.
COUNTERS = (
    "inference",
    "inference_missed",
    "inference_answered",
    "inference_answered_missed",
    "none",
    "none_false_alarm",
    "none_answered",
    "none_answered_false_alarm",
    "abstained",
    "steps",
)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def step_counters(truth: str, score: ScoredStep) -> dict[str, int]:
    """What one step contributes to each rate; an abstention counts as a miss, not an alarm."""
    counts = dict.fromkeys(COUNTERS, 0)
    counts["steps"] = 1
    counts["abstained"] = int(score.abstained)
    if truth == INFERENCE:
        missed = score.abstained or score.label != INFERENCE
        counts["inference"] = 1
        counts["inference_missed"] = int(missed)
        counts["inference_answered"] = int(not score.abstained)
        counts["inference_answered_missed"] = int(not score.abstained and missed)
    elif truth == NONE:
        alarm = not score.abstained and score.label != NONE
        counts["none"] = 1
        counts["none_false_alarm"] = int(alarm)
        counts["none_answered"] = int(not score.abstained)
        counts["none_answered_false_alarm"] = int(alarm)
    return counts


def miss_rate(totals: dict[str, int]) -> float | None:
    return rate(totals["inference_missed"], totals["inference"])


def false_alarm_rate(totals: dict[str, int]) -> float | None:
    return rate(totals["none_false_alarm"], totals["none"])


def measures(
    rows: list[dict], scored: list[ScoredStep], truth: str, *, resamples: int, seed: int
) -> dict:
    """Confusion, miss and false-alarm rates, and batch-grouped intervals for one slice.

    Only rows whose `truths[truth]` is not null count. Intervals resample whole batches, because
    runs in one batch share their contention. Each batch is summarised once, so a draw costs one
    addition per batch rather than one per step, and a rate draws only the batches that have
    its denominator.
    """
    if len(rows) != len(scored):
        raise ValueError("rows and scored steps must have the same length")
    confusion: dict[str, dict[str, int]] = {}
    totals = dict.fromkeys(COUNTERS, 0)
    batches: dict[str, dict[str, int]] = {}
    for row, score in zip(rows, scored, strict=True):
        actual = row["truths"][truth]
        if actual is None:
            continue
        predicted = ABSTAINED if score.abstained else score.label
        cell = confusion.setdefault(actual, {})
        cell[predicted] = cell.get(predicted, 0) + 1
        counts = step_counters(actual, score)
        batch = batches.setdefault(row[BATCH_KEY], dict.fromkeys(COUNTERS, 0))
        for name, value in counts.items():
            totals[name] += value
            batch[name] += value

    def interval(rate_of, denominator: str) -> dict:
        # Only batches that contribute to this rate's denominator are groups for it, so the
        # group count and the minimum-groups rule describe the batches the rate is made of.
        by_batch = [counts for _, counts in sorted(batches.items()) if counts[denominator] > 0]
        pseudo_rows = [
            {BATCH_KEY: index, "counts": counts} for index, counts in enumerate(by_batch)
        ]

        def draw(indices: list[int]) -> float | None:
            summed = dict.fromkeys(COUNTERS, 0)
            for index in indices:
                for name, value in by_batch[index].items():
                    summed[name] += value
            return rate_of(summed)

        return bootstrap_interval(
            pseudo_rows,
            draw,
            group_key=BATCH_KEY,
            resamples=resamples,
            seed=seed,
        )

    return {
        "steps": totals["steps"],
        "confusion": confusion,
        "inference_steps": totals["inference"],
        "miss_rate": miss_rate(totals),
        "answered_miss_rate": rate(
            totals["inference_answered_missed"], totals["inference_answered"]
        ),
        "none_steps": totals["none"],
        "false_alarm_rate": false_alarm_rate(totals),
        "answered_false_alarm_rate": rate(
            totals["none_answered_false_alarm"], totals["none_answered"]
        ),
        "abstention_rate": rate(totals["abstained"], totals["steps"]),
        "intervals": {
            "miss_rate": interval(miss_rate, "inference"),
            "false_alarm_rate": interval(false_alarm_rate, "none"),
        },
    }


def without_abstention(cell: dict) -> dict:
    """Abstention was never fitted: its rates are unmeasured, not zero."""
    return cell | {
        "abstention_rate": None,
        "answered_miss_rate": None,
        "answered_false_alarm_rate": None,
    }


def slice_measures(
    rows: list[dict],
    scored: list[ScoredStep],
    truth: str,
    *,
    resamples: int,
    seed: int,
    abstention_unavailable: bool,
) -> dict | None:
    """measures() for one slice, or None when no step in it has a judged truth."""
    if all(row["truths"][truth] is None for row in rows):
        return None
    result = measures(rows, scored, truth, resamples=resamples, seed=seed)
    if abstention_unavailable and scored and scored[0].scorer == CLASSIFIER_SCORER:
        return without_abstention(result)
    return result


def unscored_steps(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Steps with no judged truth, by truth and recorded reason."""
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        for truth, value in row["truths"].items():
            if value is None:
                reason = row["truth_reasons"].get(truth) or "unrecorded"
                counts.setdefault(truth, Counter())[reason] += 1
    return {truth: dict(sorted(reasons.items())) for truth, reasons in sorted(counts.items())}


def unscored_runs(summaries: list[dict]) -> dict[str, dict]:
    """Runs and batches the truth step never reached, per configuration.

    A run without a replay plan has no baseline and no truth records; live-confirmation runs
    are such runs. `load_truths` guarantees the counters and the excluded-batch list.
    """
    runs: dict[str, dict] = {}
    for summary in summaries:
        entry = runs.setdefault(
            summary["configuration"],
            {
                "runs_without_plan": {"count": 0, "reason": NO_PLAN_REASON},
                "runs_without_evidence": 0,
                "excluded_batches": 0,
            },
        )
        entry["runs_without_plan"]["count"] += summary["runs_without_plan"]
        entry["runs_without_evidence"] += summary["runs_without_evidence"]
        entry["excluded_batches"] += len(summary["excluded_batches"])
    return runs


def build_report(rows: list[dict], scores: dict[str, list[ScoredStep]], context: dict) -> dict:
    """Cells per (scorer, truth, configuration, N, phase, step type, task seen in training), the
    four rollups, and the report's fixed statements.

    `context` carries `seen_task_groups`, `refused` (counts by code), `summaries`, `model`,
    `resamples`, `seed` and `abstention_unavailable`.
    """
    seen_groups = frozenset(context["seen_task_groups"])
    seen = [task_group(row["task_id"]) in seen_groups for row in rows]
    settings = {
        "resamples": context["resamples"],
        "seed": context["seed"],
        "abstention_unavailable": context["abstention_unavailable"],
    }
    cells: list[dict] = []
    rollups: dict[str, list[dict]] = {
        "by_scorer_truth": [],
        "by_configuration": [],
        "by_task_seen": [],
        "by_configuration_task_seen": [],
    }
    slices = {
        "cells": (
            ("configuration", lambda r, s: r["configuration"]),
            ("n", lambda r, s: r["n"]),
            ("phase", lambda r, s: r["phase"]),
            ("step_type", lambda r, s: r["step_type"]),
            ("task_seen_in_training", lambda r, s: s),
        ),
        "by_scorer_truth": (),
        "by_configuration": (("configuration", lambda r, s: r["configuration"]),),
        "by_task_seen": (("task_seen_in_training", lambda r, s: s),),
        "by_configuration_task_seen": (
            ("configuration", lambda r, s: r["configuration"]),
            ("task_seen_in_training", lambda r, s: s),
        ),
    }
    truth_names = sorted({name for row in rows for name in row["truths"]})
    for scorer, scored in scores.items():
        for truth in truth_names:
            for target, dimensions in slices.items():
                groups: dict[tuple, list[int]] = {}
                for index, (row, was_seen) in enumerate(zip(rows, seen, strict=True)):
                    key = tuple(value(row, was_seen) for _, value in dimensions)
                    groups.setdefault(key, []).append(index)
                for key in sorted(groups, key=repr):
                    members = groups[key]
                    result = slice_measures(
                        [rows[i] for i in members],
                        [scored[i] for i in members],
                        truth,
                        resamples=settings["resamples"],
                        seed=settings["seed"],
                        abstention_unavailable=settings["abstention_unavailable"],
                    )
                    if result is None:
                        continue
                    entry = (
                        {"scorer": scorer, "truth": truth}
                        | {name: value for (name, _), value in zip(dimensions, key, strict=True)}
                        | result
                    )
                    (cells if target == "cells" else rollups[target]).append(entry)
    return {
        "held_out": True,
        "gate": GATE,
        "seen_task_groups": sorted(seen_groups),
        "cells": cells,
        "rollups": rollups,
        "refused": dict(sorted(context["refused"].items())),
        "unscored": {
            "steps": unscored_steps(rows),
            "runs": unscored_runs(context["summaries"]),
        },
        "limitations": list(LIMITATIONS),
        "protocol_sha256": {
            summary["configuration"]: summary["protocol_sha256"] for summary in context["summaries"]
        },
        "model": context["model"],
    }
