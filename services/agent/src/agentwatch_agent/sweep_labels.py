"""Label validation for `natural-overload-sweep-v1`: do duration labels agree with measured queue?

A step is labelled contended when its duration exceeds its N = 1 baseline mean by more than
`noise_threshold`. The reference is the step's own measured vLLM queue time against that same
threshold. Durations are runner ticks (`experiment.step_timings`); vLLM span durations are
durations in their own right and are never aligned with runner ticks or engine timestamps.

Steps without a measured span are excluded and counted by status, never treated as queue 0.
Tool and retrieval steps have no measured reference, so their label rate is reported as
`unvalidated`. A baseline needs `BASELINE_BATCHES_PER_SOURCE` replays; fewer omits the key (counted).
"""

import math
from pathlib import Path

from agentwatch_agent.coverage import read_journal
from agentwatch_agent.experiment import step_timings, variability
from agentwatch_agent.runner import JOURNAL_FILE
from agentwatch_agent.sweep_analysis import (
    MAX_ERROR_TEXT,
    _is_number,
    _read_json,
    _resolve_run_dir,
    _span_values,
    split_ramp_entries,
)
from agentwatch_agent.sweep_protocol import (
    BASELINE_BATCHES_PER_SOURCE,
    LABEL_THRESHOLD_FLOOR_SECONDS,
    LABEL_THRESHOLD_STDEV_MULTIPLIER,
    VALIDATED_MIN_PRECISION,
    VALIDATED_MIN_RECALL,
)

MIN_SPEARMAN_PAIRS = 3
BASELINE_LEVEL = 1
MODEL_STEP_TYPE = "model_call"
TOOL_STEP_TYPES = ("tool", "retrieval")
REQUESTS_RUNNING_METRIC = "inference.vllm.requests_running.v1"

BaselineKey = tuple[str, str]


class Baselines(dict):
    """`(plan_sha256, step_id)` -> baseline, with the evidence that was left out.

    `omitted` counts keys with too few values; `runs_without_plan` counts runs with no
    `replay.plan_sha256`; `excluded_not_completed` counts model steps by final state;
    `excluded_malformed` counts model steps with a non-finite or negative duration.
    """

    omitted: int = 0
    runs_without_plan: int = 0
    excluded_malformed: int = 0

    def __init__(self) -> None:
        super().__init__()
        self.excluded_not_completed: dict[str, int] = {}


def _valid_seconds(value: object) -> bool:
    return _is_number(value) and math.isfinite(value) and value >= 0


def _not_completed_state(timing: dict) -> str | None:
    """The final state of a step that has no usable completed duration, else None."""
    if timing["state"] != "completed":
        return str(timing["state"])
    return "completed_without_duration" if timing["total_seconds"] is None else None


def noise_threshold(baseline: dict) -> float:
    """`max(floor, multiplier * stdev)`; a baseline with no stdev has only the floor."""
    stdev = baseline.get("stdev_seconds")
    spread = LABEL_THRESHOLD_STDEV_MULTIPLIER * stdev if _is_number(stdev) else 0.0
    return max(LABEL_THRESHOLD_FLOOR_SECONDS, spread)


def _plan_sha256(evidence: dict) -> str | None:
    value = (evidence.get("replay") or {}).get("plan_sha256")
    return value if isinstance(value, str) and value else None


def _step_spans(evidence: dict) -> dict[str, dict]:
    """Per step id: `status` (measured, or why not) and summed measured seconds.

    A step with several calls is measured only when every call is measured.
    """
    grouped: dict[str, list[dict]] = {}
    for call in evidence.get("model_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("step_id"), str):
            grouped.setdefault(call["step_id"], []).append(call)
    steps: dict[str, dict] = {}
    for step_id, calls in grouped.items():
        queue = service = 0.0
        problem = None
        for call in calls:
            counts, values = _span_values({"model_calls": [call]})
            status = next((name for name, count in counts.items() if count), "not_recorded")
            if status != "measured":
                problem = status
                break
            if not all(values[f] for f in ("queue_seconds", "prefill_seconds", "decode_seconds")):
                problem = "incomplete"
                break
            measured = (
                values[f][0] for f in ("queue_seconds", "prefill_seconds", "decode_seconds")
            )
            if not all(_valid_seconds(v) for v in measured):
                problem = "malformed"
                break
            queue += values["queue_seconds"][0]
            service += values["prefill_seconds"][0] + values["decode_seconds"][0]
        steps[step_id] = (
            {"status": "measured", "queue_seconds": queue, "service_seconds": service}
            if problem is None
            else {"status": problem}
        )
    return steps


def _read_run(run_dir: Path) -> tuple[str | None, dict[str, dict], dict[str, dict]]:
    evidence = _read_json(run_dir / "agent-evidence.json")
    return _plan_sha256(evidence), step_timings(run_dir), _step_spans(evidence)


def baselines(n1_runs: list[Path]) -> Baselines:
    """Per `(plan_sha256, step_id)`: `variability` of `total_seconds` over completed steps.

    Adds `service_mean_seconds`, the mean measured prefill + decode of the same step (None when
    no replay measured one). Keys with fewer than `BASELINE_BATCHES_PER_SOURCE` durations are omitted
    and counted in `.omitted`. A run whose files cannot be read raises.
    """
    result = Baselines()
    durations: dict[BaselineKey, list[float]] = {}
    services: dict[BaselineKey, list[float]] = {}
    step_types: dict[BaselineKey, str] = {}
    for run_dir in n1_runs:
        plan, timings, spans = _read_run(run_dir)
        if plan is None:
            result.runs_without_plan += 1
            continue
        for step_id, timing in timings.items():
            total = timing["total_seconds"]
            state = _not_completed_state(timing)
            if state is not None:
                if timing["step_type"] == MODEL_STEP_TYPE:
                    counts = result.excluded_not_completed
                    counts[state] = counts.get(state, 0) + 1
                continue
            if not _valid_seconds(total):
                if timing["step_type"] == MODEL_STEP_TYPE:
                    result.excluded_malformed += 1
                continue
            durations.setdefault((plan, step_id), []).append(float(total))
            step_types[(plan, step_id)] = timing["step_type"]
            span = spans.get(step_id)
            if span and span["status"] == "measured":
                services.setdefault((plan, step_id), []).append(span["service_seconds"])
    for key, values in durations.items():
        if len(values) < BASELINE_BATCHES_PER_SOURCE:
            result.omitted += 1
            continue
        service = services.get(key)
        result[key] = variability(values) | {
            "step_type": step_types[key],
            "service_mean_seconds": sum(service) / len(service) if service else None,
        }
    return result


def _ranks(values: list[float]) -> list[float]:
    """1-based ranks with ties given the average of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation; None below `MIN_SPEARMAN_PAIRS` pairs or when a side is constant."""
    if len(xs) != len(ys) or len(xs) < MIN_SPEARMAN_PAIRS:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    spread = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    return covariance / spread if spread > 0 else None


def _empty_tally() -> dict:
    return {
        "confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "excess": [],
        "queue": [],
        "excluded": {},
        "not_completed": {},
        "runs_without_plan": 0,
        "no_baseline": 0,
        "explained": {"steps": 0, "excess_seconds": 0.0, "queue_seconds": 0.0},
        "service": {"steps": 0, "increase_seconds": 0.0, "without_baseline": 0},
        "tool": {"steps": 0, "contended": 0},
    }


def _count(tally: dict, group: str, key: str) -> None:
    tally[group][key] = tally[group].get(key, 0) + 1


def _tally_run(tally: dict, run_dir: Path, reference: dict) -> None:
    plan, timings, spans = _read_run(run_dir)
    if plan is None:
        tally["runs_without_plan"] += 1
        return
    for step_id, timing in timings.items():
        total = timing["total_seconds"]
        is_model = timing["step_type"] == MODEL_STEP_TYPE
        state = _not_completed_state(timing)
        if state is not None:
            if is_model:
                _count(tally, "not_completed", state)
                _count(tally, "excluded", "not_completed")
            continue
        if not _valid_seconds(total):
            if is_model:
                _count(tally, "excluded", "malformed")
            continue
        baseline = reference.get((plan, step_id))
        if baseline is None:
            tally["no_baseline"] += 1
            continue
        threshold = noise_threshold(baseline)
        excess = total - baseline["mean_seconds"]
        contended = excess > threshold
        if timing["step_type"] in TOOL_STEP_TYPES:
            tally["tool"]["steps"] += 1
            tally["tool"]["contended"] += contended
            continue
        if timing["step_type"] != MODEL_STEP_TYPE:
            continue
        span = spans.get(step_id, {"status": "not_recorded"})
        if span["status"] != "measured":
            _count(tally, "excluded", span["status"])
            continue
        queue = span["queue_seconds"]
        actual = queue > threshold
        outcome = {(True, True): "tp", (True, False): "fp", (False, True): "fn"}
        tally["confusion"][outcome.get((contended, actual), "tn")] += 1
        tally["excess"].append(excess)
        tally["queue"].append(queue)
        if contended:
            tally["explained"]["steps"] += 1
            tally["explained"]["excess_seconds"] += excess
            tally["explained"]["queue_seconds"] += queue
            service_mean = baseline.get("service_mean_seconds")
            if service_mean is None:
                tally["service"]["without_baseline"] += 1
            else:
                tally["service"]["steps"] += 1
                tally["service"]["increase_seconds"] += span["service_seconds"] - service_mean


def _verdict(precision: float | None, recall: float | None) -> str:
    if precision is None or recall is None:
        return "inconclusive"
    ok = precision >= VALIDATED_MIN_PRECISION and recall >= VALIDATED_MIN_RECALL
    return "validated" if ok else "not_validated"


def _report(tally: dict) -> dict:
    confusion = tally["confusion"]
    predicted = confusion["tp"] + confusion["fp"]
    actual = confusion["tp"] + confusion["fn"]
    precision = confusion["tp"] / predicted if predicted else None
    recall = confusion["tp"] / actual if actual else None
    tool = tally["tool"]
    return {
        "confusion": dict(confusion),
        "precision": precision,
        "recall": recall,
        "spearman": spearman(tally["excess"], tally["queue"]),
        "excess_explained": {
            "contended_steps": tally["explained"]["steps"],
            "excess_seconds": tally["explained"]["excess_seconds"],
            "queue_seconds": tally["explained"]["queue_seconds"],
            "prefill_decode_increase_seconds": tally["service"]["increase_seconds"],
            "prefill_decode_steps": tally["service"]["steps"],
            "prefill_decode_without_baseline": tally["service"]["without_baseline"],
        },
        "excluded": dict(tally["excluded"]),
        "excluded_not_completed": dict(tally["not_completed"]),
        "runs_without_plan": tally["runs_without_plan"],
        "steps_without_baseline": tally["no_baseline"],
        "tool_steps": {
            "validation": "unvalidated",
            "steps": tool["steps"],
            "contended": tool["contended"],
            "label_rate": tool["contended"] / tool["steps"] if tool["steps"] else None,
        },
        "verdict": _verdict(precision, recall),
    }


def label_validation(runs: list[Path] | dict[int, list[Path]], reference: Baselines) -> dict:
    """Confusion, precision, recall and verdict, overall and (for a level mapping) per level.

    Level 1 is the baseline level: a mapping's level 1 is skipped (listed in `levels_skipped`)
    and a flat list must hold N >= 2 runs only. `reference` may be a plain dict, in which case
    the omitted-baseline counts are reported as None (unknown), never 0.
    """
    by_level = runs if isinstance(runs, dict) else {}
    skipped = [level for level in by_level if level == BASELINE_LEVEL]
    overall = _empty_tally()
    levels: dict[int, dict] = {}
    groups = (
        [(level, paths) for level, paths in by_level.items() if level != BASELINE_LEVEL]
        if by_level
        else [(None, runs)]
    )
    for level, paths in groups:
        tally = _empty_tally()
        for run_dir in paths:
            _tally_run(tally, run_dir, reference)
        if level is not None:
            levels[level] = _report(tally)
        _merge(overall, tally)
    by_type: dict[str, int] = {}
    for baseline in reference.values():
        step_type = str(baseline.get("step_type"))
        by_type[step_type] = by_type.get(step_type, 0) + 1
    return {
        # The label rule is validated on model calls only; other baselines are counted apart.
        "label_rule_baselines": by_type.get(MODEL_STEP_TYPE, 0),
        "baselines_by_step_type": by_type,
        "baselines_omitted": getattr(reference, "omitted", None),
        "baseline_runs_without_plan": getattr(reference, "runs_without_plan", None),
        "baseline_steps_not_completed": getattr(reference, "excluded_not_completed", None),
        "baseline_steps_malformed": getattr(reference, "excluded_malformed", None),
        "levels_skipped": skipped,
        "overall": _report(overall),
        "levels": levels,
    }


def _merge(total: dict, part: dict) -> None:
    for key in ("tp", "fp", "fn", "tn"):
        total["confusion"][key] += part["confusion"][key]
    total["excess"] += part["excess"]
    total["queue"] += part["queue"]
    for group in ("excluded", "not_completed"):
        for status, count in part[group].items():
            total[group][status] = total[group].get(status, 0) + count
    total["runs_without_plan"] += part["runs_without_plan"]
    total["no_baseline"] += part["no_baseline"]
    for group in ("explained", "service", "tool"):
        for key, value in part[group].items():
            total[group][key] += value


def max_running_check(runs: list[Path], max_num_seqs: int) -> dict:
    """The highest sampled running-request count; never 0 when nothing was sampled."""
    readings: list[float] = []
    for run_dir in runs:
        _registration, records, _unusable = read_journal(run_dir / JOURNAL_FILE)
        for record in records:
            if (
                record.get("record_type") == "resource_sample"
                and record.get("metric") == REQUESTS_RUNNING_METRIC
                and record.get("availability") == "measured"
                and _is_number(record.get("value"))
            ):
                readings.append(float(record["value"]))
    if not readings:
        return {"status": "not_sampled", "max_observed": None, "max_num_seqs": max_num_seqs}
    highest = max(readings)
    return {
        "status": "contradicted" if highest > max_num_seqs else "consistent",
        "max_observed": highest,
        "max_num_seqs": max_num_seqs,
        "samples": len(readings),
    }


def ramp_run_dirs(output_dir: Path, entries: list[dict]) -> tuple[dict[int, list[Path]], int]:
    """Run folders of the usable ramp batches by level, and how many runs had no evidence.

    Batches excluded from level metrics (`split_ramp_entries`) are excluded here too.
    """
    usable, _excluded = split_ramp_entries(entries)
    runs: dict[int, list[Path]] = {}
    without_evidence = 0
    for entry in usable:
        batch_dir = output_dir / entry["folder"]
        for run in entry["runs"]:
            run_dir, _found_by = _resolve_run_dir(batch_dir, run.get("output_dir"))
            files = ("agent-evidence.json", JOURNAL_FILE)
            if run_dir is None or not all((run_dir / name).is_file() for name in files):
                without_evidence += 1
                continue
            runs.setdefault(int(entry["level"]), []).append(run_dir)
    return runs, without_evidence


def _unreadable(error: BaseException) -> dict:
    return {"status": "unreadable", "error": f"{type(error).__name__}: {error}"[:MAX_ERROR_TEXT]}


def ramp_label_results(output_dir: Path, entries: list[dict], max_num_seqs: int) -> dict:
    """Label validation of N >= 2 ramp runs against the N = 1 baselines, and the running check.

    `entries` are `sweep.json` batch entries under `output_dir`. Level keys stay ints so N = 1
    is never validated against itself. Unreadable evidence is recorded as such.
    """
    runs, without_evidence = ramp_run_dirs(output_dir, entries)
    results: dict = {}
    try:
        reference = baselines(runs.get(BASELINE_LEVEL, []))
        validation = label_validation(
            {level: paths for level, paths in runs.items() if level > BASELINE_LEVEL}, reference
        )
        results["label_validation"] = validation | {"runs_without_evidence": without_evidence}
    # CoverageError and json.JSONDecodeError are ValueErrors.
    except (OSError, ValueError, TypeError, KeyError) as error:
        results["label_validation"] = _unreadable(error)
    try:
        results["max_running_check"] = max_running_check(
            [path for paths in runs.values() for path in paths], max_num_seqs
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        results["max_running_check"] = _unreadable(error)
    return results


__all__ = [
    "Baselines",
    "baselines",
    "label_validation",
    "max_running_check",
    "noise_threshold",
    "ramp_label_results",
    "ramp_run_dirs",
    "spearman",
]
