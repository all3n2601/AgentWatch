"""Level metrics and the capacity decision for `natural-overload-sweep-v1`.

Reads what `AgentBatch` wrote (`batch.json`, each run's `agent-evidence.json` and
`live-journal.jsonl`) and applies the declared capacity rule from `sweep_protocol`.

Missing, unreadable and unsupported readings stay distinct from measured zero: a batch whose
`batch.json` cannot be read is reported `unreadable` and fails its level's `readable`
criterion rather than counting as zero throughput; a model call without a measured vLLM span is
counted by its status and excluded from span statistics; a device metric that was not sampled
keeps its availability counts and has no min/mean/max.

Units are seconds unless a name says otherwise; throughput is passed runs per minute.
"""

import json
import math
import statistics
from pathlib import Path

from agentwatch_agent.coverage import CoverageError, coverage_report
from agentwatch_agent.experiment import step_timings, variability
from agentwatch_agent.runner import VERIFICATION_FAILED, is_agent_decision_failure
from agentwatch_agent.sweep_protocol import (
    MAX_P90_FRACTION_OF_RUN_LIMIT,
    MAX_PASS_RATE_DROP,
    MIN_PASSED_RUNS_AT_ONE_AGENT,
    MIN_THROUGHPUT_GAIN,
    RAMP_LEVELS,
    declared_batches,
)

ROOT_STEP_ID = "agent-run"
P90_QUANTILE = 0.9
SPAN_FIELDS = (
    "queue_seconds",
    "prefill_seconds",
    "decode_seconds",
    "first_token_seconds",
    "e2e_seconds",
)
SPAN_STATUSES = ("measured", "unavailable", "malformed")
# (report name, metric name, aggregate kind). Gauges report min/mean/max over the window,
# cumulative counters their increase.
DEVICE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("gpu_utilization_percent", "gpu.nvml.utilization.gpu.v1", "gauge"),
    ("gpu_memory_used_bytes", "gpu.nvml.memory.used_bytes.v1", "gauge"),
    ("gpu_sm_clock_mhz", "gpu.nvml.clock.sm_mhz.v1", "gauge"),
    ("gpu_power_watts", "gpu.nvml.power.usage_watts.v1", "gauge"),
    ("gpm_sm_occupancy_percent", "gpu.gpm.sm_occupancy.v1", "gauge"),
    ("cpu_psi_some_avg10_percent", "cpu.psi.some.avg10.v1", "gauge"),
    ("cpu_cgroup_throttled_usec", "cpu.cgroup.throttled_usec.v1", "counter"),
)
SM_CLOCK = "gpu_sm_clock_mhz"
SERVER_COUNTERS = {
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
    "request_successes": "vllm:request_success_total",
}
RESET = "reset"
# Each model call's runner time outside vLLM's request span, as the runner records it.
OUTSIDE_SERVER_FIELD = "client_outside_server_seconds"
OUTSIDE_SERVER_METHOD = (
    "the runner's client_outside_server_seconds per model call (model.joined_vllm_timing): "
    "runner client wall time minus the vLLM span's e2e, two durations on two clocks, so an "
    "estimate; calls without a measured span are counted unavailable"
)


def classify_run(entry: dict) -> str:
    """`passed`, `task_failure`, `agent_failure` or `infrastructure_failure` for a batch.json run.

    A task failure is a wrong patch; an agent failure is the agent's own decision ending the
    run (a model-call or tool-call limit, or a reply with neither a tool call nor an answer,
    matched against the runner's declared reasons). Both count against the pass rate; neither
    is a hard stop. Every other failure, and any incomplete delivery, is infrastructure.
    """
    if entry.get("delivery_complete") is not True:
        return "infrastructure_failure"
    state, reason = entry.get("state"), entry.get("reason")
    if state == "completed" and entry.get("task_passed") is True:
        return "passed"
    if state == "failed" and reason == VERIFICATION_FAILED:
        return "task_failure"
    if state == "failed" and is_agent_decision_failure(reason):
        return "agent_failure"
    return "infrastructure_failure"


def p90(values: list[float]) -> float | None:
    """The sorted element at round(0.9 * (n - 1)); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round(P90_QUANTILE * (len(ordered) - 1))]


def _p50(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _spread(values: list[float]) -> dict | None:
    """Count, mean, sample stdev and CV of unitless values; None when there are none."""
    if not values:
        return None
    spread = variability(values)
    return {
        "count": spread["count"],
        "mean": spread["mean_seconds"],
        "stdev": spread["stdev_seconds"],
        "cv": spread["cv"],
        "min": spread["min_seconds"],
        "max": spread["max_seconds"],
    }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _resolve_run_dir(batch_dir: Path, output_dir: object) -> tuple[Path | None, str | None]:
    """The run's folder and how it was found.

    `BatchRun.evidence()` records `str(runner.output_dir)`: absolute, or relative to the
    working directory the sweep ran in. `recorded` is that path as given (a relative one is
    read from the current directory); `batch_relative` is `batch_dir/runs/<name>`, for a sweep
    copied elsewhere or analysed from another directory. Never joined onto `batch_dir` whole,
    which would double a relative path.
    """
    if not isinstance(output_dir, str) or not output_dir:
        return None, None
    recorded = Path(output_dir)
    if recorded.is_dir():
        return recorded, "recorded"
    fallback = batch_dir / "runs" / recorded.name
    if fallback.is_dir():
        return fallback, "batch_relative"
    return None, None


def _read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{path.name} is not a JSON object")
    return document


def _span_values(evidence: dict) -> tuple[dict[str, int], dict[str, list[float]]]:
    counts = {status: 0 for status in SPAN_STATUSES}
    counts["not_recorded"] = 0
    values: dict[str, list[float]] = {field: [] for field in SPAN_FIELDS}
    for call in evidence.get("model_calls") or []:
        timing = call.get("timing") if isinstance(call, dict) else None
        status = timing.get("status") if isinstance(timing, dict) else None
        # `joined_vllm_timing` reports "reported" when the span was measured; the span's own
        # durations are under `server_trace`.
        trace = timing.get("server_trace") if isinstance(timing, dict) else None
        if status == "reported" and isinstance(trace, dict) and trace.get("status") == "measured":
            status = "measured"
        elif status == "reported":
            status = None
        if status not in SPAN_STATUSES:
            counts["not_recorded"] += 1
            continue
        counts[status] += 1
        if status != "measured":
            continue
        for field in SPAN_FIELDS:
            value = trace.get(field)
            if _is_number(value):
                values[field].append(float(value))
    return counts, values


def _outside_server_values(evidence: dict) -> tuple[list[float], int]:
    """Each model call's recorded time outside the server, and how many calls have none."""
    values: list[float] = []
    unavailable = 0
    for call in evidence.get("model_calls") or []:
        timing = call.get("timing") if isinstance(call, dict) else None
        value = timing.get(OUTSIDE_SERVER_FIELD) if isinstance(timing, dict) else None
        if _is_number(value) and math.isfinite(value):
            values.append(float(value))
        else:
            unavailable += 1
    return values, unavailable


def _outside_server_stats(values: list[float], unavailable: int) -> dict:
    return {
        "method": OUTSIDE_SERVER_METHOD,
        "count": len(values),
        "unavailable": unavailable,
        "mean": statistics.fmean(values) if values else None,
        "p50": _p50(values),
        "p90": p90(values),
        "max": max(values) if values else None,
        "values": values,
    }


def _reply_matches(evidence: dict) -> tuple[int, int]:
    outputs = (evidence.get("replay") or {}).get("model_outputs") or []
    flags = [o.get("matches_source") for o in outputs if isinstance(o, dict)]
    return sum(1 for flag in flags if flag is True), len(flags)


def _device_for_run(run_dir: Path) -> dict:
    """Aggregates over the `agent-run` step's window, or `unreadable` with the reason."""
    try:
        report = coverage_report(run_dir)
    except (CoverageError, OSError) as error:
        return {"status": "unreadable", "error": str(error)}
    root = next((s for s in report["steps"] if s.get("step_id") == ROOT_STEP_ID), None)
    if root is None or root.get("window") is None:
        return {"status": "unreadable", "error": f"no executed {ROOT_STEP_ID} step window"}
    windows = [root.get("own") or {}, *(root.get("context") or [])]
    metrics = {}
    for name, metric, kind in DEVICE_METRICS:
        carriers = [w for w in windows if metric in (w.get("availability") or {})]
        if not carriers:
            metrics[name] = {"status": "not_sampled"}
            continue
        # A metric sampled by more than one resource: prefer the one that measured it.
        window = max(carriers, key=lambda w: w["measured"][metric].get("measured", 0))
        aggregate = window["measured"][metric]
        entry: dict = {
            "status": "measured" if aggregate.get("measured") else "unavailable",
            "availability": window["availability"][metric],
        }
        keys = ("increase",) if kind == "counter" else ("min", "mean", "max")
        for key in keys:
            entry[key] = aggregate.get(key)
        metrics[name] = entry
    return {"status": "readable", "metrics": metrics}


def _combine_device(runs: list[dict]) -> dict:
    """Across runs: min of mins, mean of run means, max of maxes, and who had no reading."""
    readable = [run for run in runs if run["status"] == "readable"]
    unreadable = [{"error": run["error"]} for run in runs if run["status"] == "unreadable"]
    metrics = {}
    for name, _metric, kind in DEVICE_METRICS:
        entries = [run["metrics"][name] for run in readable]
        measured = [e for e in entries if e["status"] == "measured"]
        statuses: dict[str, int] = {}
        for entry in entries:
            statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        combined: dict = {"runs_measured": len(measured), "run_statuses": statuses}
        if kind == "counter":
            increases = [e["increase"] for e in measured if e["increase"] is not None]
            combined["increase_sum"] = sum(increases) if increases else None
            combined["runs_with_increase"] = len(increases)
        else:
            combined["min"] = min((e["min"] for e in measured), default=None)
            combined["mean"] = statistics.fmean([e["mean"] for e in measured]) if measured else None
            combined["max"] = max((e["max"] for e in measured), default=None)
        metrics[name] = combined
    return {"runs_readable": len(readable), "runs_unreadable": unreadable, "metrics": metrics}


def _span_stats(counts: dict[str, int], values: dict[str, list[float]]) -> dict:
    return {
        "calls_by_status": counts,
        "fields": {
            field: {
                "count": len(samples),
                "mean": statistics.fmean(samples) if samples else None,
                "p50": _p50(samples),
                "p90": p90(samples),
                "max": max(samples) if samples else None,
                "values": samples,
            }
            for field, samples in values.items()
        },
    }


def _unreadable(error: str) -> dict:
    return {"status": "unreadable", "error": error}


def batch_metrics(batch_dir: Path, run_timeout_seconds: float) -> dict:
    """One batch's counts, throughput, durations, span statistics and device statistics.

    `run_timeout_seconds` is recorded so the result carries the limit its durations were
    judged against.
    """
    try:
        batch = _read_json(batch_dir / "batch.json")
        entries = batch["runs"]
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise TypeError("batch.json runs is not a list of objects")
    except (OSError, ValueError, TypeError, KeyError) as error:
        # json.JSONDecodeError is a ValueError.
        return _unreadable(f"{batch_dir / 'batch.json'}: {error}")

    classes = [classify_run(entry) for entry in entries]
    passed = classes.count("passed")
    wall = batch.get("batch_wall_seconds")
    wall_seconds = float(wall) if _is_number(wall) and math.isfinite(wall) else None
    throughput = passed / (wall_seconds / 60) if wall_seconds and wall_seconds > 0 else None

    durations: list[float] = []
    durations_missing = 0
    span_counts = {status: 0 for status in (*SPAN_STATUSES, "not_recorded")}
    span_values: dict[str, list[float]] = {field: [] for field in SPAN_FIELDS}
    matched = replies = 0
    # Runs whose agent-evidence.json is missing or unreadable: their spans and replies are
    # absent from the counts below, never zero.
    without_evidence = 0
    outside: list[float] = []
    outside_unavailable = 0
    device_runs: list[dict] = []
    run_dirs: list[dict] = []
    for entry in entries:
        run_dir, found_by = _resolve_run_dir(batch_dir, entry.get("output_dir"))
        run_dirs.append({"index": entry.get("index"), "found_by": found_by})
        if run_dir is None:
            durations_missing += 1
            without_evidence += 1
            device_runs.append(_unreadable("run folder not found"))
            continue
        try:
            total = step_timings(run_dir).get(ROOT_STEP_ID, {}).get("total_seconds")
        except (OSError, ValueError, TypeError, KeyError) as error:
            total = None
            device_runs.append(_unreadable(f"journal unreadable: {error}"))
        else:
            device_runs.append(_device_for_run(run_dir))
        if _is_number(total):
            durations.append(float(total))
        else:
            durations_missing += 1
        try:
            evidence = _read_json(run_dir / "agent-evidence.json")
        except (OSError, ValueError, TypeError):
            without_evidence += 1
            continue
        counts, values = _span_values(evidence)
        for status, count in counts.items():
            span_counts[status] += count
        for field in SPAN_FIELDS:
            span_values[field].extend(values[field])
        run_outside, run_unavailable = _outside_server_values(evidence)
        outside.extend(run_outside)
        outside_unavailable += run_unavailable
        run_matched, run_replies = _reply_matches(evidence)
        matched += run_matched
        replies += run_replies

    return {
        "status": "readable",
        "run_timeout_seconds": run_timeout_seconds,
        "runs": len(entries),
        "passed": passed,
        "task_failures": classes.count("task_failure"),
        "agent_failures": classes.count("agent_failure"),
        "infrastructure_failures": classes.count("infrastructure_failure"),
        "wall_seconds": wall_seconds,
        "throughput_per_minute": throughput,
        "run_durations": durations,
        "run_durations_missing": durations_missing,
        "reply_match_rate": matched / replies if replies else None,
        "reply_matches": {"matched": matched, "compared": replies},
        "runs_without_evidence": without_evidence,
        "spans": _span_stats(span_counts, span_values),
        OUTSIDE_SERVER_FIELD: _outside_server_stats(outside, outside_unavailable),
        "device": _combine_device(device_runs),
        "run_folders": run_dirs,
    }


ANALYSIS_ERROR = "analysis_error"
MAX_ERROR_TEXT = 1000


def batch_metrics_or_error(batch_dir: Path, run_timeout_seconds: float) -> dict:
    """`batch_metrics`, or status `analysis_error` with the error when the analysis raises.

    Unreadable files are already reported `unreadable`; this keeps a bug in the analysis of one
    batch from costing the sweep its later phases. The error is recorded on the batch, which
    then fails its level's `readable` criterion, and the folder can be analysed again offline.
    """
    try:
        return batch_metrics(batch_dir, run_timeout_seconds)
    # Any analysis bug: recorded on the batch (see the docstring), never dropped.
    except Exception as error:  # noqa: BLE001
        return {
            "status": ANALYSIS_ERROR,
            "error": f"{type(error).__name__}: {error}"[:MAX_ERROR_TEXT],
        }


def server_changes(before: dict, after: dict) -> dict:
    """Counter increases between two server-state snapshots.

    None when a counter is missing on either side; "reset" when it decreased (the server
    restarted or its counters were cleared). The prefix-cache hit ratio is None when the
    query increase is 0 or either count is unknown.
    """
    before_counters = before.get("counters") or {}
    after_counters = after.get("counters") or {}
    changes: dict[str, float | str | None] = {}
    for name, metric in SERVER_COUNTERS.items():
        start, end = before_counters.get(metric), after_counters.get(metric)
        if not _is_number(start) or not _is_number(end):
            changes[name] = None
        elif end < start:
            changes[name] = RESET
        else:
            changes[name] = end - start
    queries, hits = changes["prefix_cache_queries"], changes["prefix_cache_hits"]
    usable = all(_is_number(value) for value in (queries, hits))
    ratio = hits / queries if usable and queries > 0 else None
    return changes | {"prefix_cache_hit_ratio": ratio}


def _sum_server_changes(changes: list[dict]) -> dict:
    """Sum per counter; any unknown or reset batch makes that counter's total unknown."""
    summed: dict[str, float | str | None] = {}
    for name in SERVER_COUNTERS:
        values = [change.get(name) for change in changes]
        if not values or any(not _is_number(v) for v in values):
            summed[name] = None if not values or None in values else RESET
        else:
            summed[name] = sum(values)
    queries, hits = summed["prefix_cache_queries"], summed["prefix_cache_hits"]
    usable = all(_is_number(value) for value in (queries, hits))
    return summed | {"prefix_cache_hit_ratio": hits / queries if usable and queries > 0 else None}


def level_summary(batches: list[dict], baseline_sm_clock_mhz: float | None = None) -> dict:
    """One level's batches combined. Each batch is a `batch_metrics` result, optionally with a
    `server_changes` result under `server`.

    Statistics are the mean across batches with sample stdev and CV. Run durations pool every
    run. `sm_clock_drift_mhz` is the level's mean SM clock minus `baseline_sm_clock_mhz`
    (the N = 1 mean): reported, never used to correct anything.
    """
    readable = [batch for batch in batches if batch.get("status") == "readable"]
    durations = [d for batch in readable for d in batch["run_durations"]]
    runs = sum(batch["runs"] for batch in readable)
    passed = sum(batch["passed"] for batch in readable)
    throughputs = [
        batch["throughput_per_minute"]
        for batch in readable
        if batch["throughput_per_minute"] is not None
    ]
    pass_rates = [batch["passed"] / batch["runs"] for batch in readable if batch["runs"]]
    reply_rates = [
        batch["reply_match_rate"] for batch in readable if batch["reply_match_rate"] is not None
    ]
    span_counts = {status: 0 for status in (*SPAN_STATUSES, "not_recorded")}
    pooled: dict[str, list[float]] = {field: [] for field in SPAN_FIELDS}
    for batch in readable:
        for status, count in batch["spans"]["calls_by_status"].items():
            span_counts[status] += count
        for field in SPAN_FIELDS:
            pooled[field].extend(batch["spans"]["fields"][field]["values"])
    device = _combine_batch_devices([batch["device"] for batch in readable])
    mean_clock = device["metrics"][SM_CLOCK]["mean"]
    drift = (
        mean_clock - baseline_sm_clock_mhz
        if mean_clock is not None and baseline_sm_clock_mhz is not None
        else None
    )
    return {
        "batches": len(batches),
        "readable_batches": len(readable),
        "batches_without_throughput": sum(
            1 for batch in readable if batch["throughput_per_minute"] is None
        ),
        "unreadable_batches": [
            {"index": position, "status": batch.get("status"), "error": batch.get("error")}
            for position, batch in enumerate(batches)
            if batch.get("status") != "readable"
        ],
        "analysis_errors": sum(1 for batch in batches if batch.get("status") == ANALYSIS_ERROR),
        "runs": runs,
        "runs_without_evidence": sum(batch["runs_without_evidence"] for batch in readable),
        "passed": passed,
        "task_failures": sum(batch["task_failures"] for batch in readable),
        "agent_failures": sum(batch["agent_failures"] for batch in readable),
        "infrastructure_failures": sum(batch["infrastructure_failures"] for batch in readable),
        "throughput_per_minute": _spread(throughputs),
        "pass_rate": {"pooled": passed / runs if runs else None, "per_batch": _spread(pass_rates)},
        "reply_match_rate": _spread(reply_rates),
        "run_duration_seconds": {
            "count": len(durations),
            "missing": sum(batch["run_durations_missing"] for batch in readable),
            "p50": _p50(durations),
            "p90": p90(durations),
            "max": max(durations) if durations else None,
        },
        "spans": _span_stats(span_counts, pooled),
        OUTSIDE_SERVER_FIELD: _outside_server_stats(
            [value for batch in readable for value in batch[OUTSIDE_SERVER_FIELD]["values"]],
            sum(batch[OUTSIDE_SERVER_FIELD]["unavailable"] for batch in readable),
        ),
        "device": device,
        "server_changes": _sum_server_changes(
            [batch["server"] for batch in readable if "server" in batch]
        ),
        "sm_clock_drift_mhz": drift,
    }


def _combine_batch_devices(devices: list[dict]) -> dict:
    """Level device statistics from per-batch ones (mean of batch means, min of mins)."""
    metrics = {}
    for name, _metric, kind in DEVICE_METRICS:
        parts = [device["metrics"][name] for device in devices]
        measured = [p for p in parts if p["runs_measured"]]
        statuses: dict[str, int] = {}
        for part in parts:
            for status, count in part["run_statuses"].items():
                statuses[status] = statuses.get(status, 0) + count
        combined: dict = {
            "runs_measured": sum(p["runs_measured"] for p in parts),
            "run_statuses": statuses,
        }
        if kind == "counter":
            sums = [p["increase_sum"] for p in measured if p["increase_sum"] is not None]
            combined["increase_sum"] = sum(sums) if sums else None
            combined["runs_with_increase"] = sum(p["runs_with_increase"] for p in parts)
        else:
            combined["min"] = min((p["min"] for p in measured), default=None)
            combined["mean"] = statistics.fmean([p["mean"] for p in measured]) if measured else None
            combined["max"] = max((p["max"] for p in measured), default=None)
        metrics[name] = combined
    return {
        "runs_readable": sum(device["runs_readable"] for device in devices),
        "runs_unreadable": [e for device in devices for e in device["runs_unreadable"]],
        "metrics": metrics,
    }


def _criterion(value: object, limit: object, passed: bool, **extra: object) -> dict:
    return {"value": value, "limit": limit, "passed": passed, **extra}


def _level_criteria(
    level: dict, previous: dict | None, baseline: dict | None, hard_stops: list[str], limit: float
) -> dict[str, dict]:
    unreadable = len(level["unreadable_batches"])
    criteria = {
        "readable": _criterion(unreadable, 0, unreadable == 0),
        "no_hard_stop": _criterion(list(hard_stops), [], not hard_stops),
    }

    # Incomplete evidence must not read as a slow or failing level, nor as a passing one. A
    # summary without the counts is not evidence that nothing is missing.
    without_throughput = level.get("batches_without_throughput")
    missing_durations = (level.get("run_duration_seconds") or {}).get("missing")
    counted = all(
        isinstance(count, int) and not isinstance(count, bool)
        for count in (without_throughput, missing_durations)
    )
    complete = counted and without_throughput == 0 and missing_durations == 0
    if not counted:
        reason = {"reason": "evidence counts missing"}
    elif not complete:
        reason = {"reason": "a batch has no usable throughput or a run has no agent-run duration"}
    else:
        reason = {}
    criteria["evidence_complete"] = _criterion(
        {
            "batches_without_throughput": without_throughput,
            "runs_missing_duration": missing_durations,
        },
        {"batches_without_throughput": 0, "runs_missing_duration": 0},
        complete,
        **reason,
    )
    if previous is None:
        criteria["passed_runs"] = _criterion(
            level.get("passed", 0),
            MIN_PASSED_RUNS_AT_ONE_AGENT,
            level.get("passed", 0) >= MIN_PASSED_RUNS_AT_ONE_AGENT,
        )

    mean = (level["throughput_per_minute"] or {}).get("mean")
    if previous is None:
        # N = 1 has no previous level and passes this criterion.
        criteria["throughput_gain"] = _criterion(None, MIN_THROUGHPUT_GAIN, True)
    else:
        previous_mean = (previous["throughput_per_minute"] or {}).get("mean")
        if mean is None:
            criteria["throughput_gain"] = _criterion(
                None, MIN_THROUGHPUT_GAIN, False, reason="no throughput measured"
            )
        elif previous_mean is None:
            criteria["throughput_gain"] = _criterion(
                None, MIN_THROUGHPUT_GAIN, False, reason="previous level has no throughput"
            )
        elif previous_mean == 0:
            criteria["throughput_gain"] = _criterion(
                None,
                MIN_THROUGHPUT_GAIN,
                False,
                reason="previous level had zero throughput",
            )
        else:
            # Rounded like the pass-rate drop so exactly +10% passes.
            gain = round(mean / previous_mean - 1, 9)
            criteria["throughput_gain"] = _criterion(
                gain, MIN_THROUGHPUT_GAIN, gain >= MIN_THROUGHPUT_GAIN
            )

    rate = level["pass_rate"]["pooled"]
    baseline_rate = baseline["pass_rate"]["pooled"] if baseline is not None else None
    if rate is None or baseline_rate is None:
        criteria["pass_rate_drop"] = _criterion(
            None, MAX_PASS_RATE_DROP, False, reason="pass rate not measured"
        )
    else:
        # Rounded so 0.80 against 1.00 is exactly the allowed 20 points, not 20.0000000001.
        drop = round(baseline_rate - rate, 9)
        criteria["pass_rate_drop"] = _criterion(
            drop, MAX_PASS_RATE_DROP, drop <= MAX_PASS_RATE_DROP
        )

    p90_limit = MAX_P90_FRACTION_OF_RUN_LIMIT * limit
    p90_value = level["run_duration_seconds"]["p90"]
    if p90_value is None:
        criteria["p90_duration_seconds"] = _criterion(
            None, p90_limit, False, reason="no run durations measured"
        )
    else:
        criteria["p90_duration_seconds"] = _criterion(p90_value, p90_limit, p90_value <= p90_limit)
    return criteria


# Criteria about whether the evidence is there, not about what it measured. A level failing only
# these has an unknown outcome, so it can never be the measured knee.
EVIDENCE_CRITERIA = frozenset({"readable", "evidence_complete", "summary_present"})


def capacity_decision(
    levels: dict[int, dict],
    hard_stops: dict[int, list[str]],
    run_timeout_seconds: float,
    excluded_batches: dict[int, int] | None = None,
) -> dict:
    """Capacity is the level before the first level that fails any criterion.

    `levels` maps level to its `level_summary` (its `batches` are the usable ones);
    `excluded_batches` counts, per level, the batches left out because the sweep stopped for a
    reason that is not a hard stop. Levels are checked in `RAMP_LEVELS` order. The result is
    `incomplete`, never a claimed knee, when no level has failed before evaluation ends at:

    - a level with no hard stop and fewer usable batches than `declared_batches` (a stopped
      level's outcome is unknown, so its criteria are recorded but not judged);
    - the end of the recorded ramp, short of the ceiling;
    - or when the first failing level fails only `EVIDENCE_CRITERIA`.

    An incomplete result's capacity is the last fully passing level (None when there is none).
    """
    # Levels reloaded from JSON have string keys.
    levels = {int(key): value for key, value in levels.items()}
    hard_stops = {int(key): value for key, value in hard_stops.items()}
    excluded = {int(key): value for key, value in (excluded_batches or {}).items()}
    criteria: dict[int, dict] = {}
    batch_counts: dict[int, dict] = {}
    previous = baseline = None
    previous_level: int | None = None  # the last level before the first failure
    first_failing: int | None = None
    incomplete_reason: str | None = None
    for level in RAMP_LEVELS:
        summary = levels.get(level)
        stops = hard_stops.get(level, [])
        if summary is None and not stops and not excluded.get(level):
            break
        usable = summary.get("batches") if summary is not None else 0
        declared = declared_batches(level)
        batch_counts[level] = {
            "usable": usable,
            "declared": declared,
            "excluded": excluded.get(level, 0),
        }
        if summary is None and stops:
            # A hard stop ended the ramp before this level produced a summary.
            criteria[level] = {
                "no_hard_stop": _criterion(list(stops), [], False),
                "summary_present": _criterion(False, True, False),
            }
        elif summary is not None:
            if level == RAMP_LEVELS[0]:
                baseline = summary
            criteria[level] = _level_criteria(
                summary, previous, baseline, stops, run_timeout_seconds
            )
        short = not isinstance(usable, int) or isinstance(usable, bool) or usable < declared
        if not stops and short:
            if first_failing is None:
                incomplete_reason = (
                    f"level {level}: {usable} of {declared} declared batches usable "
                    f"({excluded.get(level, 0)} excluded) and no hard stop"
                )
            break
        # Levels past the first failure are still evaluated so the whole curve is recorded.
        if first_failing is None:
            if all(criterion["passed"] for criterion in criteria[level].values()):
                previous_level = level
            else:
                first_failing = level
        previous = summary or previous
    failed = (
        sorted(name for name, c in criteria[first_failing].items() if not c["passed"])
        if first_failing is not None
        else []
    )
    if first_failing is None and incomplete_reason is not None:
        result, capacity = "incomplete", previous_level
    elif first_failing is None and len(criteria) < len(RAMP_LEVELS):
        # The ramp ended with no failing level and short of the ceiling: no knee was found.
        result, capacity = "incomplete", previous_level
        incomplete_reason = "the recorded ramp ended before a failing level or the ceiling"
    elif first_failing is None:
        result, capacity = "at_least_ceiling", RAMP_LEVELS[-1]
    elif set(failed) <= EVIDENCE_CRITERIA:
        result, capacity = "incomplete", previous_level
        incomplete_reason = (
            f"level {first_failing} failed only evidence criteria ({', '.join(failed)}), so "
            "what it measured is unknown"
        )
    elif previous_level is None:
        result, capacity = "below_1", None
    else:
        result, capacity = "measured", previous_level
    return {
        "capacity": capacity,
        "result": result,
        "first_failing_level": first_failing,
        "incomplete_reason": incomplete_reason,
        "batch_counts": batch_counts,
        "criteria": criteria,
    }


# Batches whose record was never finished: the sweep failed while the batch ran.
BATCH_NOT_FINISHED = "batch_not_finished"


def split_ramp_entries(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """`sweep.json` ramp batch entries usable for level metrics, and the excluded ones.

    The driver marks a batch `metrics_exclusion` when the sweep stopped for a cause that is not
    a hard stop (cancellation, runs not started, resend needed, a runner error): its runs were
    cut short by the stop, so they say nothing about the level. A batch without its `server`
    reading was never finished. Each exclusion keeps its level, sequence and causes.
    """
    usable, excluded = [], []
    for entry in entries:
        if entry.get("phase") != "ramp":
            continue
        exclusion = entry.get("metrics_exclusion")
        if exclusion:
            causes = exclusion["causes"]
        elif "server" not in entry:
            causes = [{"state": BATCH_NOT_FINISHED, "reason": "the batch's record is unfinished"}]
        else:
            usable.append(entry)
            continue
        excluded.append({"sequence": entry["sequence"], "level": entry["level"], "causes": causes})
    return usable, excluded


def ramp_analysis(
    entries: list[dict],
    hard_stop_events: list[dict],
    run_timeout_seconds: float,
    metrics_of,
) -> dict:
    """Level summaries, the capacity decision and the excluded batches of a recorded ramp.

    `entries` and `hard_stop_events` are `sweep.json`'s `batches` and `hard_stops`;
    `metrics_of(entry)` returns that batch's `batch_metrics` (recorded, or read again from
    disk). Level keys are ints.
    """
    usable, excluded = split_ramp_entries(entries)
    excluded_by_level: dict[int, int] = {}
    for item in excluded:
        excluded_by_level[int(item["level"])] = excluded_by_level.get(int(item["level"]), 0) + 1
    by_level: dict[int, list[dict]] = {}
    for entry in usable:
        changes = entry["server"].get("changes")
        server = {"server": changes} if changes is not None else {}
        by_level.setdefault(int(entry["level"]), []).append(metrics_of(entry) | server)
    summaries: dict[int, dict] = {}
    baseline_clock = None
    for level in sorted(by_level):
        summary = level_summary(by_level[level], baseline_clock if level > 1 else None)
        summaries[level] = summary | {"excluded_batches": excluded_by_level.get(level, 0)}
        if level == RAMP_LEVELS[0]:
            baseline_clock = summary["device"]["metrics"][SM_CLOCK]["mean"]
    hard_stops: dict[int, list[str]] = {}
    for event in hard_stop_events:
        if event.get("phase") == "ramp":
            hard_stops.setdefault(int(event["level"]), []).append(event["code"])
    decision = capacity_decision(summaries, hard_stops, run_timeout_seconds, excluded_by_level)
    return {
        "level_summaries": summaries,
        "capacity_decision": decision,
        "excluded_batches": excluded,
    }
