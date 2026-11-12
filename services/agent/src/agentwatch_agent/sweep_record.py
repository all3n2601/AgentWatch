"""The capacity record's result and coverage tables, rendered from recorded sweep folders only.

`sweep-record DIR [DIR ...]` prints, for each sweep, the Markdown tables of
`docs/capacity-record.md` ("Results"): the verified configuration, the ramp, the capacity
decision, label validation, the live confirmation and the injected condition. With
`--coverage-jsonl FILE` it also prints the "Sampling coverage" tables (`coverage` on the first
run of the first ramp batch at each level) and writes those raw reports to FILE as JSON lines.

Every number comes from the sweep's own `sweep.json`, its injection logs and, for coverage, its
run folders, read under the sweep folder only. Nothing is computed from other sources, and a
value that is not recorded prints as "n/a", never as 0. A record or log without a field the
tables read is refused with `SweepFolderError`, never rendered with a guess.
"""

import json
from collections.abc import Callable
from pathlib import Path

from agentwatch_agent.coverage import coverage_report
from agentwatch_agent.sweep_offline import SweepFolderError, _read_record

NOT_AVAILABLE = "n/a"
FAILED_MARK = " ✗"
SECONDS_TO_MILLISECONDS = 1000
RATIO_TO_PERCENT = 100
# The record shows a hash's first 12 hex digits and a commit's first 7.
SHORT_HASH_LENGTH = 12
SHORT_COMMIT_LENGTH = 7


def _number(value: object, digits: int = 1) -> str:
    """A recorded value as the record prints it: "n/a" when absent, thousands separators."""
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    raise TypeError(f"not a number: {value!r}")


def _as_recorded(value: object) -> str:
    """A recorded value printed as is (no separators), or "n/a" when absent."""
    return NOT_AVAILABLE if value is None else str(value)


def _milliseconds(seconds: object) -> str:
    if seconds is None:
        return NOT_AVAILABLE
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        raise TypeError(f"not a duration in seconds: {seconds!r}")
    return _number(seconds * SECONDS_TO_MILLISECONDS)


def _statistic(block: dict | None, key: str = "mean") -> object:
    return (block or {}).get(key)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _configuration(record: dict) -> str:
    verified = record["configuration"]["verified"]
    rows = [
        [
            f"`{name}`",
            json.dumps(setting.get("declared")),
            json.dumps(setting.get("reported")),
            setting["status"],
            setting.get("source", ""),
        ]
        for name, setting in verified["settings"].items()
    ]
    launch = record["configuration"]["launch"]
    note = (
        f"KV cache: {_number(verified['num_gpu_blocks'])} blocks of {verified['block_size']} "
        f"tokens = {_number(verified['kv_cache_size_tokens'])} tokens; vLLM's maximum concurrency "
        f"at {record['protocol']['configuration']['max_model_len']} tokens per request: "
        f"{_number(verified['kv_cache_max_concurrency'], 2)}. Launch arguments read with "
        f"`{launch.get('executable')}` from `{launch.get('container')}`, which publishes the "
        f"measured endpoint's port {launch.get('endpoint_port')} "
        f"(`serves_endpoint: {_number(launch.get('serves_endpoint'))}`). Start attempts: "
        f"{json.dumps(record.get('start_attempts'))}."
    )
    header = ["Setting", "Declared", "Reported", "Status", "Source"]
    return _table(header, rows) + "\n\n" + note


RAMP_HEADER = [
    "N",
    "Batches / runs",
    "Passed",
    "Throughput, runs/min (mean ± sd, CV)",
    "Run p50 / p90 / max, s",
    "vLLM queue mean / p90, ms",
    "Prefill mean, ms",
    "Decode mean, ms",
    "End to end mean, ms",
    "Prefix-cache hit, %",
    "Preemptions",
    "SM clock mean, MHz",
    "SM clock drift, MHz",
    "GPU util mean, %",
    "Power mean, W",
]


def _ramp(record: dict) -> str:
    rows = []
    for level, summary in record["level_summaries"].items():
        throughput = summary["throughput_per_minute"] or {}
        durations = summary["run_duration_seconds"]
        fields = summary["spans"]["fields"]
        changes = summary["server_changes"] or {}
        device = summary["device"]["metrics"]
        hit_ratio = changes.get("prefix_cache_hit_ratio")
        rows.append(
            [
                str(level),
                f"{summary['batches']} / {summary['runs']}",
                f"{summary['passed']}",
                (
                    f"{_number(throughput.get('mean'))} ± {_number(throughput.get('stdev'))} "
                    f"({_number(throughput.get('cv'), 3)})"
                ),
                (
                    f"{_number(durations['p50'])} / {_number(durations['p90'])} / "
                    f"{_number(durations['max'])}"
                ),
                (
                    f"{_milliseconds(_statistic(fields['queue_seconds']))} / "
                    f"{_milliseconds(_statistic(fields['queue_seconds'], 'p90'))}"
                ),
                _milliseconds(_statistic(fields["prefill_seconds"])),
                _milliseconds(_statistic(fields["decode_seconds"])),
                _milliseconds(_statistic(fields["e2e_seconds"])),
                _number(None if hit_ratio is None else hit_ratio * RATIO_TO_PERCENT),
                _number(changes.get("preemptions"), 0),
                _number(_statistic(device.get("gpu_sm_clock_mhz")), 0),
                _number(summary.get("sm_clock_drift_mhz"), 0),
                _number(_statistic(device.get("gpu_utilization_percent")), 0),
                _number(_statistic(device.get("gpu_power_watts"))),
            ]
        )
    calls = [summary["spans"]["calls_by_status"] for summary in record["level_summaries"].values()]
    without_span = sum(c["unavailable"] + c["malformed"] + c["not_recorded"] for c in calls)
    measured = sum(c["measured"] for c in calls)
    note = f"Model calls with a vLLM span: {measured:,} measured, {without_span} without one."
    return _table(RAMP_HEADER, rows) + "\n\n" + note


def _marked(text: str, criterion: dict) -> str:
    return text + ("" if criterion.get("passed", True) else FAILED_MARK)


def _decision(record: dict) -> str:
    decision = record["capacity_decision"]
    reasons = decision["first_failing_reasons"]
    # A `not_sampled` or unreadable check has no observed maximum or sample count.
    check = record["max_running_check"] or {"status": NOT_AVAILABLE}
    failed = ", ".join(reasons["failed_criteria"]) if reasons else "none"
    lines = [
        f"- Result: **{decision['result']}**, capacity **{decision['capacity']}**.",
        (
            f"- First failing level: {decision['first_failing_level']} (failed criteria: "
            f"{failed}; hard stops: {reasons['hard_stops'] if reasons else []}; "
            f"unmeasured: {reasons['unmeasured_criteria'] if reasons else []})."
        ),
        (
            f"- Hard stops: {record['hard_stops'] or 'none'}. Excluded batches: "
            f"{record['excluded_batches'] or 'none'}. vLLM state read failures: "
            f"{len(record['server']['read_failures'])}."
        ),
        (
            f"- `max_running_check`: {check['status']} (most running requests observed "
            f"{_number(check.get('max_observed'), 0)}, declared `max_num_seqs` "
            f"{_number(check.get('max_num_seqs'))}, {_number(check.get('samples'))} samples)."
        ),
    ]
    rows = []
    for level, criteria in decision["criteria"].items():
        gain = criteria.get("throughput_gain", {})
        drop = criteria.get("pass_rate_drop", {})
        p90 = criteria.get("p90_duration_seconds", {})
        rows.append(
            [
                str(level),
                _marked(_number(gain.get("value"), 3), gain),
                _marked(_number(drop.get("value"), 3), drop),
                _marked(f"{_number(p90.get('value'))} (limit {_number(p90.get('limit'), 0)})", p90),
                "pass" if all(c["passed"] for c in criteria.values()) else "fail",
            ]
        )
    header = ["N", "Throughput gain (≥ 0.10)", "Pass-rate drop (≤ 0.20)", "p90 run, s", "Level"]
    return "\n".join(lines) + "\n\n" + _table(header, rows)


LABEL_HEADER = [
    "N",
    "TP / FP / FN / TN",
    "Precision",
    "Recall",
    "Spearman",
    "Excess over baseline, s",
    "of which vLLM queue, s",
    "Prefill + decode increase, s",
    "Verdict",
]


def _labels(record: dict) -> str:
    validation = record["label_validation"]
    rows = []
    entries = [*validation["levels"].items(), ("all N ≥ 2", validation["overall"])]
    for level, result in entries:
        confusion = result["confusion"]
        excess = result.get("excess_explained") or {}
        rows.append(
            [
                str(level),
                f"{confusion['tp']} / {confusion['fp']} / {confusion['fn']} / {confusion['tn']}",
                _number(result.get("precision"), 3),
                _number(result.get("recall"), 3),
                _number(result.get("spearman"), 3),
                _number(excess.get("excess_seconds")),
                _number(excess.get("queue_seconds"), 3),
                _number(excess.get("prefill_decode_increase_seconds")),
                result["verdict"],
            ]
        )
    tool = validation["overall"].get("tool_steps") or {}
    note = (
        f"Baselines: {validation['label_rule_baselines']} source steps from the N = 1 replays "
        f"({validation['baselines_omitted']} omitted). Tool steps have no measured reference: "
        f"{_number(tool.get('contended'))} of {_number(tool.get('steps'))} labelled contended "
        f"(rate {_number(tool.get('label_rate'), 3)}), `{tool.get('validation')}`."
    )
    return _table(LABEL_HEADER, rows) + "\n\n" + note


def _live(record: dict) -> str:
    rows = [
        [
            str(batch["level"]),
            str(batch["repeat"]),
            f"{batch['metrics']['passed']} / {batch['metrics']['runs']}",
            _number(batch["metrics"]["throughput_per_minute"]),
            _number(batch["metrics"]["wall_seconds"]),
        ]
        for batch in record["batches"]
        if batch["phase"] == "live"
    ]
    header = ["N", "Batch", "Passed / runs", "Throughput, runs/min", "Batch wall, s"]
    return _table(header, rows)


def _inside(sweep_dir: Path, relative: str) -> Path:
    """A path the record names relative to the sweep folder; never one outside it."""
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise SweepFolderError(f"{relative} is outside the sweep folder")
    return sweep_dir / relative


def _flood_activity(sweep_dir: Path, batch: dict) -> str:
    path = _inside(sweep_dir, batch["injection_log"])
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SweepFolderError(f"cannot read injection log {path}: {error}") from error
    # A flood whose report never arrived has no counts; those print as n/a.
    activity = log["injections"][0].get("activity") or {}
    prompt_tokens = activity.get("prompt_tokens") or {}
    return (
        f"{_as_recorded(activity.get('requests_completed'))} done / "
        f"{_as_recorded(activity.get('requests_failed'))} failed; "
        f"prompt median {_as_recorded(prompt_tokens.get('median'))}; replies not 1 token: "
        f"{_as_recorded(activity.get('completions_not_one_token'))}; "
        f"cleanup {_number(log.get('cleanup_confirmed'))}"
    )


INJECTED_HEADER = [
    "Pair",
    "Batch",
    "N",
    "Passed / runs",
    "Throughput, runs/min",
    "Run median, s",
    "vLLM queue mean, ms",
    "Prefill mean, ms",
    "Decode mean, ms",
    "Flood activity",
]


def _injected(record: dict, sweep_dir: Path) -> str:
    rows = []
    for batch in record["batches"]:
        if batch["phase"] not in ("injected", "control"):
            continue
        metrics = batch["metrics"]
        fields = (metrics.get("spans") or {}).get("fields") or {}
        durations = sorted(metrics.get("run_durations") or [])
        # The upper median, as the record has always shown it.
        median = durations[len(durations) // 2] if durations else None
        injected = batch["phase"] == "injected"
        rows.append(
            [
                str(batch["repeat"]),
                batch["phase"],
                str(batch["level"]),
                f"{metrics['passed']} / {metrics['runs']}",
                _number(metrics["throughput_per_minute"]),
                _number(median),
                _milliseconds(_statistic(fields.get("queue_seconds"))),
                _milliseconds(_statistic(fields.get("prefill_seconds"))),
                _milliseconds(_statistic(fields.get("decode_seconds"))),
                _flood_activity(sweep_dir, batch) if injected else "no flood",
            ]
        )
    return _table(INJECTED_HEADER, rows)


def _rendered(sweep_dir: Path, render: Callable[[], str]) -> str:
    """Run a renderer, refusing a record that lacks what it reads instead of guessing."""
    try:
        return render()
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise SweepFolderError(
            f"{sweep_dir / 'sweep.json'} lacks a field the record tables read: "
            f"{type(error).__name__}: {error}"
        ) from error


def _sweep_heading(record: dict, sweep_dir: Path) -> str:
    commit = record["code_revision"].get("commit") or NOT_AVAILABLE
    return (
        f"### {record['configuration']['declared']['name']} (`{sweep_dir.name}`)\n\n"
        f"Sweep `{record['sweep_id']}`, {record['started_at']} to {record['finished_at']}; "
        f"state `{record['state']}`, ramp ended `{record['ramp_end']}`, analysis scope "
        f"`{record['analysis_scope']}`, analysis errors {record['analysis_errors'] or 'none'}. "
        f"Protocol `{record['protocol']['version']}` (`protocol_sha256` "
        f"{record['protocol_sha256'][:SHORT_HASH_LENGTH]}), code {commit[:SHORT_COMMIT_LENGTH]}, "
        f"workload `{record['workload_sha256'][:SHORT_HASH_LENGTH]}`."
    )


def render_sweep_tables(sweep_dir: Path) -> str:
    """One sweep's "Results" subsection of the capacity record, as Markdown."""
    record = _read_record(sweep_dir)

    def render() -> str:
        sections = [
            _sweep_heading(record, sweep_dir),
            "#### Configuration as verified\n\n" + _configuration(record),
            "#### Ramp\n\n" + _ramp(record),
            "#### Capacity decision\n\n" + _decision(record),
            "#### Label validation\n\n" + _labels(record),
            "#### Live confirmation\n\n" + _live(record),
            "#### Injected condition\n\n" + _injected(record, sweep_dir),
        ]
        return "\n\n".join(sections) + "\n"

    return _rendered(sweep_dir, render)


COVERAGE_HEADER = [
    "N",
    "Run",
    "Source",
    "Rounds (expected)",
    "Late",
    "Missed",
    "Interval median / p90 / max, s",
]


def _interval(seconds: object) -> str:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return NOT_AVAILABLE
    return f"{seconds:.3f}"


def render_coverage_tables(sweep_dir: Path) -> tuple[str, list[dict]]:
    """One sweep's "Sampling coverage" table and the raw coverage reports it shows.

    `coverage` runs on the first run of the first ramp batch at each level, from the run folder
    under this sweep folder (never the path the runner recorded).
    """
    record = _read_record(sweep_dir)
    reports: list[dict] = []

    def render() -> str:
        rows = []
        seen_levels: set[int] = set()
        for batch in record["batches"]:
            if batch["phase"] != "ramp" or batch["level"] in seen_levels or not batch["runs"]:
                continue
            seen_levels.add(batch["level"])
            run_name = Path(batch["runs"][0]["output_dir"]).name
            run_dir = _inside(sweep_dir, batch["folder"]) / "runs" / run_name
            report = coverage_report(run_dir)
            reports.append({"sweep": sweep_dir.name, "level": batch["level"], "report": report})
            for source in report["sampling"]:
                rows.append(
                    [
                        str(batch["level"]),
                        f"`{report['run_id'][:8]}`",
                        source["service_id"],
                        f"{source['sample_rounds']} ({source['expected_rounds']})",
                        str(source["late_rounds"]),
                        str(source["missed_rounds"]),
                        (
                            f"{_interval(source['observed_interval_median_seconds'])} / "
                            f"{_interval(source['observed_interval_p90_seconds'])} / "
                            f"{_interval(source['observed_interval_max_seconds'])}"
                        ),
                    ]
                )
        name = record["configuration"]["declared"]["name"]
        return f"### `{name}`\n\n" + _table(COVERAGE_HEADER, rows) + "\n"

    return _rendered(sweep_dir, render), reports
