"""Summarize a gpu-sampled-llm-run.sh evidence directory.

Reports achieved nvidia-smi sampling intervals and the GPU readings that fall inside each
recorded agent step. Step windows come from the agent runner's live journal (the producer's
UTC wall time when each step started and ended); probes recorded before 2026-10-20 used the
removed LLM demo, whose OpenTelemetry spans are still read. Utilization is NVML's averaged reading, so it cannot resolve activity
shorter than its averaging period; treat it as coarse evidence, not saturation.

Missing, malformed, or ambiguous inputs are reported rather than summarized: samples are not
placed against spans when the sampler's UTC offset is unknown or changed during the run.

Usage: python deploy/diagnostics/summarize_gpu_run.py .data/a01-gpu-probe [...]
Exits 1 if any directory could not be summarized at all.
"""

import csv
import json
import re
import statistics
import sys
from datetime import UTC, datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path

SAMPLE_TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
MAX_EXPECTED_GAP_MS = 500
UTC_OFFSET = re.compile(r"(?P<sign>[+-])(?P<hours>\d{2})(?P<minutes>\d{2})\b")
# The first 2026-10-13 probes used `nvidia_smi_timestamp_zone=EDT-0400 (...)`.
START_OFFSET_KEYS = ("nvidia_smi_timestamp_utc_offset=", "nvidia_smi_timestamp_zone=")
END_OFFSET_KEY = "nvidia_smi_timestamp_utc_offset_end="
SUMMARIZED_COLUMNS = (
    "utilization.gpu [%]",
    "memory.used [MiB]",
    "power.draw [W]",
    "clocks.current.sm [MHz]",
)


def parse_offset(line: str) -> timezone | None:
    match = UTC_OFFSET.search(line)
    if match is None:
        return None
    sign = -1 if match["sign"] == "-" else 1
    return timezone(sign * timedelta(hours=int(match["hours"]), minutes=int(match["minutes"])))


def sampler_zone(meta_path: Path) -> tuple[timezone | None, str | None]:
    """Return the offset of nvidia-smi's local timestamps, or None and the reason it is unknown."""
    if not meta_path.is_file():
        return None, f"{meta_path.name} is missing"
    lines = meta_path.read_text().splitlines()
    start = next((parse_offset(x) for x in lines if x.startswith(START_OFFSET_KEYS)), None)
    end = next((parse_offset(x) for x in lines if x.startswith(END_OFFSET_KEY)), None)
    if start is None:
        return None, "the sampler's UTC offset was not recorded"
    if end is not None and end != start:
        return None, f"the local UTC offset changed during the run ({start} to {end})"
    return start, None


def leading_number(value: str) -> float | None:
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def read_samples(path: Path, zone: timezone | None) -> tuple[list[dict], int]:
    """Return parsed rows and the number of malformed rows skipped."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        header = next(reader, None)
        if header is None or "timestamp" not in header:
            return [], 0
        rows, malformed = [], 0
        for raw in reader:
            if len(raw) != len(header):
                malformed += 1
                continue
            row = dict(zip(header, raw, strict=True))
            try:
                # Without a known offset, UTC is only a placeholder that keeps intervals exact;
                # such samples are never placed against span times.
                sampled = datetime.strptime(row["timestamp"], SAMPLE_TIME_FORMAT).replace(
                    tzinfo=zone or UTC
                )
            except ValueError:
                malformed += 1
                continue
            row["sampled"] = sampled
            row["observed_at"] = sampled if zone else None
            rows.append(row)
    return rows, malformed


Window = tuple[str, datetime, datetime]


def read_journal_windows(journal_path: Path) -> tuple[list[Window], int]:
    """Return executed non-root step windows from a runner journal, and malformed entries.

    A window spans the `running` event's observed_at to the terminal event's observed_at.
    Steps that never started (rejected or cancelled while queued) have no execution window.
    """
    started: dict[str, datetime] = {}
    windows, malformed = [], 0
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if entry.get("kind") != "record" or entry["payload"]["record_type"] != "step_event":
                continue
            event = entry["payload"]
            observed = datetime.fromisoformat(event["observed_at"])
            step_id, state = event["step_id"], event["state"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            malformed += 1
            continue
        if observed.tzinfo is None:
            malformed += 1
            continue
        if event.get("parent_step_id") is None:
            continue  # The root step spans the whole run, like the old agent.tool_loop span.
        if state == "running":
            started[step_id] = observed
        elif state in {"completed", "failed", "cancelled"} and step_id in started:
            windows.append((step_id, started.pop(step_id), observed))
    windows.sort(key=lambda window: window[1])
    return windows, malformed


def find_windows(probe: Path) -> tuple[list[Window], int, str]:
    """Read step windows from the runner journal, or spans from a pre-runner probe."""
    journals = sorted(probe.glob("agent-runs/*/live-journal.jsonl"))
    if len(journals) > 1:
        return [], 0, f"{len(journals)} runner journals; expected one per probe"
    if journals:
        windows, malformed = read_journal_windows(journals[0])
        return windows, malformed, "agent steps"
    windows, malformed = read_windows(probe / "llm-demo/agent-spans.jsonl")
    return windows, malformed, "agent spans"


def read_windows(spans_path: Path) -> tuple[list[Window], int]:
    """Return non-root agent span windows in start order and the number of malformed spans."""
    if not spans_path.is_file():
        return [], 0
    windows, malformed = [], 0
    for line in spans_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            span = json.loads(line)
            window = (
                span["name"],
                datetime.fromisoformat(span["start_time"]),
                datetime.fromisoformat(span["end_time"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            malformed += 1
            continue
        if window[1].tzinfo is None or window[2].tzinfo is None or window[2] < window[1]:
            malformed += 1
            continue
        if window[0] != "agent.tool_loop":
            windows.append(window)
    windows.sort(key=lambda window: window[1])
    return windows, malformed


def describe(rows: list[dict], column: str) -> str:
    if not rows or column not in rows[0]:
        return f"{column}: not sampled"
    values = [v for v in (leading_number(r[column]) for r in rows) if v is not None]
    missing = len(rows) - len(values)
    if not values:
        return f"{column}: unavailable in all {len(rows)} samples"
    return (
        f"{column}: min={min(values):g} max={max(values):g} "
        f"mean={statistics.fmean(values):.1f} unavailable={missing}"
    )


def summarize(probe: Path) -> bool:
    """Print a summary; return False when the directory has no usable GPU samples."""
    print(f"== {probe.name}")
    samples_path = probe / "gpu-samples.csv"
    if not samples_path.is_file():
        print("  GPU samples unavailable: gpu-samples.csv is missing")
        return False
    zone, zone_problem = sampler_zone(probe / "probe-meta.txt")
    rows, malformed_rows = read_samples(samples_path, zone)
    print(f"  {len(rows)} GPU samples, {malformed_rows} malformed rows skipped")
    if not rows:
        print("  GPU samples unavailable: no parsable rows")
        return False
    if len(rows) >= 2:
        gaps_ms = [
            (later["sampled"] - earlier["sampled"]).total_seconds() * 1000
            for earlier, later in pairwise(rows)
        ]
        print(
            f"  interval_ms median={statistics.median(gaps_ms):.0f} min={min(gaps_ms):.0f} "
            f"max={max(gaps_ms):.0f} gaps_over_{MAX_EXPECTED_GAP_MS}ms="
            f"{sum(gap > MAX_EXPECTED_GAP_MS for gap in gaps_ms)} "
            f"backwards_steps={sum(gap < 0 for gap in gaps_ms)}"
        )
    else:
        print("  interval_ms unavailable: fewer than two samples")
    for column in SUMMARIZED_COLUMNS:
        print(f"  whole run {describe(rows, column)}")
    if zone is None:
        print(f"  per-span summary unavailable: {zone_problem}")
        return True
    windows, malformed, source = find_windows(probe)
    if malformed:
        print(f"  {malformed} malformed {source} entries skipped")
    if not windows:
        print(f"  per-step summary unavailable: no valid {source}")
        return True
    if rows[0]["observed_at"] < windows[0][1]:
        windows.insert(0, ("idle before loop", rows[0]["observed_at"], windows[0][1]))
    for name, start, end in windows:
        inside = [row for row in rows if start <= row["observed_at"] <= end]
        print(f"  {name} ({(end - start).total_seconds():.2f} s, {len(inside)} samples)")
        if not inside:
            continue
        for column in SUMMARIZED_COLUMNS:
            print(f"    {describe(inside, column)}")
        print(f"    pstates={sorted({row.get('pstate', 'not sampled') for row in inside})}")
    return True


def main(arguments: list[str]) -> int:
    if not arguments:
        print(__doc__, file=sys.stderr)
        return 2
    results = [summarize(Path(argument)) for argument in arguments]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
