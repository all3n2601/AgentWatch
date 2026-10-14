"""Summarize a gpu-sampled-llm-run.sh evidence directory.

Reports achieved nvidia-smi sampling intervals and the GPU readings that fall inside each
recorded agent span. Utilization is NVML's averaged reading, so it cannot resolve activity
shorter than its averaging period; treat it as coarse evidence, not saturation.

Usage: python deploy/diagnostics/summarize_gpu_run.py .data/a01-gpu-probe [...]
"""

import csv
import json
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path

SAMPLE_TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
MAX_EXPECTED_GAP_MS = 500
UTC_OFFSET = re.compile(r"(?P<sign>[+-])(?P<hours>\d{2})(?P<minutes>\d{2})\b")


def sampler_zone(meta_path: Path) -> timezone:
    """Return the UTC offset recorded when nvidia-smi wrote its local timestamps."""
    # The first 2026-10-13 probes used `nvidia_smi_timestamp_zone=EDT-0400 (...)`.
    keys = ("nvidia_smi_timestamp_utc_offset=", "nvidia_smi_timestamp_zone=")
    for line in meta_path.read_text().splitlines():
        if line.startswith(keys) and (match := UTC_OFFSET.search(line)):
            sign = -1 if match["sign"] == "-" else 1
            hours, minutes = int(match["hours"]), int(match["minutes"])
            return timezone(sign * timedelta(hours=hours, minutes=minutes))
    raise ValueError(f"{meta_path} does not record the nvidia-smi UTC offset")


def leading_number(value: str) -> float | None:
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def read_samples(path: Path, zone: timezone) -> list[dict]:
    with path.open() as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        header = next(reader)
        rows = [dict(zip(header, raw, strict=True)) for raw in reader if len(raw) == len(header)]
    for row in rows:
        row["observed_at"] = datetime.strptime(row["timestamp"], SAMPLE_TIME_FORMAT).replace(
            tzinfo=zone
        )
    return rows


def describe(rows: list[dict], column: str) -> str:
    values = [v for v in (leading_number(r[column]) for r in rows) if v is not None]
    missing = len(rows) - len(values)
    if not values:
        return f"{column}: unavailable in all {len(rows)} samples"
    return (
        f"{column}: min={min(values):g} max={max(values):g} "
        f"mean={statistics.fmean(values):.1f} unavailable={missing}"
    )


def summarize(probe: Path) -> None:
    rows = read_samples(probe / "gpu-samples.csv", sampler_zone(probe / "probe-meta.txt"))
    print(f"== {probe.name}: {len(rows)} GPU samples")
    if len(rows) < 2:
        print("  not enough samples to measure intervals")
        return
    gaps_ms = [
        (later["observed_at"] - earlier["observed_at"]).total_seconds() * 1000
        for earlier, later in pairwise(rows)
    ]
    print(
        f"  interval_ms median={statistics.median(gaps_ms):.0f} min={min(gaps_ms):.0f} "
        f"max={max(gaps_ms):.0f} gaps_over_{MAX_EXPECTED_GAP_MS}ms="
        f"{sum(gap > MAX_EXPECTED_GAP_MS for gap in gaps_ms)}"
    )
    spans = [
        json.loads(line)
        for line in (probe / "llm-demo/agent-spans.jsonl").read_text().splitlines()
        if line.strip()
    ]
    windows = [
        (
            span["name"],
            datetime.fromisoformat(span["start_time"]),
            datetime.fromisoformat(span["end_time"]),
        )
        for span in spans
        if span["name"] != "agent.tool_loop"
    ]
    windows.sort(key=lambda window: window[1])
    windows.insert(0, ("idle before loop", rows[0]["observed_at"], windows[0][1]))
    for name, start, end in windows:
        inside = [row for row in rows if start <= row["observed_at"] <= end]
        print(f"  {name} ({(end - start).total_seconds():.2f} s, {len(inside)} samples)")
        if not inside:
            continue
        for column in (
            "utilization.gpu [%]",
            "memory.used [MiB]",
            "power.draw [W]",
            "clocks.current.sm [MHz]",
        ):
            print(f"    {describe(inside, column)}")
        print(f"    pstates={sorted({row['pstate'] for row in inside})}")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        summarize(Path(argument))
