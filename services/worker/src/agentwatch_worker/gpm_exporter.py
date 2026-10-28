"""Write NVML GPM metrics as a Prometheus text file, one window per interval (ADR 0008).

Run on Windows when the runner is in WSL2, which refuses GPM samples:

    uv run --package agentwatch-worker python -m agentwatch_worker.gpm_exporter \\
        --output .data/gpm/gpm.prom

Each window replaces the file atomically. It carries a `sequence` that increases with every
window, so a reader can tell a new window from one it has already read, the window's length on
this process's monotonic clock, each metric's NVML status, and this process's own CPU time. A
metric NVML refused is omitted, never written as zero. The file names no host paths.
"""

import argparse
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from agentwatch_worker.gpm import GpmSampler, MetricResult
from agentwatch_worker.nvml import NvmlDevice, NvmlError, NvmlUnavailable

PREFIX = "agentwatch_gpm"
DEFAULT_INTERVAL_SECONDS = 0.25
MIN_INTERVAL_SECONDS = 0.1
# A reader on Windows can hold the file open briefly; replacing it then fails.
REPLACE_ATTEMPTS = 3
REPLACE_RETRY_SECONDS = 0.01


def render(
    gpu_index: int,
    sequence: int,
    window_seconds: float,
    results: dict[str, MetricResult],
    units: dict[str, str],
    cpu_seconds: float,
    write_failures: int,
) -> str:
    label = f'{{gpu="{gpu_index}"}}'
    lines = [
        f"# TYPE {PREFIX}_sequence counter",
        f"{PREFIX}_sequence{label} {sequence}",
        f"{PREFIX}_window_seconds{label} {window_seconds:.6f}",
        f"{PREFIX}_exporter_cpu_seconds_total{label} {cpu_seconds:.6f}",
        f"{PREFIX}_write_failures_total{label} {write_failures}",
    ]
    for name, result in results.items():
        lines.append(f'{PREFIX}_metric_status{{gpu="{gpu_index}",metric="{name}"}} {result.code}')
        if result.value is not None and math.isfinite(result.value):
            lines.append(f"{PREFIX}_{name}_{units[name]}{label} {result.value:.6f}")
    return "\n".join(lines) + "\n"


def replace_atomically(path: Path, text: str, sleep: Callable[[float], None]) -> bool:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return True
        except PermissionError:
            if attempt + 1 < REPLACE_ATTEMPTS:
                sleep(REPLACE_RETRY_SECONDS)
    temporary.unlink(missing_ok=True)
    return False


def export(
    sampler: GpmSampler,
    output: Path,
    *,
    gpu_index: int,
    interval_seconds: float,
    duration_seconds: float | None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    cpu_seconds: Callable[[], float] = time.process_time,
) -> dict:
    """Write windows until the duration ends (or forever); return counts for the operator."""
    units = {metric.name: metric.unit for metric in sampler.metrics}
    started = monotonic()
    next_due = started
    previous_at: float | None = None
    sequence = write_failures = 0
    while duration_seconds is None or monotonic() - started < duration_seconds:
        sampled_at = monotonic()
        results = sampler.window()
        if results is not None and previous_at is not None:
            sequence += 1
            text = render(
                gpu_index,
                sequence,
                sampled_at - previous_at,
                results,
                units,
                cpu_seconds(),
                write_failures,
            )
            if not replace_atomically(output, text, sleep):
                write_failures += 1
        previous_at = sampled_at
        # A fixed cadence, so slow windows do not accumulate drift.
        next_due += interval_seconds
        sleep(max(next_due - monotonic(), 0))
    return {"windows": sequence, "write_failures": write_failures}


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not MIN_INTERVAL_SECONDS <= seconds < float("inf"):
        raise argparse.ArgumentTypeError(f"must be at least {MIN_INTERVAL_SECONDS} s")
    return seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=positive_seconds, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument(
        "--duration", type=positive_seconds, help="Stop after this many seconds (default: never)"
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        device = NvmlDevice.open(args.gpu_index)
    except NvmlUnavailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        sampler = GpmSampler(device)
    except NvmlUnavailable as error:
        device.close()
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Writing GPM windows of {device.identity.name} to {args.output}; Ctrl-C stops")
    try:
        counts = export(
            sampler,
            args.output,
            gpu_index=args.gpu_index,
            interval_seconds=args.interval,
            duration_seconds=args.duration,
        )
    except KeyboardInterrupt:
        counts = None
    except NvmlError as error:
        # A sample that cannot be taken ends the exporter; readers then see no new window.
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        sampler.close()
        device.close()
    print(f"Stopped: {counts}" if counts else "Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
