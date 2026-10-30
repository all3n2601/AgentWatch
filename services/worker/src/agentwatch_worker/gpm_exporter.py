"""Write NVML GPM metrics as a Prometheus text file, one window per interval (ADR 0008).

Run on Windows when the runner is in WSL2, which refuses GPM samples:

    uv run --package agentwatch-worker python -m agentwatch_worker.gpm_exporter \\
        --output .data/gpm/gpm.prom

Each window replaces the file atomically. It carries a random `instance` for this process and a
`sequence` that increases with every window, so a reader can tell a new window from one it has
already read even across exporter restarts, the window's length on this process's monotonic
clock, each metric's NVML status, and this process's own CPU time. A metric NVML refused is
omitted, never written as zero. The file names no host paths.
"""

import argparse
import math
import os
import secrets
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
# A reader holds the file for milliseconds; this many failed windows in a row (10 s at the
# default interval) means the output cannot be written, so the exporter stops with an error.
MAX_CONSECUTIVE_WRITE_FAILURES = 40


class ExportFailed(Exception):
    """The exporter cannot write its output and stops; readers then see no new window."""


def render(
    gpu_index: int,
    instance: str,
    sequence: int,
    window_seconds: float,
    results: dict[str, MetricResult],
    units: dict[str, str],
    cpu_seconds: float,
    write_failures: int,
) -> str:
    label = f'{{gpu="{gpu_index}"}}'
    lines = [
        f'{PREFIX}_exporter_info{{gpu="{gpu_index}",instance="{instance}"}} 1',
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


def replace_atomically(path: Path, text: str, sleep: Callable[[float], None]) -> OSError | None:
    """Replace `path` with `text`; return None once written, else the error that dropped it.

    A held file (PermissionError) is retried; any other file error, such as a full disk, drops
    the window at once. The temporary file never outlives a failed attempt.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                return None
            except PermissionError as error:
                failure: OSError = error
                if attempt + 1 < REPLACE_ATTEMPTS:
                    sleep(REPLACE_RETRY_SECONDS)
    except OSError as error:
        failure = error
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        pass  # The next window's write replaces it; the original failure is what is reported.
    return failure


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
    instance: str | None = None,
) -> dict:
    """Write windows until the duration ends (or forever); return counts for the operator."""
    units = {metric.name: metric.unit for metric in sampler.metrics}
    instance = instance or secrets.token_hex(8)
    started = monotonic()
    next_due = started
    previous_at: float | None = None
    sequence = write_failures = consecutive_failures = 0
    while duration_seconds is None or monotonic() - started < duration_seconds:
        sampled_at = monotonic()
        results = sampler.window()
        if results is not None and previous_at is not None:
            sequence += 1
            # A failed write cannot publish itself: readers see it in the next window written.
            text = render(
                gpu_index,
                instance,
                sequence,
                sampled_at - previous_at,
                results,
                units,
                cpu_seconds(),
                write_failures,
            )
            error = replace_atomically(output, text, sleep)
            if error is None:
                consecutive_failures = 0
            else:
                write_failures += 1
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_WRITE_FAILURES:
                    raise ExportFailed(
                        f"{consecutive_failures} windows in a row could not be written to "
                        f"{output.name} ({type(error).__name__}: {error})"
                    )
        previous_at = sampled_at
        # A fixed cadence, so slow windows do not accumulate drift. An overrun skips the slots it
        # missed instead of sampling back to back to catch up.
        next_due += interval_seconds
        now = monotonic()
        if next_due < now:
            next_due += math.ceil((now - next_due) / interval_seconds) * interval_seconds
        sleep(next_due - now)
    # `windows` counts every window computed, including the ones whose write failed.
    return {
        "windows": sequence,
        "windows_written": sequence - write_failures,
        "write_failures": write_failures,
    }


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
    except (NvmlError, ExportFailed) as error:
        # A sample that cannot be taken, or an output that cannot be written, ends the exporter;
        # readers then see no new window.
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        sampler.close()
        device.close()
    print(f"Stopped: {counts}" if counts else "Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
