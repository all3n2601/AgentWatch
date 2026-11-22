"""Judge saved overhead measurements against the instrumentation overhead budget (A07).

Applies protocol `instrumentation-overhead-acceptance-v1`
(docs/instrumentation-overhead-protocol.md) to the JSON that the agent's `overhead`
(`agent-overhead-v1`) and `sampling-overhead` (`resource-sampling-overhead-v1`/`-v2`)
commands write. It reads their saved output only and never reruns a measurement.

Per attempt, the uninstrumented repetition at index 2k and the instrumented one at 2k + 1 form
a pair. The statistic is `median(instrumented) / median(uninstrumented) - 1` over pairs, with a
paired percentile bootstrap interval using the optimization contract's resamples and level
and a recorded seed. The budget and the baseline stability gate also come from that contract.
Attempts are combined only when their settings match: protocol, task, model and request,
runner version, sampling interval, sampled probe metrics, OS and machine, and the host label
given on the command line (OS and machine alone cannot tell two Linux x86_64 hosts apart).
Fields an attempt does not record are left out rather than compared as null. The same file
given twice is refused. A completed attempt needs the declared 30 pairs. Failed attempts and
attempts saved without per-repetition walls are kept as `not_evaluated`, and a group in which
nothing could be evaluated is `not_evaluated` too.

The agent's `admission-overhead` measurements (`admission-overhead-v1`) are judged by the same
rule against the contract's admission overhead budget (`max_median_admission_overhead`):
`admission_off` takes the uninstrumented side of each pair and `admission_on` the instrumented
one, so the report's "uninstrumented"/"instrumented" fields mean "without"/"with admission".

Exit codes: 0 when the report was written, whatever the verdict (it is in the report and the
printed summary); 1 when an attempt file, the host label or the contract is invalid, in which
case nothing is written.
"""

import argparse
import hashlib
import math
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.extract import parse_json
from agentwatch_data.files import json_text, write_atomic
from agentwatch_data.optimization_comparison import bootstrap, coefficient_of_variation
from agentwatch_data.optimization_manifest import (
    ManifestError,
    OptimizationContract,
    load_optimization_contract,
)

PROTOCOL = "instrumentation-overhead-acceptance-v1"
METHOD = "overhead-verdict-v1"
REPORT_FILE = "overhead-verdict.json"
DEFAULT_REPORTS = Path(".data/reports/overhead")
DEFAULT_SEED = 0
# Fewer completed attempts cannot show that a difference is stable between attempts.
MIN_COMPLETED_ATTEMPTS = 2
# The repetitions per condition both measurement protocols declare; fewer is not that protocol.
MIN_PAIRS = 30
# A short name for the measured host, never a path or host name, e.g. "benoy-rtx5070-wsl2".
HOST_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# (uninstrumented, instrumented) condition names for each measurement protocol.
CONDITIONS = {
    "agent-overhead-v1": ("uninstrumented", "instrumented"),
    "resource-sampling-overhead-v1": ("unsampled", "sampled"),
    "resource-sampling-overhead-v2": ("unsampled", "sampled"),
    "admission-overhead-v1": ("admission_off", "admission_on"),
}
ADMISSION_PROTOCOL = "admission-overhead-v1"
STATUSES = ("completed", "failed", "cancelled")
WITHIN, OVER, INCONCLUSIVE, NOT_EVALUATED = (
    "within_budget",
    "over_budget",
    "inconclusive",
    "not_evaluated",
)
# Values that must match for attempts to be combined into one verdict.
SETTING_FIELDS = (
    "task_id",
    "model",
    "request_sha256",
    "runner_version",
    "sampling_interval_seconds",
)


class AttemptError(ValueError):
    """An attempt file is unreadable or not a valid overhead measurement."""


@dataclass(frozen=True)
class Attempt:
    number: int  # 1-based position on the command line; tells same-named files apart
    source: str  # parent folder and file name only, never the host path
    source_sha256: str
    status: str
    error: str | None
    settings: dict
    # (uninstrumented, instrumented) wall seconds; None when repetitions were not saved.
    pairs: tuple[tuple[float, float], ...] | None
    unpaired_repetitions: int


def relative_increase(pairs: list[tuple[float, float]]) -> float:
    baseline = statistics.median(base for base, _ in pairs)
    return statistics.median(treated for _, treated in pairs) / baseline - 1


def _wall(measurement, index: int) -> float:
    wall = measurement.get("wall_seconds")
    if isinstance(wall, bool) or not isinstance(wall, int | float):
        raise AttemptError(f"measurement {index} wall_seconds is not a number")
    if not math.isfinite(wall) or wall <= 0:
        raise AttemptError(f"measurement {index} wall_seconds must be finite and positive")
    return float(wall)


def _pairs(measurements, names: tuple[str, str]) -> tuple[tuple[tuple[float, float], ...], int]:
    if not isinstance(measurements, list):
        raise AttemptError("measurements must be a list")
    walls = []
    for position, measurement in enumerate(measurements):
        if not isinstance(measurement, dict):
            raise AttemptError(f"measurement {position} is not an object")
        if measurement.get("index") != position:
            raise AttemptError(f"measurement {position} has index {measurement.get('index')!r}")
        if measurement.get("condition") != names[position % 2]:
            raise AttemptError(
                f"measurement {position} is {measurement.get('condition')!r}; conditions must "
                f"alternate {names[0]}, {names[1]}"
            )
        walls.append(_wall(measurement, position))
    pairs = tuple(zip(walls[0::2], walls[1::2], strict=False))
    return pairs, len(walls) - 2 * len(pairs)


def _setting_value(value, name: str):
    """A comparable setting: a string or a finite number, never an object or NaN."""
    if isinstance(value, str) or (
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    ):
        return value
    raise AttemptError(f"setting {name} must be a string or a finite number, not {value!r}")


def _probe_metrics(document: dict) -> list[str] | None:
    """The sampled metrics, which identify the probe set (a GPU probe adds gpu.* metrics)."""
    sampling = document.get("sampling")
    counts = sampling.get("availability_counts") if isinstance(sampling, dict) else None
    if counts is None:
        return None
    if not isinstance(counts, dict) or not all(isinstance(key, str) for key in counts):
        raise AttemptError("sampling.availability_counts must map metric names to counts")
    return sorted(counts)


def _settings(document: dict, version: str) -> dict:
    """The attempt's recorded settings; a field it does not record is left out, not null."""
    settings: dict = {"protocol": version}
    host = document.get("host")
    if isinstance(host, dict):
        for field in ("system", "machine"):
            if field in host:
                settings[f"host_{field}"] = _setting_value(host[field], f"host.{field}")
    for field in SETTING_FIELDS:
        if field in document:
            settings[field] = _setting_value(document[field], field)
    metrics = _probe_metrics(document)
    if metrics is not None:
        settings["probe_metrics"] = metrics
    return settings


def _parse_attempt(document) -> tuple[str, str | None, dict, tuple | None, int]:
    if not isinstance(document, dict):
        raise AttemptError("the attempt is not a JSON object")
    protocol = document.get("protocol")
    version = protocol.get("version") if isinstance(protocol, dict) else None
    if version not in CONDITIONS:
        raise AttemptError(f"unknown protocol {version!r}; expected one of {sorted(CONDITIONS)}")
    # `overhead` saves its file only when the measurement completed, without a status.
    status = document.get("status", "completed")
    if status not in STATUSES:
        raise AttemptError(f"status {status!r} is not one of {STATUSES}")
    error = document.get("error")
    if "measurements" not in document:
        return status, error, _settings(document, version), None, 0
    pairs, unpaired = _pairs(document["measurements"], CONDITIONS[version])
    return status, error, _settings(document, version), pairs, unpaired


def read_attempt(path: Path, number: int = 1) -> Attempt:
    source = "/".join(path.parts[-2:])
    try:
        data = path.read_bytes()
    except OSError as error:
        raise AttemptError(f"{source} cannot be read: {error.strerror or error}") from error
    try:
        status, error, settings, pairs, unpaired = _parse_attempt(parse_json(data.decode("utf-8")))
    # AttemptError is a ValueError, so malformed JSON and invalid fields end here alike.
    except (UnicodeDecodeError, ValueError) as error:
        raise AttemptError(f"{source}: {error}") from error
    return Attempt(
        number=number,
        source=source,
        source_sha256=hashlib.sha256(data).hexdigest(),
        status=status,
        error=error if isinstance(error, str) else None,
        settings=settings,
        pairs=pairs,
        unpaired_repetitions=unpaired,
    )


def budget_for(protocol: str, contract: OptimizationContract) -> float:
    """The contract's budget for what a measurement protocol adds to the workload."""
    if protocol == ADMISSION_PROTOCOL:
        return contract.max_median_admission_overhead
    return contract.max_median_instrumentation_overhead


def _judge(
    attempt: Attempt, contract: OptimizationContract, seed: int
) -> tuple[str, str, float | None, tuple[float, float] | None, float | None]:
    """Verdict, reason, relative increase, interval and baseline CV for one attempt."""
    if attempt.status != "completed":
        return NOT_EVALUATED, f"Attempt {attempt.status}: {attempt.error}", None, None, None
    if attempt.pairs is None:
        reason = "Per-repetition wall times were not saved, so pairs cannot be formed"
        return NOT_EVALUATED, reason, None, None, None
    pairs = list(attempt.pairs)
    increase = relative_increase(pairs) if pairs else None
    cv = coefficient_of_variation([base for base, _ in pairs])
    interval = bootstrap(
        pairs, relative_increase, contract.bootstrap_resamples, contract.interval_level, seed
    )
    budget = budget_for(attempt.settings["protocol"], contract)
    if len(pairs) < MIN_PAIRS:
        reason = f"{len(pairs)} pairs, fewer than the protocol's {MIN_PAIRS}"
        return INCONCLUSIVE, reason, increase, interval, cv
    if cv > contract.max_baseline_cv:
        reason = f"Baseline coefficient of variation {cv:.3f} exceeds {contract.max_baseline_cv}"
        return INCONCLUSIVE, reason, increase, interval, cv
    if interval[1] <= budget:
        return WITHIN, f"Interval upper bound at most {budget}", increase, interval, cv
    if interval[0] > budget:
        return OVER, f"Interval lower bound above {budget}", increase, interval, cv
    return INCONCLUSIVE, f"Interval includes the budget {budget}", increase, interval, cv


def _median(pairs: tuple | None, side: int) -> float | None:
    return statistics.median(pair[side] for pair in pairs) if pairs else None


def attempt_entry(attempt: Attempt, contract: OptimizationContract, seed: int) -> dict:
    verdict, reason, increase, interval, cv = _judge(attempt, contract, seed)
    return {
        "attempt": attempt.number,
        "source": attempt.source,
        "source_sha256": attempt.source_sha256,
        "status": attempt.status,
        "pairs": len(attempt.pairs) if attempt.pairs is not None else None,
        "unpaired_repetitions": attempt.unpaired_repetitions,
        "median_uninstrumented_seconds": _median(attempt.pairs, 0),
        "median_instrumented_seconds": _median(attempt.pairs, 1),
        "relative_increase": increase,
        "interval": list(interval) if interval is not None else None,
        "baseline_cv": cv,
        "verdict": verdict,
        "reason": reason,
    }


def combined_verdict(attempts: list[dict]) -> str:
    """A verdict only when enough attempts completed and every completed one agrees.

    `not_evaluated` when no attempt could be evaluated at all, so unavailable evidence is never
    reported as an inconclusive measurement.
    """
    if all(entry["verdict"] == NOT_EVALUATED for entry in attempts):
        return NOT_EVALUATED
    completed = [entry["verdict"] for entry in attempts if entry["status"] == "completed"]
    agreed = len(completed) >= MIN_COMPLETED_ATTEMPTS and len(set(completed)) == 1
    return completed[0] if agreed and completed[0] in (WITHIN, OVER) else INCONCLUSIVE


def _grouped(attempts: list[Attempt], host_label: str) -> list[tuple[dict, list[Attempt]]]:
    groups: dict[str, tuple[dict, list[Attempt]]] = {}
    for attempt in attempts:
        settings = attempt.settings | {"host_label": host_label}
        groups.setdefault(json_text(settings), (settings, []))[1].append(attempt)
    return list(groups.values())


def evaluate(
    paths: list[Path], contract: OptimizationContract, host_label: str, seed: int = DEFAULT_SEED
) -> dict:
    """Judge attempts all measured on the host named `host_label`."""
    if not HOST_LABEL.fullmatch(host_label):
        raise AttemptError(
            f"host label {host_label!r} must be 1-64 letters, digits, '.', '_' or '-'"
        )
    attempts = [read_attempt(path, number) for number, path in enumerate(paths, start=1)]
    seen: dict[str, int] = {}
    for attempt in attempts:
        if attempt.source_sha256 in seen:
            raise AttemptError(
                f"attempt {attempt.number} ({attempt.source}) is the same file as attempt "
                f"{seen[attempt.source_sha256]}; one file cannot count as two attempts"
            )
        seen[attempt.source_sha256] = attempt.number
    groups = []
    for settings, members in _grouped(attempts, host_label):
        entries = [attempt_entry(attempt, contract, seed) for attempt in members]
        groups.append(
            {"settings": settings, "attempts": entries, "verdict": combined_verdict(entries)}
        )
    return {
        "protocol": PROTOCOL,
        "method": METHOD,
        "contract": {
            "version": contract.version,
            "status": contract.status,
            "sha256": contract.sha256[:8],
        },
        "budget": {
            "max_median_instrumentation_overhead": contract.max_median_instrumentation_overhead,
            "max_median_admission_overhead": contract.max_median_admission_overhead,
            "max_baseline_cv": contract.max_baseline_cv,
            "min_completed_attempts": MIN_COMPLETED_ATTEMPTS,
            "min_pairs": MIN_PAIRS,
        },
        "bootstrap": {
            "method": "paired_bootstrap_percentile",
            "resamples": contract.bootstrap_resamples,
            "level": contract.interval_level,
            "seed": seed,
        },
        "groups": groups,
        "limitations": [
            "Judges saved measurements only; describes the measured hosts, not other hardware",
            "A failed `overhead` command saves no file, so its failure is not in this report",
            (
                "Runner wall time only; server durations and sampler cost are reported by the "
                "measurement commands without a gate"
            ),
        ],
    }


def _print_summary(report: dict, path: Path, write: Callable[[str], object] = print) -> None:
    write(f"overhead verdict: {path}")
    for group in report["groups"]:
        settings = group["settings"]
        write(f"  {settings['protocol']} on {settings['host_label']}: {group['verdict']}")
        for entry in group["attempts"]:
            increase = entry["relative_increase"]
            shown = "n/a" if increase is None else f"{increase:+.2%}"
            write(
                f"    attempt {entry['attempt']} {entry['source']}: {entry['verdict']} "
                f"({shown}); {entry['reason']}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("attempts", type=Path, nargs="+", help="overhead or sampling-overhead JSON")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--host-label",
        required=True,
        help="short name of the host every attempt was measured on (not a path or host name)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap seed (recorded)")
    parser.add_argument("--output", type=Path, help=f"report directory (default {DEFAULT_REPORTS})")
    args = parser.parse_args(argv)
    try:
        contract = load_optimization_contract(args.contract)
        report = evaluate(args.attempts, contract, args.host_label, args.seed)
    except (ManifestError, AttemptError) as error:
        print(f"Overhead verdict failed: {error}", file=sys.stderr)
        return 1
    path = (args.output or DEFAULT_REPORTS) / REPORT_FILE
    write_atomic(path, json_text(report))
    _print_summary(report, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
