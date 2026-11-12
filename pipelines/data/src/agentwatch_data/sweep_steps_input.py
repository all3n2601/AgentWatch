"""Read and validate the recorder's `steps.jsonl` and `steps.json` before any labeling.

The files are written by `capacity-sweep-steps` (`agentwatch_agent.sweep_steps`). This package
does not import the agent, so the format constants below are re-declared and must equal that
module's. Everything is checked before a record is returned: a refusal names the file and, for
a record, its 1-based line number and field, and no partial result exists.

`steps.json` records the SHA-256 of `steps.jsonl`, so records from one write are never read with
the summary of another. A reading that was not measured is null; it is never coerced to zero
here.
"""

import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agentwatch_data.files import file_sha256
from agentwatch_data.labels import LabelingError

# Re-declared from agentwatch_agent.sweep_steps; keep equal.
STEPS_FORMAT = "sweep-steps-v1"
STEPS_FILE = "steps.jsonl"
SUMMARY_FILE = "steps.json"
RECORD_KEYS = frozenset(
    {
        "run_id",
        "run_folder",
        "step_id",
        "step_type",
        "parent_step_id",
        "plan_sha256",
        "task_id",
        "configuration",
        "n",
        "phase",
        "batch_sequence",
        "batch_id",
        "state",
        "duration_seconds",
        "queue_seconds",
        "execution_seconds",
        "span_status",
        "vllm_queue_seconds",
        "vllm_service_seconds",
        "baseline",
    }
)
SPAN_STATUSES = ("measured", "unavailable", "malformed", "incomplete", "not_recorded")
PHASES = ("ramp", "live", "injected", "control")
PROTOCOL_VERSION = "natural-overload-sweep-v1"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
# A step's runner ticks split into queue and execution time; the parts must add up to the
# duration within the live contracts' timing-breakdown tolerance.
TIMING_TOLERANCE_SECONDS = 1e-6

SECONDS_FIELDS = (
    "duration_seconds",
    "queue_seconds",
    "execution_seconds",
    "vllm_queue_seconds",
    "vllm_service_seconds",
)
VLLM_FIELDS = ("vllm_queue_seconds", "vllm_service_seconds")
TEXT_FIELDS = ("step_id", "step_type", "state", "run_folder", "batch_id", "configuration")
NULLABLE_TEXT_FIELDS = ("parent_step_id", "plan_sha256", "task_id")
BASELINE_KEYS = frozenset({"count", "mean_seconds", "stdev_seconds"})
SUMMARY_KEYS = frozenset(
    {
        "format",
        "sweep_folder",
        "sweep_id",
        "configuration",
        "protocol_version",
        "protocol_sha256",
        "sweep_json_sha256",
        "held_out",
        "steps",
        "steps_sha256",
        "excluded_batches",
        "runs_without_evidence",
        "runs_without_plan",
    }
)


class SweepStepsError(LabelingError):
    """Recorder output that cannot be trusted as labeling input."""


@dataclass(frozen=True)
class SweepSteps:
    summary: dict
    records: tuple[dict, ...]
    steps_sha256: str
    summary_sha256: str


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_seconds(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_uuid(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except OSError as error:
        raise SweepStepsError(f"{path.name} cannot be read from {path.parent}: {error}") from error
    except UnicodeDecodeError as error:
        raise SweepStepsError(f"{path.name} is not valid UTF-8: {error}") from error


def _summary_problem(
    summary: dict, accepted_protocols: dict[str, str], line_count: int
) -> tuple[str, str] | None:
    """The (field, problem) of the first thing wrong with the summary, if any."""
    missing, unknown = SUMMARY_KEYS - summary.keys(), summary.keys() - SUMMARY_KEYS
    if missing or unknown:
        field = min(missing | unknown)
        return field, f"missing {sorted(missing)}, unknown {sorted(unknown)}"
    if summary["format"] != STEPS_FORMAT:
        return "format", f"must be {STEPS_FORMAT!r}, got {summary['format']!r}"
    if summary["held_out"] is not True:
        return "held_out", f"must be true, got {summary['held_out']!r}"
    version = summary["protocol_version"]
    if version != PROTOCOL_VERSION:
        return "protocol_version", f"must be {PROTOCOL_VERSION!r}, got {version!r}"
    configuration, digest = summary["configuration"], summary["protocol_sha256"]
    if not isinstance(configuration, str) or configuration not in accepted_protocols:
        return "configuration", f"{configuration!r} is not one of {sorted(accepted_protocols)}"
    if digest not in accepted_protocols.values():
        return "protocol_sha256", f"{digest!r} is not an accepted protocol hash"
    if accepted_protocols[configuration] != digest:
        return "protocol_sha256", f"{digest!r} is not the accepted hash for {configuration}"
    if not _is_int(summary["steps"]) or summary["steps"] < 0:
        return "steps", f"must be a nonnegative integer, got {summary['steps']!r}"
    if summary["steps"] != line_count:
        return "steps", f"says {summary['steps']} but {STEPS_FILE} has {line_count} records"
    if not _is_text(summary["sweep_folder"]):
        return "sweep_folder", f"must be a nonempty string, got {summary['sweep_folder']!r}"
    for field in ("sweep_json_sha256", "steps_sha256"):
        if not isinstance(summary[field], str) or not SHA256_HEX.fullmatch(summary[field]):
            return field, f"must be 64 lowercase hex digits, got {summary[field]!r}"
    if summary["sweep_id"] is not None and not isinstance(summary["sweep_id"], str):
        return "sweep_id", f"must be a string or null, got {summary['sweep_id']!r}"
    excluded = summary["excluded_batches"]
    if not isinstance(excluded, list) or not all(isinstance(item, dict) for item in excluded):
        return "excluded_batches", "must be a list of objects"
    for field in ("runs_without_evidence", "runs_without_plan"):
        if not _is_int(summary[field]) or summary[field] < 0:
            return field, f"must be a nonnegative integer, got {summary[field]!r}"
    return None


def _baseline_problem(baseline) -> tuple[str, str] | None:
    if baseline is None:
        return None
    if not isinstance(baseline, dict):
        return "baseline", "must be an object or null"
    missing, unknown = BASELINE_KEYS - baseline.keys(), baseline.keys() - BASELINE_KEYS
    if missing or unknown:
        return "baseline", f"missing {sorted(missing)}, unknown {sorted(unknown)}"
    if not _is_int(baseline["count"]) or baseline["count"] < 1:
        return "baseline.count", f"must be an integer of at least 1, got {baseline['count']!r}"
    mean = baseline["mean_seconds"]
    if not _is_seconds(mean):
        return "baseline.mean_seconds", f"must be finite and >= 0, got {mean!r}"
    stdev = baseline["stdev_seconds"]
    if stdev is not None and not _is_seconds(stdev):
        return "baseline.stdev_seconds", f"must be null or finite and >= 0, got {stdev!r}"
    return None


def _record_problem(record: dict) -> tuple[str, str] | None:
    """The (field, problem) of the first thing wrong with one record, if any."""
    missing, unknown = RECORD_KEYS - record.keys(), record.keys() - RECORD_KEYS
    if missing or unknown:
        field = min(missing | unknown)
        return field, f"missing {sorted(missing)}, unknown {sorted(unknown)}"
    if not _is_uuid(record["run_id"]):
        return "run_id", f"must be a UUID, got {record['run_id']!r}"
    for field in TEXT_FIELDS:
        if not _is_text(record[field]):
            return field, f"must be a nonempty string, got {record[field]!r}"
    folder = PurePosixPath(record["run_folder"])
    if folder.is_absolute() or ".." in folder.parts or "\\" in record["run_folder"]:
        return "run_folder", f"must be relative and stay inside the sweep, got {folder}"
    for field in NULLABLE_TEXT_FIELDS:
        if record[field] is not None and not isinstance(record[field], str):
            return field, f"must be a string or null, got {record[field]!r}"
    if not _is_int(record["n"]) or record["n"] < 1:
        return "n", f"must be a positive integer, got {record['n']!r}"
    if not _is_int(record["batch_sequence"]) or record["batch_sequence"] < 0:
        return "batch_sequence", f"must be a nonnegative integer, got {record['batch_sequence']!r}"
    if record["phase"] not in PHASES:
        return "phase", f"must be one of {PHASES}, got {record['phase']!r}"
    for field in SECONDS_FIELDS:
        if record[field] is not None and not _is_seconds(record[field]):
            return field, f"must be null or a finite number >= 0, got {record[field]!r}"
    parts = (record["queue_seconds"], record["execution_seconds"], record["duration_seconds"])
    if None not in parts and abs(parts[0] + parts[1] - parts[2]) > TIMING_TOLERANCE_SECONDS:
        return "duration_seconds", (
            f"queue_seconds + execution_seconds is {parts[0] + parts[1]!r}, not {parts[2]!r}"
        )
    if record["span_status"] not in SPAN_STATUSES:
        return "span_status", f"must be one of {SPAN_STATUSES}, got {record['span_status']!r}"
    if record["span_status"] != "measured":
        for field in VLLM_FIELDS:
            if record[field] is not None:
                return field, f"must be null when span_status is {record['span_status']!r}"
    return _baseline_problem(record["baseline"])


def _parse_records(text: str) -> list[tuple[int, dict]]:
    """Each non-empty line as (1-based line number, object); empty lines are skipped."""
    parsed = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SweepStepsError(f"{STEPS_FILE} line {number}: not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise SweepStepsError(f"{STEPS_FILE} line {number}: must be a JSON object")
        parsed.append((number, record))
    return parsed


def read_steps(directory: Path, accepted_protocols: dict[str, str]) -> SweepSteps:
    """The validated summary and records of `directory`, or `SweepStepsError`.

    `accepted_protocols` maps a configuration name to its full protocol SHA-256.
    """
    steps_path, summary_path = directory / STEPS_FILE, directory / SUMMARY_FILE
    try:
        summary = json.loads(_read_text(summary_path))
    except json.JSONDecodeError as error:
        raise SweepStepsError(f"{SUMMARY_FILE} is not valid JSON: {error}") from error
    if not isinstance(summary, dict):
        raise SweepStepsError(f"{SUMMARY_FILE} must be a JSON object")
    parsed = _parse_records(_read_text(steps_path))
    problem = _summary_problem(summary, accepted_protocols, len(parsed))
    if problem is not None:
        field, detail = problem
        raise SweepStepsError(f"{SUMMARY_FILE} field {field}: {detail}")
    steps_sha256 = file_sha256(steps_path)
    if summary["steps_sha256"] != steps_sha256:
        raise SweepStepsError(
            f"{SUMMARY_FILE} field steps_sha256: does not match {STEPS_FILE} ({steps_sha256}); "
            "the two files are not from the same write"
        )
    seen: set[tuple[str, str]] = set()
    for number, record in parsed:
        problem = _record_problem(record)
        if problem is not None:
            field, detail = problem
            raise SweepStepsError(f"{STEPS_FILE} line {number} field {field}: {detail}")
        if record["configuration"] != summary["configuration"]:
            raise SweepStepsError(
                f"{STEPS_FILE} line {number} field configuration: {record['configuration']!r} "
                f"differs from the summary's {summary['configuration']!r}"
            )
        identity = (record["run_id"], record["step_id"])
        if identity in seen:
            raise SweepStepsError(
                f"{STEPS_FILE} line {number} field step_id: repeats run {identity[0]} "
                f"step {identity[1]!r}"
            )
        seen.add(identity)
    return SweepSteps(
        summary=summary,
        records=tuple(record for _number, record in parsed),
        steps_sha256=steps_sha256,
        summary_sha256=file_sha256(summary_path),
    )
