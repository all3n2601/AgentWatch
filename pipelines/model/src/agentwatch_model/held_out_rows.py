"""Feature rows for held-out capacity-sweep truths.

`make sweep-truth` (`agentwatch_data.sweep_truth`) writes, per sweep, `truth.jsonl` (one record
per recorded step, each naming its `run_folder` relative to the sweep folder) and `truth.json`
(the summary). This module joins every judged truth record, on (run, step), to the `steps-v1.3`
feature row computed from that run's own `live-journal.jsonl`. A record whose three truths are
all null is unjudged (a root step, or a step with no reference): it is returned apart and never
feature-joined, so its missing journal is not a refusal. Sweep runs have no `experiment.json`, so the per-replay
join of `agentwatch_data.replay_rows` is composed from its public pieces.

A step the evidence cannot support is refused under a `replay_rows` code and counted, never
guessed: `feature_evidence_unreadable` (journal or evidence unreadable, or a `run_folder` that
leaves the sweep folder), `feature_row_missing` (the journal belongs to another run, or has no
terminal row for the step), `feature_timing_mismatch` (the journal's timings differ from the
truth record's), `declared_interval_invalid`.

Truth files are another package's output, so they are checked at this boundary before any
join: a summary or record outside the `sweep-truth-v2` format, a summary of another label
version or that does not match its records (step count, `truth.jsonl` hash), or a sweep or step passed twice raises HeldOutError, which fails the attempt.
"""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from uuid import UUID

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.files import file_sha256
from agentwatch_data.replay_evidence import EVIDENCE_FILE, JOURNAL_FILE
from agentwatch_data.replay_journal import read_journal
from agentwatch_data.replay_rows import (
    FEATURE_EVIDENCE_UNREADABLE,
    FEATURE_ROW_MISSING,
    JOINED_TIMINGS,
    FeatureJoinError,
    ReplayFeatures,
    declared_interval_seconds,
    feature_values,
    joined_feature,
    same_run,
)
from agentwatch_data.sweep_truth import TRUTH_FILE, TRUTH_NAMES, TRUTH_SUMMARY_FILE
from agentwatch_features import compute_feature_rows

# The `sweep-truth-v2` rule's label version: the rule's name plus the fingerprint of its values
# and logic (`load_truth_rule().label_version`; a test keeps this equal to it). Truths labelled
# by any other rule are refused.
ACCEPTED_TRUTH_LABEL_VERSION = "sweep-truth-v2+c211f3e8"
# The protocol hashes of the two held-out configurations, as pinned in sweep-truth-v2.toml.
ACCEPTED_PROTOCOL_SHA256: dict[str, str] = {
    "cfg-default": "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9",
    "cfg-hardware": "1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd",
}
TRUTH_CLASSES = frozenset({"inference", "none"})
TRUTH_PHASES = frozenset({"ramp", "live", "injected", "control"})
RECORD_STRINGS = ("run_id", "step_id", "run_folder", "batch_id", "step_type", "task_id")
SUMMARY_COUNTERS = ("runs_without_plan", "runs_without_evidence")
# Hashes `make sweep-truth` records of the steps it read and of the recorder that wrote them.
SUMMARY_HASHES = ("steps_sha256", "steps_summary_sha256", "recorder_sha256")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class HeldOutError(ValueError):
    """Truth files that are not accepted held-out sweep truths."""


def is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_seconds(value: object) -> bool:
    """A finite number of seconds, or None for an unmeasured reading."""
    if value is None:
        return True
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def summary_problem(summary: dict, record_count: int, truth_sha256: str) -> str | None:
    """Why the summary is not an accepted `sweep-truth-v2` summary of `record_count` records."""
    if summary.get("held_out") is not True:
        return "truths are not marked held_out"
    if summary.get("label_version") != ACCEPTED_TRUTH_LABEL_VERSION:
        return (
            f"label_version {summary.get('label_version')!r} is not the accepted "
            f"{ACCEPTED_TRUTH_LABEL_VERSION!r}"
        )
    configuration = summary.get("configuration")
    accepted = (
        ACCEPTED_PROTOCOL_SHA256.get(configuration) if isinstance(configuration, str) else None
    )
    if accepted is None:
        return f"configuration {configuration!r} is not an accepted held-out configuration"
    if summary.get("protocol_sha256") != accepted:
        return f"protocol_sha256 is not the accepted {configuration} protocol hash"
    folder = summary.get("sweep_folder")
    if (
        not isinstance(folder, str)
        or folder in {"", ".", ".."}
        or "/" in folder
        or "\\" in folder
        or ":" in folder
    ):
        return "sweep_folder must be a single folder name"
    sweep_id = summary.get("sweep_id")
    if not isinstance(sweep_id, str) or not sweep_id:
        return f"sweep_id must be a non-empty string, found {sweep_id!r}"
    for name in SUMMARY_COUNTERS:
        if not is_count(summary.get(name)) or summary[name] < 0:
            return f"{name} must be a non-negative integer, found {summary.get(name)!r}"
    if not isinstance(summary.get("excluded_batches"), list):
        return f"excluded_batches must be a list, found {summary.get('excluded_batches')!r}"
    for name in SUMMARY_HASHES:
        value = summary.get(name)
        if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
            return f"{name} must be 64 lowercase hex characters, found {value!r}"
    steps = summary.get("steps")
    if not is_count(steps) or steps != record_count:
        return (
            f"steps is {steps!r} but {TRUTH_FILE} holds {record_count} records; the summary "
            "does not describe these records (stale or incomplete truth files)"
        )
    if summary.get("truth_sha256") != truth_sha256:
        return (
            f"truth_sha256 is {summary.get('truth_sha256')!r} but {TRUTH_FILE} hashes to "
            f"{truth_sha256}; the summary does not describe these records"
        )
    return None


def truths_problem(record: dict) -> str | None:
    """Why the record's truths are not three classes-or-null, each null one with a reason."""
    truths, reasons = record.get("truths"), record.get("truth_reasons")
    for field, value in (("truths", truths), ("truth_reasons", reasons)):
        if not isinstance(value, dict) or set(value) != set(TRUTH_NAMES):
            return f"{field} must be an object with exactly the keys {sorted(TRUTH_NAMES)}"
    for name in sorted(TRUTH_NAMES):
        truth, reason = truths[name], reasons[name]
        if truth is None:
            if not isinstance(reason, str) or not reason:
                return f"truths.{name} is null without a reason in truth_reasons.{name}"
        elif not isinstance(truth, str) or truth not in TRUTH_CLASSES:
            return f"truths.{name} is {truth!r}, not 'inference', 'none' or null"
        elif reason is not None:
            return f"truth_reasons.{name} must be null when truths.{name} is {truth!r}"
    return None


def record_problem(record: dict, configuration: str) -> str | None:
    """Why the record is not a `sweep-truth-v2` step record of the summary's configuration."""
    for field in RECORD_STRINGS:
        if not isinstance(record.get(field), str):
            return f"{field} must be a string, found {record.get(field)!r}"
    if not is_count(record.get("n")):
        return f"n must be an integer, found {record.get('n')!r}"
    phase = record.get("phase")
    if not isinstance(phase, str) or phase not in TRUTH_PHASES:
        return f"phase must be one of {sorted(TRUTH_PHASES)}, found {phase!r}"
    if record.get("configuration") != configuration:
        return (
            f"configuration is {record.get('configuration')!r}, the summary's is {configuration!r}"
        )
    for field in JOINED_TIMINGS:
        if field not in record or not is_seconds(record[field]):
            return f"{field} must be a finite number or null, found {record.get(field)!r}"
    return truths_problem(record)


def load_truths(directory: Path) -> tuple[dict, list[dict]]:
    """The summary and truth records of one sweep truth folder, refused unless accepted.

    Refusals name the file and, for a record, its index among the records and its line.
    """
    summary_file, truth_file = directory / TRUTH_SUMMARY_FILE, directory / TRUTH_FILE
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        truth_bytes = truth_file.read_bytes()
        lines = truth_bytes.decode("utf-8").splitlines()
        numbered = [
            (number, json.loads(line)) for number, line in enumerate(lines, 1) if line.strip()
        ]
    except (OSError, ValueError) as error:
        raise HeldOutError(f"{directory}: truth files cannot be read: {error}") from error
    if not isinstance(summary, dict):
        raise HeldOutError(f"{summary_file}: the summary must be a JSON object")
    problem = summary_problem(summary, len(numbered), hashlib.sha256(truth_bytes).hexdigest())
    if problem is not None:
        raise HeldOutError(f"{summary_file}: {problem}")
    for index, (number, record) in enumerate(numbered):
        if not isinstance(record, dict):
            problem = "the record must be a JSON object"
        else:
            problem = record_problem(record, summary["configuration"])
        if problem is not None:
            raise HeldOutError(f"{truth_file}: record {index} (line {number}): {problem}")
    return summary, [record for _, record in numbered]


def refuse_repeats(loaded: list[tuple[Path, dict, list[dict]]]) -> None:
    """Refuse a sweep passed twice, by id or folder, and a (run, step) recorded twice: either
    would count the same steps more than once."""
    sweeps: dict[tuple[str, str], Path] = {}
    steps: dict[tuple[str, str], Path] = {}
    for truth_dir, summary, records in loaded:
        for field in ("sweep_id", "sweep_folder"):
            key = (field, summary[field])
            if key in sweeps:
                raise HeldOutError(
                    f"{truth_dir} and {sweeps[key]} both hold {field} {summary[field]!r}; "
                    "pass each sweep's truths once"
                )
            sweeps[key] = truth_dir
        truth_file = truth_dir / TRUTH_FILE
        for record in records:
            step = (record["run_id"], record["step_id"])
            if step in steps:
                raise HeldOutError(
                    f"{truth_file}: run {step[0]} step {step[1]} appears more than once "
                    f"(first in {steps[step]})"
                )
            steps[step] = truth_file


def run_directory(sweep_dir: Path, run_folder: object) -> Path:
    """The run folder inside its sweep folder; refuses an absolute or parent-relative path."""
    if not isinstance(run_folder, str) or "\\" in run_folder or ":" in run_folder:
        raise FeatureJoinError(FEATURE_EVIDENCE_UNREADABLE, f"run_folder {run_folder!r} is invalid")
    relative = PurePosixPath(run_folder)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise FeatureJoinError(
            FEATURE_EVIDENCE_UNREADABLE, f"run_folder {run_folder!r} is outside the sweep folder"
        )
    return sweep_dir / relative


def run_features(run_dir: Path) -> tuple[ReplayFeatures, UUID | None]:
    """One run's features and the run its journal records, from its own two files."""
    try:
        declared = declared_interval_seconds(run_dir)
        journal = read_journal(run_dir / JOURNAL_FILE)
        source_sha256 = canonical_sha256(
            {
                EVIDENCE_FILE: file_sha256(run_dir / EVIDENCE_FILE),
                JOURNAL_FILE: file_sha256(run_dir / JOURNAL_FILE),
            }
        )
    except FeatureJoinError:
        raise
    except (OSError, ValueError) as error:  # ReplayEvidenceError is a ValueError too
        raise FeatureJoinError(
            FEATURE_EVIDENCE_UNREADABLE, f"{run_dir.name}: evidence cannot be read: {error}"
        ) from error
    try:
        computed = compute_feature_rows(journal.events, journal.samples, declared)
    except ValueError as error:
        raise FeatureJoinError(
            FEATURE_ROW_MISSING, f"features cannot be computed: {error}"
        ) from error
    records = (*journal.events, *journal.samples)
    features = ReplayFeatures(
        by_step={row.step_id: row for row in computed.rows},
        declared_interval_seconds=declared,
        configuration_sha256=journal.configuration_sha256,
        recorded_at=min((event.observed_at for event in journal.events), default=None),
        source_sha256=source_sha256,
    )
    return features, records[0].run_id if records else None


class HeldOutSet(NamedTuple):
    """Joined rows, refusal counts by code, the summaries, and the steps left unjudged."""

    rows: list[dict]
    refused: dict[str, int]
    summaries: list[dict]
    unjudged: list[dict]


def is_unjudged(record: dict) -> bool:
    """A record the rule judged on none of its truths; there is nothing to score it against."""
    return all(record["truths"][name] is None for name in TRUTH_NAMES)


def held_out_rows(truth_dirs: list[Path], sweep_root: Path) -> HeldOutSet:
    """Joined rows (truth record, feature columns, telemetry_status), refusal counts by code,
    the summaries, and the unjudged records.

    `sweep_root` holds the sweep folders; each summary names its own by `sweep_folder`.
    Unjudged records are set aside before any journal is read, so they neither join nor
    count as refused. Each run's journal is read once. Unacceptable truth files, a sweep
    passed twice and a repeated (run, step) raise HeldOutError before any journal is read.
    """
    loaded = [(truth_dir, *load_truths(truth_dir)) for truth_dir in truth_dirs]
    refuse_repeats(loaded)
    rows: list[dict] = []
    refused: Counter[str] = Counter()
    summaries: list[dict] = []
    unjudged: list[dict] = []
    for _, summary, records in loaded:
        summaries.append(summary)
        sweep_dir = sweep_root / summary["sweep_folder"]
        by_run: dict[object, list[dict]] = {}
        for record in records:
            if is_unjudged(record):
                unjudged.append(record)
            else:
                by_run.setdefault(record.get("run_folder"), []).append(record)
        for run_folder, run_records in by_run.items():
            try:
                features, journal_run = run_features(run_directory(sweep_dir, run_folder))
            except FeatureJoinError as error:
                refused[error.code] += len(run_records)
                continue
            for record in run_records:
                try:
                    if journal_run is None or not same_run(record["run_id"], journal_run):
                        raise FeatureJoinError(
                            FEATURE_ROW_MISSING,
                            f"{JOURNAL_FILE} records run_id {journal_run}, "
                            f"the truth names {record['run_id']}",
                        )
                    feature = joined_feature(record, features)
                except FeatureJoinError as error:
                    refused[error.code] += 1
                    continue
                rows.append(
                    record
                    | feature_values(feature)
                    | {"telemetry_status": feature.telemetry.status}
                )
    return HeldOutSet(rows, dict(sorted(refused.items())), summaries, unjudged)
