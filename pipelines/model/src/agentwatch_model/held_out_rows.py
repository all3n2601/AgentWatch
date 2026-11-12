"""Feature rows for held-out capacity-sweep truths.

`capacity-sweep-truth` writes, per sweep, `truth.jsonl` (one record per journaled step, each
naming its `run_folder` relative to the sweep folder) and `truth.json` (the summary). This
module joins every truth record, on (run, step), to the `steps-v1.3` feature row computed from
that run's own `live-journal.jsonl`. Sweep runs have no `experiment.json`, so the per-replay
join of `agentwatch_data.replay_rows` is composed from its public pieces.

A step the evidence cannot support is refused under a `replay_rows` code and counted, never
guessed: `feature_evidence_unreadable` (journal or evidence unreadable, or a `run_folder` that
leaves the sweep folder), `feature_row_missing` (the journal belongs to another run, or has no
terminal row for the step), `feature_timing_mismatch` (the journal's timings differ from the
truth record's), `declared_interval_invalid`.
"""

import json
from collections import Counter
from pathlib import Path, PurePosixPath
from uuid import UUID

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.files import file_sha256
from agentwatch_data.replay_evidence import EVIDENCE_FILE, JOURNAL_FILE
from agentwatch_data.replay_journal import read_journal
from agentwatch_data.replay_rows import (
    FEATURE_EVIDENCE_UNREADABLE,
    FEATURE_ROW_MISSING,
    FeatureJoinError,
    ReplayFeatures,
    declared_interval_seconds,
    feature_values,
    joined_feature,
    same_run,
)
from agentwatch_features import compute_feature_rows

# Declared by services/agent/src/agentwatch_agent/sweep_truth.py (TRUTH_METHOD and
# ACCEPTED_PROTOCOL_SHA256); re-declared because pipelines must not import the agent service.
# Keep both in step with it.
ACCEPTED_TRUTH_METHOD = "sweep-truth-v1"
ACCEPTED_PROTOCOL_SHA256: dict[str, str] = {
    "cfg-default": "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9",
    "cfg-hardware": "1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd",
}
TRUTH_FILE = "truth.jsonl"
SUMMARY_FILE = "truth.json"


class HeldOutError(ValueError):
    """Truth files that are not accepted held-out sweep truths."""


def load_truths(directory: Path) -> tuple[dict, list[dict]]:
    """The summary and truth records of one sweep truth folder, refused unless accepted."""
    try:
        summary = json.loads((directory / SUMMARY_FILE).read_text(encoding="utf-8"))
        lines = (directory / TRUTH_FILE).read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as error:
        raise HeldOutError(f"{directory}: truth files cannot be read: {error}") from error
    if not isinstance(summary, dict) or not all(isinstance(r, dict) for r in records):
        raise HeldOutError(f"{directory}: truth files must hold JSON objects")
    if summary.get("held_out") is not True:
        raise HeldOutError(f"{directory}: truths are not marked held_out")
    if summary.get("method") != ACCEPTED_TRUTH_METHOD:
        raise HeldOutError(f"{directory}: method {summary.get('method')!r} is not accepted")
    if summary.get("protocol_sha256") not in ACCEPTED_PROTOCOL_SHA256.values():
        raise HeldOutError(f"{directory}: protocol hash is not an accepted held-out protocol")
    folder = summary.get("sweep_folder")
    if (
        not isinstance(folder, str)
        or folder in {"", ".", ".."}
        or "/" in folder
        or "\\" in folder
        or ":" in folder
    ):
        raise HeldOutError(f"{directory}: sweep_folder must be a single folder name")
    return summary, records


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


def held_out_rows(
    truth_dirs: list[Path], sweep_root: Path
) -> tuple[list[dict], dict[str, int], list[dict]]:
    """Joined rows (truth record, feature columns, telemetry_status), refusal counts by code,
    and the summaries.

    `sweep_root` holds the sweep folders; each summary names its own by `sweep_folder`.
    Each run's journal is read once. Unacceptable truth files raise HeldOutError.
    """
    rows: list[dict] = []
    refused: Counter[str] = Counter()
    summaries: list[dict] = []
    for truth_dir in truth_dirs:
        summary, records = load_truths(truth_dir)
        summaries.append(summary)
        sweep_dir = sweep_root / summary["sweep_folder"]
        by_run: dict[object, list[dict]] = {}
        for record in records:
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
    return rows, dict(sorted(refused.items())), summaries
