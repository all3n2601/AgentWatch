"""Release rows for one replay experiment: labels-v2 rows joined to step-features-v1.

Each labeled step is joined, on (run_id, step_id), to the feature row computed from its own
replay's `live-journal.jsonl`. The join is refused, rejecting only the affected steps, when the
journal belongs to another run or has no terminal row for the step, and when the feature row's
timings differ from the `experiment.json` timings the label used: the label and the features
would then describe different windows. Readings the journal lacks stay null.

A read defect is not tolerated per replay: when any labeled replay's journal or
`agent-evidence.json` cannot be read (`feature_evidence_unreadable`) or its declared sampling
interval is invalid (`declared_interval_invalid`), every labeled step of the experiment is
rejected with that code, naming the defective replay.

Rows hold every release column except dataset_version, which the release build adds. Each row
is validated on its own, so one invalid step rejects only itself.

Usage: python -m agentwatch_data.replay_rows <experiment directory>
"""

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path, PurePosixPath
from uuid import UUID

from agentwatch_features import FEATURE_VERSION, FeatureRow, compute_feature_rows

from agentwatch_data.extract import Rejection, canonical_sha256, parse_json
from agentwatch_data.files import file_sha256
from agentwatch_data.labels import LabelingError
from agentwatch_data.replay_evidence import (
    EVIDENCE_FILE,
    JOURNAL_FILE,
    MANIFEST_FILE,
    Experiment,
    ReplayEvidenceError,
    read_evidence_text,
    read_experiment,
)
from agentwatch_data.replay_journal import read_journal
from agentwatch_data.replay_labels import ReplayLabelConfig, label_experiment, load_replay_config
from agentwatch_data.schema import SCHEMA_VERSION, SchemaError, validate_rows

logger = logging.getLogger(__name__)

REPLAY_KIND = "agent_replay"
FEATURE_TIMING_TOLERANCE_SECONDS = 1e-6  # absolute, as for contract timing breakdowns
ROLE_CONDITIONS = {"baseline": "baseline", "contended": "contended", "clean_control": "clean"}
REPLAY_STAGES = ("replay_label", "replay_features", "replay_schema")
LABEL_STAGE, FEATURE_STAGE, SCHEMA_STAGE = REPLAY_STAGES
# The timings the label used, compared with the feature row's before the two are joined.
JOINED_TIMINGS = ("duration_seconds", "queue_seconds", "execution_seconds")
FEATURE_ROW_MISSING = "feature_row_missing"
FEATURE_TIMING_MISMATCH = "feature_timing_mismatch"
FEATURE_EVIDENCE_UNREADABLE = "feature_evidence_unreadable"
DECLARED_INTERVAL_INVALID = "declared_interval_invalid"
# Read defects reject the whole experiment (the D03 spec allows no per-replay read tolerance);
# the other codes reject only the affected replay or step.
EXPERIMENT_DEFECTS = (FEATURE_EVIDENCE_UNREADABLE, DECLARED_INTERVAL_INVALID)


class FeatureJoinError(Exception):
    """A labeled step cannot be joined to its features; the message starts with its code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ReplayFeatures:
    """What one replay's journal contributes to its rows."""

    # Terminal feature rows by step_id. The journal's run is checked against the manifest's
    # before these are kept, so step_id alone is the (run_id, step_id) join key.
    by_step: dict[str, FeatureRow]
    declared_interval_seconds: float | None
    configuration_sha256: str | None
    recorded_at: datetime | None  # earliest observed_at of the replay's step events
    source_sha256: str


def replay_source_sha256(directory: Path, evidence_dir: str) -> str:
    """Identify one replay's evidence: the experiment manifest and the replay's two files."""
    relative = PurePosixPath(evidence_dir)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReplayEvidenceError(f"Path {evidence_dir!r} is outside the experiment directory")
    replay_dir = directory / relative
    return canonical_sha256(
        {
            MANIFEST_FILE: file_sha256(directory / MANIFEST_FILE),
            EVIDENCE_FILE: file_sha256(replay_dir / EVIDENCE_FILE),
            JOURNAL_FILE: file_sha256(replay_dir / JOURNAL_FILE),
        }
    )


def evidence_dirs(directory: Path) -> dict[int, str]:
    """Each replay's evidence directory by index, as the manifest names it.

    `read_experiment` has already confined and de-duplicated these paths; `Replay` does not
    keep them, and changing it would change the labels-v2 fingerprint.
    """
    try:
        manifest = parse_json(read_evidence_text(directory / MANIFEST_FILE))
    except ValueError as error:  # ReplayEvidenceError is a ValueError too
        raise ReplayEvidenceError(f"{MANIFEST_FILE}: {error}") from error
    entries = manifest.get("replays") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ReplayEvidenceError(f"{MANIFEST_FILE}: replays must be a list of objects")
    directories = {entry.get("index"): entry.get("evidence_dir") for entry in entries}
    if not all(isinstance(path, str) for path in directories.values()):
        raise ReplayEvidenceError(f"{MANIFEST_FILE}: evidence_dir must be a string")
    return directories


def declared_interval_seconds(replay_dir: Path) -> float | None:
    """The replay's configured sampling interval, or None when it was not recorded."""
    try:
        evidence = parse_json(read_evidence_text(replay_dir / EVIDENCE_FILE))
    except ValueError as error:  # ReplayEvidenceError is a ValueError too
        raise FeatureJoinError(
            FEATURE_EVIDENCE_UNREADABLE, f"{EVIDENCE_FILE} cannot be read: {error}"
        ) from error
    configuration = evidence.get("configuration") if isinstance(evidence, dict) else None
    if configuration is None:
        return None
    if not isinstance(configuration, dict):
        raise FeatureJoinError(
            DECLARED_INTERVAL_INVALID, f"{EVIDENCE_FILE} configuration is not an object"
        )
    value = configuration.get("sampling_interval_seconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise FeatureJoinError(
            DECLARED_INTERVAL_INVALID,
            f"{EVIDENCE_FILE} configuration.sampling_interval_seconds must be a finite number, "
            f"got {value!r}",
        )
    return float(value)


def same_run(manifest_run_id: str, journal_run_id: UUID) -> bool:
    try:
        return UUID(manifest_run_id) == journal_run_id
    except ValueError:
        return False


def replay_features(directory: Path, evidence_dir: str, run_id: str) -> ReplayFeatures:
    """Compute the features of one replay's own run.

    FeatureJoinError rejects all the replay's steps, or with an EXPERIMENT_DEFECTS code all the
    experiment's labeled steps.
    """
    replay_dir = directory / evidence_dir
    declared = declared_interval_seconds(replay_dir)
    try:
        journal = read_journal(replay_dir / JOURNAL_FILE)
    except ValueError as error:  # ReplayEvidenceError is a ValueError too
        raise FeatureJoinError(
            FEATURE_EVIDENCE_UNREADABLE, f"{JOURNAL_FILE} cannot be read: {error}"
        ) from error
    records = (*journal.events, *journal.samples)
    # read_journal guarantees every record shares one run; it must be the manifest's run.
    if records and not same_run(run_id, records[0].run_id):
        raise FeatureJoinError(
            FEATURE_ROW_MISSING,
            f"{JOURNAL_FILE} records run_id {records[0].run_id}, the manifest names {run_id}",
        )
    try:
        computed = compute_feature_rows(journal.events, journal.samples, declared)
    except ValueError as error:
        raise FeatureJoinError(
            FEATURE_ROW_MISSING, f"features cannot be computed: {error}"
        ) from error
    return ReplayFeatures(
        by_step={row.step_id: row for row in computed.rows},
        declared_interval_seconds=declared,
        configuration_sha256=journal.configuration_sha256,
        recorded_at=min((event.observed_at for event in journal.events), default=None),
        source_sha256=replay_source_sha256(directory, evidence_dir),
    )


def joined_feature(label: dict, features: ReplayFeatures) -> FeatureRow:
    """The step's feature row, refused when its timings differ from the label's."""
    feature = features.by_step.get(label["step_id"])
    if feature is None:
        raise FeatureJoinError(
            FEATURE_ROW_MISSING, f"{JOURNAL_FILE} has no terminal {FEATURE_VERSION} row"
        )
    for name in JOINED_TIMINGS:
        labeled, measured = label[name], getattr(feature.timings, name)
        if (labeled is None) != (measured is None) or (
            labeled is not None
            and measured is not None
            and not math.isclose(
                labeled, measured, rel_tol=0, abs_tol=FEATURE_TIMING_TOLERANCE_SECONDS
            )
        ):
            raise FeatureJoinError(
                FEATURE_TIMING_MISMATCH,
                f"{name} is {labeled!r} in {MANIFEST_FILE}, {measured!r} in {JOURNAL_FILE}",
            )
    return feature


def counter_delta(value: float | None) -> int | float | None:
    """The throttling delta as an int64 when integral; otherwise left for the schema to reject."""
    return int(value) if value is not None and value.is_integer() else value


def release_row(
    experiment: Experiment, label: dict, feature: FeatureRow, features: ReplayFeatures
) -> dict:
    values = feature.features
    return {
        "run_id": label["run_id"],
        "step_id": label["step_id"],
        "task_id": label["task_id"],
        "kind": REPLAY_KIND,
        "source": "measured",
        "recorded_at": features.recorded_at,
        "queue_seconds": values.queue_seconds,
        "execution_seconds": values.execution_seconds,
        "duration_seconds": values.duration_seconds,
        "cpu_seconds": values.cpu_seconds,
        "queue_fraction": values.queue_fraction,
        "workload_iterations": None,
        "tests_per_step": None,
        "worker_capacity": None,
        "cpu_pressure_max_avg10": values.cpu_pressure_max_avg10,
        "cpu_pressure_available": values.cpu_pressure_available,
        "cgroup_throttled_usec_delta": counter_delta(values.cgroup_throttled_usec_delta),
        "cgroup_available": values.cgroup_available,
        "telemetry_sample_count": values.telemetry_sample_count,
        "telemetry_coverage": values.telemetry_coverage,
        "label_resource": label["label_resource"],
        "label_excess_fraction": label["label_excess_fraction"],
        "rule_prediction": None,  # replay evidence has no rule output
        "rule_version": None,
        "condition": ROLE_CONDITIONS[label["role"]],
        "workload": f"agent replay {experiment.task_id}",
        "injection_method": label["injection_method"],
        "agent_model": label["agent_model"],
        "run_passed": experiment.passed,  # None stays None and fails the schema
        "failed_checks": label["failed_checks"],
        "telemetry_status": feature.telemetry.status,
        "sampling_interval_seconds": features.declared_interval_seconds,
        "measured_sampling_interval_seconds": feature.telemetry.observed_interval_seconds,
        "baseline_mean_seconds": label["baseline_mean_seconds"],
        "noise_threshold_seconds": label["noise_threshold_seconds"],
        "label_basis": label["label_basis"],
        "label_version": label["label_version"],
        "schema_version": SCHEMA_VERSION,
        "source_sha256": features.source_sha256,
        "feature_version": FEATURE_VERSION,
        "task_group": label["task_group"],
        "experiment": label["experiment"],
        "plan_sha256": label["plan_sha256"],
        "replay_index": label["replay_index"],
        "step_type": label["step_type"],
        "replay_role": label["role"],
        "replay_condition": label["condition"],
        "label_resources": label["label_resources"],
        "control_exceeded_noise": label["control_exceeded_noise"],
        "configuration_sha256": features.configuration_sha256,
    }


def experiment_rows(
    directory: Path, config: ReplayLabelConfig
) -> tuple[list[dict], dict[str, list[Rejection]]]:
    """Release rows of one experiment, ordered by (replay_index, step_id), and rejections.

    Rejections are grouped by REPLAY_STAGES. Unreadable experiment evidence raises
    ReplayEvidenceError (or OSError) instead of rejecting.
    """
    root = directory.resolve()
    experiment = read_experiment(root)
    labels, label_rejections = label_experiment(experiment, config)
    rejections: dict[str, list[Rejection]] = {stage: [] for stage in REPLAY_STAGES}
    rejections[LABEL_STAGE].extend(label_rejections)

    def reject(stage: str, label: dict, reason: str) -> None:
        where = (
            f"experiment {experiment.name} replay {label['replay_index']:02d} "
            f"step {label['step_id']}"
        )
        logger.warning("Rejected %s at %s: %s", where, stage, reason)
        rejections[stage].append(Rejection(where, reason))

    directories = evidence_dirs(root) if labels else {}
    replays = [
        (index, list(group))
        for index, group in groupby(labels, key=lambda label: label["replay_index"])
    ]
    # Every labeled replay is read before any row is kept, so a read defect in one replay can
    # reject the whole experiment.
    computed: dict[int, ReplayFeatures | FeatureJoinError] = {}
    for index, replay_labels in replays:
        if index not in directories:
            raise ReplayEvidenceError(f"{MANIFEST_FILE} changed while it was being read")
        try:
            computed[index] = replay_features(root, directories[index], replay_labels[0]["run_id"])
        except FeatureJoinError as error:
            if error.code in EXPERIMENT_DEFECTS:
                reason = f"{error.code}: replay {index:02d} {error.detail}"
                for label in labels:
                    reject(FEATURE_STAGE, label, reason)
                return [], rejections
            computed[index] = error
    rows = []
    for index, replay_labels in replays:
        features = computed[index]
        if isinstance(features, FeatureJoinError):
            for label in replay_labels:
                reject(FEATURE_STAGE, label, str(features))
            continue
        for label in replay_labels:
            try:
                row = release_row(experiment, label, joined_feature(label, features), features)
            except FeatureJoinError as error:
                reject(FEATURE_STAGE, label, str(error))
                continue
            try:
                validate_rows([row])
            except SchemaError as error:
                reject(SCHEMA_STAGE, label, str(error))
                continue
            rows.append(row)
    rows.sort(key=lambda row: (row["replay_index"], row["step_id"]))
    return rows, rejections


def rejection_code(stage: str, reason: str) -> str:
    """The code a rejection starts with; for schema rejections, the failing column."""
    if stage == SCHEMA_STAGE:
        # validate_rows reasons read "Row 0 (<step>): <column> <problem>".
        return reason.partition(": ")[2].split(" ", 1)[0]
    return reason.partition(":")[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment", type=Path, help="replay experiment directory")
    args = parser.parse_args(argv)
    try:
        rows, rejections = experiment_rows(args.experiment, load_replay_config())
    except (ReplayEvidenceError, LabelingError, OSError) as error:
        print(f"Replay rows failed: {error}", file=sys.stderr)
        return 1
    counts = {
        stage: dict(sorted(Counter(rejection_code(stage, r.reason) for r in found).items()))
        for stage, found in rejections.items()
    }
    print(json.dumps({"rows": len(rows), "rejections": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
