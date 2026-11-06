"""Build a versioned step dataset release from recorded evidence.

    python -m agentwatch_data.build --source local:data/evidence --output data/datasets

Rows come from recorded runs (local or API sources) and from replay experiments (directories
holding experiment.json, found under local sources only): run rows first, then replay rows.
A release is an immutable directory named by its dataset version, holding steps.parquet,
schema.json (the column contract), and manifest.json, which describe only the dataset content.
From steps-v1.3 the manifest also carries content statistics: counts per label version,
feature version, kind and label resource, and the null count of every column.
The version is derived from the rows, the contract, and the release format, so rebuilding from
the same evidence with the same rules and code reuses the same release, while a release in an
older format or with an older contract is left untouched and superseded by a new version.

Every successful build, new or reused, also writes its own report to builds/<build id>.json
with that build's sources, duplicate copies, and rejected runs. A rebuild that reuses a
release therefore still records evidence rejected this time.

A release is published only when a committed build report names it; consumers select
releases with published_releases(). Builds never delete releases, so concurrent builds cannot
remove each other's datasets. A build whose report cannot be committed fails and may leave an
unpublished release directory, which the next successful build verifies and publishes. A
failed build otherwise writes nothing:
unreadable sources, invalid rules, empty results, and defects in the builder itself stop it,
while untrustworthy individual runs, including runs whose rows break the dataset schema, are
rejected and reported.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from agentwatch_data import __version__
from agentwatch_data.evidence import experiment_files
from agentwatch_data.extract import (
    Extraction,
    ExtractionError,
    RawRun,
    Rejection,
    canonical_sha256,
    combine,
    extract_api,
    extract_local,
    summarize,
)
from agentwatch_data.files import file_sha256, fsync_directory, fsync_file, json_text
from agentwatch_data.flatten import flatten_runs
from agentwatch_data.labels import LabelConfig, LabelingError, label_rows, load_config
from agentwatch_data.replay_evidence import MANIFEST_FILE as EXPERIMENT_MANIFEST
from agentwatch_data.replay_evidence import ReplayEvidenceError
from agentwatch_data.replay_labels import ReplayLabelConfig, load_replay_config
from agentwatch_data.replay_rows import LABEL_STAGE as REPLAY_LABEL_STAGE
from agentwatch_data.replay_rows import REPLAY_STAGES, experiment_rows
from agentwatch_data.schema import (
    COLUMNS,
    SCHEMA_VERSION,
    Column,
    SchemaError,
    contract,
    validate_rows,
)

logger = logging.getLogger("agentwatch_data.build")

DATASET_FILE, MANIFEST_FILE, BUILDS_DIR = "steps.parquet", "manifest.json", "builds"
SCHEMA_FILE = "schema.json"  # the column contract, for tools that cannot import this package
# The files a release holds. Format 1 (before schema.json) had only the Parquet and manifest;
# bump this whenever the set of files or the manifest's meaning changes.
RELEASE_FORMAT = 2
# Schema versions whose manifests predate content statistics. Their releases are checked
# against the manifest they shipped, so it must stay recomputable in its original shape.
MANIFEST_WITHOUT_STATISTICS = frozenset({"steps-v1", "steps-v1.1", "steps-v1.2"})
NO_RESOURCE = "none"  # how an empty label_resources (no confirmed resource) is counted
CONDITION_ORDER = {"baseline": 0, "clean": 1, "contended": 2}
ARROW_TYPES = {
    "string": pa.string(),
    "float64": pa.float64(),
    "int64": pa.int64(),
    "bool": pa.bool_(),
    "timestamp[us, tz=UTC]": pa.timestamp("us", tz="UTC"),
}


def arrow_schema(columns: tuple[Column, ...]) -> pa.Schema:
    return pa.schema(
        [pa.field(column.name, ARROW_TYPES[column.dtype], column.nullable) for column in columns]
    )


ARROW_SCHEMA = arrow_schema(COLUMNS)
ALL_COLUMNS = frozenset(column.name for column in COLUMNS)
# Written explicitly rather than left to pyarrow's defaults, so the bytes of a release depend
# only on its rows, this code, and the locked pyarrow version.
PARQUET_OPTIONS = {
    "compression": "snappy",
    "version": "2.6",
    "data_page_version": "1.0",
    "use_dictionary": True,
    "write_statistics": True,
}
RELEASE_NAME = re.compile(r"[a-z]+-v\d+\.\d+\+[0-9a-f]{12}")


class BuildError(RuntimeError):
    """The build cannot produce a trustworthy release; nothing was written."""


@dataclass(frozen=True)
class BuildResult:
    release: Path  # the immutable release directory
    report: Path  # this build's report: sources, duplicates, and rejections
    reused: bool  # the release already existed and was verified, not rewritten


@dataclass(frozen=True)
class Source:
    type: Literal["local", "api"]
    location: str

    @classmethod
    def parse(cls, text: str) -> "Source":
        kind, separator, location = text.partition(":")
        if not separator or kind not in ("local", "api") or not location.strip():
            raise ValueError(f"Source {text!r} must be local:<directory> or api:<url>")
        return cls(kind, location)

    def __str__(self) -> str:
        return f"{self.type}:{self.location}"


class Redactor:
    """Replace host paths in anything written to a release with short source labels.

    Local sources are extracted by their resolved path, so every origin and message names
    one exact form of the path. Labels use only the directory name, never the path.
    """

    def __init__(self, sources: list[Source]):
        local = [source for source in sources if source.type == "local"]
        names = Counter(Path(source.location).resolve().name for source in local)
        self.labels: dict[Source, str] = {}
        replacements = {}
        for index, source in enumerate(local, start=1):
            resolved = Path(source.location).resolve()
            name = resolved.name or "root"
            label = f"<local:{name}>" if names[resolved.name] == 1 else f"<local:{name}#{index}>"
            self.labels[source] = label
            replacements[str(resolved)] = label
        home = home_directory()
        if home is not None:
            replacements[str(home)] = "~"
        # Longest first, so a source inside the home directory keeps its own label.
        self.replacements = sorted(replacements.items(), key=lambda item: -len(item[0]))

    def label(self, source: Source) -> str:
        return self.labels.get(source, str(source))

    def __call__(self, text: str) -> str:
        for path, label in self.replacements:
            text = text.replace(path, label)
        return text


def home_directory() -> Path | None:
    """The resolved home directory, or None where there is none worth redacting.

    Containers may run as a user with no home (Path.home raises), or with HOME=/, which
    would turn every slash in a report into "~".
    """
    try:
        home = Path.home().resolve()
    except (RuntimeError, OSError):
        return None
    return None if home == Path(home.anchor) else home


def extract(source: Source) -> Extraction:
    if source.type == "local":
        return extract_local(Path(source.location).resolve())
    return extract_api(source.location)


def find_experiments(root: Path) -> list[Path]:
    """Every non-hidden directory under root holding experiment.json, without following links.

    An experiment's own subdirectories are not searched. Linked and unreadable directories are
    not reported here: extract_local walks the same trees by the same rules, without stopping
    at experiments, and has already rejected each of them by the time a build gets here.
    """
    found = []
    for current, directories, files in os.walk(root):  # links not followed
        directories[:] = sorted(name for name in directories if not name.startswith("."))
        if EXPERIMENT_MANIFEST in files:
            found.append(Path(current))
            directories[:] = []
    return found


def experiment_hashes(directory: Path) -> dict[str, str]:
    """The SHA-256 of each evidence file of an experiment, by relative POSIX path."""
    return {name: file_sha256(directory / name) for name in experiment_files(directory)}


def experiment_directories(
    sources: list[Source], redact: Redactor
) -> tuple[list[Path], list[dict], int]:
    """The replay experiments under the local sources, sorted by name, the rejection records,
    and the number of identical copies dropped. The API serves runs only.

    An experiment's directory name identifies its rows (the experiment column), so a name
    found more than once is kept once when every copy holds identical files; otherwise every
    copy is rejected, since nothing says which one is the evidence.
    """
    copies: dict[str, list[Path]] = {}
    for source in sources:
        if source.type == "local":
            for directory in find_experiments(Path(source.location).resolve()):
                copies.setdefault(directory.name, []).append(directory)
    found, rejected, duplicates = [], [], 0
    for name, paths in sorted(copies.items()):
        paths = sorted(set(paths))  # a source listed twice finds the same directory twice
        if len(paths) > 1:
            try:
                hashes = [experiment_hashes(path) for path in paths]
                agree = all(copy == hashes[0] for copy in hashes)
                reason = f"Experiment {name} has conflicting evidence in sources"
            except (OSError, ValueError) as error:
                agree = False
                reason = f"Experiment {name} has copies that cannot be compared: {summarize(error)}"
            if not agree:
                for path in paths:
                    logger.warning("Rejected experiment evidence from %s: %s", path, reason)
                    rejected.append(Rejection(str(path), reason))
                continue
            duplicates += len(paths) - 1
        found.append(paths[0])
    return found, rejection_records("extract", rejected, redact), duplicates


def step_order(row: dict) -> tuple:
    step_id = row["step_id"]
    repeat = int(step_id.rsplit("-", 1)[1]) if row["condition"] == "baseline" else 0
    return row["run_id"], CONDITION_ORDER[row["condition"]], repeat


def rows_digest(rows: list[dict]) -> str:
    """Identify the dataset content independently of file encoding."""
    serializable = [
        # In UTC, as Parquet stores it: the same moment written with another offset (the API
        # returns its database session's time zone) must not change the version.
        row
        | {
            "recorded_at": row["recorded_at"].astimezone(UTC).isoformat()
            if row["recorded_at"]
            else None
        }
        for row in rows
    ]
    return canonical_sha256(serializable)


def release_digest(rows_sha256: str, release_contract: dict | None = None) -> str:
    """Identify a whole release: its rows, the column contract it ships, and its format.

    Releases are never rewritten, so anything a release must contain is part of its identity;
    otherwise an older release with the same rows would be reused without it. A release
    checked later passes the contract it shipped, which may predate the current one.
    """
    shipped = contract() if release_contract is None else release_contract
    return canonical_sha256(
        {"release_format": RELEASE_FORMAT, "contract": shipped, "rows_sha256": rows_sha256}
    )


def release_manifest(
    rows: list[dict],
    version: str,
    digest: str,
    label_version: str,
    parquet_sha256: str,
    schema_sha256: str,
    schema_version: str = SCHEMA_VERSION,
) -> dict:
    """A release's manifest: content-derived fields only, so its rows determine it."""
    manifest = {
        "release_format": RELEASE_FORMAT,
        "dataset_version": version,
        "schema_version": schema_version,
        "label_version": label_version,
        "builder_version": __version__,
        "rows": len(rows),
        "runs": len({row["run_id"] for row in rows}),
        "tasks": len({row["task_id"] for row in rows}),
        "rows_sha256": digest,
        "parquet_sha256": parquet_sha256,
        "schema_sha256": schema_sha256,
        "label_resource": counts(rows, "label_resource"),
        "label_basis": counts(rows, "label_basis"),
        "condition": counts(rows, "condition"),
        "telemetry_status": counts(rows, "telemetry_status"),
    }
    if schema_version in MANIFEST_WITHOUT_STATISTICS:
        return manifest
    return manifest | {
        "label_versions": counts(rows, "label_version"),
        "feature_version": counts(rows, "feature_version"),
        "kind": counts(rows, "kind"),
        "label_resources": resource_counts(rows),
        # Every column, zeros included, so a column that is never null is shown as such. Built
        # rows hold every column; a value absent from a row (only rows that break the contract,
        # which integrity refuses separately) is missing too, never a measured value.
        "missing": {
            column.name: sum(row.get(column.name) is None for row in rows) for column in COLUMNS
        },
    }


def resource_counts(rows: list[dict]) -> dict:
    """Rows per confirmed resource; a mixed row counts once for each of its resources."""
    found = Counter(
        resource
        for row in rows
        for resource in (
            row["label_resources"].split(",") if row["label_resources"] else [NO_RESOURCE]
        )
    )
    return dict(sorted(found.items()))


def release_label_version(rows: list[dict]) -> str:
    """Every label version the rows were labeled under, sorted and comma-separated."""
    return ",".join(sorted({row["label_version"] for row in rows}))


def contract_text() -> str:
    return json_text(contract())


def rejection_records(stage: str, rejections: list[Rejection], redact: Redactor) -> list[dict]:
    return [
        {"stage": stage, "origin": redact(item.origin), "reason": redact(item.reason)}
        for item in rejections
    ]


def counts(rows: list[dict], column: str) -> dict:
    return dict(sorted(Counter(str(row[column]) for row in rows).items()))


def build(
    sources: list[Source],
    output_dir: Path,
    rules: Path | None = None,
    now: datetime | None = None,
    replay_rules: Path | None = None,
) -> BuildResult:
    """Build or reuse a release under output_dir, and write this build's report."""
    if not sources:
        raise BuildError("At least one source is required")
    redact = Redactor(sources)
    try:
        config = load_config(rules)
    except LabelingError as error:
        raise BuildError(f"Label rules are invalid: {error}") from error
    try:
        replay_config = load_replay_config(replay_rules)
    except LabelingError as error:
        raise BuildError(redact(f"Replay label rules are invalid: {error}")) from error

    extractions, source_summaries = [], []
    for source in sources:
        try:
            extraction = extract(source)
        except ExtractionError as error:
            raise BuildError(redact(f"Source {source} could not be read: {error}")) from error
        extractions.append(extraction)
        source_summaries.append(
            {
                "source": redact.label(source),
                "runs": len(extraction.runs),
                "rejections": len(extraction.rejections),
            }
        )
    merged = combine(*extractions)
    experiments, experiment_rejections, experiment_duplicates = experiment_directories(
        sources, redact
    )
    labeled, stage_rejections = derive_rows(merged.runs, config, experiments, replay_config)
    rejections = rejection_records("extract", merged.rejections, redact) + experiment_rejections
    for stage, items in stage_rejections.items():
        rejections += rejection_records(stage, items, redact)
    if not labeled:
        raise BuildError(
            f"No usable rows: every run and experiment was rejected ({len(rejections)} rejections)"
        )
    label_version = release_label_version(labeled)

    digest = rows_digest(labeled)
    version = f"{SCHEMA_VERSION}+{release_digest(digest)[:12]}"
    final = [row | {"dataset_version": version} for row in labeled]
    try:  # every run already passed; a failure here is a defect in the builder itself
        validate_rows(final, ALL_COLUMNS)
    except SchemaError as error:
        raise BuildError(f"Release rows violate the dataset schema: {error}") from error

    built_at = now or datetime.now(UTC)
    build_id = f"{built_at:%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    release = output_dir / version
    report_path = output_dir / BUILDS_DIR / f"{build_id}.json"

    def report(reused: bool) -> dict:
        return {
            "build_id": build_id,
            "built_at": built_at.isoformat(),
            "dataset_version": version,
            "release": version,
            "reused_release": reused,
            "label_version": label_version,
            "builder_version": __version__,
            "rows": len(final),
            "runs": len({row["run_id"] for row in final}),
            "experiments": len(experiments),
            "sources": source_summaries,
            "duplicate_copies": len(merged.duplicates) + experiment_duplicates,
            "rejections": rejections,
        }

    # Publication is the report commit. The report is staged before the release directory
    # appears, so most write failures happen before anything is visible. Nothing is ever
    # deleted after publication starts: a release another build may already have verified
    # and reported must survive this build's failure.
    staged = None
    try:
        reused = release.exists() and verify_existing(release, digest)
        staged = stage_report(report_path, report(reused))
        if not reused:
            reused = write_release(release, final, version, digest, label_version)
            if reused:  # a concurrent build published the identical release first
                staged = stage_report(report_path, report(reused))
        os.replace(staged, report_path)
        staged = None
        fsync_directory(report_path.parent)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    logger.info(
        "%s %s: %d rows, %d rejections; report %s",
        "Reused" if reused else "Built",
        version,
        len(final),
        len(rejections),
        report_path.name,
    )
    return BuildResult(release, report_path, reused)


def derive_rows(
    runs: list[RawRun],
    config: LabelConfig,
    experiments: Sequence[Path] = (),
    replay_config: ReplayLabelConfig | None = None,
) -> tuple[list[dict], dict[str, list[Rejection]]]:
    """The rows runs and experiments produce, in release order, with each stage's rejections.

    Run rows come first, sorted by step_order, then replay rows sorted by (experiment,
    replay_index, step_id). Rejections are keyed by stage: flatten, label and schema for runs,
    then REPLAY_STAGES. replay_config defaults to the packaged labels-v2 rules.

    The one derivation of release rows from evidence: the build uses it, and integrity uses
    it to prove a release is what its stored evidence builds.
    """
    rows, flatten_rejections = flatten_runs(runs)
    labeled, label_rejections = label_rows(rows, config)
    origins = {run.run_id: run.origin for run in runs}
    labeled, schema_rejections = validate_runs(labeled, origins)
    labeled.sort(key=step_order)
    replayed, replay_rejections = derive_replay_rows(experiments, replay_config)
    return labeled + replayed, {
        "flatten": flatten_rejections,
        "label": label_rejections,
        "schema": schema_rejections,
        **replay_rejections,
    }


def derive_replay_rows(
    experiments: Sequence[Path], config: ReplayLabelConfig | None
) -> tuple[list[dict], dict[str, list[Rejection]]]:
    """Replay rows of every experiment, sorted, and rejections keyed by REPLAY_STAGES.

    An experiment whose evidence cannot be read is rejected whole at the label stage, so the
    others still build and the report says why it contributed nothing.
    """
    rejections: dict[str, list[Rejection]] = {stage: [] for stage in REPLAY_STAGES}
    if not experiments:
        return [], rejections
    rules = config if config is not None else load_replay_config()
    rows = []
    for directory in experiments:
        try:
            found, stages = experiment_rows(directory, rules)
        except (ReplayEvidenceError, OSError) as error:
            reason = f"Experiment cannot be read: {summarize(error)}"
            logger.warning("Rejected experiment %s: %s", directory, reason)
            rejections[REPLAY_LABEL_STAGE].append(Rejection(str(directory), reason))
            continue
        rows.extend(found)
        for stage, items in stages.items():
            rejections[stage].extend(items)
    rows.sort(key=lambda row: (row["experiment"], row["replay_index"], row["step_id"]))
    return rows, rejections


def parquet_bytes(rows: list[dict], schema: pa.Schema = ARROW_SCHEMA) -> bytes:
    """steps.parquet for rows, exactly as a release stores it."""
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), buffer, **PARQUET_OPTIONS)
    return buffer.getvalue().to_pybytes()


def validate_runs(rows: list[dict], origins: dict[str, str]) -> tuple[list[dict], list[Rejection]]:
    """Check each run's rows against the schema; a run that fails is rejected, not fatal.

    The ingestion contract is looser than the dataset schema (it allows an empty
    rule_version, or PSI above 100), so one such run must not stop every later build.
    """
    runs: dict[str, list[dict]] = {}
    for row in rows:
        runs.setdefault(row["run_id"], []).append(row)
    valid, rejections = [], []
    for run_id, run_rows in runs.items():
        try:
            validate_rows(run_rows)
        except SchemaError as error:
            logger.warning("Rejected run %s at schema validation: %s", run_id, error)
            rejections.append(Rejection(origins.get(run_id) or f"run {run_id}", str(error)))
            continue
        valid.extend(run_rows)
    return valid, rejections


def write_release(
    release: Path, rows: list[dict], version: str, digest: str, label_version: str
) -> bool:
    """Write a release atomically. Returns True if a concurrent build had already written it.

    The manifest holds only content-derived fields and Parquet is written with declared
    options, so with the same code and locked dependencies, rebuilding the same evidence into
    an empty output directory reproduces both files byte for byte. Build events live in
    reports. Files are fsynced before the rename, so after a crash a release is whole or
    absent.
    """
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=release.parent))
    try:
        # mkdtemp creates the directory owner-only; releases are read by other tools
        # (DVC, Airflow workers, training jobs) that may run as other users.
        staging.chmod(0o755)
        dataset = staging / DATASET_FILE
        dataset.write_bytes(parquet_bytes(rows))
        (staging / SCHEMA_FILE).write_text(contract_text(), encoding="utf-8")
        manifest = release_manifest(
            rows,
            version,
            digest,
            label_version,
            file_sha256(dataset),
            file_sha256(staging / SCHEMA_FILE),
        )
        (staging / MANIFEST_FILE).write_text(json_text(manifest), encoding="utf-8")
        # Durable before the rename publishes it: after a crash a release is whole or absent.
        for name in (DATASET_FILE, SCHEMA_FILE, MANIFEST_FILE):
            fsync_file(staging / name)
        fsync_directory(staging)
        try:
            os.replace(staging, release)  # atomic on one filesystem: the release appears whole
        except OSError:
            if not release.exists():
                raise
            return verify_existing(release, digest)  # a concurrent build finished first
        fsync_directory(release.parent)
        return False
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def stage_report(path: Path, report: dict) -> Path:
    """Write a build report next to its final name, readable by other users, uncommitted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    try:
        staged.write_text(json_text(report), encoding="utf-8")
        staged.chmod(0o644)
        fsync_file(staged)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def published_releases(output_dir: Path) -> list[Path]:
    """Releases named by at least one committed build report, by their latest report.

    The last one is the release the most recent successful build created or reused, so
    `make dataset` followed by `make validate-dataset` checks what was just built. A release
    directory without a report came from a build that failed before publishing; it is not
    published until a later build verifies it and commits a report. A report naming anything
    but a real release directory of that version publishes nothing.
    """
    builds = output_dir / BUILDS_DIR
    latest_report: dict[str, tuple[str, str]] = {}
    for path in builds.glob("*.json") if builds.is_dir() else []:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # an unreadable report publishes nothing
        if not isinstance(report, dict):
            continue
        name, built_at = report.get("release"), report.get("built_at")
        if not isinstance(name, str) or not is_release(output_dir, name):
            continue
        key = (built_at if isinstance(built_at, str) else "", path.name)
        latest_report[name] = max(latest_report.get(name, key), key)
    return [output_dir / name for name in sorted(latest_report, key=latest_report.get)]


def is_release(output_dir: Path, name: str) -> bool:
    """A release directory named like a version, not a link, whose manifest says so."""
    # The name pattern also keeps a report from pointing outside the output directory.
    if not RELEASE_NAME.fullmatch(name):
        return False
    directory = output_dir / name
    try:
        if directory.is_symlink() or not directory.is_dir():
            return False
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and manifest.get("dataset_version") == name


def verify_existing(release: Path, digest: str) -> bool:
    """A release with this version already exists; confirm it is intact and identical."""
    expected_contract = hashlib.sha256(contract_text().encode()).hexdigest()
    try:
        manifest = json.loads((release / MANIFEST_FILE).read_text(encoding="utf-8"))
        intact = (
            manifest["release_format"] == RELEASE_FORMAT
            and manifest["rows_sha256"] == digest
            and manifest["parquet_sha256"] == file_sha256(release / DATASET_FILE)
            and manifest["schema_sha256"] == expected_contract
            and file_sha256(release / SCHEMA_FILE) == expected_contract
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise BuildError(f"Existing release {release.name} is unreadable: {error}") from error
    if not intact:
        raise BuildError(f"Existing release {release.name} does not match its manifest")
    logger.info("Release %s already exists and is intact", release.name)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="local:<directory> or api:<url>; repeat to combine sources",
    )
    parser.add_argument("--output", type=Path, default=Path("data/datasets"))
    parser.add_argument("--rules", type=Path, help="label rules (default: packaged labels.toml)")
    parser.add_argument(
        "--replay-rules",
        type=Path,
        help="replay label rules (default: packaged labels-v2.toml)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sources = [Source.parse(text) for text in args.source]
        result = build(sources, args.output, args.rules, replay_rules=args.replay_rules)
    except (ValueError, BuildError) as error:
        print(f"Dataset build failed: {error}", file=sys.stderr)
        return 1
    print(f"release: {result.release}")
    print(f"report: {result.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
