"""Verify the evidence store and dataset store before they are versioned.

    python -m agentwatch_data.integrity --evidence data/evidence --datasets data/datasets \\
        [--dvc-status <output of dvc data status --granular --unchanged --json>]

A version should never record damage, and every versioned release must be rebuildable from
versioned evidence. Refused, with every problem listed:

- an evidence entry that is not a valid run stored under its own <kind prefix>-<run_id> name,
  holds anything besides results.json and samples.jsonl, or repeats another entry's run ID;
- a replay experiment entry (replay-...) that holds anything besides its experiment evidence
  files, cannot be read (every journal strictly), is not stored under the name its own
  experiment.json gives it, or names a replay run ID that another entry (a run or another
  experiment's replay) already has: a run ID identifies one run's evidence in every release;
- a release that does not match its manifest and name (a superseded older release format is
  checked against what that format recorded), whose rows break the frozen column contract of
  its schema version, or that uses a run (or replay) whose exact evidence is not in the
  evidence store;
- a release not yet versioned that is not exactly what its stored evidence builds: its runs
  and the experiments its rows name are rebuilt under the current label rules (run rules and
  replay rules) and must reproduce its rows (and, for the current format, its Parquet bytes).
  A versioned release was rebuilt when it was versioned, and the append-only check keeps it
  unchanged, so a later rule or code change never blocks versioning;
- a build report that is unreadable or names a release that does not exist;
- a quality-gate attempt that is damaged (an unknown verdict), whose statistics do not match
  the hash its result recorded, or that holds unexpected files;
- anything stored as a symbolic link, or as a folder where a file belongs (or the reverse): a
  version must hold the content itself, not a pointer to files outside the stores, and
  anything that cannot be read, which is reported rather than stopping the check;
- a split file that does not hash to its id, sit beside the release it names, or refer to a
  release that exists;
- with --dvc-status, any file DVC already versioned that was deleted or modified: both stores
  are append-only, so a failed verdict, a release, or evidence cannot quietly disappear.

Hidden entries are staging left by a running or killed command, or tool state; DVC excludes
them from versions and the dataset build skips them, so they are only counted.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from agentwatch_data.build import (
    BUILDS_DIR,
    DATASET_FILE,
    MANIFEST_FILE,
    RELEASE_NAME,
    SCHEMA_FILE,
    arrow_schema,
    derive_rows,
    parquet_bytes,
    release_digest,
    release_label_version,
    release_manifest,
    rows_digest,
)
from agentwatch_data.evidence import (
    EVIDENCE_FILES,
    REPLAY_PREFIX,
    EvidenceError,
    entry_name,
    experiment_entry_name,
    hidden_entries,
    read_bounded,
    read_entry,
    read_stored_experiment,
)
from agentwatch_data.extract import RawRun, summarize
from agentwatch_data.files import file_sha256
from agentwatch_data.labels import LabelConfig, load_config
from agentwatch_data.replay_evidence import EVIDENCE_FILE, JOURNAL_FILE, ReplayEvidenceError
from agentwatch_data.replay_evidence import MANIFEST_FILE as EXPERIMENT_MANIFEST
from agentwatch_data.replay_labels import ReplayLabelConfig, load_replay_config
from agentwatch_data.replay_rows import evidence_dirs, replay_source_sha256
from agentwatch_data.schema import Column, SchemaError, columns_from, validate_rows
from agentwatch_data.splits import (
    RESULT_FILE,
    SPLITS_DIR,
    VALIDATIONS_DIR,
    SplitError,
    latest_verdict,
    read_object,
    read_split_file,
    verify_release,
)

STATISTICS_FILE = "statistics.pbtxt"  # written beside result.json by the quality gate
RELEASE_FILES = {DATASET_FILE, SCHEMA_FILE, MANIFEST_FILE}
SUPERSEDED_RELEASE_FILES = {DATASET_FILE, MANIFEST_FILE}  # format 1, before schema.json
# Recorded by the builder that wrote a release, not derived from its rows.
NOT_CONTENT_DERIVED = {"builder_version"}
ATTEMPT_FILES = {RESULT_FILE, STATISTICS_FILE}
# Stages that reject a run's rows. Replay stages reject some steps of a valid experiment by
# design (a warm-up, a failed control), so a rebuild compares the rows, not their absence.
RUN_STAGES = ("flatten", "label", "schema")
# Every column contract a release was ever built under, by schema version, exactly as the
# builder shipped it. A schema change freezes the new contract beside the old ones, so the
# releases of every version stay verifiable (test_dataset_schema enforces this).
CONTRACTS = {
    path.stem: path.read_text(encoding="utf-8")
    for path in (Path(__file__).parent / "contracts").glob("*.json")
}


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    runs: int = 0
    releases: int = 0
    superseded: int = 0
    reports: int = 0
    attempts: int = 0
    splits: int = 0
    hidden: int = 0
    evidence: dict[str, RawRun] = field(default_factory=dict)  # by run ID
    experiment_entries: int = 0
    experiments: dict[str, Path] = field(default_factory=dict)  # by entry name
    # The source_sha256 every row of a stored replay carries, by replay run ID.
    replay_digests: dict[str, str] = field(default_factory=dict)
    rebuilt: int = 0  # releases rebuilt from evidence (the rest were already versioned)


@dataclass(frozen=True)
class Release:
    """A release whose files, contract, and identity checked out."""

    rows: list[dict]
    manifest: dict
    columns: tuple[Column, ...]
    current: bool  # False for a superseded format


def is_file(path: Path) -> bool:
    """A regular file, not a link to one: a version must hold the content, not a pointer."""
    return path.is_file() and not path.is_symlink()


def is_directory(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def visible(directory: Path, report: Report) -> list[Path]:
    """Entries of directory, sorted, with hidden entries counted and skipped."""
    try:
        listed = sorted(directory.iterdir())
    except OSError as error:
        report.problems.append(f"{directory}: cannot be listed: {error.strerror}")
        return []
    entries = []
    for path in listed:
        if path.name.startswith("."):
            report.hidden += 1
        else:
            entries.append(path)
    return entries


def check_evidence(store: Path, report: Report) -> None:
    stored_as: dict[str, str] = {}  # entry name by run ID, runs and replays alike
    for path in visible(store, report):
        label = f"evidence/{path.name}"
        if path.name.startswith(f"{REPLAY_PREFIX}-"):
            check_experiment_entry(path, report, stored_as)
            continue
        if not is_directory(path):
            report.problems.append(f"{label}: not a run directory")
            continue
        extra = sorted(
            child.name
            for child in visible(path, report)
            if child.name not in EVIDENCE_FILES or not is_file(child)
        )
        if extra:
            report.problems.append(f"{label}: holds more than run evidence: {extra}")
            continue
        try:
            run = read_entry(path)
        except EvidenceError as error:
            report.problems.append(f"{label}: {error}")
            continue
        if path.name != entry_name(path, run):
            report.problems.append(f"{label}: stored under the wrong name")
            continue
        if run.run_id in stored_as:
            report.problems.append(f"{label}: run {run.run_id} is also {stored_as[run.run_id]}")
            continue
        stored_as[run.run_id] = path.name
        report.evidence[run.run_id] = run
        report.runs += 1


def check_experiment_entry(path: Path, report: Report, stored_as: dict[str, str]) -> None:
    """Record a stored replay experiment and its replays' evidence hashes, if it is intact.

    stored_as maps every run ID already accepted to its entry and gains this entry's replay
    run IDs. A replay without both evidence files can produce no rows, so it has no hash.
    """
    label = f"evidence/{path.name}"
    if not is_directory(path):
        report.problems.append(f"{label}: not an experiment directory")
        return
    try:
        experiment, hashes = read_stored_experiment(path)
        # Skipped by read_stored_experiment and counted, as inside run entries.
        report.hidden += len(hidden_entries(path))
        manifest = read_bounded(path / EXPERIMENT_MANIFEST)
        if path.name != experiment_entry_name(manifest, experiment):
            report.problems.append(f"{label}: stored under the wrong name")
            return
        directories = evidence_dirs(path)
        digests = {}
        for replay in experiment.replays:
            if not isinstance(directories.get(replay.index), str):
                raise ReplayEvidenceError(f"{EXPERIMENT_MANIFEST} changed while it was read")
            evidence_dir = PurePosixPath(directories[replay.index])
            files = (f"{evidence_dir}/{EVIDENCE_FILE}", f"{evidence_dir}/{JOURNAL_FILE}")
            if replay.run_id is not None and all(file in hashes for file in files):
                digests[replay.run_id] = replay_source_sha256(path, str(evidence_dir))
    except EvidenceError as error:
        report.problems.append(f"{label}: {error}")
        return
    except (OSError, ValueError) as error:  # changed or unreadable since it was read
        report.problems.append(
            f"{label}: Stored evidence {path.name} is unreadable: {summarize(error)}"
        )
        return
    run_ids = sorted(replay.run_id for replay in experiment.replays if replay.run_id is not None)
    repeated = [run_id for run_id in run_ids if run_id in stored_as]
    if repeated:
        report.problems.append(
            f"{label}: "
            + "; ".join(f"replay run {run_id} is also {stored_as[run_id]}" for run_id in repeated)
        )
        return
    stored_as.update(dict.fromkeys(run_ids, path.name))
    report.replay_digests.update(digests)
    report.experiments[path.name] = path
    report.experiment_entries += 1


def check_release(path: Path, report: Report) -> Release | None:
    """The release, if it is exactly what its rows produce, by the format it declares.

    Matching file hashes only prove the manifest describes the files; a release edited along
    with its manifest would still pass that. So the rows are first held to the frozen column
    contract of their schema version, and then the identity is recomputed from the content,
    as the build computes it: the row digest, the content-addressed name (from the rows and
    the column contract the release ships), and every content-derived manifest field.
    """
    label = f"datasets/{path.name}"
    try:
        manifest = read_object(path / MANIFEST_FILE, "Manifest")
    except SplitError as error:
        report.problems.append(f"{label}: {error}")
        return None
    current = "release_format" in manifest
    expected_files = RELEASE_FILES if current else SUPERSEDED_RELEASE_FILES
    entries = visible(path, report)
    names = {entry.name for entry in entries}
    if names != expected_files or not all(is_file(entry) for entry in entries):
        report.problems.append(f"{label}: holds {sorted(names)}, not {sorted(expected_files)}")
        return None
    try:
        if current:
            verify_release(path)
        elif manifest.get("parquet_sha256") != file_sha256(path / DATASET_FILE):
            raise SplitError(f"Release {path.name} does not match its manifest (no format)")
        # Read once: the identity and the evidence checks share the rows.
        columns = release_columns(path, manifest, current)
        rows = read_rows(path, columns)
        problem = identity_problem(path, manifest, current, rows)
    except SplitError as error:
        problem = str(error)
    except OSError as error:  # hash-consistent so far, but a file cannot be read
        problem = f"cannot be read: {error.strerror}: {Path(error.filename or path).name}"
    if problem:
        report.problems.append(f"{label}: {problem}")
        return None
    if current:
        report.releases += 1
    else:
        report.superseded += 1
    return Release(rows, manifest, columns, current)


def release_columns(path: Path, manifest: dict, current: bool) -> tuple[Column, ...]:
    """The column contract the release's rows must satisfy: the one of its schema version.

    A current release must ship exactly that contract; a superseded one shipped none.
    """
    version = manifest.get("schema_version")
    frozen = CONTRACTS.get(version) if isinstance(version, str) else None
    if frozen is None:
        raise SplitError(f"schema version {version!r} has no frozen column contract")
    if current and (path / SCHEMA_FILE).read_bytes() != frozen.encode():
        raise SplitError(f"{SCHEMA_FILE} is not the frozen {version} column contract")
    return columns_from(json.loads(frozen))


def read_rows(path: Path, columns: tuple[Column, ...]) -> list[dict]:
    """The release's rows, refused unless they have exactly the contract's columns and values.

    Hash-consistent bytes are not necessarily a release's rows: they may not be Parquet, may
    lack a column the checks rely on, or may hold values the contract forbids.
    """
    try:
        table = pq.read_table(path / DATASET_FILE)
        rows = table.to_pylist()
    except (pa.ArrowException, OSError, ValueError, OverflowError) as error:
        raise SplitError(f"{DATASET_FILE} is not readable Parquet: {error}") from error
    if not table.schema.equals(arrow_schema(columns), check_metadata=False):
        raise SplitError(f"{DATASET_FILE} does not have its contract's columns and types")
    try:
        validate_rows(rows, frozenset(column.name for column in columns), columns)
    except SchemaError as error:
        raise SplitError(f"rows break their column contract: {error}") from error
    return rows


def identity_problem(path: Path, manifest: dict, current: bool, rows: list[dict]) -> str | None:
    """Why the release's content does not produce its name and manifest, or None.

    rows must already satisfy the release's column contract (read_rows checks it).
    """
    for column, declared in (
        ("dataset_version", path.name),
        ("label_version", manifest.get("label_version")),
        ("schema_version", manifest.get("schema_version")),
    ):
        if not isinstance(declared, str):
            return f"rows do not all carry the manifest's {column} {declared!r}"
        # A release may hold rows of several label rules: its label_version names them all.
        expected = set(declared.split(",")) if column == "label_version" else {declared}
        if {row.get(column) for row in rows} != expected:
            return f"rows do not all carry the manifest's {column} {declared!r}"
    digest = rows_digest([{k: v for k, v in row.items() if k != "dataset_version"} for row in rows])
    if current:
        shipped = read_object(path / SCHEMA_FILE, "Column contract")
        name = f"{manifest['schema_version']}+{release_digest(digest, shipped)[:12]}"
    else:
        name = f"{manifest['schema_version']}+{digest[:12]}"
    if name != path.name:
        return f"its content produces {name}, not its name; it was changed after it was built"
    expected = release_manifest(
        rows,
        name,
        digest,
        release_label_version(rows),  # sorted, as the build writes it
        manifest.get("parquet_sha256"),
        manifest.get("schema_sha256"),
        manifest["schema_version"],
    )
    if not current:
        del expected["release_format"], expected["schema_sha256"]
    differing = sorted(
        key
        for key in (expected.keys() | manifest.keys()) - NOT_CONTENT_DERIVED
        if expected.get(key) != manifest.get(key)
    )
    if differing:
        return f"manifest fields {differing} do not match its rows"
    return None


def check_reproducible(
    path: Path,
    release: Release,
    report: Report,
    rules: LabelConfig | None,
    replay_rules: ReplayLabelConfig,
) -> None:
    """Every row's run must have exactly the evidence the row names in the evidence store,
    and, unless rules is None (already versioned), the release must be what it builds.

    A replay row's run is one replay of a stored experiment, so its evidence hash is the
    replay's (Report.replay_digests); run IDs are unique across the store, so each is either.

    Checked row by row: one run's rows must all name the same evidence hash, so a row
    attributed to a different source artifact cannot hide behind its run's other rows.
    """
    digests: dict[str, set] = {}
    for row in release.rows:
        digests.setdefault(row["run_id"], set()).add(row["source_sha256"])
    mixed = sorted(run for run, found in digests.items() if len(found) > 1)
    if mixed:
        report.problems.append(
            f"datasets/{path.name}: rows of one run name different evidence hashes "
            f"(runs {mixed[:3]}{'...' if len(mixed) > 3 else ''})"
        )
        return
    stored = {run: entry.source_sha256 for run, entry in report.evidence.items()}
    stored |= report.replay_digests
    missing = sorted(run for run, [digest] in digests.items() if stored.get(run) != digest)
    if missing:
        report.problems.append(
            f"datasets/{path.name}: built from evidence that is not in the evidence store "
            f"(runs {missing[:3]}{'...' if len(missing) > 3 else ''}), so it cannot be rebuilt"
        )
        return
    if rules is None:
        return
    # Rows of a schema version without the experiment column are run rows only.
    named = sorted({row["experiment"] for row in release.rows if row.get("experiment") is not None})
    unstored = [name for name in named if name not in report.experiments]
    if unstored:
        report.problems.append(
            f"datasets/{path.name}: names experiments that are not in the evidence store "
            f"({unstored[:3]}{'...' if len(unstored) > 3 else ''}), so it cannot be rebuilt"
        )
        return
    report.rebuilt += 1
    runs = [report.evidence[run] for run in digests if run in report.evidence]
    experiments = [report.experiments[name] for name in named]
    problem = rebuild_problem(path, release, runs, experiments, rules, replay_rules)
    if problem:
        report.problems.append(f"datasets/{path.name}: {problem}")


# A release that was never versioned may be removed; see retire.py.
RETIRE = "; retire it with make retire-release RELEASE={name} and build again"


def rebuild_problem(
    path: Path,
    release: Release,
    runs: list[RawRun],
    experiments: list[Path],
    rules: LabelConfig,
    replay_rules: ReplayLabelConfig,
) -> str | None:
    """Why building runs and experiments (its stored evidence) would not reproduce the
    release, or None.

    Agreeing hashes only prove the release names its evidence; this proves the rows were
    derived from it, so an edited value, a dropped row, or a re-encoded file is refused.
    """
    current = sorted({rules.label_version, replay_rules.label_version})
    labeled = sorted({row["label_version"] for row in release.rows})  # identity checked
    outside = [version for version in labeled if version not in current]
    if outside:
        problem = (
            f"it was labeled under {', '.join(outside)}, but this pipeline labels under "
            f"{' and '.join(current)}, so it cannot be rebuilt from its evidence"
        )
    else:
        problem = derived_problem(path, release, runs, experiments, rules, replay_rules)
    return problem and problem + RETIRE.format(name=path.name)


def derived_problem(
    path: Path,
    release: Release,
    runs: list[RawRun],
    experiments: list[Path],
    rules: LabelConfig,
    replay_rules: ReplayLabelConfig,
) -> str | None:
    rows, stage_rejections = derive_rows(runs, rules, experiments, replay_rules)
    refused = [
        f"{stage}: {item.reason}"
        for stage in RUN_STAGES
        for item in stage_rejections.get(stage, [])
    ]
    if refused:
        return f"its evidence no longer builds ({refused[0]})"
    if rows_digest(rows) != release.manifest["rows_sha256"]:
        return "its rows are not what its stored evidence builds; it was edited or built elsewhere"
    if release.current:
        final = [row | {"dataset_version": path.name} for row in rows]
        encoded = parquet_bytes(final, arrow_schema(release.columns))
        if hashlib.sha256(encoded).hexdigest() != release.manifest["parquet_sha256"]:
            return f"its {DATASET_FILE} is not the bytes its rows build"
    return None


def check_datasets(store: Path, report: Report, versioned: set[Path]) -> None:
    # Release directories present, intact or not: a damaged release is reported once, by
    # itself, rather than again as "missing" by its reports, gate attempts, and splits.
    releases = set()
    rules = load_config()
    replay_rules = load_replay_config()
    for path in visible(store, report):
        if path.name in (BUILDS_DIR, VALIDATIONS_DIR, SPLITS_DIR) and is_directory(path):
            continue
        if not RELEASE_NAME.fullmatch(path.name) or not is_directory(path):
            report.problems.append(f"datasets/{path.name}: unexpected entry")
            continue
        releases.add(path.name)
        release = check_release(path, report)
        if release is not None:
            files = {(path / name).resolve() for name in (DATASET_FILE, MANIFEST_FILE)}
            rebuild = None if files <= versioned else rules
            check_reproducible(path, release, report, rebuild, replay_rules)
    check_reports(store / BUILDS_DIR, releases, report)
    check_validations(store, releases, report)
    check_splits(store / SPLITS_DIR, releases, report)


def check_reports(builds: Path, releases: set[str], report: Report) -> None:
    for path in visible(builds, report) if is_directory(builds) else []:
        label = f"datasets/{BUILDS_DIR}/{path.name}"
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if is_file(path) else None
        except (OSError, ValueError, RecursionError):
            value = None
        if path.suffix != ".json" or not isinstance(value, dict):
            report.problems.append(f"{label}: unreadable build report")
            continue
        named = value.get("release")
        if not isinstance(named, str) or named not in releases:
            report.problems.append(f"{label}: names release {named!r}, which is missing")
            continue
        report.reports += 1


def check_validations(store: Path, releases: set[str], report: Report) -> None:
    validations = store / VALIDATIONS_DIR
    for folder in visible(validations, report) if is_directory(validations) else []:
        label = f"datasets/{VALIDATIONS_DIR}/{folder.name}"
        if folder.name not in releases or not is_directory(folder):
            report.problems.append(f"{label}: no such release")
            continue
        try:
            latest_verdict(store / folder.name)
        except SplitError as error:
            report.problems.append(f"{label}: {error}")
            continue
        except OSError as error:
            report.problems.append(f"{label}: cannot be listed: {error.strerror}")
            continue
        for attempt in visible(folder, report):
            check_attempt(attempt, f"{label}/{attempt.name}", report)


def check_attempt(attempt: Path, label: str, report: Report) -> None:
    if not is_directory(attempt):
        report.problems.append(f"{label}: not a gate attempt")
        return
    entries = visible(attempt, report)
    names = {entry.name for entry in entries}
    if names != ATTEMPT_FILES or not all(is_file(entry) for entry in entries):
        report.problems.append(f"{label}: holds {sorted(names)}, not {sorted(ATTEMPT_FILES)}")
        return
    try:
        result = json.loads((attempt / RESULT_FILE).read_text(encoding="utf-8"))
        # The gate hashes the statistics text it wrote; the drift baseline relies on it.
        text = (attempt / STATISTICS_FILE).read_text(encoding="utf-8")
    except (OSError, ValueError, RecursionError) as error:
        report.problems.append(f"{label}: unreadable: {type(error).__name__}")
        return
    if not isinstance(result, dict):
        report.problems.append(f"{label}: {RESULT_FILE} is not a JSON object")
        return
    if hashlib.sha256(text.encode()).hexdigest() != result.get("statistics_sha256"):
        report.problems.append(f"{label}: statistics do not match the hash their result recorded")
        return
    report.attempts += 1


def check_splits(splits: Path, releases: set[str], report: Report) -> None:
    for folder in visible(splits, report) if is_directory(splits) else []:
        if folder.name not in releases or not is_directory(folder):
            report.problems.append(f"datasets/{SPLITS_DIR}/{folder.name}: no such release")
            continue
        for path in visible(folder, report):
            if not is_file(path):
                report.problems.append(
                    f"datasets/{SPLITS_DIR}/{folder.name}/{path.name}: not a file"
                )
                continue
            try:
                read_split_file(path)
            except SplitError as error:
                report.problems.append(f"datasets/{SPLITS_DIR}/{folder.name}: {error}")
                continue
            report.splits += 1


def check_append_only(status: dict, stores: list[Path], report: Report) -> None:
    """Files DVC already versioned may be added to but never deleted or modified.

    status is `dvc data status --granular --json`: "uncommitted" compares the workspace with the
    last snapshot, "committed" compares that snapshot with the git commit. Directory entries
    (ending in "/") change whenever a file is added, so only file entries count.
    """
    roots = [store.resolve() for store in stores]
    for section in ("committed", "uncommitted"):
        for change in ("deleted", "modified"):
            for entry in status_files(status, section, change):
                path = Path(entry).resolve()
                if not any(path.is_relative_to(root) for root in roots):
                    continue
                report.problems.append(
                    f"{entry}: versioned file was {change}; the stores are append-only"
                )


def status_files(status: dict | None, *keys: str) -> list[str]:
    """The file entries DVC status lists under keys, ignoring anything malformed.

    Directory entries (ending in "/") change whenever a file is added, so only files count.
    """
    value = status
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, str) and not entry.endswith("/")]


def versioned_files(status: dict | None) -> set[Path]:
    """Files DVC reports unchanged since the committed version (needs --unchanged)."""
    return {Path(entry).resolve() for entry in status_files(status, "unchanged")}


def known_files(status: dict | None) -> set[Path]:
    """Files DVC already holds in a snapshot or a commit: all it lists but files added since
    the last snapshot (needs --unchanged). Removing one would break the append-only rule."""
    listed = status_files(status, "unchanged")
    for change in ("added", "modified", "deleted"):
        listed += status_files(status, "committed", change)
    for change in ("modified", "deleted"):
        listed += status_files(status, "uncommitted", change)
    return {Path(entry).resolve() for entry in listed}


class StatusError(RuntimeError):
    """DVC status output that cannot be used."""


def read_status(path: Path) -> dict:
    """`dvc data status --granular --unchanged --json` output, or StatusError saying why not."""
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as error:
        raise StatusError(f"cannot read DVC status: {error}") from error
    if not isinstance(status, dict):
        raise StatusError("DVC status is not a JSON object")
    return status


def verify(evidence: Path, datasets: Path, dvc_status: dict | None = None) -> Report:
    report = Report()
    for store, name in ((evidence, "evidence"), (datasets, "datasets")):
        if not store.is_dir():
            report.problems.append(f"{name} store {store} does not exist")
    if not report.problems:
        check_evidence(evidence, report)
        check_datasets(datasets, report, versioned_files(dvc_status))
    if dvc_status is not None:
        check_append_only(dvc_status, [evidence, datasets], report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evidence", type=Path, default=Path("data/evidence"))
    parser.add_argument("--datasets", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--dvc-status",
        type=Path,
        help="JSON from `dvc data status --granular --unchanged --json`",
    )
    args = parser.parse_args(argv)
    status = None
    if args.dvc_status is not None:
        try:
            status = read_status(args.dvc_status)
        except StatusError as error:
            print(f"Not versioning: {error}", file=sys.stderr)
            return 1
    report = verify(args.evidence, args.datasets, status)
    for problem in report.problems:
        print(f"problem: {problem}", file=sys.stderr)
    if report.problems:
        print(f"Not versioning: {len(report.problems)} problem(s) found", file=sys.stderr)
        return 1
    print(
        f"verified: {report.runs} runs, {report.experiment_entries} experiments,"
        f" {report.releases} releases"
        f" (+{report.superseded} superseded), {report.reports} build reports,"
        f" {report.attempts} gate attempts, {report.splits} splits;"
        f" {report.rebuilt} releases rebuilt from evidence"
        f" ({report.hidden} hidden entries excluded)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
