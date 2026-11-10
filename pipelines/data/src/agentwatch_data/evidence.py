"""Curate recorded runs and replay experiments into the versioned evidence store.

    python -m agentwatch_data.evidence --store data/evidence .data/coding-demo [...]

The store holds two kinds of entry. A recorded run is one directory named
<kind prefix>-<run_id> (for example coding-<uuid>), with the run's results.json and
samples.jsonl copied byte for byte. A replay experiment (a directory with an experiment.json)
is one directory named replay-<task_id>-<started_at in UTC>-<first 6 hex of the SHA-256 of
experiment.json>, holding exactly the experiment's evidence files (EXPERIMENT_FILE) byte for
byte. The experiment's own content names it, so the same experiment delivered under two folder
names is one entry. Hidden files and directories (a name starting with ".", such as a Finder
.DS_Store) are tool state, not evidence: they are never copied into an entry, and inside a
stored entry they are skipped, as inside run entries and by .dvcignore.

Evidence is validated before it is added: a run with the same rules as dataset extraction, an
experiment with `read_experiment` and every journal read strictly. An experiment that failed
its protocol checks is still added; labeling rejects it with a recorded reason. A store entry
is never changed: adding evidence that is already stored does nothing, and adding different
evidence under a stored name is refused. Re-running a demo that overwrites its output
directory therefore cannot change versioned evidence, and a release built from the store can
be rebuilt from it anywhere.
"""

import argparse
import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path, PurePosixPath

from agentwatch_data.extract import (
    RESULTS_FILE,
    SAMPLES_FILE,
    RawRun,
    directory_prefix,
    parse_json,
    read_run,
    read_run_files,
    run_from_bytes,
    summarize,
)
from agentwatch_data.files import fsync_directory, fsync_file
from agentwatch_data.replay_evidence import (
    JOURNAL_FILE,
    MANIFEST_FILE,
    MAX_EVIDENCE_BYTES,
    Experiment,
    ReplayEvidenceError,
    read_experiment,
)
from agentwatch_data.replay_journal import read_journal

EVIDENCE_FILES = (RESULTS_FILE, SAMPLES_FILE)  # all an entry may hold
STAGING_PREFIX = ".adding-"  # hidden, and excluded from DVC snapshots by .dvcignore
REPLAY_PREFIX = "replay"
# Every file a replay experiment entry may hold, as a POSIX path relative to the entry.
# [0-9], not \d, which would also match non-ASCII digits.
EXPERIMENT_FILE = re.compile(
    r"\A(?:experiment\.json"
    r"|(?:source|replays/[0-9]{2})/(?:agent-evidence\.json|live-journal\.jsonl)"
    r"|injections/[0-9]{2}\.json)\Z"
)
# The task ID becomes part of a directory name: no separators, no hidden or relative names.
ENTRY_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class EvidenceError(RuntimeError):
    """Evidence that cannot be added to, or read from, the evidence store; it is unchanged."""


@dataclass(frozen=True)
class Stored:
    name: str  # the store entry: <kind prefix>-<run_id>, or replay-... for an experiment
    added: bool  # False when identical evidence was already stored


def entry_name(run_dir: Path, run: RawRun) -> str:
    return f"{directory_prefix(run_dir.name)}-{run.run_id}"


def same_evidence(stored: RawRun, run: RawRun) -> bool:
    """The raw evidence hash and samples-file presence decide; formatting does not."""
    return (stored.source_sha256, stored.telemetry_status) == (
        run.source_sha256,
        run.telemetry_status,
    )


def read_entry(path: Path) -> RawRun:
    try:
        return read_run(path)
    except (OSError, ValueError) as error:
        raise EvidenceError(
            f"Stored evidence {path.name} is unreadable: {summarize(error)}"
        ) from error


def add_run(run_dir: Path, store: Path) -> Stored:
    """Validate run_dir and copy it into store, unless identical evidence is already there.

    Each file is read once and the bytes that were validated are the bytes stored, so a
    producer rewriting its output meanwhile cannot slip unchecked evidence into the store.
    """
    try:
        results, samples = read_run_files(run_dir)
        run = run_from_bytes(run_dir, results, samples)
    except (OSError, ValueError) as error:
        raise EvidenceError(f"{run_dir} is not usable evidence: {summarize(error)}") from error
    name = entry_name(run_dir, run)
    target = store / name
    if target.exists():
        return existing(target, run)
    # One run ID is one run: the same ID under another kind prefix is a conflict too.
    others = sorted(p.name for p in store.glob(f"*-{run.run_id}") if p.name != name)
    if others:
        raise EvidenceError(f"Run {run.run_id} is already stored as {others[0]}")
    store.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=store))
    try:
        staging.chmod(0o755)  # mkdtemp is owner-only; DVC, CI, and Airflow may run as others
        for file, data in ((RESULTS_FILE, results), (SAMPLES_FILE, samples)):
            if data is not None:
                (staging / file).write_bytes(data)
                (staging / file).chmod(0o644)
                fsync_file(staging / file)
        fsync_directory(staging)
        # Read the copy back before publishing it: the store must hold what was validated.
        staged_samples = staging / SAMPLES_FILE
        copy = run_from_bytes(
            target,
            (staging / RESULTS_FILE).read_bytes(),
            staged_samples.read_bytes() if staged_samples.exists() else None,
        )
        if not same_evidence(copy, run):
            raise EvidenceError(f"The copy of {name} does not match the validated evidence")
        try:
            os.replace(staging, target)
        except OSError:
            if not target.exists():
                raise
            return existing(target, run)  # a concurrent add stored this run first
        fsync_directory(store)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return Stored(name, added=True)


def existing(target: Path, run: RawRun) -> Stored:
    if not same_evidence(read_entry(target), run):
        raise EvidenceError(f"{target.name} is already stored with different evidence")
    return Stored(target.name, added=False)


def experiment_entry_name(manifest: bytes, experiment: Experiment) -> str:
    """replay-<task_id>-<started_at in UTC>-<first 6 hex of the manifest bytes' SHA-256>."""
    if experiment.started_at is None:
        raise EvidenceError(f"{MANIFEST_FILE} has no started_at to name the store entry")
    if not ENTRY_TASK_ID.fullmatch(experiment.task_id):
        raise EvidenceError(f"task_id {experiment.task_id!r} cannot name a store entry")
    started = experiment.started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(manifest).hexdigest()[:6]
    return f"{REPLAY_PREFIX}-{experiment.task_id}-{started}-{digest}"


def add_experiment(directory: Path, store: Path) -> Stored:
    """Validate a replay experiment directory and copy it into store, unless already there.

    Each file is read once, and the staged copy of those bytes is what gets validated and
    published, so a producer rewriting its output meanwhile cannot slip unchecked evidence in.
    """
    try:
        files = {name: read_bounded(directory / name) for name in experiment_files(directory)}
    except (OSError, ValueError) as error:
        raise EvidenceError(f"{directory} is not usable evidence: {summarize(error)}") from error
    if MANIFEST_FILE not in files:
        raise EvidenceError(f"{directory} is not usable evidence: it has no {MANIFEST_FILE}")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    store.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=store))
    try:
        write_staged(staging, files)
        try:
            experiment, staged = read_experiment_entry(staging)
            name = experiment_entry_name(files[MANIFEST_FILE], experiment)
        except (OSError, ValueError, EvidenceError) as error:
            reason = error if isinstance(error, EvidenceError) else summarize(error)
            raise EvidenceError(f"{directory} is not usable evidence: {reason}") from error
        if staged != hashes:
            raise EvidenceError(f"The copy of {name} does not match the evidence read")
        target = store / name
        if target.exists():
            return existing_experiment(target, hashes)
        # Serialize the identity check and publication, including differently named
        # manifests. Refuse a concurrent writer rather than waiting without a bound.
        lock = store / ".replay-publication-lock"
        try:
            lock.mkdir()
        except FileExistsError as error:
            raise EvidenceError("Another replay publication is in progress; retry later") from error
        try:
            if target.exists():
                return existing_experiment(target, hashes)
            incoming = {r.run_id for r in experiment.replays if r.run_id is not None}
            for entry in sorted(store.iterdir()):
                if entry.name.startswith(".") or not entry.is_dir():
                    continue
                if (entry / MANIFEST_FILE).exists():
                    stored, _ = read_stored_experiment(entry)
                    identities = {r.run_id for r in stored.replays if r.run_id is not None}
                    if incoming & identities:
                        raise EvidenceError(
                            f"Replay run identity is already stored as {entry.name}"
                        )
            try:
                os.replace(staging, target)
            except OSError:
                if not target.exists():
                    raise
                return existing_experiment(target, hashes)
            fsync_directory(store)
        finally:
            lock.rmdir()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return Stored(name, added=True)


def existing_experiment(target: Path, hashes: dict[str, str]) -> Stored:
    _, stored = read_stored_experiment(target)
    if stored != hashes:
        raise EvidenceError(f"{target.name} is already stored with different evidence")
    return Stored(target.name, added=False)


def read_stored_experiment(path: Path) -> tuple[Experiment, dict[str, str]]:
    """A stored experiment and the SHA-256 of each of its files, by relative POSIX path."""
    try:
        return read_experiment_entry(path)
    except (OSError, ValueError) as error:
        raise EvidenceError(
            f"Stored evidence {path.name} is unreadable: {summarize(error)}"
        ) from error


def read_experiment_entry(path: Path) -> tuple[Experiment, dict[str, str]]:
    hashes = {
        name: hashlib.sha256(read_bounded(path / name)).hexdigest()
        for name in experiment_files(path)
    }
    experiment = read_experiment(path)
    refuse_hidden_references(read_bounded(path / MANIFEST_FILE))
    for name in hashes:
        if name.endswith(f"/{JOURNAL_FILE}"):
            read_journal(path / name)
    return experiment, hashes


def refuse_hidden_references(manifest: bytes) -> None:
    """Refuse a manifest that names a hidden or relative path for a replay's evidence.

    Hidden entries are skipped, so no hash would cover the files such a path points at.
    read_experiment has already validated the manifest's structure.
    """
    for entry in parse_json(manifest.decode("utf-8"))["replays"]:
        for key in ("evidence_dir", "injection_log"):
            value = entry.get(key)
            if isinstance(value, str) and any(
                part.startswith(".") for part in PurePosixPath(value).parts
            ):
                raise ReplayEvidenceError(f"{MANIFEST_FILE} names hidden path {value!r}")


def experiment_files(directory: Path) -> list[str]:
    """The experiment's files as relative POSIX paths; anything but EXPERIMENT_FILE is refused.

    Raises ValueError for a symlink, a special file or an unexpected file. Directories carry
    no evidence of their own and are not stored; only the files in them are. Hidden entries
    are skipped (see hidden_entries).
    """
    return walk_experiment(directory)[0]


def hidden_entries(directory: Path) -> list[str]:
    """The hidden files and directories experiment_files skips; a hidden directory counts once.

    Raises ValueError like experiment_files.
    """
    return walk_experiment(directory)[1]


def walk_experiment(directory: Path) -> tuple[list[str], list[str]]:
    """(evidence files, skipped hidden entries) of an experiment, as relative POSIX paths."""
    if directory.is_symlink() or not directory.is_dir():
        raise ReplayEvidenceError(f"{directory.name} must be a directory without symlinks")
    names, hidden = [], []

    def unreadable(error: OSError) -> None:
        raise error  # os.walk would otherwise skip an unreadable directory silently

    for current, directories, files in os.walk(directory, onerror=unreadable):
        # Hidden directories are pruned in place, so os.walk never descends into them.
        hidden.extend(
            Path(current, entry).relative_to(directory).as_posix()
            for entry in sorted(directories + files)
            if entry.startswith(".")
        )
        directories[:] = [entry for entry in directories if not entry.startswith(".")]
        for entry in sorted(directories + [file for file in files if not file.startswith(".")]):
            path = Path(current, entry)
            name = path.relative_to(directory).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ReplayEvidenceError(f"{name} is a symlink")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode) or not EXPERIMENT_FILE.fullmatch(name):
                raise ReplayEvidenceError(f"{name} is not an experiment evidence file")
            names.append(name)
    return sorted(names), sorted(hidden)


def read_bounded(path: Path) -> bytes:
    """A regular file's bytes, refusing a symlink swapped in after listing and oversize files."""
    with open(
        path, "rb", opener=lambda file, flags: os.open(file, flags | os.O_NOFOLLOW)
    ) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ReplayEvidenceError(f"{path.name} must be a regular file")
        data = stream.read(MAX_EVIDENCE_BYTES + 1)
    if len(data) > MAX_EVIDENCE_BYTES:
        raise ReplayEvidenceError(f"{path.name} is larger than {MAX_EVIDENCE_BYTES} bytes")
    return data


def write_staged(staging: Path, files: dict[str, bytes]) -> None:
    """Write files under staging durably, readable by others (DVC, CI and Airflow users)."""
    directories = {staging}
    for name, data in files.items():
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o644)
        fsync_file(path)
        directories.update(parent for parent in path.parents if parent.is_relative_to(staging))
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o755)  # mkdtemp is owner-only, and mkdir follows the umask
        fsync_directory(directory)


def add(path: Path, store: Path) -> Stored:
    """Add a replay experiment (it has an experiment.json) or a recorded run."""
    if (path / MANIFEST_FILE).exists():
        return add_experiment(path, store)
    return add_run(path, store)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="recorded run or replay experiment directories to add",
    )
    parser.add_argument("--store", type=Path, default=Path("data/evidence"))
    args = parser.parse_args(argv)
    failed = False
    for run_dir in args.runs:
        try:
            stored = add(run_dir, args.store)
        except EvidenceError as error:
            print(f"refused: {error}", file=sys.stderr)
            failed = True
            continue
        print(f"{'added' if stored.added else 'already stored'}: {args.store / stored.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
