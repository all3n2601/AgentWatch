"""Curate recorded runs into the versioned evidence store that dataset releases are built from.

    python -m agentwatch_data.evidence --store data/evidence .data/coding-demo [...]

The store holds one directory per run, named <kind prefix>-<run_id> (for example
coding-<uuid>), with the run's results.json and samples.jsonl copied byte for byte. A run is
validated with the same rules as dataset extraction before it is added, and a store entry is
never changed: adding evidence that is already stored does nothing, and adding different
evidence under a stored run ID is refused. Re-running a demo that overwrites its output
directory therefore cannot change versioned evidence, and a release built from the store can
be rebuilt from it anywhere.
"""

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.extract import (
    RESULTS_FILE,
    SAMPLES_FILE,
    RawRun,
    directory_prefix,
    read_run,
    read_run_files,
    run_from_bytes,
    summarize,
)
from agentwatch_data.files import fsync_directory, fsync_file

EVIDENCE_FILES = (RESULTS_FILE, SAMPLES_FILE)  # all an entry may hold
STAGING_PREFIX = ".adding-"  # hidden, and excluded from DVC snapshots by .dvcignore


class EvidenceError(RuntimeError):
    """A run that cannot be added to the evidence store; the store is unchanged."""


@dataclass(frozen=True)
class Stored:
    name: str  # the store entry, <kind prefix>-<run_id>
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path, nargs="+", help="recorded run directories to add")
    parser.add_argument("--store", type=Path, default=Path("data/evidence"))
    args = parser.parse_args(argv)
    failed = False
    for run_dir in args.runs:
        try:
            stored = add_run(run_dir, args.store)
        except EvidenceError as error:
            print(f"refused: {error}", file=sys.stderr)
            failed = True
            continue
        print(f"{'added' if stored.added else 'already stored'}: {args.store / stored.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
