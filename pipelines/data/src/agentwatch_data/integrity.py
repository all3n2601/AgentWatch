"""Verify the evidence store and dataset store before they are versioned.

    python -m agentwatch_data.integrity --evidence data/evidence --datasets data/datasets

A version should never record damage. Every evidence entry must be valid evidence stored
under its own <kind prefix>-<run_id> name; every release must match its manifest and name;
every build report must be readable; every committed quality-gate attempt must be readable
(an unknown verdict is damage); and every split file must hash to its id, sit beside the
release it names, and refer to a release that exists. Hidden staging left by a running or
killed command is not part of a version (DVC excludes it) and is only counted.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agentwatch_data.build import BUILDS_DIR, RELEASE_NAME
from agentwatch_data.evidence import EvidenceError, entry_name, read_entry
from agentwatch_data.splits import (
    SPLITS_DIR,
    VALIDATIONS_DIR,
    SplitError,
    latest_verdict,
    read_split_file,
    verify_release,
)


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    runs: int = 0
    releases: int = 0
    reports: int = 0
    splits: int = 0
    staging: int = 0


def visible(directory: Path, report: Report) -> list[Path]:
    """Entries of directory, sorted, with hidden staging counted and skipped."""
    entries = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            report.staging += 1
        else:
            entries.append(path)
    return entries


def check_evidence(store: Path, report: Report) -> None:
    for path in visible(store, report):
        if not path.is_dir() or path.is_symlink():
            report.problems.append(f"evidence/{path.name}: not a run directory")
            continue
        try:
            run = read_entry(path)
        except EvidenceError as error:
            report.problems.append(f"evidence/{path.name}: {error}")
            continue
        if path.name != entry_name(path, run):
            report.problems.append(f"evidence/{path.name}: stored under the wrong name")
            continue
        report.runs += 1


def check_datasets(store: Path, report: Report) -> None:
    releases = set()
    for path in visible(store, report):
        if path.name in (BUILDS_DIR, VALIDATIONS_DIR, SPLITS_DIR) and path.is_dir():
            continue
        if not RELEASE_NAME.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
            report.problems.append(f"datasets/{path.name}: unexpected entry")
            continue
        try:
            verify_release(path)
        except SplitError as error:
            report.problems.append(f"datasets/{path.name}: {error}")
            continue
        releases.add(path.name)
        report.releases += 1
    check_reports(store / BUILDS_DIR, report)
    check_validations(store, releases, report)
    check_splits(store / SPLITS_DIR, releases, report)


def check_reports(builds: Path, report: Report) -> None:
    for path in visible(builds, report) if builds.is_dir() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = None
        if path.suffix != ".json" or not isinstance(value, dict):
            report.problems.append(f"datasets/{BUILDS_DIR}/{path.name}: unreadable build report")
            continue
        report.reports += 1


def check_validations(store: Path, releases: set[str], report: Report) -> None:
    validations = store / VALIDATIONS_DIR
    for path in visible(validations, report) if validations.is_dir() else []:
        if path.name not in releases:
            report.problems.append(f"datasets/{VALIDATIONS_DIR}/{path.name}: no such release")
            continue
        try:
            latest_verdict(store / path.name)
        except SplitError as error:
            report.problems.append(f"datasets/{VALIDATIONS_DIR}/{path.name}: {error}")


def check_splits(splits: Path, releases: set[str], report: Report) -> None:
    for folder in visible(splits, report) if splits.is_dir() else []:
        if folder.name not in releases or not folder.is_dir():
            report.problems.append(f"datasets/{SPLITS_DIR}/{folder.name}: no such release")
            continue
        for path in visible(folder, report):
            try:
                read_split_file(path)
            except SplitError as error:
                report.problems.append(f"datasets/{SPLITS_DIR}/{folder.name}: {error}")
                continue
            report.splits += 1


def verify(evidence: Path, datasets: Path) -> Report:
    report = Report()
    for store, name in ((evidence, "evidence"), (datasets, "datasets")):
        if not store.is_dir():
            report.problems.append(f"{name} store {store} does not exist")
    if not report.problems:
        check_evidence(evidence, report)
        check_datasets(datasets, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evidence", type=Path, default=Path("data/evidence"))
    parser.add_argument("--datasets", type=Path, default=Path("data/datasets"))
    args = parser.parse_args(argv)
    report = verify(args.evidence, args.datasets)
    for problem in report.problems:
        print(f"problem: {problem}", file=sys.stderr)
    if report.problems:
        print(f"Not versioning: {len(report.problems)} problem(s) found", file=sys.stderr)
        return 1
    print(
        f"verified: {report.runs} runs, {report.releases} releases, {report.reports} build "
        f"reports, {report.splits} splits ({report.staging} hidden staging entries excluded)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
