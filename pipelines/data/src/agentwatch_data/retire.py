"""Remove a dataset release that was never versioned, with everything that names it.

    python -m agentwatch_data.retire --datasets data/datasets \\
        --dvc-status <output of dvc data status --granular --unchanged --json> <release>

Versioned releases are permanent: the stores are append-only and the integrity check refuses
any change to a file DVC already holds. A release that was never versioned, though, may be
one the current pipeline can no longer rebuild (built before the label rules or the code
changed), and integrity refuses to version it. Retiring removes it, its build reports, its
quality-gate attempts, and its splits, so the store can be versioned again; the next build
publishes the current pipeline's release of the same evidence.

Refused, removing nothing, when any of those files is already in a DVC snapshot or commit.
Files are removed release last, so an interrupted retire is finished by running it again.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from agentwatch_data.build import BUILDS_DIR, RELEASE_NAME
from agentwatch_data.integrity import StatusError, known_files, read_status
from agentwatch_data.splits import SPLITS_DIR, VALIDATIONS_DIR


class RetireError(RuntimeError):
    """The release cannot be retired; nothing was removed."""


def names_release(report: Path, release: str) -> bool:
    try:
        value = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return False  # an unreadable report names nothing; integrity reports it
    return isinstance(value, dict) and value.get("release") == release


def retire(datasets: Path, release: str, status: dict) -> list[Path]:
    """Remove release and everything naming it; returns what was removed, in order."""
    if not RELEASE_NAME.fullmatch(release):
        raise RetireError(f"{release!r} is not a release name")
    builds = datasets / BUILDS_DIR
    reports = sorted(
        path
        for path in (builds.iterdir() if builds.is_dir() else [])
        if path.suffix == ".json" and not path.is_symlink() and names_release(path, release)
    )
    folders = [datasets / VALIDATIONS_DIR / release, datasets / SPLITS_DIR / release]
    targets = [path for path in folders if path.exists() or path.is_symlink()]
    targets += reports
    if (datasets / release).exists() or (datasets / release).is_symlink():
        targets.append(datasets / release)  # last: a rerun can still find what remains
    if not targets:
        raise RetireError(f"No release {release} or anything naming it in {datasets}")
    files = set()
    for target in targets:
        inside = target.rglob("*") if target.is_dir() and not target.is_symlink() else []
        files |= {target.resolve(), *(path.resolve() for path in inside)}
    versioned = sorted(str(path) for path in files & known_files(status))
    if versioned:
        raise RetireError(
            f"{release} is already versioned ({versioned[0]}"
            f"{' and more' if len(versioned) > 1 else ''}); versioned releases are permanent"
        )
    for target in targets:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("release", help="the release to remove, for example steps-v1.2+<hex>")
    parser.add_argument("--datasets", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--dvc-status",
        type=Path,
        required=True,
        help="JSON from `dvc data status --granular --unchanged --json`",
    )
    args = parser.parse_args(argv)
    try:
        removed = retire(args.datasets, args.release, read_status(args.dvc_status))
    except (RetireError, StatusError, OSError) as error:
        print(f"Not retiring: {error}", file=sys.stderr)
        return 1
    for path in removed:
        print(f"removed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
