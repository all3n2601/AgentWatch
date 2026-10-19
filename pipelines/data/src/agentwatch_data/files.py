"""Hashing and durable writes shared by every stage that stores pipeline output.

Every stored file is written so that a reader, or a crash, never sees it half-written: content
is staged under a hidden name, fsynced, renamed into place, and the directory is fsynced so
the rename survives. Staged names stay hidden so DVC excludes them (see .dvcignore).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_text(value) -> str:
    """Canonical, human-readable JSON: sorted keys, and no NaN or infinity."""
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic(path: Path, text: str) -> None:
    """Write text so readers never see a partial file and it survives a crash.

    Readable by other users (DVC, CI, and Airflow workers may run as someone else).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        staged.chmod(0o644)
        os.replace(staged, path)
        fsync_directory(path.parent)
    finally:
        staged.unlink(missing_ok=True)
