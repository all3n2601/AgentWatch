"""Hashing and durable writes shared by every stage that stores pipeline output.

Every stored file is written so that a reader, or a crash, never sees it half-written: content
is staged under a hidden name, fsynced, renamed into place, and the directory is fsynced so
the rename survives. Staged names stay hidden so DVC excludes them (see .dvcignore).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePath

# Readers of untrusted evidence directories. replay_evidence.py keeps its own copies on purpose:
# its source is fingerprinted into the labels-v2 label version, so moving them there would
# change a frozen label version without changing any label.


def confined_path(root: Path, relative: str, error: type[Exception]) -> Path:
    """Resolve an untrusted relative path inside root; reject absolute, linked and escaping paths.

    `error` is the caller's exception type, so each reader keeps reporting its own errors.
    """
    if PurePath(relative).is_absolute():
        raise error(f"Path {relative!r} is outside the experiment directory")
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)):
        raise error(f"Path {relative!r} contains a symlink")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise error(f"Path {relative!r} is outside the experiment directory")
    return resolved


def read_bounded_bytes(path: Path, max_bytes: int, error: type[Exception]) -> bytes:
    """The bytes of an untrusted regular file of at most max_bytes; symlinks are refused."""
    if path.is_symlink() or not path.is_file():
        raise error(f"{path.name} must be a regular file without symlinks")
    try:
        # Read one byte past the limit so an oversize file is detected without loading it all.
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as cause:
        raise error(f"{path.name} cannot be read: {cause}") from cause
    if len(raw) > max_bytes:
        raise error(f"{path.name} is larger than {max_bytes} bytes")
    return raw


def decode_utf8(raw: bytes, name: str, error: type[Exception]) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as cause:
        raise error(f"{name} is not valid UTF-8: {cause}") from cause


def read_bounded_text(path: Path, max_bytes: int, error: type[Exception]) -> str:
    """UTF-8 text of an untrusted regular file of at most max_bytes; symlinks are refused."""
    return decode_utf8(read_bounded_bytes(path, max_bytes, error), path.name, error)


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
