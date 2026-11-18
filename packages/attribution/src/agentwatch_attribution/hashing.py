"""SHA-256 identities shared by dataset building, model training and serving."""

import hashlib
import json
from pathlib import Path

HASH_BLOCK_BYTES = 1 << 20


def canonical_sha256(value: object) -> str:
    """Hash JSON-serialisable data independent of key order; NaN and infinity are refused."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
