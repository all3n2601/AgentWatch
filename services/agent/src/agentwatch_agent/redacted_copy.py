"""Write a redacted copy of an evidence folder for sharing (A07).

Local evidence stays complete: replays resend the recorded conversation exactly, so it is
never redacted in place. Sharing uses a copy instead. Every string value in each `.json` and
`.jsonl` file is passed through `redact_text`, and so is every object key (a map keyed by a
path or email would otherwise pass unchanged); numbers, booleans and nulls are kept, so the
copy keeps its structure. Keys of one object that redact to the same text are kept apart with
a numeric suffix (`<path>`, `<path>#2`) rather than merged, and counted in `keys_redacted`.
A file with no replacements and no changed keys is copied byte for byte, so its hash is
unchanged; any other file is re-serialized. Files that cannot be scanned are not copied and
are listed, and hidden files (tool state such as `.DS_Store`) are skipped. `redaction.json`
in the copy records the method, each file's source SHA-256 and replacements by kind, without
host paths.

The copy is built in a hidden staging folder beside the output and renamed into place, so a
failure leaves no partial output. An existing output is never overwritten.
"""

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from agentwatch_agent.redaction import LIMITATIONS, REDACTION_METHOD, redact_text

MANIFEST_FILE = "redaction.json"
SCANNED_SUFFIXES = (".json", ".jsonl")


class RedactionError(ValueError):
    """The folder cannot be copied safely."""


def _redact_string(text: str, counts: Counter[str]) -> str:
    redacted, replaced = redact_text(text)
    counts.update(replaced)
    return redacted


def _unique_key(key: str, taken: dict) -> str:
    """Return key, or key with the first free `#n` suffix (n >= 2) when it is already taken."""
    if key not in taken:
        return key
    suffix = 2
    while f"{key}#{suffix}" in taken:
        suffix += 1
    return f"{key}#{suffix}"


def _redact_value(value, counts: Counter[str], keys: Counter[str]):
    """Redact strings and object keys; `keys` counts the keys that changed."""
    if isinstance(value, str):
        return _redact_string(value, counts)
    if isinstance(value, list):
        return [_redact_value(item, counts, keys) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            redacted_key = _unique_key(_redact_string(key, counts), result)
            if redacted_key != key:
                keys["redacted"] += 1
            result[redacted_key] = _redact_value(item, counts, keys)
        return result
    return value


def _redact_file(data: bytes, relative: str, counts: Counter[str], keys: Counter[str]) -> bytes:
    """Return the bytes to write: the original when nothing changed, else the redacted text."""
    try:
        redacted = _redact_document(data, relative, counts, keys)
    except RedactionError as error:
        if str(error).startswith(relative):
            raise
        raise RedactionError(f"{relative}: {error}") from error
    if not counts and not keys["redacted"]:
        return data
    return redacted.encode("utf-8")


def _redact_document(data: bytes, relative: str, counts: Counter[str], keys: Counter[str]) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RedactionError(f"{relative} is not UTF-8 text") from error
    if relative.endswith(".json"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise RedactionError(f"{relative} is not valid JSON: {error.msg}") from error
        return json.dumps(_redact_value(value, counts, keys), indent=2, ensure_ascii=False) + "\n"
    lines = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RedactionError(f"{relative} line {number} is not valid JSON") from error
        lines.append(json.dumps(_redact_value(value, counts, keys), ensure_ascii=False) + "\n")
    return "".join(lines)


def _source_files(source: Path) -> list[Path]:
    """Every file under source in a stable order, skipping hidden names; links are refused."""
    files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink():
            raise RedactionError(f"{relative.as_posix()} is a symbolic link; copy its target")
        if path.is_file():
            files.append(path)
    return files


def _check_locations(source: Path, output: Path) -> None:
    if not source.is_dir():
        raise RedactionError(f"{source.name} is not a folder")
    if output.exists() or output.is_symlink():
        raise RedactionError(f"{output.name} already exists; choose a new output folder")
    if output.resolve().is_relative_to(source.resolve()):
        raise RedactionError("The output folder cannot be inside the source folder")
    if (source / MANIFEST_FILE).exists():
        raise RedactionError(f"{source.name} is already a redacted copy ({MANIFEST_FILE})")


def write_redacted_copy(source: Path, output: Path) -> dict:
    """Copy source to output with sensitive values redacted; return the manifest."""
    _check_locations(source, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        entries, not_copied = [], []
        totals: Counter[str] = Counter()
        for path in _source_files(source):
            relative = path.relative_to(source).as_posix()
            if not relative.endswith(SCANNED_SUFFIXES):
                not_copied.append(relative)
                continue
            data = path.read_bytes()
            counts: Counter[str] = Counter()
            keys: Counter[str] = Counter()
            redacted = _redact_file(data, relative, counts, keys)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(redacted)
            totals.update(counts)
            entries.append(
                {
                    "path": relative,
                    "source_sha256": hashlib.sha256(data).hexdigest(),
                    "replacements": dict(sorted(counts.items())),
                    "keys_redacted": keys["redacted"],
                }
            )
        manifest = {
            "method": REDACTION_METHOD,
            "source": source.name,
            "files": entries,
            "not_copied": not_copied,
            "totals": dict(sorted(totals.items())),
            "limitations": [
                *LIMITATIONS,
                "Files other than .json and .jsonl are not copied",
                "Review the copy before sharing",
            ],
        }
        (staging / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
