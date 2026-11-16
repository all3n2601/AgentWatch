"""Write a redacted copy of an evidence folder for sharing (A07).

Local evidence stays complete: replays resend the recorded conversation exactly, so it is
never redacted in place. Sharing uses a copy instead. Every string value in each `.json` and
`.jsonl` file is passed through `redact_text`; keys, numbers, booleans and nulls are kept, so
the copy keeps its structure. Files that cannot be scanned are not copied and are listed, and
hidden files (tool state such as `.DS_Store`) are skipped. `redaction.json` in the copy records
the method, each file's source SHA-256 and replacements by kind, without host paths.

The copy is built in a hidden staging folder beside the output and renamed into place, so a
failure leaves no partial output. An existing output is never overwritten.
"""

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from agentwatch_agent.redaction import REDACTION_METHOD, redact_text

MANIFEST_FILE = "redaction.json"
SCANNED_SUFFIXES = (".json", ".jsonl")


class RedactionError(ValueError):
    """The folder cannot be copied safely."""


def _redact_value(value, counts: Counter[str]):
    if isinstance(value, str):
        text, replaced = redact_text(value)
        counts.update(replaced)
        return text
    if isinstance(value, list):
        return [_redact_value(item, counts) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item, counts) for key, item in value.items()}
    return value


def _redact_file(data: bytes, relative: str, counts: Counter[str]) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RedactionError(f"{relative} is not UTF-8 text") from error
    if relative.endswith(".json"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise RedactionError(f"{relative} is not valid JSON: {error.msg}") from error
        return json.dumps(_redact_value(value, counts), indent=2, ensure_ascii=False) + "\n"
    lines = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RedactionError(f"{relative} line {number} is not valid JSON") from error
        lines.append(json.dumps(_redact_value(value, counts), ensure_ascii=False) + "\n")
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
            redacted = _redact_file(data, relative, counts)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(redacted, encoding="utf-8")
            totals.update(counts)
            entries.append(
                {
                    "path": relative,
                    "source_sha256": hashlib.sha256(data).hexdigest(),
                    "replacements": dict(sorted(counts.items())),
                }
            )
        manifest = {
            "method": REDACTION_METHOD,
            "source": source.name,
            "files": entries,
            "not_copied": not_copied,
            "totals": dict(sorted(totals.items())),
            "limitations": [
                (
                    "Pattern-based: an unlabeled random secret is indistinguishable from data, "
                    "so review the copy before sharing"
                ),
                "Files other than .json and .jsonl are not copied",
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
