"""Check an evidence bundle before sharing it, and refuse it if anything sensitive is left (A07).

`check_bundle` reads a folder or a `.zip` file and never changes it. Every file is scanned,
hidden files included, because everything in a bundle is shared. Findings use the same rules
as `redact` (`trace-redaction-v1`), so a copy written by `redact` passes unless it was edited
afterwards. A private key's `BEGIN` line alone also counts, since text files are read line by
line.

`.json` files are parsed and every string value and object key is checked, so an escaped
Windows path (`C:\\\\Users`) is read as the path it is, and a repeated key, whose earlier value
a normal parser drops, is still checked. `.jsonl` files are parsed line by line. A JSON file
that does not parse is checked as text and listed. Other files are checked line by line as
UTF-8 text.

The bundle is refused when any file has a finding or cannot be checked: not UTF-8 text (such
as images, nested archives or `.DS_Store`), larger than `MAX_FILE_BYTES`, an encrypted zip
member, or a symbolic link. Something that could not be read is never reported as clean.
Findings give the file, the line or JSON location and the kinds found, never the value, and
file names and JSON keys in the report are themselves redacted.

A `redact` copy made before this check existed is refused: two of the limitations its
`redaction.json` records gave examples that the rules match. Make the copy again.
"""

import json
import os
import re
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from agentwatch_agent.redaction import LIMITATIONS, REDACTION_METHOD, redact_text

CHECK_METHOD = "evidence-redaction-check-v1"
CLEAN, REFUSED = "clean", "refused"
# Evidence files are at most tens of MB; a larger file is not read rather than read slowly.
MAX_FILE_BYTES = 256 * 1024 * 1024
_PRIVATE_KEY_HEADER = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_ZIP_ENCRYPTED_FLAG = 0x1


class BundleError(ValueError):
    """The bundle cannot be checked at all: missing, not a folder or zip, or empty."""


class _Unscannable(Exception):
    """One file cannot be checked; the reason is safe to report."""


@dataclass(frozen=True)
class _Entry:
    """One file in the bundle, by its path inside the bundle."""

    name: str
    size_bytes: int
    link: bool = False
    encrypted: bool = False


class _PairList(list):
    """A JSON object as its key-value pairs, so a repeated key keeps every value."""


def sensitive_kinds(text: str) -> dict[str, int]:
    """Sensitive values in text, counted by kind; the values themselves are not returned."""
    _, counts = redact_text(text)
    if "private_key" not in counts and _PRIVATE_KEY_HEADER.search(text):
        counts["private_key"] = 1
    return counts


def _safe(text: str) -> str:
    """Text fit for the report: names and keys can themselves hold a path or address."""
    return redact_text(text)[0]


def _pointer_segment(key: str) -> str:
    return _safe(key).replace("~", "~0").replace("/", "~1")


def _check_json_value(value, pointer: str, findings: list[tuple[str, dict]]) -> None:
    if isinstance(value, str):
        if kinds := sensitive_kinds(value):
            findings.append((pointer or "/", kinds))
    elif isinstance(value, _PairList):
        for key, item in value:
            child = f"{pointer}/{_pointer_segment(key)}"
            if kinds := sensitive_kinds(key):
                findings.append((f"{child} (key)", kinds))
            _check_json_value(item, child, findings)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_value(item, f"{pointer}/{index}", findings)


def _parse_json(text: str):
    return json.loads(text, object_pairs_hook=_PairList)


def _check_text_lines(text: str) -> list[tuple[str, dict]]:
    return [
        (f"line {number}", kinds)
        for number, line in enumerate(text.splitlines(), start=1)
        if (kinds := sensitive_kinds(line))
    ]


def _check_content(name: str, data: bytes) -> tuple[list[tuple[str, dict]], bool]:
    """Findings as (location, kinds), and whether a JSON file had to be read as plain text."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _Unscannable("not UTF-8 text") from error
    lowered = name.lower()
    if lowered.endswith(".json"):
        try:
            value = _parse_json(text)
        except json.JSONDecodeError:
            return _check_text_lines(text), True
        findings: list[tuple[str, dict]] = []
        _check_json_value(value, "", findings)
        return findings, False
    if lowered.endswith(".jsonl"):
        findings = []
        as_text = False
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = _parse_json(line)
            except json.JSONDecodeError:
                as_text = True
                if kinds := sensitive_kinds(line):
                    findings.append((f"line {number}", kinds))
                continue
            line_findings: list[tuple[str, dict]] = []
            _check_json_value(value, "", line_findings)
            findings.extend((f"line {number} {where}", kinds) for where, kinds in line_findings)
        return findings, as_text
    return _check_text_lines(text), False


def _folder_entries(folder: Path) -> Iterator[tuple[_Entry, Path]]:
    """Every file and link under folder, in a stable order; links are listed, not followed."""
    for root, directories, files in os.walk(folder, followlinks=False):
        directories.sort()
        here = Path(root)
        for name in sorted([*directories, *files]):
            path = here / name
            relative = path.relative_to(folder).as_posix()
            if path.is_symlink():
                yield _Entry(relative, 0, link=True), path
            elif name in files and path.is_file():
                yield _Entry(relative, path.stat().st_size), path


def _read_folder_file(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(MAX_FILE_BYTES + 1)


def _read_zip_member(archive: zipfile.ZipFile, name: str) -> bytes:
    # A member's declared size can be wrong, so the read itself is bounded.
    with archive.open(name) as handle:
        return handle.read(MAX_FILE_BYTES + 1)


def _too_large() -> _Unscannable:
    return _Unscannable(f"larger than the {MAX_FILE_BYTES // (1024 * 1024)} MiB limit")


def _scan(entry: _Entry, read: Callable[[], bytes], report: dict, totals: Counter[str]) -> None:
    shown = _safe(entry.name)
    if name_kinds := sensitive_kinds(entry.name):
        report["findings"].append({"file": shown, "location": "file name", "kinds": name_kinds})
        totals.update(name_kinds)
    try:
        if entry.link:
            raise _Unscannable("symbolic link; include the file it points to instead")
        if entry.encrypted:
            raise _Unscannable("encrypted zip member")
        if entry.size_bytes > MAX_FILE_BYTES:
            raise _too_large()
        try:
            data = read()
        except (OSError, zipfile.BadZipFile, zlib.error) as error:
            raise _Unscannable(f"could not be read ({type(error).__name__})") from error
        if len(data) > MAX_FILE_BYTES:
            raise _too_large()
        findings, as_text = _check_content(entry.name, data)
    except _Unscannable as reason:
        report["unscanned"].append({"file": shown, "reason": str(reason)})
        return
    report["files_scanned"] += 1
    if as_text:
        report["json_read_as_text"].append(shown)
    for location, kinds in findings:
        report["findings"].append({"file": shown, "location": location, "kinds": kinds})
        totals.update(kinds)


def check_bundle(bundle: Path) -> dict:
    """Check a folder or `.zip` bundle; the report's verdict is `clean` or `refused`."""
    report: dict = {
        "method": CHECK_METHOD,
        "redaction_method": REDACTION_METHOD,
        "bundle": _safe(bundle.name),
        "files_scanned": 0,
        "findings": [],
        "unscanned": [],
        "json_read_as_text": [],
    }
    totals: Counter[str] = Counter()
    if bundle.is_symlink():
        raise BundleError(f"{report['bundle']} is a symbolic link; check its target")
    entries = 0
    if bundle.is_dir():
        for entry, path in _folder_entries(bundle):
            entries += 1
            _scan(entry, lambda path=path: _read_folder_file(path), report, totals)
    elif bundle.is_file() and zipfile.is_zipfile(bundle):
        try:
            with zipfile.ZipFile(bundle) as archive:
                for info in sorted(archive.infolist(), key=lambda item: item.filename):
                    if info.is_dir():
                        continue
                    entries += 1
                    entry = _Entry(
                        info.filename,
                        info.file_size,
                        encrypted=bool(info.flag_bits & _ZIP_ENCRYPTED_FLAG),
                    )
                    _scan(
                        entry,
                        lambda name=info.filename: _read_zip_member(archive, name),
                        report,
                        totals,
                    )
        except (zipfile.BadZipFile, NotImplementedError) as error:
            raise BundleError(f"{report['bundle']} is not a readable zip file: {error}") from error
    elif bundle.exists():
        raise BundleError(f"{report['bundle']} is neither a folder nor a zip file")
    else:
        raise BundleError(f"{report['bundle']} does not exist")
    if entries == 0:
        raise BundleError(f"{report['bundle']} contains no files")
    refused = bool(report["findings"] or report["unscanned"])
    return report | {
        "totals": dict(sorted(totals.items())),
        "verdict": REFUSED if refused else CLEAN,
        "limitations": [
            *LIMITATIONS,
            "A clean result means no rule matched, not that the bundle holds no secret",
        ],
    }
