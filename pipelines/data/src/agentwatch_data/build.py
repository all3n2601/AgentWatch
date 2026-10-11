"""Build a versioned step dataset release from recorded evidence.

    python -m agentwatch_data.build --source local:.data --output .data/datasets

A release is a directory named by its dataset version, holding steps.parquet and
manifest.json. The version is derived from the rows themselves, so rebuilding from the same
evidence with the same rules and code produces the same version. A build either writes a
complete release or nothing: unreadable sources, invalid rules, schema violations, and empty
results fail the build, while untrustworthy individual runs are rejected and reported.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from agentwatch_data import __version__
from agentwatch_data.extract import (
    Extraction,
    ExtractionError,
    Rejection,
    canonical_sha256,
    combine,
    extract_api,
    extract_local,
)
from agentwatch_data.flatten import flatten_runs
from agentwatch_data.labels import LabelingError, label_rows, load_config
from agentwatch_data.schema import COLUMNS, SCHEMA_VERSION, SchemaError, validate_rows

logger = logging.getLogger("agentwatch_data.build")

DATASET_FILE, MANIFEST_FILE = "steps.parquet", "manifest.json"
CONDITION_ORDER = {"baseline": 0, "clean": 1, "contended": 2}
ARROW_TYPES = {
    "string": pa.string(),
    "float64": pa.float64(),
    "int64": pa.int64(),
    "bool": pa.bool_(),
    "timestamp[us, tz=UTC]": pa.timestamp("us", tz="UTC"),
}
ARROW_SCHEMA = pa.schema(
    [pa.field(column.name, ARROW_TYPES[column.dtype], column.nullable) for column in COLUMNS]
)
ALL_COLUMNS = frozenset(column.name for column in COLUMNS)


class BuildError(RuntimeError):
    """The build cannot produce a trustworthy release; nothing was written."""


@dataclass(frozen=True)
class Source:
    type: Literal["local", "api"]
    location: str

    @classmethod
    def parse(cls, text: str) -> "Source":
        kind, separator, location = text.partition(":")
        if not separator or kind not in ("local", "api") or not location.strip():
            raise ValueError(f"Source {text!r} must be local:<directory> or api:<url>")
        return cls(kind, location)

    def __str__(self) -> str:
        return f"{self.type}:{self.location}"


class Redactor:
    """Replace host paths in anything written to a release with short source labels.

    Local sources are extracted by their resolved path, so every origin and message names
    one exact form of the path. Labels use only the directory name, never the path.
    """

    def __init__(self, sources: list[Source]):
        local = [source for source in sources if source.type == "local"]
        names = Counter(Path(source.location).resolve().name for source in local)
        self.labels: dict[Source, str] = {}
        replacements = {}
        for index, source in enumerate(local, start=1):
            resolved = Path(source.location).resolve()
            name = resolved.name or "root"
            label = f"<local:{name}>" if names[resolved.name] == 1 else f"<local:{name}#{index}>"
            self.labels[source] = label
            replacements[str(resolved)] = label
        replacements[str(Path.home().resolve())] = "~"
        # Longest first, so a source inside the home directory keeps its own label.
        self.replacements = sorted(replacements.items(), key=lambda item: -len(item[0]))

    def label(self, source: Source) -> str:
        return self.labels.get(source, str(source))

    def __call__(self, text: str) -> str:
        for path, label in self.replacements:
            text = text.replace(path, label)
        return text


def extract(source: Source) -> Extraction:
    if source.type == "local":
        return extract_local(Path(source.location).resolve())
    return extract_api(source.location)


def step_order(row: dict) -> tuple:
    step_id = row["step_id"]
    repeat = int(step_id.rsplit("-", 1)[1]) if row["condition"] == "baseline" else 0
    return row["run_id"], CONDITION_ORDER[row["condition"]], repeat


def rows_digest(rows: list[dict]) -> str:
    """Identify the dataset content independently of file encoding."""
    serializable = [
        row | {"recorded_at": row["recorded_at"].isoformat() if row["recorded_at"] else None}
        for row in rows
    ]
    return canonical_sha256(serializable)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rejection_records(stage: str, rejections: list[Rejection], redact: Redactor) -> list[dict]:
    return [
        {"stage": stage, "origin": redact(item.origin), "reason": redact(item.reason)}
        for item in rejections
    ]


def counts(rows: list[dict], column: str) -> dict:
    return dict(sorted(Counter(str(row[column]) for row in rows).items()))


def build(
    sources: list[Source],
    output_dir: Path,
    rules: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Build a release under output_dir and return its directory."""
    if not sources:
        raise BuildError("At least one source is required")
    redact = Redactor(sources)
    try:
        config = load_config(rules)
    except LabelingError as error:
        raise BuildError(f"Label rules are invalid: {error}") from error

    extractions, source_summaries = [], []
    for source in sources:
        try:
            extraction = extract(source)
        except ExtractionError as error:
            raise BuildError(redact(f"Source {source} could not be read: {error}")) from error
        extractions.append(extraction)
        source_summaries.append(
            {
                "source": redact.label(source),
                "runs": len(extraction.runs),
                "rejections": len(extraction.rejections),
            }
        )
    merged = combine(*extractions)
    rows, flatten_rejections = flatten_runs(merged.runs)
    labeled, label_rejections = label_rows(rows, config)
    rejections = (
        rejection_records("extract", merged.rejections, redact)
        + rejection_records("flatten", flatten_rejections, redact)
        + rejection_records("label", label_rejections, redact)
    )
    if not labeled:
        raise BuildError(f"No usable rows: every run was rejected ({len(rejections)} rejections)")

    labeled.sort(key=step_order)
    digest = rows_digest(labeled)
    version = f"{SCHEMA_VERSION}+{digest[:12]}"
    final = [row | {"dataset_version": version} for row in labeled]
    try:
        validate_rows(final, ALL_COLUMNS)
    except SchemaError as error:
        raise BuildError(f"Rows violate the dataset schema: {error}") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    release = output_dir / version
    if release.exists():
        return verify_existing(release, digest)

    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=output_dir))
    try:
        # mkdtemp creates the directory owner-only; releases are read by other tools
        # (DVC, Airflow workers, training jobs) that may run as other users.
        staging.chmod(0o755)
        dataset = staging / DATASET_FILE
        pq.write_table(pa.Table.from_pylist(final, schema=ARROW_SCHEMA), dataset)
        manifest = {
            "dataset_version": version,
            "schema_version": SCHEMA_VERSION,
            "label_version": config.label_version,
            "builder_version": __version__,
            "created_at": (now or datetime.now(UTC)).isoformat(),
            "rows": len(final),
            "runs": len({row["run_id"] for row in final}),
            "tasks": len({row["task_id"] for row in final}),
            "rows_sha256": digest,
            "parquet_sha256": file_sha256(dataset),
            "label_resource": counts(final, "label_resource"),
            "label_basis": counts(final, "label_basis"),
            "condition": counts(final, "condition"),
            "telemetry_status": counts(final, "telemetry_status"),
            "sources": source_summaries,
            "duplicate_copies": len(merged.duplicates),
            "rejections": rejections,
        }
        (staging / MANIFEST_FILE).write_text(json_text(manifest), encoding="utf-8")
        try:
            os.replace(staging, release)  # atomic on one filesystem: the release appears whole
        except OSError:
            if not release.exists():
                raise
            return verify_existing(release, digest)  # a concurrent build finished first
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    logger.info(
        "Built %s: %d rows from %d runs, %d rejections",
        version,
        len(final),
        manifest["runs"],
        len(rejections),
    )
    return release


def verify_existing(release: Path, digest: str) -> Path:
    """A release with this version already exists; confirm it is intact and identical."""
    try:
        manifest = json.loads((release / MANIFEST_FILE).read_text(encoding="utf-8"))
        intact = manifest["rows_sha256"] == digest and manifest["parquet_sha256"] == file_sha256(
            release / DATASET_FILE
        )
    except (OSError, ValueError, KeyError) as error:
        raise BuildError(f"Existing release {release.name} is unreadable: {error}") from error
    if not intact:
        raise BuildError(f"Existing release {release.name} does not match its manifest")
    logger.info("Release %s already exists and is intact", release.name)
    return release


def json_text(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="local:<directory> or api:<url>; repeat to combine sources",
    )
    parser.add_argument("--output", type=Path, default=Path(".data/datasets"))
    parser.add_argument("--rules", type=Path, help="label rules (default: packaged labels.toml)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sources = [Source.parse(text) for text in args.source]
        release = build(sources, args.output, args.rules)
    except (ValueError, BuildError) as error:
        print(f"Dataset build failed: {error}", file=sys.stderr)
        return 1
    print(release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
