"""Extract completed runs from the ingestion API or local experiment directories.

Every run is validated with the ingestion contracts before it reaches the dataset.
Two kinds of failure are kept apart:

- Invalid evidence (a run that breaks the contracts) becomes a ``Rejection`` with a reason,
  and extraction continues with the other runs.
- An unreadable source (network failure after bounded retries, malformed API responses,
  a missing directory) raises ``ExtractionError``, so no partial dataset is ever published
  as if it were complete.
"""

import hashlib
import http.client
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from agentwatch_worker.contracts import RunResult, Sample
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError, field_validator

logger = logging.getLogger(__name__)

# Local run directories are named <prefix>-<anything>; the Makefile writes cpu-demo,
# coding-demo, and llm-demo. The LLM loop records coding experiments.
KIND_BY_DIRECTORY_PREFIX = {"cpu": "cpu", "coding": "coding", "llm": "coding"}
# Directories whose runs must carry the LLM agent evidence (result["agent"]).
AGENT_DIRECTORY_PREFIXES = frozenset({"llm"})
# Experiment check names are identifiers; anything else could not be stored unambiguously
# in the comma-separated failed_checks column.
CHECK_NAME = re.compile(r"[a-z][a-z0-9_]*")
RESULTS_FILE, SAMPLES_FILE = "results.json", "samples.jsonl"
API_MAX_PAGE_SIZE = 100  # GET /v1/runs rejects larger limits
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
# Bounds on one API response, so a slow or oversized response cannot stall extraction.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
# RFC 3339 date and time with an explicit offset; numeric strings are not timestamps.
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})")
# Recorded evidence nests 4 levels deep (samples 2). The limit is explicit because whether
# deep JSON fails to parse depends on the Python version: 3.13 raises RecursionError, while
# 3.14.7 parses it and leaves later stages to fail differently.
MAX_JSON_DEPTH = 64
# When identical copies of a run differ only in telemetry status, keep the most informative.
TELEMETRY_PREFERENCE = {"present": 0, "empty": 1, "missing": 2}

# Whether a run carried resource samples at all; distinct from samples that miss a step.
TELEMETRY_PRESENT, TELEMETRY_EMPTY, TELEMETRY_MISSING = "present", "empty", "missing"


class ExtractionError(RuntimeError):
    """A source could not be read reliably; the extraction must not be used."""


class UnusableRunDirectory(ValueError):
    """A run directory that cannot hold trustworthy evidence, whatever its contents."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retries with exponential backoff, for transient network failures only."""

    attempts: int = 3
    initial_backoff_seconds: float = 0.5
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self):
        if self.attempts < 1 or self.initial_backoff_seconds < 0:
            raise ValueError("Retry attempts must be positive and backoff nonnegative")


DEFAULT_RETRY = RetryPolicy()


@dataclass(frozen=True)
class RawRun:
    run_id: str
    kind: str
    source: str
    result: dict  # contract-normalized
    samples: list[dict]  # contract-normalized, ordered by timestamp, duplicates collapsed
    telemetry_status: str
    source_sha256: str  # results and samples exactly as received, for lineage
    content_sha256: str  # normalized kind, result, and samples, for duplicate detection
    origin: str
    recorded_at: datetime | None = None


@dataclass(frozen=True)
class Rejection:
    origin: str
    reason: str


@dataclass(frozen=True)
class Duplicate:
    origin: str  # an identical extra copy that was not kept
    run_id: str


@dataclass
class Extraction:
    runs: list[RawRun] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    duplicates: list[Duplicate] = field(default_factory=list)
    # Run IDs with conflicting evidence. Kept as data, not only as rejection text, so a
    # copy of the same run arriving later from another source is rejected too.
    conflicted: set[str] = field(default_factory=set)


class ApiRunRow(BaseModel):
    """One entry of GET /v1/runs, validated before its evidence is trusted.

    Strict, so an integer created_at is not read as a Unix timestamp, and created_at must be
    an RFC 3339 date and time: pydantic would otherwise read "1791200000" as one.
    """

    model_config = ConfigDict(extra="ignore", strict=True)
    run_id: UUID
    kind: Literal["coding", "cpu"]
    result: dict
    created_at: AwareDatetime

    @field_validator("created_at", mode="plain")
    @classmethod
    def rfc3339(cls, value) -> datetime:
        # Replaces pydantic's parsing: under any validator, strict mode stops accepting strings.
        if not isinstance(value, str) or not TIMESTAMP.fullmatch(value):
            raise ValueError("created_at must be an RFC 3339 date and time with an offset")
        return datetime.fromisoformat(value)  # also rejects impossible dates


def canonical_sha256(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def evidence_hash(result: dict, samples: Iterable[dict] = ()) -> str:
    """Identify a run's evidence: its results and every resource sample."""
    return canonical_sha256({"result": result, "samples": list(samples)})


def unique_keys(pairs: list[tuple]) -> dict:
    """A JSON object whose keys are all distinct; json would silently keep the last one."""
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"JSON object repeats keys {repeated}")
    return dict(pairs)


def parse_json(text: str):
    """Strict JSON: repeated keys are an error, and so is nesting beyond MAX_JSON_DEPTH.

    Some Python versions raise RecursionError (not ValueError) on deep nesting, which left
    alone would abort the whole extraction instead of rejecting one run; others parse it.
    Both end in the same ValueError, so behavior does not depend on the interpreter.
    """
    too_deep = f"JSON is nested more than {MAX_JSON_DEPTH} levels deep"
    try:
        value = json.loads(text, object_pairs_hook=unique_keys)
    except RecursionError as error:
        raise ValueError(too_deep) from error
    if nesting_depth(value) > MAX_JSON_DEPTH:
        raise ValueError(too_deep)
    return value


def nesting_depth(value) -> int:
    """How many objects and arrays deep value goes, counted without recursion."""
    deepest, pending = 0, [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, dict | list):
            deepest = max(deepest, depth)
            children = item.values() if isinstance(item, dict) else item
            pending.extend((child, depth + 1) for child in children)
    return deepest


def parse_jsonl(text: str, name: str) -> list[dict]:
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(parse_json(line))
        except ValueError as error:
            raise ValueError(f"{name} line {number}: {getattr(error, 'msg', error)}") from error
    return rows


def collapse_duplicate_samples(samples: list[dict]) -> list[dict]:
    """Keep one copy of identical samples; reject samples that disagree at one timestamp."""
    collapsed: list[dict] = []
    for sample in samples:
        if collapsed and collapsed[-1]["timestamp_ns"] == sample["timestamp_ns"]:
            if collapsed[-1] != sample:
                raise ValueError(f"Samples disagree at timestamp_ns {sample['timestamp_ns']}")
            continue
        collapsed.append(sample)
    return collapsed


def validate(
    result: dict,
    samples: list[dict],
    kind: str,
    recorded_at: datetime | None = None,
    *,
    origin: str = "",
    telemetry_status: str | None = None,
) -> RawRun:
    """Return contract-normalized evidence after strict validation.

    Validation is strict and JSON-mode: the ingestion contracts would otherwise turn true
    into 1 and "10" into 10, so evidence of the wrong type would pass every later check.
    """
    run = RunResult.model_validate_json(json.dumps(result), strict=True)
    checked = [
        Sample.model_validate_json(json.dumps(sample), strict=True).model_dump(mode="json")
        for sample in samples
    ]
    check_experiment(run.model_dump(mode="json"), kind)
    order = sorted(range(len(samples)), key=lambda index: checked[index]["timestamp_ns"])
    normalized_samples = collapse_duplicate_samples([checked[index] for index in order])
    normalized_result = run.model_dump(mode="json")
    if telemetry_status is None:
        telemetry_status = TELEMETRY_PRESENT if normalized_samples else TELEMETRY_EMPTY
    return RawRun(
        run_id=str(run.run_id),
        kind=kind,
        source="measured",
        result=normalized_result,
        samples=normalized_samples,
        telemetry_status=telemetry_status,
        source_sha256=evidence_hash(result, [samples[index] for index in order]),
        content_sha256=canonical_sha256(
            {"kind": kind, "result": normalized_result, "samples": normalized_samples}
        ),
        origin=origin,
        recorded_at=recorded_at,
    )


def check_experiment(result: dict, kind: str) -> None:
    """Require the evidence to agree with its declared kind and to name checks plainly."""
    tests = result["experiment"].get("tests_per_step")
    if kind == "coding" and not (type(tests) is int and tests >= 1):
        raise ValueError("A coding run must declare tests_per_step; this evidence is a CPU run")
    if kind == "cpu" and tests is not None:
        raise ValueError("A cpu run must not declare tests_per_step; this evidence is a coding run")
    for name in result["checks"]:
        if not CHECK_NAME.fullmatch(name):
            raise ValueError(f"Experiment check name {name!r} is not a lowercase identifier")


def reject(extraction: Extraction, origin: str, reason: str) -> None:
    logger.warning("Rejected run evidence from %s: %s", origin, reason)
    extraction.rejections.append(Rejection(origin, reason))


def directory_prefix(name: str) -> str:
    return name.split("-", 1)[0]


def kind_for_directory(name: str) -> str | None:
    return KIND_BY_DIRECTORY_PREFIX.get(directory_prefix(name))


def run_directories(data_dir: Path, extraction: Extraction) -> list[Path]:
    """Every directory under data_dir holding results.json, without following links.

    A linked run directory would otherwise vanish without a trace, so it is rejected.
    """
    found = []
    for root, directories, files in os.walk(data_dir):  # does not follow directory links
        # Hidden directories hold staging or tool state, never evidence; DVC excludes them too.
        directories[:] = sorted(name for name in directories if not name.startswith("."))
        for name in directories:
            linked = Path(root) / name
            if linked.is_symlink() and (linked / RESULTS_FILE).exists():
                reason = "Run directory is a symbolic link, which is not followed; copy it"
                reject(extraction, str(linked), reason)
        if RESULTS_FILE in files:
            found.append(Path(root))
    return sorted(found)


def require_kind(run_dir: Path) -> str:
    kind = kind_for_directory(run_dir.name)
    if kind is None:
        prefixes = ", ".join(f"{prefix}-*" for prefix in KIND_BY_DIRECTORY_PREFIX)
        raise UnusableRunDirectory(f"Unknown experiment kind; name it {prefixes}")
    return kind


def read_run(run_dir: Path) -> RawRun:
    """Read and validate one local run directory; ValueError or OSError says why it is unusable.

    The directory name gives the experiment kind (cpu-*, coding-*, llm-*), so the same rules
    apply wherever a run is read: building a dataset or curating evidence.
    """
    return run_from_bytes(run_dir, *read_run_files(run_dir))


def read_run_files(run_dir: Path) -> tuple[bytes, bytes | None]:
    """A run directory's results.json bytes and samples.jsonl bytes (None when absent)."""
    require_kind(run_dir)
    results_path, samples_path = run_dir / RESULTS_FILE, run_dir / SAMPLES_FILE
    if results_path.is_symlink() or samples_path.is_symlink():
        # A link can point anywhere, so the evidence would not be the directory's own.
        raise UnusableRunDirectory("Evidence files must not be symbolic links")
    results = results_path.read_bytes()
    samples = samples_path.read_bytes() if samples_path.exists() else None
    return results, samples


def run_from_bytes(run_dir: Path, results: bytes, samples: bytes | None) -> RawRun:
    """Validate a run from the exact bytes read, so a caller can store exactly what it checked.

    run_dir supplies the kind (from its name) and the origin; its files are not read again.
    """
    kind = require_kind(run_dir)
    result = parse_json(results.decode("utf-8"))
    rows = [] if samples is None else parse_jsonl(samples.decode("utf-8"), SAMPLES_FILE)
    status = TELEMETRY_MISSING if samples is None else None
    run = validate(result, rows, kind, origin=str(run_dir), telemetry_status=status)
    if directory_prefix(run_dir.name) in AGENT_DIRECTORY_PREFIXES and not isinstance(
        run.result.get("agent"), dict
    ):
        raise ValueError("An llm-* run has no agent evidence; the LLM loop did not finish")
    return run


def extract_local(data_dir: Path) -> Extraction:
    """Read every run directory (one containing results.json) anywhere under data_dir."""
    if not data_dir.is_dir():
        raise ExtractionError(f"Local evidence directory {data_dir} does not exist")
    extraction = Extraction()
    for run_dir in run_directories(data_dir, extraction):
        try:
            run = read_run(run_dir)
        except (OSError, ValueError) as error:  # pydantic ValidationError is a ValueError
            reject(extraction, str(run_dir), summarize(error))
            continue
        extraction.runs.append(run)
    return finish(extraction, str(data_dir))


def read_body(
    response, url: str, timeout: float, clock: Callable[[], float] = time.monotonic
) -> bytes:
    """Read a response within one overall deadline and size limit.

    The socket timeout bounds each read, not the whole response, so a server sending a few
    bytes at a time could stall extraction indefinitely. Exceeding the deadline is a
    retryable timeout; an oversized response is not.
    """
    deadline = clock() + timeout
    body = bytearray()
    while chunk := response.read1(READ_CHUNK_BYTES):
        body += chunk
        if len(body) > MAX_RESPONSE_BYTES:
            raise ExtractionError(f"GET {url} returned more than {MAX_RESPONSE_BYTES} bytes")
        if clock() > deadline:
            raise TimeoutError(f"GET {url} took longer than {timeout}s")
    return bytes(body)


def get_json(url: str, timeout: float):
    """One HTTP GET; JSON or newline-delimited JSON depending on the response type."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        text = read_body(response, url, timeout).decode()
        if "ndjson" in response.headers.get("Content-Type", ""):
            return [parse_json(line) for line in text.splitlines() if line]
        return parse_json(text)


def fetch_json(url: str, timeout: float, retry: RetryPolicy):
    """GET with bounded retries for transient failures; anything else fails immediately."""
    delay = retry.initial_backoff_seconds
    for attempt in range(1, retry.attempts + 1):
        try:
            return get_json(url, timeout)
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_HTTP_STATUS:
                raise ExtractionError(f"GET {url} returned HTTP {error.code}") from error
            failure: Exception = error
        except (OSError, http.client.HTTPException) as error:  # URLError, timeouts, resets
            failure = error
        except ValueError as error:
            raise ExtractionError(f"GET {url} returned malformed JSON: {error}") from error
        if attempt < retry.attempts:
            logger.warning("GET %s failed (%s); retry %d in %.1fs", url, failure, attempt, delay)
            retry.sleep(delay)
            delay *= 2
    raise ExtractionError(f"GET {url} failed after {retry.attempts} attempts: {failure}")


def parse_page(page, url: str, limit: int) -> tuple[list, int | None]:
    if not isinstance(page, dict) or not isinstance(page.get("runs"), list):
        raise ExtractionError(f"GET {url} did not return a run history page")
    runs = page["runs"]
    if len(runs) > limit:
        raise ExtractionError(f"GET {url} returned {len(runs)} runs for a limit of {limit}")
    # A missing cursor is a malformed page, not the last page: treating it as null would
    # end extraction successfully and silently omit every older run.
    if "next_cursor" not in page:
        raise ExtractionError(f"GET {url} returned a history page without next_cursor")
    cursor = page["next_cursor"]
    if cursor is not None and (type(cursor) is not int or cursor <= 0):
        raise ExtractionError(f"GET {url} returned an invalid next_cursor {cursor!r}")
    # The API sets a cursor only when more runs exist, after a full page; a cursor after a
    # short or empty page would let a broken server keep extraction paging indefinitely.
    if cursor is not None and len(runs) != limit:
        raise ExtractionError(f"GET {url} returned a cursor after a partial page")
    return runs, cursor


def extract_api_row(base: str, row, extraction: Extraction, timeout: float, retry: RetryPolicy):
    origin = f"{base}/v1/runs/{row.get('run_id') if isinstance(row, dict) else '?'}"
    try:
        listed = ApiRunRow.model_validate_json(json.dumps(row))
    except ValidationError as error:
        reject(extraction, origin, summarize(error))
        return
    # Transport failures here propagate: a listed run whose samples cannot be read
    # must stop the extraction rather than silently lose its telemetry.
    samples = fetch_json(f"{base}/v1/runs/{listed.run_id}/artifacts/{SAMPLES_FILE}", timeout, retry)
    if not isinstance(samples, list):
        raise ExtractionError(f"{origin}: samples artifact is not a list")
    try:
        run = validate(listed.result, samples, listed.kind, listed.created_at, origin=origin)
    except ValueError as error:
        reject(extraction, origin, summarize(error))
        return
    if run.run_id != str(listed.run_id):
        reject(extraction, origin, f"Listed run ID does not match its result ({run.run_id})")
        return
    extraction.runs.append(run)


def extract_api(
    api_url: str,
    page_size: int = API_MAX_PAGE_SIZE,
    timeout: float = 15,
    retry: RetryPolicy | None = None,
) -> Extraction:
    """Read every saved run, following the history cursor until the last page."""
    if not 1 <= page_size <= API_MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {API_MAX_PAGE_SIZE}")
    parsed = urllib.parse.urlparse(api_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"API URL {api_url!r} must be http:// or https:// with a host")
    retry = retry or DEFAULT_RETRY  # read at call time, so callers and tests can replace it
    base = api_url.rstrip("/")
    extraction = Extraction()
    before = None
    while True:
        query = {"limit": page_size} | ({"before": before} if before else {})
        url = f"{base}/v1/runs?{urllib.parse.urlencode(query)}"
        rows, cursor = parse_page(fetch_json(url, timeout, retry), url, page_size)
        for row in rows:
            extract_api_row(base, row, extraction, timeout, retry)
        if cursor is None:
            return finish(extraction, base)
        # History is ordered by descending sequence, so each cursor must be smaller.
        if before is not None and cursor >= before:
            raise ExtractionError(f"GET {url} returned a cursor that does not advance")
        before = cursor


def deduplicate(extraction: Extraction) -> Extraction:
    """Keep one copy of runs seen more than once; reject runs whose copies disagree.

    Which identical copy is kept never depends on source order, because its recorded_at
    and source_sha256 reach the rows and so the dataset version: a copy with recorded_at
    (from the API) is preferred, then the earliest recorded_at, then the most telemetry
    (present, then empty, then missing), then the smallest source_sha256 and origin. Once a
    run ID has conflicting evidence, every copy of it is rejected, including copies that
    arrive later through combine().
    """
    copies: dict[str, list[RawRun]] = {}
    for run in extraction.runs:
        copies.setdefault(run.run_id, []).append(run)
    unique = Extraction(
        rejections=list(extraction.rejections),
        duplicates=list(extraction.duplicates),
        conflicted=set(extraction.conflicted),
    )
    for run_id, found in copies.items():
        if run_id in extraction.conflicted:
            for copy in found:
                reject(unique, copy.origin, f"Run {run_id} has conflicting evidence elsewhere")
            continue
        if len({copy.content_sha256 for copy in found}) > 1:
            unique.conflicted.add(run_id)
            for copy in found:
                reject(unique, copy.origin, f"Run {run_id} has conflicting evidence in sources")
            continue
        keep = min(found, key=preference)
        unique.runs.append(keep)
        unique.duplicates.extend(Duplicate(c.origin, run_id) for c in found if c is not keep)
    # A copy dropped earlier as an identical duplicate is rejected once its run conflicts.
    still_duplicates = []
    for duplicate in unique.duplicates:
        if duplicate.run_id in unique.conflicted:
            reject(unique, duplicate.origin, f"Run {duplicate.run_id} has conflicting evidence")
        else:
            still_duplicates.append(duplicate)
    unique.duplicates = still_duplicates
    return unique


def preference(copy: RawRun) -> tuple:
    recorded = (
        copy.recorded_at.astimezone(UTC) if copy.recorded_at else datetime.max.replace(tzinfo=UTC)
    )
    return (
        copy.recorded_at is None,
        recorded,
        TELEMETRY_PREFERENCE[copy.telemetry_status],
        copy.source_sha256,
        copy.origin,
    )


def combine(*extractions: Extraction) -> Extraction:
    """Merge extractions from several sources, detecting duplicates across them."""
    merged = Extraction()
    for extraction in extractions:
        merged.runs.extend(extraction.runs)
        merged.rejections.extend(extraction.rejections)
        merged.duplicates.extend(extraction.duplicates)
        merged.conflicted |= extraction.conflicted
    return deduplicate(merged)


def finish(extraction: Extraction, source: str) -> Extraction:
    unique = deduplicate(extraction)
    logger.info(
        "Extracted %d runs from %s (%d rejected, %d duplicate copies)",
        len(unique.runs),
        source,
        len(unique.rejections),
        len(unique.duplicates),
    )
    if not unique.runs:
        logger.warning("No usable runs were found in %s", source)
    return unique


def summarize(error: Exception) -> str:
    if isinstance(error, UnusableRunDirectory):
        return str(error)
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "run"
        return f"{location}: {first['msg']}"
    return f"{type(error).__name__}: {error}"
