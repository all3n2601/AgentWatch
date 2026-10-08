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
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from agentwatch_worker.contracts import RunResult, Sample
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

# Local run directories are named <prefix>-<anything>; the Makefile writes cpu-demo,
# coding-demo, and llm-demo. The LLM loop records coding experiments.
KIND_BY_DIRECTORY_PREFIX = {"cpu": "cpu", "coding": "coding", "llm": "coding"}
RESULTS_FILE, SAMPLES_FILE = "results.json", "samples.jsonl"
API_MAX_PAGE_SIZE = 100  # GET /v1/runs rejects larger limits
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

# Whether a run carried resource samples at all; distinct from samples that miss a step.
TELEMETRY_PRESENT, TELEMETRY_EMPTY, TELEMETRY_MISSING = "present", "empty", "missing"


class ExtractionError(RuntimeError):
    """A source could not be read reliably; the extraction must not be used."""


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


@dataclass
class Extraction:
    runs: list[RawRun] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)  # origins of identical extra copies


class ApiRunRow(BaseModel):
    """One entry of GET /v1/runs, validated before its evidence is trusted."""

    model_config = ConfigDict(extra="ignore")
    run_id: UUID
    kind: Literal["coding", "cpu"]
    result: dict
    created_at: AwareDatetime


def canonical_sha256(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def evidence_hash(result: dict, samples: Iterable[dict] = ()) -> str:
    """Identify a run's evidence: its results and every resource sample."""
    return canonical_sha256({"result": result, "samples": list(samples)})


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name} line {number}: {error.msg}") from error
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
    """Return contract-normalized evidence, so coerced values carry their declared types."""
    run = RunResult.model_validate(result)
    checked = [Sample.model_validate(sample).model_dump(mode="json") for sample in samples]
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


def reject(extraction: Extraction, origin: str, reason: str) -> None:
    logger.warning("Rejected run evidence from %s: %s", origin, reason)
    extraction.rejections.append(Rejection(origin, reason))


def kind_for_directory(name: str) -> str | None:
    return KIND_BY_DIRECTORY_PREFIX.get(name.split("-", 1)[0])


def extract_local(data_dir: Path) -> Extraction:
    """Read every run directory (one containing results.json) anywhere under data_dir."""
    if not data_dir.is_dir():
        raise ExtractionError(f"Local evidence directory {data_dir} does not exist")
    extraction = Extraction()
    for results_path in sorted(data_dir.rglob(RESULTS_FILE)):
        run_dir = results_path.parent
        kind = kind_for_directory(run_dir.name)
        if kind is None:
            prefixes = ", ".join(f"{prefix}-*" for prefix in KIND_BY_DIRECTORY_PREFIX)
            reject(extraction, str(run_dir), f"Unknown experiment kind; name it {prefixes}")
            continue
        samples_path = run_dir / SAMPLES_FILE
        try:
            result = json.loads(results_path.read_text(encoding="utf-8"))
            samples = read_jsonl(samples_path) if samples_path.exists() else []
            status = None if samples_path.exists() else TELEMETRY_MISSING
            run = validate(result, samples, kind, origin=str(run_dir), telemetry_status=status)
        except (OSError, ValueError) as error:  # pydantic ValidationError is a ValueError
            reject(extraction, str(run_dir), summarize(error))
            continue
        extraction.runs.append(run)
    return finish(extraction, str(data_dir))


def get_json(url: str, timeout: float):
    """One HTTP GET; JSON or newline-delimited JSON depending on the response type."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if "ndjson" in response.headers.get("Content-Type", ""):
            return [json.loads(line) for line in response.read().decode().splitlines() if line]
        return json.load(response)


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


def parse_page(page, url: str) -> tuple[list, int | None]:
    if not isinstance(page, dict) or not isinstance(page.get("runs"), list):
        raise ExtractionError(f"GET {url} did not return a run history page")
    cursor = page.get("next_cursor")
    if cursor is not None and (type(cursor) is not int or cursor <= 0):
        raise ExtractionError(f"GET {url} returned an invalid next_cursor {cursor!r}")
    return page["runs"], cursor


def extract_api_row(base: str, row, extraction: Extraction, timeout: float, retry: RetryPolicy):
    origin = f"{base}/v1/runs/{row.get('run_id') if isinstance(row, dict) else '?'}"
    try:
        listed = ApiRunRow.model_validate(row)
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
    retry: RetryPolicy = DEFAULT_RETRY,
) -> Extraction:
    """Read every saved run, following the history cursor until the last page."""
    if not 1 <= page_size <= API_MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {API_MAX_PAGE_SIZE}")
    base = api_url.rstrip("/")
    extraction = Extraction()
    before = None
    while True:
        query = {"limit": page_size} | ({"before": before} if before else {})
        url = f"{base}/v1/runs?{urllib.parse.urlencode(query)}"
        rows, cursor = parse_page(fetch_json(url, timeout, retry), url)
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

    A copy with a recorded_at timestamp (from the API) is preferred over a local copy.
    """
    copies: dict[str, list[RawRun]] = {}
    for run in extraction.runs:
        copies.setdefault(run.run_id, []).append(run)
    unique = Extraction(
        rejections=list(extraction.rejections), duplicates=list(extraction.duplicates)
    )
    for run_id, found in copies.items():
        if len({copy.content_sha256 for copy in found}) > 1:
            for copy in found:
                reject(unique, copy.origin, f"Run {run_id} has conflicting evidence in sources")
            continue
        keep = next((copy for copy in found if copy.recorded_at is not None), found[0])
        unique.runs.append(keep)
        unique.duplicates.extend(copy.origin for copy in found if copy is not keep)
    return unique


def combine(*extractions: Extraction) -> Extraction:
    """Merge extractions from several sources, detecting duplicates across them."""
    merged = Extraction()
    for extraction in extractions:
        merged.runs.extend(extraction.runs)
        merged.rejections.extend(extraction.rejections)
        merged.duplicates.extend(extraction.duplicates)
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
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "run"
        return f"{location}: {first['msg']}"
    return f"{type(error).__name__}: {error}"
