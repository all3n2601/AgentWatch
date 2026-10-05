"""Extract completed runs from the ingestion API or local experiment directories.

Every run is validated with the ingestion contracts before it reaches the dataset.
Runs that fail validation are reported as rejections instead of being silently dropped.
"""

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agentwatch_worker.contracts import RunResult, Sample
from pydantic import ValidationError

# Directories written by the Makefile experiment commands, and the kind each one records.
LOCAL_RUN_DIRS = {"cpu-demo": "cpu", "coding-demo": "coding", "llm-demo": "coding"}


@dataclass(frozen=True)
class RawRun:
    run_id: str
    kind: str
    source: str
    result: dict
    samples: list[dict]
    source_sha256: str
    recorded_at: datetime | None = None


@dataclass(frozen=True)
class Rejection:
    origin: str
    reason: str


@dataclass
class Extraction:
    runs: list[RawRun] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)


def evidence_hash(result: dict) -> str:
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(result: dict, samples: list[dict], kind: str, recorded_at=None) -> RawRun:
    run = RunResult.model_validate(result)
    checked = [Sample.model_validate(sample).model_dump(mode="json") for sample in samples]
    return RawRun(
        run_id=str(run.run_id),
        kind=kind,
        source="measured",
        result=result,
        samples=checked,
        source_sha256=evidence_hash(result),
        recorded_at=recorded_at,
    )


def extract_local(data_dir: Path) -> Extraction:
    extraction = Extraction()
    for name, kind in LOCAL_RUN_DIRS.items():
        run_dir = data_dir / name
        if not (run_dir / "results.json").exists():
            continue
        try:
            result = json.loads((run_dir / "results.json").read_text())
            extraction.runs.append(validate(result, read_jsonl(run_dir / "samples.jsonl"), kind))
        except (OSError, ValueError, ValidationError) as error:
            extraction.rejections.append(Rejection(str(run_dir), summarize(error)))
    return extraction


def get_json(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if "ndjson" in response.headers.get("Content-Type", ""):
            return [json.loads(line) for line in response.read().decode().splitlines() if line]
        return json.load(response)


def extract_api(api_url: str, page_size: int = 100, timeout: float = 15) -> Extraction:
    """Read every saved run, following the history cursor until the last page."""
    base = api_url.rstrip("/")
    extraction = Extraction()
    before = None
    while True:
        query = {"limit": page_size} | ({"before": before} if before else {})
        page = get_json(f"{base}/v1/runs?{urllib.parse.urlencode(query)}", timeout)
        for row in page["runs"]:
            origin = f"{base}/v1/runs/{row['run_id']}"
            try:
                samples = get_json(f"{origin}/artifacts/samples.jsonl", timeout)
                recorded_at = datetime.fromisoformat(row["created_at"])
                extraction.runs.append(validate(row["result"], samples, row["kind"], recorded_at))
            except (OSError, ValueError, ValidationError) as error:
                extraction.rejections.append(Rejection(origin, summarize(error)))
        before = page["next_cursor"]
        if before is None:
            return extraction


def summarize(error: Exception) -> str:
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "run"
        return f"{location}: {first['msg']}"
    return f"{type(error).__name__}: {error}"
