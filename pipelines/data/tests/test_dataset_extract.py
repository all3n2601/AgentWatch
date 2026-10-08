import json
import logging
import urllib.error
from uuid import uuid4

import pytest
from agentwatch_data import extract
from agentwatch_data.extract import (
    TELEMETRY_EMPTY,
    TELEMETRY_MISSING,
    TELEMETRY_PRESENT,
    ExtractionError,
    RetryPolicy,
    combine,
    evidence_hash,
    extract_api,
    extract_local,
)
from dataset_fixtures import BASE_NS, SAMPLE, make_result, write_run


@pytest.fixture
def result():
    return make_result()


def sample(offset, avg10=1.0):
    return {
        "timestamp_ns": BASE_NS + offset,
        "cpu_pressure": {"some": {"avg10": avg10}},
        "cgroup_cpu_stat": None,
        "unavailable": [],
    }


class FakeApi:
    """In-memory GET /v1/runs history with scripted failures."""

    def __init__(self, base="http://api"):
        self.base = base
        self.pages = {}
        self.samples = {}
        self.failures = {}
        self.calls = []

    def add_page(self, before, rows, next_cursor, limit=100):
        query = f"limit={limit}" + (f"&before={before}" if before else "")
        self.pages[f"{self.base}/v1/runs?{query}"] = {"runs": rows, "next_cursor": next_cursor}

    def row(self, result, kind="cpu", created_at="2026-10-05T12:00:00+00:00", samples=()):
        self.samples[result["run_id"]] = list(samples)
        return {
            "run_id": result["run_id"],
            "kind": kind,
            "result": result,
            "created_at": created_at,
        }

    def fail(self, url_suffix, *errors):
        self.failures[url_suffix] = list(errors)

    def __call__(self, url, timeout):
        self.calls.append(url)
        for suffix, errors in self.failures.items():
            if url.endswith(suffix) and errors:
                raise errors.pop(0)
        if url.endswith("/artifacts/samples.jsonl"):
            return self.samples[url.split("/v1/runs/")[1].split("/")[0]]
        return self.pages[url]


@pytest.fixture
def api(monkeypatch):
    fake = FakeApi()
    monkeypatch.setattr(extract, "get_json", fake)
    return fake


@pytest.fixture
def no_sleep():
    delays = []
    return RetryPolicy(attempts=3, initial_backoff_seconds=0.5, sleep=delays.append), delays


def http_error(code):
    return urllib.error.HTTPError("http://api", code, "error", hdrs=None, fp=None)


# Local directories


def test_local_runs_are_validated_and_tagged(tmp_path, result):
    write_run(tmp_path / "cpu-demo", result)
    extraction = extract_local(tmp_path)
    assert extraction.rejections == []
    [run] = extraction.runs
    assert (run.run_id, run.kind, run.source) == (result["run_id"], "cpu", "measured")
    assert run.samples == [SAMPLE]
    assert run.telemetry_status == TELEMETRY_PRESENT
    assert run.origin == str(tmp_path / "cpu-demo")
    assert run.source_sha256 == evidence_hash(result, [SAMPLE])


def test_runs_are_discovered_in_any_named_directory(tmp_path):
    for name in ("cpu-300000", "coding-run-7", "nested/llm-demo"):
        write_run(tmp_path / name, make_result())
    extraction = extract_local(tmp_path)
    assert sorted(run.kind for run in extraction.runs) == ["coding", "coding", "cpu"]
    assert extraction.rejections == []


def test_unknown_directory_kind_is_rejected_not_skipped(tmp_path, result):
    write_run(tmp_path / "experiment-1", result)
    extraction = extract_local(tmp_path)
    assert extraction.runs == []
    [rejection] = extraction.rejections
    assert "Unknown experiment kind" in rejection.reason and "cpu-*" in rejection.reason


def test_missing_data_directory_is_an_error(tmp_path):
    with pytest.raises(ExtractionError, match="does not exist"):
        extract_local(tmp_path / "absent")


def test_empty_data_directory_finds_nothing_and_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="agentwatch_data.extract"):
        extraction = extract_local(tmp_path)
    assert extraction.runs == [] and extraction.rejections == []
    assert "No usable runs" in caplog.text


def test_missing_and_empty_samples_are_distinguished(tmp_path, result):
    write_run(tmp_path / "cpu-empty", result, samples=())
    missing = make_result()
    write_run(tmp_path / "cpu-missing", missing)
    (tmp_path / "cpu-missing" / "samples.jsonl").unlink()
    status = {run.run_id: run.telemetry_status for run in extract_local(tmp_path).runs}
    assert status == {result["run_id"]: TELEMETRY_EMPTY, missing["run_id"]: TELEMETRY_MISSING}


def test_invalid_local_run_is_rejected_with_reason(tmp_path, result):
    result["clean"]["queue_seconds"] = 5.0
    write_run(tmp_path / "coding-demo", result)
    extraction = extract_local(tmp_path)
    assert extraction.runs == []
    [rejection] = extraction.rejections
    assert rejection.origin.endswith("coding-demo")
    assert "Duration does not match" in rejection.reason


def test_malformed_samples_line_is_rejected_with_line_number(tmp_path, result):
    write_run(tmp_path / "cpu-demo", result)
    with (tmp_path / "cpu-demo" / "samples.jsonl").open("a") as stream:
        stream.write("{not json\n")
    [rejection] = extract_local(tmp_path).rejections
    assert "samples.jsonl line 2" in rejection.reason


def test_malformed_results_file_is_rejected(tmp_path, result):
    write_run(tmp_path / "cpu-demo", result)
    (tmp_path / "cpu-demo" / "results.json").write_text("{")
    [rejection] = extract_local(tmp_path).rejections
    assert "JSONDecodeError" in rejection.reason


# Evidence identity and normalization


def test_hash_ignores_key_and_sample_order(result):
    samples = [sample(0), sample(10)]
    reordered = dict(reversed(list(result.items())))
    assert evidence_hash(result, samples) == evidence_hash(reordered, samples)
    assert (
        extract.validate(result, samples, "cpu").source_sha256
        == extract.validate(result, samples[::-1], "cpu").source_sha256
    )


def test_hash_changes_when_telemetry_changes(result):
    full = extract.validate(result, [sample(0), sample(10)], "cpu")
    partial = extract.validate(result, [sample(0)], "cpu")
    assert full.source_sha256 != partial.source_sha256
    assert full.content_sha256 != partial.content_sha256


def test_samples_are_sorted_by_timestamp(result):
    run = extract.validate(result, [sample(20), sample(0), sample(10)], "cpu")
    assert [s["timestamp_ns"] for s in run.samples] == [BASE_NS, BASE_NS + 10, BASE_NS + 20]


def test_identical_duplicate_samples_are_collapsed(result):
    run = extract.validate(result, [sample(0), sample(0), sample(10)], "cpu")
    assert len(run.samples) == 2


def test_conflicting_duplicate_samples_are_rejected(tmp_path, result):
    write_run(tmp_path / "cpu-demo", result, samples=(sample(0, 1.0), sample(0, 9.0)))
    [rejection] = extract_local(tmp_path).rejections
    assert "Samples disagree at timestamp_ns" in rejection.reason


def test_coerced_values_are_stored_with_contract_types(result):
    original = json.loads(json.dumps(result))
    result["clean"]["submitted_ns"] = str(result["clean"]["submitted_ns"])
    result["experiment"]["iterations"] = "10"
    run = extract.validate(result, [SAMPLE | {"timestamp_ns": "1"}], "cpu")
    assert run.result["clean"]["submitted_ns"] == original["clean"]["submitted_ns"]
    assert run.result["experiment"]["iterations"] == 10
    assert run.samples[0]["timestamp_ns"] == 1
    assert run.source_sha256 != evidence_hash(original, [SAMPLE])
    normalized = extract.validate(original, [SAMPLE], "cpu")
    assert run.content_sha256 == normalized.content_sha256


def test_validated_result_keeps_extra_evidence(result):
    result["agent"] = {"model": "qwen3:1.7b"}
    assert extract.validate(result, [], "coding").result["agent"] == {"model": "qwen3:1.7b"}


# Duplicate runs across sources


def test_same_run_from_two_sources_is_kept_once(tmp_path, api, result):
    write_run(tmp_path / "cpu-demo", result)
    api.add_page(None, [api.row(json.loads(json.dumps(result)), samples=[SAMPLE])], None)
    local, remote = extract_local(tmp_path), extract_api("http://api")
    merged = combine(local, remote)
    [run] = merged.runs
    assert run.recorded_at is not None  # the API copy carries the saved time
    assert merged.duplicates == [str(tmp_path / "cpu-demo")]
    assert merged.rejections == []


def test_same_run_with_conflicting_evidence_is_rejected(tmp_path, api, result):
    write_run(tmp_path / "cpu-demo", result)
    changed = json.loads(json.dumps(result))
    changed["contended"]["cpu_seconds"] = 0.5
    api.add_page(None, [api.row(changed, samples=[SAMPLE])], None)
    merged = combine(extract_local(tmp_path), extract_api("http://api"))
    assert merged.runs == []
    assert len(merged.rejections) == 2
    assert all("conflicting evidence" in r.reason for r in merged.rejections)


def test_run_listed_twice_by_one_source_is_kept_once(tmp_path, result):
    write_run(tmp_path / "cpu-a", result)
    write_run(tmp_path / "cpu-b", result)
    extraction = extract_local(tmp_path)
    assert len(extraction.runs) == 1 and len(extraction.duplicates) == 1


# Ingestion API


def test_api_extraction_follows_cursor(api, result):
    second = make_result()
    api.add_page(None, [api.row(result)], 7, limit=1)
    api.add_page(7, [api.row(second, kind="coding", created_at="2026-10-05T12:05:00Z")], None, 1)
    extraction = extract_api("http://api/", page_size=1)
    assert [run.kind for run in extraction.runs] == ["cpu", "coding"]
    assert extraction.runs[0].recorded_at.isoformat() == "2026-10-05T12:00:00+00:00"
    assert extraction.runs[0].telemetry_status == TELEMETRY_EMPTY
    assert extraction.rejections == []


def test_transient_failures_are_retried_with_backoff(api, no_sleep, result):
    retry, delays = no_sleep
    api.add_page(None, [api.row(result, samples=[SAMPLE])], None)
    api.fail("/v1/runs?limit=100", TimeoutError("timed out"))
    api.fail("samples.jsonl", urllib.error.URLError("reset"), http_error(503))
    extraction = extract_api("http://api", retry=retry)
    assert len(extraction.runs) == 1
    assert delays == [0.5, 0.5, 1.0]


def test_persistent_network_failure_stops_extraction(api, no_sleep, result):
    retry, delays = no_sleep
    api.add_page(None, [api.row(result)], None)
    api.fail("samples.jsonl", *[urllib.error.URLError("reset")] * 3)
    with pytest.raises(ExtractionError, match="failed after 3 attempts"):
        extract_api("http://api", retry=retry)
    assert delays == [0.5, 1.0]


def test_client_errors_are_not_retried(api, no_sleep, result):
    retry, delays = no_sleep
    api.add_page(None, [api.row(result)], None)
    api.fail("samples.jsonl", http_error(404))
    with pytest.raises(ExtractionError, match="HTTP 404"):
        extract_api("http://api", retry=retry)
    assert delays == []


@pytest.mark.parametrize(
    "page",
    [ValueError("Expecting value"), [], {"next_cursor": None}, {"runs": [], "next_cursor": "7"}],
)
def test_malformed_history_pages_stop_extraction(api, page):
    if isinstance(page, Exception):
        api.fail("/v1/runs?limit=100", page)
    else:
        api.pages["http://api/v1/runs?limit=100"] = page
    with pytest.raises(ExtractionError):
        extract_api("http://api")


def test_cursor_that_does_not_advance_stops_extraction(api):
    api.add_page(None, [], 7)
    api.add_page(7, [], 7)
    with pytest.raises(ExtractionError, match="does not advance"):
        extract_api("http://api")
    assert len(api.calls) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"created_at": None},
        {"created_at": "2026-10-05T12:00:00"},
        {"kind": "gpu"},
        {"run_id": "not-a-uuid"},
    ],
)
def test_invalid_history_rows_are_rejected_and_others_kept(api, result, change):
    good = make_result()
    bad = api.row(result) | change
    api.add_page(None, [bad, api.row(good)], None)
    extraction = extract_api("http://api")
    assert [run.run_id for run in extraction.runs] == [good["run_id"]]
    assert len(extraction.rejections) == 1


def test_history_row_without_result_is_rejected(api, result):
    row = api.row(result)
    row.pop("result")
    api.add_page(None, [row], None)
    [rejection] = extract_api("http://api").rejections
    assert rejection.reason.startswith("result")


def test_listed_run_id_must_match_its_result(api, result):
    row = api.row(result) | {"run_id": str(uuid4())}
    api.samples[row["run_id"]] = []
    api.add_page(None, [row], None)
    extraction = extract_api("http://api")
    assert extraction.runs == []
    assert "does not match its result" in extraction.rejections[0].reason


def test_invalid_api_evidence_is_rejected(api, result):
    result["contended"]["queue_fraction"] = 2.0
    api.add_page(None, [api.row(result)], None)
    [rejection] = extract_api("http://api").rejections
    assert "contended.queue_fraction" in rejection.reason


@pytest.mark.parametrize("page_size", [0, 101])
def test_page_size_must_match_api_limits(page_size):
    with pytest.raises(ValueError, match="page_size"):
        extract_api("http://api", page_size=page_size)


def test_extraction_logs_a_summary(api, result, caplog):
    api.add_page(None, [api.row(result)], None)
    with caplog.at_level(logging.INFO, logger="agentwatch_data.extract"):
        extract_api("http://api")
    assert "Extracted 1 runs from http://api (0 rejected, 0 duplicate copies)" in caplog.text


@pytest.mark.parametrize("attempts", [0, -1])
def test_retry_policy_requires_at_least_one_attempt(attempts):
    with pytest.raises(ValueError):
        RetryPolicy(attempts=attempts)


def test_blank_samples_lines_are_ignored(tmp_path, result):
    write_run(tmp_path / "cpu-demo", result)
    path = tmp_path / "cpu-demo" / "samples.jsonl"
    path.write_text("\n" + path.read_text() + "\n\n")
    assert extract_local(tmp_path).runs[0].samples == [SAMPLE]


def test_samples_artifact_must_be_a_list(api, result):
    api.add_page(None, [api.row(result)], None)
    api.samples[result["run_id"]] = {"samples": []}
    with pytest.raises(ExtractionError, match="not a list"):
        extract_api("http://api")


def test_get_json_reads_json_and_ndjson_over_http():
    """Exercise the real HTTP client against an in-process localhost server."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            ndjson = self.path.endswith(".jsonl")
            body = b'{"a": 1}\n{"a": 2}\n' if ndjson else b'{"runs": [], "next_cursor": null}'
            self.send_response(200)
            kind = "application/x-ndjson" if ndjson else "application/json"
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert extract.get_json(f"{base}/v1/runs", 5) == {"runs": [], "next_cursor": None}
        assert extract.get_json(f"{base}/samples.jsonl", 5) == [{"a": 1}, {"a": 2}]
    finally:
        server.shutdown()
        server.server_close()
