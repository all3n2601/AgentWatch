import json
from uuid import uuid4

import pytest
from agentwatch_data import extract
from agentwatch_data.extract import evidence_hash, extract_api, extract_local
from dataset_fixtures import SAMPLE, make_result, write_run


@pytest.fixture
def result():
    return make_result()


def test_local_runs_are_validated_and_tagged(tmp_path, result):
    write_run(tmp_path / "cpu-demo", result)
    extraction = extract_local(tmp_path)
    assert extraction.rejections == []
    [run] = extraction.runs
    assert (run.run_id, run.kind, run.source) == (result["run_id"], "cpu", "measured")
    assert run.samples == [SAMPLE]
    assert run.source_sha256 == evidence_hash(result)


def test_hash_ignores_key_order(result):
    assert evidence_hash(result) == evidence_hash(dict(reversed(list(result.items()))))


def test_invalid_local_run_is_rejected_with_reason(tmp_path, result):
    result["clean"]["queue_seconds"] = 5.0
    write_run(tmp_path / "coding-demo", result)
    extraction = extract_local(tmp_path)
    assert extraction.runs == []
    [rejection] = extraction.rejections
    assert rejection.origin.endswith("coding-demo")
    assert "Duration does not match" in rejection.reason


def test_missing_local_directories_are_skipped(tmp_path):
    extraction = extract_local(tmp_path)
    assert extraction.runs == [] and extraction.rejections == []


def test_api_extraction_follows_cursor(monkeypatch, result):
    second = json.loads(json.dumps(result)) | {"run_id": str(uuid4())}
    pages = {
        "http://api/v1/runs?limit=1": {
            "runs": [
                {
                    "run_id": result["run_id"],
                    "kind": "cpu",
                    "result": result,
                    "created_at": "2026-10-05T12:00:00+00:00",
                }
            ],
            "next_cursor": 7,
        },
        "http://api/v1/runs?limit=1&before=7": {
            "runs": [
                {
                    "run_id": second["run_id"],
                    "kind": "coding",
                    "result": second,
                    "created_at": "2026-10-05T12:05:00+00:00",
                }
            ],
            "next_cursor": None,
        },
    }

    def fake_get(url, timeout):
        return [SAMPLE] if url.endswith("samples.jsonl") else pages[url]

    monkeypatch.setattr(extract, "get_json", fake_get)
    extraction = extract_api("http://api/", page_size=1)
    assert [run.kind for run in extraction.runs] == ["cpu", "coding"]
    assert extraction.runs[0].recorded_at.isoformat() == "2026-10-05T12:00:00+00:00"
    assert extraction.rejections == []
