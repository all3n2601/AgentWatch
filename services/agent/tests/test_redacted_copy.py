"""Redacted copies of evidence folders for sharing. All content is a synthetic fixture."""

import json
import os

import pytest
from agentwatch_agent.cli import main
from agentwatch_agent.redacted_copy import (
    MANIFEST_FILE,
    RedactionError,
    write_redacted_copy,
)

EMAIL = "alice@example.test"
HOME_PATH = "/Users/alice/project/stats.py"
# A dummy value in the URL user-info position, built from parts for secret scanners.
DUMMY_URL_VALUE = "dummy" + "1234"


def evidence_folder(root):
    source = root / "run"
    (source / "replays/01").mkdir(parents=True)
    evidence = {
        "run_id": "97fb140d-be6f-42f9-b31c-0d1ab305aabf",
        "content_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "configuration": {"sampling_interval_seconds": 0.25, "model_url": "http://127.0.0.1:11434"},
        "messages": [
            {"role": "user", "content": f"Fix stats.py; report to {EMAIL}"},
            {"role": "tool", "content": f"Traceback in {HOME_PATH}\nAssertionError"},
        ],
        "delivery": {"last_error": f"401 for https://bob:{DUMMY_URL_VALUE}@api.example.test/v1"},
        "counts": [1, 2.5, None, True],
    }
    (source / "agent-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    lines = [
        {"kind": "record", "payload": {"reason": f"GPU read failed at {HOME_PATH}"}},
        {"kind": "record", "payload": {"value": 3, "reason": None}},
    ]
    (source / "live-journal.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    (source / "replays/01/agent-evidence.json").write_text(
        json.dumps({"final_answer": f"Done; contact {EMAIL}"}), encoding="utf-8"
    )
    (source / "notes.txt").write_text(f"unscanned {EMAIL}", encoding="utf-8")
    (source / ".DS_Store").write_bytes(b"\x00")
    return source


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_json_and_jsonl_strings_are_redacted_and_everything_else_is_kept(tmp_path):
    source = evidence_folder(tmp_path)
    output = tmp_path / "shared"
    manifest = write_redacted_copy(source, output)

    evidence = read(output / "agent-evidence.json")
    assert evidence["messages"][0]["content"] == "Fix stats.py; report to <email>"
    assert evidence["messages"][1]["content"] == "Traceback in <path>\nAssertionError"
    assert evidence["delivery"]["last_error"] == "401 for https://<credentials>@api.example.test/v1"
    # Identifiers, hashes, URLs without credentials, numbers and nulls are untouched.
    original = read(source / "agent-evidence.json")
    for key in ("run_id", "content_sha256", "configuration", "counts"):
        assert evidence[key] == original[key]

    journal = [
        json.loads(line) for line in (output / "live-journal.jsonl").read_text().splitlines()
    ]
    assert journal[0]["payload"]["reason"] == "GPU read failed at <path>"
    assert journal[1] == {"kind": "record", "payload": {"value": 3, "reason": None}}
    assert (
        read(output / "replays/01/agent-evidence.json")["final_answer"] == "Done; contact <email>"
    )

    assert manifest == read(output / MANIFEST_FILE)
    files = {entry["path"]: entry for entry in manifest["files"]}
    assert files["agent-evidence.json"]["replacements"] == {
        "email": 1,
        "path": 1,
        "url_credentials": 1,
    }
    assert files["live-journal.jsonl"]["replacements"] == {"path": 1}
    assert manifest["totals"] == {"email": 2, "path": 2, "url_credentials": 1}
    assert len(files["agent-evidence.json"]["source_sha256"]) == 64


def test_files_that_cannot_be_scanned_are_left_out_and_listed(tmp_path):
    source = evidence_folder(tmp_path)
    manifest = write_redacted_copy(source, tmp_path / "shared")
    assert manifest["not_copied"] == ["notes.txt"]
    assert not (tmp_path / "shared/notes.txt").exists()
    # Hidden tool state is neither copied nor listed.
    assert not (tmp_path / "shared/.DS_Store").exists()


def test_the_manifest_names_no_host_path(tmp_path):
    source = evidence_folder(tmp_path)
    manifest = write_redacted_copy(source, tmp_path / "shared")
    assert manifest["source"] == "run"
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["method"] == "trace-redaction-v1"


def test_an_existing_output_is_never_overwritten(tmp_path):
    source = evidence_folder(tmp_path)
    output = tmp_path / "shared"
    output.mkdir()
    (output / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(RedactionError, match="already exists"):
        write_redacted_copy(source, output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_the_output_cannot_be_inside_the_source(tmp_path):
    source = evidence_folder(tmp_path)
    with pytest.raises(RedactionError, match="inside"):
        write_redacted_copy(source, source / "shared")


def test_a_malformed_file_stops_the_copy_and_leaves_no_output(tmp_path):
    source = evidence_folder(tmp_path)
    (source / "live-journal.jsonl").write_text('{"kind": "record"}\n{broken\n', encoding="utf-8")
    output = tmp_path / "shared"
    with pytest.raises(RedactionError, match=r"live-journal.jsonl line 2"):
        write_redacted_copy(source, output)
    assert not output.exists()


def test_symbolic_links_are_refused(tmp_path):
    source = evidence_folder(tmp_path)
    os.symlink(tmp_path / "elsewhere.json", source / "linked.json")
    with pytest.raises(RedactionError, match="symbolic link"):
        write_redacted_copy(source, tmp_path / "shared")


def test_a_missing_source_is_refused(tmp_path):
    with pytest.raises(RedactionError, match="not a folder"):
        write_redacted_copy(tmp_path / "missing", tmp_path / "shared")


def test_redact_command_writes_the_copy_and_reports_replacements(tmp_path, capsys):
    source = evidence_folder(tmp_path)
    output = tmp_path / "shared"
    assert main(["redact", str(source), "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "5 replacements" in printed and "notes.txt" in printed
    assert (output / MANIFEST_FILE).is_file()


def test_redact_command_exits_2_when_the_copy_is_refused(tmp_path, capsys):
    source = evidence_folder(tmp_path)
    assert main(["redact", str(source), "--output", str(source / "inner")]) == 2
    assert "inside" in capsys.readouterr().err
