import json
import shutil
import stat
from pathlib import Path

import pytest
from agentwatch_data import evidence as evidence_module
from agentwatch_data.evidence import EvidenceError, add_run, main
from agentwatch_data.extract import extract_local
from dataset_fixtures import SAMPLE, make_result, write_run

GOLDEN = next((Path(__file__).parent / "fixtures").iterdir())


@pytest.fixture
def store(tmp_path):
    return tmp_path / "evidence"


def golden_copy(tmp_path, name="coding-copy"):
    target = tmp_path / "recorded" / name
    shutil.copytree(GOLDEN, target)
    return target


def files_of(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_a_recorded_run_is_copied_byte_for_byte_under_its_run_id(tmp_path, store):
    run_dir = golden_copy(tmp_path)
    stored = add_run(run_dir, store)
    run_id = json.loads((run_dir / "results.json").read_text())["run_id"]
    assert stored.added and stored.name == f"coding-{run_id}"
    entry = store / stored.name
    expected = {
        name: data
        for name, data in files_of(run_dir).items()
        if name in evidence_module.EVIDENCE_FILES
    }
    assert files_of(entry) == expected  # only evidence files, exactly as recorded
    assert stat.S_IMODE(entry.stat().st_mode) == 0o755
    assert {stat.S_IMODE(p.stat().st_mode) for p in entry.iterdir()} == {0o644}
    assert [p.name for p in store.iterdir()] == [stored.name]  # no staging left behind


def test_the_store_builds_like_the_original_runs(tmp_path, store):
    run_dir = golden_copy(tmp_path)
    add_run(run_dir, store)
    [original] = extract_local(run_dir.parent).runs
    [stored] = extract_local(store).runs
    assert (stored.source_sha256, stored.content_sha256) == (
        original.source_sha256,
        original.content_sha256,
    )


def test_adding_the_same_evidence_again_changes_nothing(tmp_path, store):
    run_dir = golden_copy(tmp_path)
    first = add_run(run_dir, store)
    before = files_of(store / first.name)
    # Same evidence, different formatting: still the same run.
    result = json.loads((run_dir / "results.json").read_text())
    (run_dir / "results.json").write_text(json.dumps(result, indent=4, sort_keys=True))
    again = add_run(run_dir, store)
    assert (again.name, again.added) == (first.name, False)
    assert files_of(store / first.name) == before


def test_different_evidence_under_a_stored_run_id_is_refused(tmp_path, store):
    run_dir = golden_copy(tmp_path)
    stored = add_run(run_dir, store)
    before = files_of(store / stored.name)
    result = json.loads((run_dir / "results.json").read_text())
    result["contended"]["cpu_seconds"] = 0.123
    (run_dir / "results.json").write_text(json.dumps(result))
    with pytest.raises(EvidenceError, match="already stored with different evidence"):
        add_run(run_dir, store)
    assert files_of(store / stored.name) == before


def test_a_missing_samples_file_is_different_evidence_from_an_empty_one(tmp_path, store):
    result = make_result()
    write_run(tmp_path / "a" / "cpu-run", result, samples=())
    (tmp_path / "a" / "cpu-run" / "samples.jsonl").unlink()  # telemetry missing
    stored = add_run(tmp_path / "a" / "cpu-run", store)
    assert files_of(store / stored.name).keys() == {"results.json"}
    write_run(tmp_path / "b" / "cpu-run", result, samples=())  # telemetry empty
    with pytest.raises(EvidenceError, match="different evidence"):
        add_run(tmp_path / "b" / "cpu-run", store)


def test_one_run_id_cannot_be_stored_under_two_kinds(tmp_path, store):
    result = make_result(kind="coding", agent_model="qwen3:1.7b")
    write_run(tmp_path / "coding-run", result)
    write_run(tmp_path / "llm-run", result)
    add_run(tmp_path / "coding-run", store)
    with pytest.raises(EvidenceError, match=f"already stored as coding-{result['run_id']}"):
        add_run(tmp_path / "llm-run", store)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("mystery-run", None, "Unknown experiment kind"),
        ("cpu-broken", "{not json", "is not usable evidence"),
        ("cpu-empty", "", "is not usable evidence"),
    ],
)
def test_runs_that_fail_extraction_rules_are_refused(tmp_path, store, name, content, message):
    write_run(tmp_path / name, make_result())
    if content is not None:
        (tmp_path / name / "results.json").write_text(content)
    with pytest.raises(EvidenceError, match=message):
        add_run(tmp_path / name, store)
    assert not store.exists()


def test_a_damaged_stored_entry_is_reported_not_overwritten(tmp_path, store):
    run_dir = golden_copy(tmp_path)
    stored = add_run(run_dir, store)
    (store / stored.name / "results.json").write_text("{damaged")
    with pytest.raises(EvidenceError, match=f"Stored evidence {stored.name} is unreadable"):
        add_run(run_dir, store)
    assert (store / stored.name / "results.json").read_text() == "{damaged"


def test_a_concurrent_add_of_the_same_run_is_accepted(tmp_path, store, monkeypatch):
    run_dir = golden_copy(tmp_path)
    real_replace = evidence_module.os.replace

    def another_add_finishes_first(source, target):
        real_replace(source, target)  # the other command's identical copy lands first
        raise OSError("target exists")

    monkeypatch.setattr(evidence_module.os, "replace", another_add_finishes_first)
    stored = add_run(run_dir, store)
    assert stored.added is False
    assert [p.name for p in store.iterdir()] == [stored.name]


def test_a_failed_copy_leaves_no_entry(tmp_path, store, monkeypatch):
    def disk_full(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(evidence_module.os, "replace", disk_full)
    with pytest.raises(OSError, match="disk full"):
        add_run(golden_copy(tmp_path), store)
    assert list(store.iterdir()) == []


def test_cli_adds_runs_and_reports_refusals(tmp_path, store, capsys):
    good = golden_copy(tmp_path)
    write_run(tmp_path / "mystery-run", make_result(), samples=(SAMPLE,))
    assert main(["--store", str(store), str(good)]) == 0
    assert capsys.readouterr().out.startswith("added: ")
    assert main(["--store", str(store), str(good), str(tmp_path / "mystery-run")]) == 1
    captured = capsys.readouterr()
    assert captured.out.startswith("already stored: ") and "refused: " in captured.err


def test_module_runs_as_a_command(tmp_path, store, monkeypatch):
    import runpy
    import sys

    monkeypatch.setattr(
        sys, "argv", ["agentwatch_data.evidence", "--store", str(store), str(golden_copy(tmp_path))]
    )
    monkeypatch.delitem(sys.modules, "agentwatch_data.evidence", raising=False)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("agentwatch_data.evidence", run_name="__main__")
    assert exited.value.code == 0
