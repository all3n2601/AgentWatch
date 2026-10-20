import shutil
import subprocess

import pytest
from agentwatch_agent.tasks import TaskDefinitionError, content_sha256, load_task, task_names
from agentwatch_agent.workspace import Workspace
from agentwatch_sandbox.project_tests import run_project_tests

CORRECT_PATCH = {
    "fix-mean-empty": "return-none-for-empty",
    "fix-window-count": "include-final-window",
}


def test_every_indexed_task_loads_with_a_derived_identity():
    assert task_names() == sorted(CORRECT_PATCH)
    for name in task_names():
        task = load_task(name)
        assert task.task_id == f"{name}-r{task.revision}"
        assert len(task.content_sha256) == 64
        assert len(task.patches) >= 2


@pytest.mark.parametrize("name", sorted(CORRECT_PATCH))
def test_fixture_tests_fail_unpatched_pass_only_with_the_documented_patch(name):
    """The task is meaningful: retrieval decides between patches that all apply cleanly."""
    task = load_task(name)
    outcomes = {}
    for patch in (None, *task.patches):
        workspace = Workspace.create(task)
        try:
            if patch is not None:
                workspace.apply(patch)
            result = run_project_tests(workspace.root, timeout_seconds=30)
            outcomes[patch.patch_id if patch else None] = result.outcome
        finally:
            assert workspace.remove()
    assert outcomes.pop(None) == "failed"
    assert outcomes.pop(CORRECT_PATCH[name]) == "passed"
    assert set(outcomes.values()) == {"failed"}


def copy_tasks(tmp_path):
    from agentwatch_agent.tasks import TASKS_ROOT

    root = tmp_path / "task_fixtures"
    shutil.copytree(TASKS_ROOT, root)
    return root


def test_changed_content_without_revision_bump_is_refused(tmp_path):
    root = copy_tasks(tmp_path)
    (root / "fix-mean-empty/docs/statistics-contract.md").write_text("# Changed\n\nmean is 0\n")
    with pytest.raises(TaskDefinitionError, match="without a revision bump"):
        load_task("fix-mean-empty", root)


def test_content_hash_ignores_bytecode_caches(tmp_path):
    root = copy_tasks(tmp_path)
    before = content_sha256(root / "fix-mean-empty")
    cache = root / "fix-mean-empty/project/__pycache__"
    cache.mkdir()
    (cache / "stats.cpython-313.pyc").write_bytes(b"\0")
    assert content_sha256(root / "fix-mean-empty") == before


def test_patch_that_does_not_match_exactly_once_is_refused(tmp_path):
    root = copy_tasks(tmp_path)
    definition = root / "fix-mean-empty/task.toml"
    definition.write_text(definition.read_text().replace("    return sum(values)", "    return x"))
    # Re-pin so the patch check, not the content hash, is what fails.
    index = root / "index.toml"
    pinned = content_sha256(root / "fix-mean-empty")
    index.write_text(index.read_text().replace(load_task("fix-mean-empty").content_sha256, pinned))
    with pytest.raises(TaskDefinitionError, match="exactly once"):
        load_task("fix-mean-empty", root)


def test_unknown_task_lists_available_names():
    with pytest.raises(TaskDefinitionError, match="fix-mean-empty"):
        load_task("does-not-exist")


def test_fixtures_are_lf_only_and_pinned_to_lf_on_checkout():
    """Task hashes and patches are byte-exact, so CRLF checkouts must not happen (PR #27)."""
    from agentwatch_agent.tasks import TASKS_ROOT, task_files

    files = task_files(TASKS_ROOT)
    assert files
    assert [path for path in files if b"\r" in path.read_bytes()] == []
    sample = TASKS_ROOT / "fix-mean-empty/project/stats.py"
    attributes = subprocess.run(
        ["git", "check-attr", "eol", "text", "--", str(sample)],
        capture_output=True,
        text=True,
        check=True,
        cwd=TASKS_ROOT,
    ).stdout
    assert "eol: lf" in attributes and "text: set" in attributes
