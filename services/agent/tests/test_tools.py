import json
import threading
import time

import pytest
from agentwatch_agent.config import RunLimits
from agentwatch_agent.errors import RunCancelled
from agentwatch_agent.retrieval import DocumentIndex, Passage, split_sections
from agentwatch_agent.tools import ToolDispatcher, ToolFailed, ToolRejected
from agentwatch_agent.workspace import Workspace


@pytest.fixture
def dispatcher(mean_task):
    workspace = Workspace.create(mean_task)
    yield ToolDispatcher(
        mean_task, workspace, DocumentIndex.from_directory(mean_task.docs_dir), RunLimits()
    )
    assert workspace.remove()


def run(dispatcher, name, arguments, cancel=None, deadline=None):
    prepared = dispatcher.prepare(name, arguments)
    return dispatcher.execute(
        prepared,
        cancel=cancel or threading.Event(),
        deadline_monotonic=deadline or time.monotonic() + 60,
    )


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("rm_rf", {}, "Unknown tool"),
        (None, {}, "Unknown tool"),
        ("run_tests", {"verbose": "yes"}, "takes exactly"),
        ("run_tests", "{}", "must be a JSON object"),
        ("read_file", {}, "takes exactly: path"),
        ("read_file", {"path": "../../etc/passwd"}, "Unknown file"),
        ("read_file", {"path": "/etc/passwd"}, "Unknown file"),
        ("read_file", {"path": 7}, "nonblank string"),
        ("apply_patch", {"patch_id": "rewrite-everything"}, "Unknown patch_id"),
        ("apply_patch", {"patch_id": "x", "file": "stats.py"}, "takes exactly"),
        ("search_docs", {"query": " "}, "nonblank string"),
        ("search_docs", {"query": "x" * 201}, "at most 200"),
    ],
)
def test_invalid_calls_are_rejected_before_execution(dispatcher, name, arguments, message):
    original = (dispatcher.workspace.root / "stats.py").read_text()
    with pytest.raises(ToolRejected, match=message):
        dispatcher.prepare(name, arguments)
    assert (dispatcher.workspace.root / "stats.py").read_text() == original


def test_tool_kinds_map_to_step_types_and_resources(dispatcher):
    assert dispatcher.prepare("search_docs", {"query": "mean"}).step_type == "retrieval"
    assert dispatcher.prepare("search_docs", {"query": "mean"}).resource == "docs_index"
    assert dispatcher.prepare("run_tests", {}).step_type == "tool"


def test_definitions_expose_only_allowlisted_values(dispatcher):
    definitions = {d["function"]["name"]: d["function"] for d in dispatcher.definitions()}
    assert set(definitions) == {"search_docs", "read_file", "apply_patch", "run_tests"}
    patch_ids = definitions["apply_patch"]["parameters"]["properties"]["patch_id"]["enum"]
    assert patch_ids == ["return-none-for-empty", "return-zero-for-empty"]
    paths = definitions["read_file"]["parameters"]["properties"]["path"]["enum"]
    assert paths == ["stats.py", "test_stats.py"]
    assert all(not d["parameters"]["additionalProperties"] for d in definitions.values())


def test_search_ranks_the_contract_section_first_and_is_deterministic(dispatcher):
    first = run(dispatcher, "search_docs", {"query": "what does mean return for empty input"})
    second = run(dispatcher, "search_docs", {"query": "what does mean return for empty input"})
    assert first.content == second.content
    top = first.content["results"][0]
    assert (top["document"], top["section"]) == ("statistics-contract.md", "Mean of an empty input")
    assert first.evidence["method"] == "bm25-markdown-sections-v1"
    assert first.cpu_seconds is None


def test_search_without_matching_terms_returns_no_results(dispatcher):
    assert run(dispatcher, "search_docs", {"query": "kubernetes"}).content == {"results": []}


def test_read_patch_and_test_flow(dispatcher):
    assert "len(values)" in run(dispatcher, "read_file", {"path": "stats.py"}).content["content"]
    failing = run(dispatcher, "run_tests", {})
    assert failing.content["outcome"] == "failed"
    assert failing.cpu_scope == "process_and_children"
    run(dispatcher, "apply_patch", {"patch_id": "return-none-for-empty"})
    passing = run(dispatcher, "run_tests", {})
    assert passing.content["outcome"] == "passed"
    assert passing.evidence["applied_patches"] == ["return-none-for-empty"]
    assert passing.evidence["tests_run"] == 5


def test_second_patch_to_the_same_code_fails_without_changing_files(dispatcher):
    run(dispatcher, "apply_patch", {"patch_id": "return-none-for-empty"})
    patched = (dispatcher.workspace.root / "stats.py").read_text()
    for patch_id in ("return-none-for-empty", "return-zero-for-empty"):
        with pytest.raises(ToolFailed):
            run(dispatcher, "apply_patch", {"patch_id": patch_id})
    assert (dispatcher.workspace.root / "stats.py").read_text() == patched


def test_test_timeout_is_a_failed_tool_with_evidence(dispatcher, monkeypatch):
    (dispatcher.workspace.root / "test_slow.py").write_text(
        "import time, unittest\n\n\nclass Slow(unittest.TestCase):\n"
        "    def test_slow(self):\n        time.sleep(30)\n"
    )
    dispatcher.limits = RunLimits(test_timeout_seconds=0.3)
    with pytest.raises(ToolFailed, match="timeout") as failure:
        run(dispatcher, "run_tests", {})
    assert failure.value.evidence["outcome"] == "timed_out"


def test_cancellation_during_tests_stops_the_run(dispatcher):
    (dispatcher.workspace.root / "test_slow.py").write_text(
        "import time, unittest\n\n\nclass Slow(unittest.TestCase):\n"
        "    def test_slow(self):\n        time.sleep(30)\n"
    )
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(RunCancelled):
        run(dispatcher, "run_tests", {}, cancel=cancel)


def test_exhausted_run_time_fails_tests_without_starting_them(dispatcher):
    with pytest.raises(ToolFailed, match="No run time remains"):
        run(dispatcher, "run_tests", {}, deadline=time.monotonic() - 1)


def test_tool_output_is_bounded(dispatcher):
    (dispatcher.workspace.root / "stats.py").write_text("#" * 10_000)
    dispatcher.limits = RunLimits(max_tool_output_chars=500)
    content = run(dispatcher, "read_file", {"path": "stats.py"}).content["content"]
    assert len(content) <= 500
    assert content.endswith("[truncated]")
    json.dumps(content)


def test_sections_split_on_headings_and_keep_preamble():
    passages = split_sections("doc.md", "intro\n# Title\n\nbody one\n## Next\nbody two\n")
    assert passages == [
        Passage("doc.md", "doc.md", "intro"),
        Passage("doc.md", "Title", "body one"),
        Passage("doc.md", "Next", "body two"),
    ]


def test_empty_index_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        DocumentIndex.from_directory(tmp_path)


def test_interruption_before_a_request_reports_that_nothing_was_started():
    from agentwatch_agent.errors import RunFailure
    from agentwatch_agent.model import wait_interruptibly

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RunCancelled) as cancelled:
        wait_interruptibly(dict, cancel=cancel, deadline_monotonic=time.monotonic() + 9)
    assert cancelled.value.request_started is False
    with pytest.raises(RunFailure) as expired:
        wait_interruptibly(dict, cancel=threading.Event(), deadline_monotonic=time.monotonic() - 1)
    assert expired.value.request_started is False


def fake_result(**changes):
    from agentwatch_sandbox.project_tests import ProjectTestResult

    values = {
        "outcome": "passed",
        "returncode": 0,
        "tests_run": 5,
        "output_tail": "OK",
        "wall_seconds": 0.5,
        "cpu_seconds": None,
        "isolation": "container",
    }
    return ProjectTestResult(**(values | changes))


def test_a_sandbox_limit_is_a_failed_tool_not_a_failing_test(dispatcher, monkeypatch):
    from agentwatch_agent import tools as tools_module

    limited = fake_result(outcome="resource_limit", returncode=None, limit_exceeded="memory")
    monkeypatch.setattr(tools_module, "run_project_tests", lambda *a, **k: limited)
    with pytest.raises(ToolFailed, match=r"sandbox limit \(memory\)") as failure:
        run(dispatcher, "run_tests", {})
    assert failure.value.evidence["outcome"] == "resource_limit"


def test_unmeasured_cpu_time_claims_no_cpu_scope(dispatcher, monkeypatch):
    from agentwatch_agent import tools as tools_module

    monkeypatch.setattr(tools_module, "run_project_tests", lambda *a, **k: fake_result())
    outcome = run(dispatcher, "run_tests", {})
    assert (outcome.cpu_seconds, outcome.cpu_scope) == (None, None)
