"""The agent's allowlisted tools: documentation search, file reads, reviewed patches, tests.

Model output is untrusted. A call is validated completely before it starts: unknown tools,
missing or extra arguments, wrong types and values outside the task's allowlists are
rejected without executing anything. The model can only choose among reviewed patches, so
no model-written code ever runs; the test subprocess is a trusted-fixture boundary, not an
isolation sandbox.
"""

import json
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from agentwatch_sandbox.project_tests import run_project_tests

from agentwatch_agent.config import RunLimits
from agentwatch_agent.errors import RunCancelled
from agentwatch_agent.retrieval import RETRIEVAL_METHOD, DocumentIndex
from agentwatch_agent.tasks import Task, task_files
from agentwatch_agent.workspace import Workspace, WorkspaceError

MAX_QUERY_CHARS = 200
MAX_PASSAGE_CHARS = 800
SEARCH_RESULTS = 3
ResourceKey = Literal["tool_host", "docs_index"]


class ToolRejected(Exception):
    """The call is invalid and was not executed; the message is safe to show the model."""


class ToolFailed(Exception):
    """A valid call started but could not complete; the message is safe to show the model."""

    def __init__(self, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.evidence = evidence or {}


@dataclass(frozen=True)
class PreparedCall:
    name: str
    arguments: dict
    step_type: Literal["tool", "retrieval"]
    resource: ResourceKey


@dataclass(frozen=True)
class ToolOutcome:
    content: dict
    evidence: dict
    cpu_seconds: float | None = None
    cpu_scope: Literal["process_and_children"] | None = None


def _function(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": sorted(properties),
                "additionalProperties": False,
            },
        },
    }


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15] + "\n…[truncated]"


def tool_definitions(task: Task, readable_files: Iterable[str] | None = None) -> tuple[dict, ...]:
    """The tool schemas offered to the model, with the task's allowlists as enums.

    Without `readable_files` the task's own fixture files are used, so callers that never
    create a workspace (such as the GPU diagnostics) send exactly the runner's definitions.
    """
    if readable_files is None:
        readable_files = (
            path.relative_to(task.project_dir).as_posix() for path in task_files(task.project_dir)
        )
    patch_ids = [patch.patch_id for patch in task.patches]
    return (
        _function(
            "search_docs",
            "Search the project's reference documentation. Returns the best-matching sections.",
            {"query": {"type": "string", "maxLength": MAX_QUERY_CHARS}},
        ),
        _function(
            "read_file",
            "Read one file of the project.",
            {"path": {"type": "string", "enum": sorted(readable_files)}},
        ),
        _function(
            "apply_patch",
            "Apply one of the task's reviewed patches to the project.",
            {"patch_id": {"type": "string", "enum": patch_ids}},
        ),
        _function("run_tests", "Run the project's test suite and report the result.", {}),
    )


class ToolDispatcher:
    def __init__(self, task: Task, workspace: Workspace, index: DocumentIndex, limits: RunLimits):
        self.task = task
        self.workspace = workspace
        self.index = index
        self.limits = limits

    def definitions(self) -> tuple[dict, ...]:
        return tool_definitions(self.task, self.workspace.files)

    def prepare(self, name: object, arguments: object) -> PreparedCall:
        specs = {
            "search_docs": ("retrieval", "docs_index", {"query"}),
            "read_file": ("tool", "tool_host", {"path"}),
            "apply_patch": ("tool", "tool_host", {"patch_id"}),
            "run_tests": ("tool", "tool_host", set()),
        }
        if not isinstance(name, str) or name not in specs:
            raise ToolRejected(f"Unknown tool. Available tools: {', '.join(sorted(specs))}")
        step_type, resource, required = specs[name]
        if not isinstance(arguments, dict):
            raise ToolRejected(f"{name} arguments must be a JSON object")
        if set(arguments) != required:
            expected = ", ".join(sorted(required)) or "no arguments ({})"
            raise ToolRejected(f"{name} takes exactly: {expected}")
        for key, value in arguments.items():
            if not isinstance(value, str) or not value.strip():
                raise ToolRejected(f"{name}.{key} must be a nonblank string")
        if name == "search_docs" and len(arguments["query"]) > MAX_QUERY_CHARS:
            raise ToolRejected(f"search_docs.query must be at most {MAX_QUERY_CHARS} characters")
        if name == "read_file" and arguments["path"] not in self.workspace.files:
            raise ToolRejected(f"Unknown file. Readable files: {sorted(self.workspace.files)}")
        if name == "apply_patch" and self.task.patch(arguments["patch_id"]) is None:
            ids = [p.patch_id for p in self.task.patches]
            raise ToolRejected(f"Unknown patch_id. Available patches: {ids}")
        return PreparedCall(name, dict(arguments), step_type, resource)

    def execute(
        self, call: PreparedCall, *, cancel: threading.Event, deadline_monotonic: float
    ) -> ToolOutcome:
        if call.name == "search_docs":
            return self._search(call.arguments["query"])
        if call.name == "read_file":
            return self._read(call.arguments["path"])
        if call.name == "apply_patch":
            return self._apply(call.arguments["patch_id"])
        return self.run_tests(cancel=cancel, deadline_monotonic=deadline_monotonic)

    def _search(self, query: str) -> ToolOutcome:
        hits = self.index.search(query, limit=SEARCH_RESULTS)
        results = [
            {
                "document": hit.passage.document,
                "section": hit.passage.heading,
                "text": _truncate(hit.passage.text, MAX_PASSAGE_CHARS),
            }
            for hit in hits
        ]
        evidence = {
            "method": RETRIEVAL_METHOD,
            "query": query,
            "hits": [
                {
                    "document": hit.passage.document,
                    "section": hit.passage.heading,
                    "score": round(hit.score, 6),
                }
                for hit in hits
            ],
        }
        return ToolOutcome({"results": results}, evidence)

    def _read(self, path: str) -> ToolOutcome:
        try:
            text = self.workspace.read(path)
        except WorkspaceError as error:
            raise ToolFailed(str(error)) from error
        content = _truncate(text, self.limits.max_tool_output_chars)
        return ToolOutcome({"path": path, "content": content}, {"path": path, "chars": len(text)})

    def _apply(self, patch_id: str) -> ToolOutcome:
        patch = self.task.patch(patch_id)
        try:
            self.workspace.apply(patch)
        except WorkspaceError as error:
            raise ToolFailed(str(error)) from error
        return ToolOutcome(
            {"applied": patch_id, "file": patch.file},
            {"patch_id": patch_id, "file": patch.file},
        )

    def run_tests(self, *, cancel: threading.Event, deadline_monotonic: float) -> ToolOutcome:
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise ToolFailed("No run time remains for the test suite")
        timeout = min(self.limits.test_timeout_seconds, remaining)
        result = run_project_tests(self.workspace.root, timeout_seconds=timeout, cancel=cancel)
        evidence = {
            "outcome": result.outcome,
            "returncode": result.returncode,
            "tests_run": result.tests_run,
            "timeout_seconds": timeout,
            "subprocess_wall_seconds": result.wall_seconds,
            "applied_patches": list(self.workspace.applied_patches),
            "output_tail": result.output_tail,
        }
        if result.outcome == "cancelled":
            raise RunCancelled("Cancelled while the test suite was running")
        if result.outcome == "timed_out":
            raise ToolFailed(f"The test suite exceeded its {timeout:g} s timeout", evidence)
        if result.outcome == "resource_limit":
            # Only the isolated runner reports this; it is a limit, not a failing test.
            raise ToolFailed(
                f"The test suite exceeded a sandbox limit ({result.limit_exceeded})", evidence
            )
        content = {
            "outcome": result.outcome,
            "tests_run": result.tests_run,
            "output": _truncate(result.output_tail, self.limits.max_tool_output_chars),
        }
        # CPU scope is only claimed with a measurement; an unmeasured CPU time stays None.
        scope = "process_and_children" if result.cpu_seconds is not None else None
        return ToolOutcome(content, evidence, result.cpu_seconds, scope)


def tool_message(name: str, content: dict) -> dict:
    return {"role": "tool", "tool_name": name, "content": json.dumps(content)}
