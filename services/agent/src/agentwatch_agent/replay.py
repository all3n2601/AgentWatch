"""Replay a recorded agent run so every experiment condition does the same work.

A replay sends the model exactly the inputs it received in the source run, call by call, and
executes exactly the tool calls the source model requested, in order, whatever the model
replies this time. Steps therefore keep the source's step IDs and inputs across clean and
contended replays, so a step can be compared with its own clean baseline. The model really
generates on every call (that is the inference work being measured), and whether its reply
matches the source is recorded, never assumed: temperature 0 does not guarantee it.

A replay is an ordinary live run (same lifecycle, cleanup, sampling and delivery as `run`).
Its registered configuration digest covers the replay plan's content, not the experiment
condition: clean and contended replays of one plan share a digest, and the condition and
injector configuration stay in the experiment manifest, outside anything a feature can read.
"""

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from agentwatch_agent.config import RunConfiguration
from agentwatch_agent.errors import RunFailure
from agentwatch_agent.model import ModelResponseError, RequestedCall, parse_reply
from agentwatch_agent.runner import EVIDENCE_FILE, AgentRunner, _RunContext
from agentwatch_agent.tasks import Task

REPLAY_METHOD = "agent-replay-v1"
# Tool results whose full text varies between identical executions (test timings): only their
# outcome is compared with the source.
OUTCOME_ONLY_TOOLS = {"run_tests": ("outcome", "tests_run")}


class ReplayPlanError(ValueError):
    """The source run cannot be replayed faithfully."""


def canonical_sha256(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def reply_signature(message: dict) -> str:
    """Digest of what a model reply said: its text and its calls' names and arguments."""
    try:
        reply = parse_reply({"message": message})
    except ModelResponseError as error:
        raise ReplayPlanError(f"Recorded assistant message is malformed: {error}") from None
    calls = [[call.name, call.arguments] for call in reply.calls]
    return canonical_sha256({"content": reply.content, "calls": calls})


def tool_signature(message: dict) -> str:
    """Digest of a tool result; for outcome-only tools, of the outcome alone."""
    name = message.get("tool_name")
    try:
        content = json.loads(message.get("content", ""))
    except (TypeError, json.JSONDecodeError):
        content = message.get("content")
    keys = OUTCOME_ONLY_TOOLS.get(name)
    if keys is not None and isinstance(content, dict):
        content = {key: content.get(key) for key in keys}
    return canonical_sha256({"tool": name, "content": content})


@dataclass(frozen=True)
class ReplayTurn:
    """One source model call: its exact input, the calls it requested and their results."""

    messages: tuple[dict, ...]
    calls: tuple[RequestedCall, ...]
    reply_sha256: str
    tool_result_sha256: tuple[str, ...]


@dataclass(frozen=True)
class ReplayPlan:
    source_run_id: str
    source_evidence_sha256: str
    task_id: str
    task_content_sha256: str
    model: str
    chat_options: dict
    tools: tuple[dict, ...]
    turns: tuple[ReplayTurn, ...]

    @property
    def tool_call_count(self) -> int:
        return sum(len(turn.calls) for turn in self.turns)

    def sha256(self) -> str:
        """Content digest of the workload; the source run's identity is not part of it."""
        return canonical_sha256(
            {
                "method": REPLAY_METHOD,
                "task_id": self.task_id,
                "task_content_sha256": self.task_content_sha256,
                "model": self.model,
                "chat_options": self.chat_options,
                "tools": list(self.tools),
                "turns": [
                    {
                        "messages": list(turn.messages),
                        "calls": [[call.name, call.arguments] for call in turn.calls],
                    }
                    for turn in self.turns
                ],
            }
        )

    def search_queries(self) -> list[str]:
        """The documentation queries the source issued, in order (valid strings only)."""
        return [
            call.arguments["query"]
            for turn in self.turns
            for call in turn.calls
            if call.name == "search_docs"
            and isinstance(call.arguments, dict)
            and isinstance(call.arguments.get("query"), str)
            and call.arguments["query"].strip()
        ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayPlanError(message)


def _turns(messages: list, model_calls: list) -> tuple[ReplayTurn, ...]:
    _require(
        len(messages) >= 3
        and all(isinstance(message, dict) for message in messages)
        and [m.get("role") for m in messages[:2]] == ["system", "user"],
        "Recorded conversation must start with the system and user messages",
    )
    turns = []
    position = 2
    while position < len(messages):
        message = messages[position]
        _require(message.get("role") == "assistant", f"Expected a model reply at {position}")
        signature = reply_signature(message)  # refuses a malformed reply first
        calls = parse_reply({"message": message}).calls
        results = messages[position + 1 : position + 1 + len(calls)]
        _require(
            len(results) == len(calls) and all(r.get("role") == "tool" for r in results),
            f"Model reply at {position} is not followed by one tool result per call",
        )
        turns.append(
            ReplayTurn(
                messages=tuple(messages[:position]),
                calls=calls,
                reply_sha256=signature,
                tool_result_sha256=tuple(tool_signature(result) for result in results),
            )
        )
        position += 1 + len(calls)
    _require(not turns[-1].calls, "Recorded conversation must end with the model's answer")
    _require(
        len(model_calls) == len(turns)
        and all(
            isinstance(call, dict) and call.get("requested_tool_calls") == len(turn.calls)
            for call, turn in zip(model_calls, turns, strict=True)
        ),
        "Recorded model calls do not match the recorded conversation",
    )
    return tuple(turns)


def load_replay_plan(run_dir: Path, task: Task) -> ReplayPlan:
    """Build a plan from a completed, verified source run of `task`; refuse anything less."""
    path = run_dir / EVIDENCE_FILE
    try:
        raw = path.read_bytes()
        evidence = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayPlanError(f"Cannot read source evidence {path}: {error}") from None
    _require(isinstance(evidence, dict), "Source evidence is not a JSON object")
    _require(
        evidence.get("status") == "completed" and evidence.get("task_passed") is True,
        "Only a completed run whose verification passed can be replayed",
    )
    recorded_task = evidence.get("task")
    _require(
        isinstance(recorded_task, dict)
        and recorded_task.get("task_id") == task.task_id
        and recorded_task.get("content_sha256") == task.content_sha256,
        f"Source run is not of task {task.task_id} with its current content",
    )
    configuration = evidence.get("configuration")
    _require(isinstance(configuration, dict), "Source evidence has no configuration")
    model, options, tools = (configuration.get(k) for k in ("model", "chat_options", "tools"))
    _require(
        isinstance(model, str)
        and model.strip()
        and isinstance(options, dict)
        and isinstance(tools, list),
        "Source configuration lacks its model, chat options or tool definitions",
    )
    messages, model_calls = evidence.get("messages"), evidence.get("model_calls")
    _require(
        isinstance(messages, list) and isinstance(model_calls, list),
        "Source evidence lacks its conversation or model calls",
    )
    run_id = evidence.get("run_id")
    _require(isinstance(run_id, str) and run_id, "Source evidence has no run ID")
    return ReplayPlan(
        source_run_id=run_id,
        source_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        task_id=task.task_id,
        task_content_sha256=task.content_sha256,
        model=model,
        chat_options=options,
        tools=tuple(tools),
        turns=_turns(messages, model_calls),
    )


@dataclass(frozen=True)
class ReplayConfiguration(RunConfiguration):
    """A run configuration that also identifies the replayed workload."""

    replay: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.replay:
            raise ValueError("A replay configuration must identify its replay plan")


class ReplayRunner(AgentRunner):
    """Runs a replay plan as one live run, with the agent runner's lifecycle and evidence."""

    def __init__(self, *, plan: ReplayPlan, **settings):
        if settings["model"].name != plan.model:
            raise ValueError(f"Replay plan was recorded with model {plan.model!r}")
        if settings.get("chat_options") not in (None, plan.chat_options):
            raise ValueError("A replay uses the source run's chat options")
        super().__init__(**(settings | {"chat_options": plan.chat_options}))
        if (
            len(plan.turns) > self.limits.max_model_calls
            or plan.tool_call_count > self.limits.max_tool_calls
        ):
            raise ValueError("Replay plan exceeds this run's model-call or tool-call limits")
        self.plan = plan

    def _configuration(self) -> RunConfiguration:
        base = super()._configuration()
        settings = {item.name: getattr(base, item.name) for item in fields(RunConfiguration)}
        return ReplayConfiguration(
            **settings, replay={"method": REPLAY_METHOD, "plan_sha256": self.plan.sha256()}
        )

    def _converse(self, context: _RunContext) -> None:
        plan = self.plan
        if canonical_sha256(list(self._tools)) != canonical_sha256(list(plan.tools)):
            raise RunFailure(
                "Replay tool definitions differ from the source run",
                "The task's tool schemas changed since the source run was recorded",
            )
        record = {
            "method": REPLAY_METHOD,
            "source_run_id": plan.source_run_id,
            "source_evidence_sha256": plan.source_evidence_sha256,
            "plan_sha256": plan.sha256(),
            "model_outputs": [],
            "tool_results": [],
        }
        self.evidence["replay"] = record
        tool_index = 0
        reply = None
        for number, turn in enumerate(plan.turns, 1):
            self._checkpoint(context.deadline_monotonic)
            reply = self._model_step(context, list(turn.messages), number)
            record["model_outputs"].append(
                {
                    "step_id": f"model-call-{number}",
                    "matches_source": reply_signature(reply.history_message) == turn.reply_sha256,
                }
            )
            for call, source_result in zip(turn.calls, turn.tool_result_sha256, strict=True):
                self._checkpoint(context.deadline_monotonic)
                tool_index += 1
                result = self._tool_step(context, call, tool_index)
                record["tool_results"].append(
                    {
                        "step_id": self.evidence["tool_calls"][-1]["step_id"],
                        "matches_source": tool_signature(result) == source_result,
                    }
                )
        self.evidence["final_answer"] = reply.content if reply is not None else None
