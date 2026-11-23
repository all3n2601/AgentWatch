"""Test doubles for the optimization batch runner (O06). Synthetic, never recorded evidence.

Builds experiment directories by hand in the shapes the data pipeline writes, a policy service
held in memory, and a host whose runners replay scripted model responses.
"""

import copy
import json
from pathlib import Path
from urllib.error import HTTPError

from agent_test_doubles import SOLVE_MEAN, ScriptedModel, chat_response, policy_version
from agentwatch_agent.admission import (
    AdmissionController,
    AdmissionSettings,
    QueueBounds,
    RunnerAdmission,
)
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.optimization_run import Host
from agentwatch_agent.replay import canonical_sha256
from agentwatch_agent.tasks import load_task
from agentwatch_injectors.prefill import PREFILL_METHOD
from agentwatch_sdk.policy import PolicyChangeResult
from test_runner_admission import make_runner
from test_sweep import FakeInjector

# The model gives up at once, so verification fails: a real failed task, not a runner error.
GIVE_UP = [chat_response("I could not find the defect.")]
SCRIPTS = {"fix-mean-empty": SOLVE_MEAN, "fix-median-even": GIVE_UP}
TASK_NAMES = tuple(SCRIPTS)
OFFSETS = {"fix-mean-empty": 0.0, "fix-median-even": 0.2}
BOUNDS = {"inference_requests": (8, 60.0), "test_jobs": (8, 60.0)}
PREFILL = {
    "kind": "inference",
    "method": PREFILL_METHOD,
    "intensity": {"concurrency": 2, "prompt_tokens": 2048},
}
DECLARED = {
    "model": {"backend": "ollama", "name": "scripted-test-double", "digest": "test-digest"},
    "hardware": {
        "host_id": "test-host",
        "cpu_model": "Synthetic CPU",
        "logical_cpus": 8,
        "memory_bytes": 2**34,
        "os": "Test OS",
        "kernel": "test-kernel",
        "gpus": [],
    },
    "power": {"source": "ac", "profile": "Balanced"},
    "state": {"tests": "task-fixtures-test", "retrieval": "not-used"},
}


class FakePolicyService:
    """The O02 policy API, in memory: versions only grow; a change must name the active one."""

    def __init__(self, ignore_service: bool = False):
        self.versions = [policy_version(0)]
        self.changes = []
        self.failures: list[BaseException] = []  # raised by the next change calls, in order
        self.ignore_service = ignore_service

    def get_policy(self, target="coding-agent"):
        return self.versions[-1]

    def change_policy(self, change, target="coding-agent"):
        if self.failures:
            raise self.failures.pop(0)
        current = self.versions[-1]
        if change.expected_version != current.version:
            raise HTTPError("policy", 409, "Policy changed", {}, None)
        limits = change.limits.model_dump(exclude_none=True) if change.limits else {}
        service = current.controlled_services["inference_requests"].service_id
        if change.inference_service is not None and not self.ignore_service:
            service = change.inference_service
        new = policy_version(
            current.version + 1,
            inference=limits.get("inference_requests"),
            tests=limits.get("test_jobs"),
            inference_service=service,
            action=change.action,
        )
        self.versions.append(new)
        self.changes.append(change)
        return PolicyChangeResult(created=True, policy=new)


def bounds() -> dict[str, QueueBounds]:
    return {control: QueueBounds(*values) for control, values in BOUNDS.items()}


def make_host(server, policy, *, observe=None, injector=None, run_timeout=60.0) -> Host:
    def runner(task, directory, clock, admission):
        model = ScriptedModel(copy.deepcopy(SCRIPTS[task.name]))
        return make_runner(task, server, model, directory, admission=admission, clock=clock)

    def make_injector(planned, clock, bound_seconds):
        return (injector or (lambda c, b: FakeInjector(c, b)))(clock, bound_seconds)

    return Host(
        policy=policy,
        inference_service_id="ollama",
        make_runner=runner,
        make_injector=make_injector,
        observe_identity=observe or (lambda: copy.deepcopy(DECLARED)),
        wait_until_idle=lambda _cancel: None,
        run_timeout_seconds=run_timeout,
        policy_poll_seconds=0.05,
        drain_seconds=5.0,
        ramp_seconds=0.0,
    )


def task_entries(host: Host) -> list[dict]:
    """Each frozen task as the manifest preregisters it, digests from this host's runners."""
    controller = AdmissionController(
        AdmissionSettings(**bounds()), inference_service_id=host.inference_service_id
    )
    admission = RunnerAdmission(controller, host.policy)
    entries = []
    for name in TASK_NAMES:
        task = load_task(name)
        runner = host.make_runner(task, Path("unused"), MonotonicClock(), admission)
        entries.append(
            {
                "task_id": task.task_id,
                "content_sha256": task.content_sha256,
                "configuration_digest": runner.configuration_digest(),
            }
        )
    return entries


def manifest_document(host: Host, **changes) -> dict:
    tasks = task_entries(host)
    document = {
        "format": "opt-experiment-manifest-v1",
        "experiment_id": "0b7e7c55-2a37-4f6b-8b1b-3c9d7a1f2e10",
        "created_at": "2026-11-22T10:00:00Z",
        "contract": {"version": "opt-contract-test", "sha256": "0" * 64},
        "purpose": "development",
        "workload": {
            "unit": "batch",
            "tasks": tasks,
            "arrivals": {
                "method": "fixed_offsets_v1",
                "tolerance_seconds": 0.25,
                "offsets": [
                    {"task_id": entry["task_id"], "offset_seconds": OFFSETS[name]}
                    for entry, name in zip(tasks, TASK_NAMES, strict=True)
                ],
            },
        },
        "runner": {"version": "agent-runner-v4"},
        **copy.deepcopy(DECLARED),
        "conditions": [
            {"name": "clean", "injectors": []},
            {"name": "contended", "injectors": [PREFILL]},
        ],
        "policies": {
            "baseline": {"limits": {}},
            "candidates": [{"name": "inference-1", "limits": {"inference_requests": 1}}],
        },
        "queue_bounds": {
            control: {"max_length": length, "max_wait_seconds": wait}
            for control, (length, wait) in BOUNDS.items()
        },
        "repetitions": {"pairs": 5, "warmup_batches": 1, "seed": 7},
    }
    document.update(changes)
    return document


def slot(index, condition="clean", configuration="baseline", kind="pair", pair_index=0):
    candidate = "inference-1" if kind == "pair" else None
    return {
        "index": index,
        "kind": kind,
        "condition": condition,
        "configuration": configuration,
        "candidate": candidate,
        "pair_index": pair_index if kind == "pair" else None,
        "policy_sha256": ("c" if configuration == "candidate" else "b") * 64,
        "limits": {"inference_requests": 1} if configuration == "candidate" else {},
    }


def write_experiment(root: Path, host: Host, slots: list[dict], **changes) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = manifest_document(host, **changes)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    schedule = {"manifest_sha256": canonical_sha256(manifest), "batches": slots}
    (root / "schedule.json").write_text(json.dumps(schedule, indent=2))
    return manifest


def record(root: Path, index: int) -> dict:
    return json.loads((root / "batches" / f"{index:03d}.json").read_text())
