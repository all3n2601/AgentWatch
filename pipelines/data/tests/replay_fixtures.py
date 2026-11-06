"""Synthetic replay experiments for tests. Never recorded evidence.

Writes the directory layout of `agent-replay-experiment-v1` with the key names the agent's
`ReplayExperiment._replay` and the injectors' `InjectionSet.log()` produce, so readers and
labelers are tested against the real shapes with invented durations. With `live=True` the
journals hold SDK-valid live-contracts-v1 lines, as the agent runner writes them, so the
evidence store and feature computation can read them too.
"""

import hashlib
import json
import shutil
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT_STEP = "agent-run"
# Real injector methods and the resource classes they contend (services/injectors/README.md).
INJECTOR_BY_KIND = {
    "cpu": ("cpu-spin-processes-v1", "cpu_sandbox"),
    "inference": ("competing-ollama-generate-v1", "inference"),
    "retrieval": ("in-process-index-query-flood-v1", "retrieval"),
}
MIXED_KINDS = ("cpu", "inference")
STEP_TYPES = {ROOT_STEP: "agent_run", "model-call-1": "model_call", "tool-call-1": "tool_call"}

DEFAULT_STEPS = {ROOT_STEP: 1.0, "model-call-1": 0.6, "tool-call-1": 0.2}
# The protocol checks labeling requires; the real manifest always records every check.
PASSING_CHECKS = {
    "baseline_replays_complete": True,
    "stable_baseline": True,
    "same_steps_every_replay": True,
    "injectors_cleaned_up": True,
}

# Live journals (live=True): live-contracts-v1 step types and the runner's resources.
LIVE_STEP_TYPES = {ROOT_STEP: "other", "model-call-1": "model_call", "tool-call-1": "tool"}
LIVE_RESOURCES = {
    "model_call": ("model-server-1", "ollama", "inference"),
    "tool": ("tool-host-1", "agentwatch-agent-runner", "cpu_sandbox"),
}
LIVE_CONFIGURATION = {"sampling_interval_seconds": 0.25}
SAMPLING_INTERVAL_NS = 250_000_000
QUEUE_NS = 1_000_000  # every live step waits 1 ms between submission and start
CLOCK_ID = "synthetic-clock"
OBSERVED_AT = "2026-11-05T00:00:00Z"
LIVE_IDS = uuid.UUID("6d1c4b52-3f0e-4c47-9a39-0d4f6a0c2b7e")  # uuid5 namespace for record IDs


def write_experiment(
    root: Path,
    *,
    name: str = "synthetic-exp",
    task_id: str = "fix-a-r1",
    baseline: list[dict[str, float]] | None = None,
    contended: dict[str, dict[str, float]] | None = None,
    controls: dict[str, dict[str, float]] | None = None,
    status: str = "completed",
    checks: dict[str, bool] | None = None,
    mismatched: Iterable[tuple[int, str]] = (),
    parents: Mapping[str, str] | None = None,
    live: bool = False,
    started_at: str = "2026-11-05T00:00:00+00:00",
    passed: bool | None = True,
    run_namespace: int = 0,
    plan_sha256: str = "0" * 64,
) -> Path:
    """Write one synthetic experiment under root/name and return its directory.

    Order follows `ExperimentSettings.schedule()`: a warm-up, the baselines, then each
    contended condition followed by a clean control. Replay indexes start at 0. A control
    defaults to the first baseline's seconds. With `live=True` the manifest timings are the
    journal's tick differences, so features computed from the journal reproduce them exactly.
    Experiments stored together need distinct run_namespace values (and, to be independent
    for split-v2, distinct plan_sha256 values), because replay run IDs repeat per index.
    """
    baseline = baseline if baseline is not None else [dict(DEFAULT_STEPS) for _ in range(3)]
    contended = contended if contended is not None else {"cpu": DEFAULT_STEPS}
    controls = controls or {}
    schedule: list[tuple[str, str, dict[str, float]]] = [("warmup", "clean", baseline[0])]
    schedule += [("baseline", "clean", seconds) for seconds in baseline]
    for condition, seconds in contended.items():
        schedule.append(("contended", condition, seconds))
        schedule.append(("clean_control", "clean", controls.get(condition, baseline[0])))

    directory = root / name
    mismatches = set(mismatched)
    replays = []
    for index, (role, condition, seconds) in enumerate(schedule):
        number = f"{index:02d}"
        kinds = (
            () if condition == "clean" else MIXED_KINDS if condition == "mixed" else (condition,)
        )
        replay_dir = directory / "replays" / number
        replay_dir.mkdir(parents=True)
        step_ids = list(seconds)
        # The fourth group keeps the RFC 4122 variant bits (8-b) for every namespace below 0x4000.
        run_id = f"00000000-0000-4000-{0x8000 + run_namespace:04x}-{index:012d}"
        evidence = {
            "run_id": run_id,
            "replay": {
                "plan_sha256": plan_sha256,
                "model_outputs": [
                    {"step_id": step_id, "matches_source": (index, step_id) not in mismatches}
                    for step_id in step_ids
                    if step_id.startswith("model-call")
                ],
                "tool_results": [
                    {"step_id": step_id, "matches_source": (index, step_id) not in mismatches}
                    for step_id in step_ids
                    if step_id.startswith("tool-call")
                ],
            },
        }
        step_parents = {s: ROOT_STEP for s in step_ids if s != ROOT_STEP} | dict(parents or {})
        if live:
            evidence["configuration"] = LIVE_CONFIGURATION
            evidence["configuration_digest"] = configuration_digest()
            lines = live_journal(run_id, task_id, seconds, step_parents)
        else:
            lines = [
                {
                    "kind": "record",
                    "payload": {
                        "record_type": "step_event",
                        "step_id": step_id,
                        "parent_step_id": step_parents.get(step_id),
                    },
                }
                for step_id in step_ids
            ]
        (replay_dir / "agent-evidence.json").write_text(json.dumps(evidence))
        (replay_dir / "live-journal.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines)
        )
        if kinds:
            (directory / "injections").mkdir(exist_ok=True)
            log = {
                "injectors": list(kinds),
                "injections": [
                    {
                        "injector": kind,
                        "method": INJECTOR_BY_KIND[kind][0],
                        "contended_resource_class": INJECTOR_BY_KIND[kind][1],
                    }
                    for kind in kinds
                ],
            }
            (directory / "injections" / f"{number}.json").write_text(json.dumps(log))
        replays.append(
            {
                "index": index,
                "role": role,
                "condition": condition,
                "evidence_dir": f"replays/{number}",
                "injection_log": f"injections/{number}.json" if kinds else None,
                "state": "completed",
                "task_passed": True,
                "run_id": run_id,
                "injection_active": True if kinds else None,
                "steps": {
                    step_id: live_timing(step_id, total) if live else static_timing(step_id, total)
                    for step_id, total in seconds.items()
                },
            }
        )
    if live:
        (directory / "source").mkdir()
        for file in ("agent-evidence.json", "live-journal.jsonl"):
            shutil.copyfile(directory / "replays" / "00" / file, directory / "source" / file)
    manifest = {
        "protocol": "agent-replay-experiment-v1",
        "status": status,
        "started_at": started_at,
        "checks": checks if checks is not None else dict(PASSING_CHECKS),
        "passed": passed,
        "plan": {"task_id": task_id, "plan_sha256": plan_sha256, "model": "synthetic-model"},
        "replays": replays,
    }
    (directory / "experiment.json").write_text(json.dumps(manifest))
    return directory


def static_timing(step_id: str, total: float) -> dict:
    return {
        "step_type": STEP_TYPES.get(step_id, "step"),
        "state": "completed",
        "total_seconds": total,
        "queue_seconds": 0.001,
        "execution_seconds": max(total - 0.001, 0.0),
    }


def live_timing(step_id: str, total: float) -> dict:
    queue_ns, execution_ns = live_durations_ns(total)
    return {
        "step_type": LIVE_STEP_TYPES.get(step_id, "other"),
        "state": "completed",
        "total_seconds": (queue_ns + execution_ns) / 1e9,
        "queue_seconds": queue_ns / 1e9,
        "execution_seconds": execution_ns / 1e9,
    }


def live_durations_ns(total_seconds: float) -> tuple[int, int]:
    """(queue, execution) nanoseconds; a step shorter than the queue delay never waits."""
    total_ns = round(total_seconds * 1e9)
    queue_ns = min(QUEUE_NS, total_ns)
    return queue_ns, total_ns - queue_ns


def configuration_digest() -> str:
    canonical = json.dumps(LIVE_CONFIGURATION, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def live_id(*parts: object) -> str:
    """A deterministic UUID, so the same fixture always writes the same journal bytes."""
    return str(uuid.uuid5(LIVE_IDS, "/".join(map(str, parts))))


def live_journal(
    run_id: str, task_id: str, seconds: Mapping[str, float], parents: Mapping[str, str]
) -> list[dict]:
    """Registration, run lifecycle, queued/running/completed events per step, then samples.

    Steps run one after another on one clock, each starting on a sampling tick so even a step
    shorter than the interval holds a sample. Pressure is sampled every interval on the tool
    host from tick 0 until the last step finishes; the value is invented and constant.
    """
    record = {
        "contract_version": "live-contracts-v1",
        "run_id": run_id,
        "task_id": task_id,
        "observed_at": OBSERVED_AT,
        "clock_id": CLOCK_ID,
    }
    tool_host = LIVE_RESOURCES["tool"]
    lines: list[dict] = [
        {
            "kind": "registration",
            "payload": {
                "contract_version": "live-contracts-v1",
                "request_id": live_id(run_id, "registration"),
                "task_id": task_id,
                "configuration_digest": configuration_digest(),
                "clock_ids": [CLOCK_ID],
                "resources": [
                    {"resource_id": rid, "service_id": sid, "resource_class": rclass}
                    for rid, sid, rclass in LIVE_RESOURCES.values()
                ],
            },
        },
        {"kind": "run_created", "payload": {"run_id": run_id, "created": True}},
        run_change(run_id, 0, "running"),
    ]
    started_ns = SAMPLING_INTERVAL_NS
    finished_ns = started_ns
    for number, (step_id, total) in enumerate(seconds.items(), start=1):
        queue_ns, execution_ns = live_durations_ns(total)
        finished_ns = started_ns + execution_ns
        step_type = LIVE_STEP_TYPES.get(step_id, "other")
        resource_id, service_id, _ = LIVE_RESOURCES.get(step_type, (None, None, None))
        step = record | {
            "record_type": "step_event",
            "step_id": step_id,
            "step_type": step_type,
            "parent_step_id": parents.get(step_id),
            "trace_id": uuid.UUID(run_id).hex,
            "span_id": f"{number:016x}",
            "resource_id": resource_id,
            "service_id": service_id,
            "submitted_tick_ns": started_ns - queue_ns,
            "cpu_seconds": None,
            "cpu_scope": None,
            "reason": None,
        }
        for sequence, state in enumerate(("queued", "running", "completed")):
            event = step | {
                "event_id": live_id(run_id, step_id, sequence),
                "sequence": sequence,
                "state": state,
                "started_tick_ns": started_ns if sequence else None,
                "finished_tick_ns": finished_ns if state == "completed" else None,
            }
            lines.append({"kind": "record", "payload": event})
        # The next step starts on a later tick, at least one interval after this one ends.
        started_ns = (finished_ns // SAMPLING_INTERVAL_NS + 2) * SAMPLING_INTERVAL_NS
    for tick_ns in range(0, finished_ns + 1, SAMPLING_INTERVAL_NS):
        sample = record | {
            "record_type": "resource_sample",
            "sample_id": live_id(run_id, "sample", tick_ns),
            "resource_id": tool_host[0],
            "service_id": tool_host[1],
            "resource_class": tool_host[2],
            "tick_ns": tick_ns,
            "metric": "cpu.psi.some.avg10.v1",
            "unit": "percent",
            "value": 10.0,
            "availability": "measured",
            "reason": None,
        }
        lines.append({"kind": "record", "payload": sample})
    lines.append(run_change(run_id, 1, "completed"))
    return lines


def run_change(run_id: str, expected_revision: int, state: str) -> dict:
    return {
        "kind": "run_change",
        "payload": {
            "change_id": live_id(run_id, "change", expected_revision),
            "expected_revision": expected_revision,
            "state": state,
            "reason": None,
        },
    }
