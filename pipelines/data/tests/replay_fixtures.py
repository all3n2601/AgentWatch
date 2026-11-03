"""Synthetic replay experiments for tests. Never recorded evidence.

Writes the directory layout of `agent-replay-experiment-v1` with the key names the agent's
`ReplayExperiment._replay` and the injectors' `InjectionSet.log()` produce, so readers and
labelers are tested against the real shapes with invented durations.
"""

import json
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
) -> Path:
    """Write one synthetic experiment under root/name and return its directory.

    Order follows `ExperimentSettings.schedule()`: a warm-up, the baselines, then each
    contended condition followed by a clean control. Replay indexes start at 0. A control
    defaults to the first baseline's seconds.
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
        evidence = {
            "replay": {
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
            }
        }
        (replay_dir / "agent-evidence.json").write_text(json.dumps(evidence))
        step_parents = {s: ROOT_STEP for s in step_ids if s != ROOT_STEP} | dict(parents or {})
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
                "run_id": f"00000000-0000-4000-8000-{index:012d}",
                "injection_active": True if kinds else None,
                "steps": {
                    step_id: {
                        "step_type": STEP_TYPES.get(step_id, "step"),
                        "state": "completed",
                        "total_seconds": total,
                        "queue_seconds": 0.001,
                        "execution_seconds": max(total - 0.001, 0.0),
                    }
                    for step_id, total in seconds.items()
                },
            }
        )
    manifest = {
        "protocol": "agent-replay-experiment-v1",
        "status": status,
        "checks": checks if checks is not None else dict(PASSING_CHECKS),
        "plan": {"task_id": task_id, "plan_sha256": "0" * 64, "model": "synthetic-model"},
        "replays": replays,
    }
    (directory / "experiment.json").write_text(json.dumps(manifest))
    return directory
