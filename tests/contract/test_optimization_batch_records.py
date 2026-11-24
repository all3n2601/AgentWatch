"""The optimization batch runner (O06) writes records the O04 checker accepts.

The runner (`services/agent`) and the checker (`pipelines/data`) declare `opt-batch-v1`
separately, because neither package imports the other. This test runs the runner on a manifest
the checker validated and a schedule the checker derived, then reads every record back through
the checker: any rejection is a disagreement between the two declarations.

Models, the live API, the policy service and injectors are the agent's test doubles; the records
are synthetic, never evidence.
"""

import json
from pathlib import Path

import pytest
from agent_test_doubles import FakeLiveServer
from agentwatch_agent.optimization_experiment import read_experiment
from agentwatch_agent.optimization_run import ExperimentRunner
from agentwatch_data.optimization_comparison import compare, schedule_document
from agentwatch_data.optimization_manifest import load_optimization_contract, parse_manifest
from optimization_doubles import DECLARED, FakePolicyService, make_host, manifest_document

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = load_optimization_contract(ROOT / "pipelines/model/optimization-contract.toml")
# Enough slots for a warm-up and one complete pair of the first condition.
RECORDED = 3


def experiment(root: Path, host, first_condition: str) -> None:
    """A development manifest the checker accepts, and the schedule it derives."""
    document = manifest_document(
        host,
        contract={"version": CONTRACT.version, "sha256": CONTRACT.sha256},
        # Two tasks need 10 pairs to reach the contract's completed-task minimum.
        repetitions={"pairs": 10, "warmup_batches": 1, "seed": 11},
    )
    document["conditions"].sort(key=lambda condition: condition["name"] != first_condition)
    manifest = parse_manifest(document, CONTRACT)
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(document, indent=2))
    (root / "schedule.json").write_text(json.dumps(schedule_document(manifest), indent=2))


def stop_after(count: int):
    """Identity readings for `count` slots, then an unreadable host: the run stops cleanly."""
    calls = {"n": 0}

    def observe() -> dict:
        calls["n"] += 1
        if calls["n"] > count:
            raise OSError("test stop")
        return json.loads(json.dumps(DECLARED))

    return observe


@pytest.mark.parametrize("first_condition", ["clean", "contended"])
def test_the_runners_records_pass_the_checker(tmp_path, first_condition):
    root = tmp_path / "experiment"
    host = make_host(FakeLiveServer(), FakePolicyService(), observe=stop_after(RECORDED))
    experiment(root, host, first_condition)
    runner = ExperimentRunner(read_experiment(root), host)
    runner.preflight()
    summary = runner.run()
    assert summary.attempted == list(range(RECORDED)) and summary.restored is True

    report = compare(root, CONTRACT)
    # Only the attempted slots have records, and the checker accepts every one.
    assert len(report["batches"]) == RECORDED
    assert all(batch["accepted"] for batch in report["batches"]), report["batches"]
    assert report["missing_batches"][:1] == [RECORDED]
    first = next(item for item in report["comparisons"] if item["condition"] == first_condition)
    assert first["pairs"]["complete"] == 1
    # One task solves, the other gives up: both outcomes reach the comparison as recorded.
    assert first["attempts"]["candidate"]["tasks_by_state"] == {"completed": 1, "failed": 1}


def test_an_interrupted_batchs_record_passes_the_checker(tmp_path, monkeypatch):
    # The process stops after the first slot's tasks ran but before its record was written;
    # the resumed runner records that attempt as `failed` from what it left.
    root = tmp_path / "experiment"
    host = make_host(FakeLiveServer(), FakePolicyService())
    experiment(root, host, "clean")
    write = ExperimentRunner._write

    def killed(self, path, entry):
        raise SystemExit("stopped before the record was written")

    monkeypatch.setattr(ExperimentRunner, "_write", killed)
    with pytest.raises(SystemExit):
        ExperimentRunner(read_experiment(root), host).run()
    monkeypatch.setattr(ExperimentRunner, "_write", write)
    summary = ExperimentRunner(read_experiment(root), host).run()
    assert summary.statuses == {0: "failed"}

    (batch,) = compare(root, CONTRACT)["batches"]
    assert batch["accepted"], batch
