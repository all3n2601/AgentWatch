"""Held-out scoring accepts the truth folders `sweep-truth-v1` writes for the frozen A06 sweeps.

The scorer (`pipelines/model`) re-checks the truth format it reads instead of importing the
agent service, so this test, which may import both packages, records each frozen sweep's steps,
labels them and feeds the result to the scorer's loader. The queue-class counts are those pinned
in `test_sweep_truth_frozen.py`: a mismatch is a defect to diagnose, not a value to update.
"""

from pathlib import Path

import pytest
from agentwatch_agent import sweep_steps as recorder
from agentwatch_data.sweep_truth import write_truths
from agentwatch_model.held_out_rows import load_truths

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "data" / "held-out" / "a06-20261111" / "capacity-sweeps"
MODEL_CALL = "model_call"

# folder -> model calls whose queue truth is a class (inference or none), not null.
QUEUE_CLASSED_MODEL_CALLS = {
    "cfg-default-20261111T211237Z-2d75ea": 825,
    "cfg-hardware-20261111T213039Z-b4ad4a": 915,
}


@pytest.mark.parametrize("folder", sorted(QUEUE_CLASSED_MODEL_CALLS))
def test_real_truth_folders_load(folder, tmp_path):
    sweep = SWEEPS / folder
    if not (sweep / "sweep.json").is_file():
        pytest.skip(f"{sweep} is not pulled (DVC data)")

    steps_dir = recorder.write_steps(sweep, tmp_path / "steps")
    truth_dir = write_truths(steps_dir, tmp_path / "truth")

    summary, records = load_truths(truth_dir)

    assert summary["sweep_folder"] == folder
    queue_classed = sum(
        record["step_type"] == MODEL_CALL and record["truths"]["queue"] is not None
        for record in records
    )
    assert queue_classed == QUEUE_CLASSED_MODEL_CALLS[folder]
