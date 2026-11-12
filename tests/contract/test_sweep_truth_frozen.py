"""sweep-truth-v1 keeps agreeing with the recorder and with the frozen A06 sweeps.

The recorder (`services/agent`) and the rule (`pipelines/data`) declare the steps format
separately because the data pipeline must not import the agent service; the first test keeps the
two declarations equal. The pinned counts below are those of the scoring branch's attempt
`20261112T025531Z-220363` for model calls, plus the spec's root-step and no-queue-reference rows.
They are measured numbers: a mismatch is a defect to diagnose, not a value to update.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
from agentwatch_agent import sweep_protocol
from agentwatch_agent import sweep_steps as recorder
from agentwatch_data import sweep_steps_input as reader
from agentwatch_data.sweep_truth import load_truth_rule, write_truths

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "data" / "held-out" / "a06-20261111" / "capacity-sweeps"
MODEL_CALL = "model_call"

# folder -> (steps, model calls, queue, slower, protocol_clean, root_step, no_queue_reference).
# queue / slower / protocol_clean map "class or reason" to a model-call count.
PINNED = {
    "cfg-default-20261111T211237Z-2d75ea": {
        "steps": 1896,
        "model_calls": 825,
        "queue": {"inference": 686, "none": 139},
        "slower": {"inference": 715, "none": 80, "not_applicable": 30},
        "protocol_clean": {"none": 45, "not_applicable": 780},
        "root_step": 165,
        "no_queue_reference": 906,
    },
    "cfg-hardware-20261111T213039Z-b4ad4a": {
        "steps": 2104,
        "model_calls": 915,
        "queue": {"inference": 8, "none": 907},
        "slower": {"inference": 663, "none": 222, "not_applicable": 30},
        "protocol_clean": {"none": 30, "not_applicable": 885},
        "root_step": 183,
        "no_queue_reference": 1006,
    },
}


def test_recorder_and_rule_agree_on_the_format():
    assert recorder.STEPS_FORMAT == reader.STEPS_FORMAT
    assert recorder.RECORD_KEYS == reader.RECORD_KEYS
    assert recorder.SPAN_STATUSES == reader.SPAN_STATUSES
    rule = load_truth_rule()
    assert rule.protocols == recorder.ACCEPTED_PROTOCOL_SHA256
    # The recorder declares the thresholds in the protocol it hashes; the rule must apply the same.
    assert rule.min_seconds == sweep_protocol.LABEL_THRESHOLD_FLOOR_SECONDS
    assert rule.stdev_multiplier == sweep_protocol.LABEL_THRESHOLD_STDEV_MULTIPLIER


def _by_truth(records: list[dict], name: str) -> Counter:
    """Model calls tallied by class, or by reason where the truth is null."""
    return Counter(
        record["truths"][name] or record["truth_reasons"][name]
        for record in records
        if record["step_type"] == MODEL_CALL
    )


@pytest.mark.parametrize("folder", sorted(PINNED))
def test_frozen_sweeps_keep_their_model_call_truths(folder, tmp_path):
    sweep = SWEEPS / folder
    if not (sweep / "sweep.json").is_file():
        pytest.skip(f"{sweep} is not pulled (DVC data)")
    expected = PINNED[folder]

    steps_dir = recorder.write_steps(sweep, tmp_path / "steps")
    truth_dir = write_truths(steps_dir, tmp_path / "truth")
    records = [
        json.loads(line)
        for line in (truth_dir / "truth.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    summary = json.loads((truth_dir / "truth.json").read_text(encoding="utf-8"))
    assert summary["label_version"] == load_truth_rule().label_version
    assert len(records) == expected["steps"]
    assert sum(record["step_type"] == MODEL_CALL for record in records) == expected["model_calls"]
    for name in ("queue", "slower", "protocol_clean"):
        assert dict(_by_truth(records, name)) == expected[name], name
    for name in ("queue", "slower", "protocol_clean"):
        reasons = Counter(
            record["truth_reasons"][name] for record in records if record["step_type"] != MODEL_CALL
        )
        assert reasons["root_step"] == expected["root_step"], name
        assert reasons["no_queue_reference"] == expected["no_queue_reference"], name
