"""sweep-truth-v2 keeps agreeing with the recorder and with the frozen A06 sweeps.

The recorder (`services/agent`) and the rule (`pipelines/data`) declare the steps format
separately because the data pipeline must not import the agent service; the first test keeps the
two declarations equal. The pinned counts below are those of the scoring branch's attempt
`20261112T025531Z-220363` for model calls, plus the spec's root-step and no-queue-reference rows.
They are measured numbers: a mismatch is a defect to diagnose, not a value to update.
"""

import hashlib
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

# folder -> (steps, model calls, queue, slower, protocol_clean, root_step, no_queue_reference,
# and the recorder's excluded batches and runs it could not record). queue / slower /
# protocol_clean map "class or reason" to a model-call count. runs_without_plan counts the live
# runs, which have no replay plan. The three hashes identify the written files: steps.jsonl,
# steps.json (which carries the recorder's code fingerprint) and truth.jsonl.
PINNED = {
    "cfg-default-20261111T211237Z-2d75ea": {
        "steps": 1896,
        "model_calls": 825,
        "queue": {"inference": 686, "none": 139},
        "slower": {"inference": 715, "none": 80, "not_applicable": 30},
        "protocol_clean": {"none": 45, "not_applicable": 780},
        "root_step": 165,
        "no_queue_reference": 906,
        "excluded_batches": [],
        "runs_without_evidence": 0,
        "runs_without_plan": 9,
        "steps_sha256": "7598a42e00fc0a3cf46dcc945646bf723339e280b0f008a0005c527796e81162",
        "steps_summary_sha256": "4cfd4804ee0165b9d57d4ae2c9ae24a363f3ffef6d9680d0594f4228a4289b70",
        "truth_sha256": "92da54bca1be5b897e32ec72e6dc1f34dad1bb6695fd8a643b84bf3d19d0debe",
    },
    "cfg-hardware-20261111T213039Z-b4ad4a": {
        "steps": 2104,
        "model_calls": 915,
        "queue": {"inference": 8, "none": 907},
        "slower": {"inference": 663, "none": 222, "not_applicable": 30},
        "protocol_clean": {"none": 30, "not_applicable": 885},
        "root_step": 183,
        "no_queue_reference": 1006,
        "excluded_batches": [],
        "runs_without_evidence": 0,
        "runs_without_plan": 24,
        "steps_sha256": "0ebe5446d0ef9a571d211dff32174ff8dece618dd1e302926bbf495c2d496bfe",
        "steps_summary_sha256": "099c05a492ec34f66fdca7cd7b100c2b5b441688ad353e22e4ab682810aaa1ad",
        "truth_sha256": "069a8303ac376c58a821b453ade440347fee94d591ce92a5f22f06a838b71308",
    },
}


def test_recorder_and_rule_agree_on_the_format():
    assert recorder.STEPS_FORMAT == reader.STEPS_FORMAT
    assert recorder.RECORD_KEYS == reader.RECORD_KEYS
    assert recorder.SPAN_STATUSES == reader.SPAN_STATUSES
    assert recorder.SUMMARY_KEYS == reader.SUMMARY_KEYS
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
    rule = load_truth_rule()
    assert summary["label_version"] == rule.label_version
    assert summary["rule"] == rule.values
    truth_bytes = (truth_dir / "truth.jsonl").read_bytes()
    assert summary["truth_sha256"] == hashlib.sha256(truth_bytes).hexdigest()
    steps_bytes = (steps_dir / "steps.jsonl").read_bytes()
    steps_summary_bytes = (steps_dir / "steps.json").read_bytes()
    assert summary["steps_sha256"] == hashlib.sha256(steps_bytes).hexdigest()
    assert summary["steps_summary_sha256"] == hashlib.sha256(steps_summary_bytes).hexdigest()
    assert summary["recorder_sha256"] == recorder.recorder_sha256()
    assert hashlib.sha256(steps_bytes).hexdigest() == expected["steps_sha256"]
    assert hashlib.sha256(steps_summary_bytes).hexdigest() == expected["steps_summary_sha256"]
    assert summary["truth_sha256"] == expected["truth_sha256"]
    for name in ("excluded_batches", "runs_without_evidence", "runs_without_plan"):
        assert summary[name] == expected[name], name
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
