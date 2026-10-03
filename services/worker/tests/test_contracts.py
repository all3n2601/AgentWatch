from copy import deepcopy

import pytest
from agentwatch_worker.contracts import RunIngest
from pydantic import ValidationError


def test_valid_evidence_preserves_unavailable_samples(evidence):
    parsed = RunIngest.model_validate(evidence)
    assert parsed.samples[0].cpu_pressure is None
    assert parsed.result.clean.submitted_ns == evidence["result"]["clean"]["submitted_ns"]


@pytest.mark.parametrize(
    "change", ["timing", "fraction", "step_id", "repeat_count", "checks", "foreign_span"]
)
def test_rejects_inconsistent_evidence(evidence, change):
    result = evidence["result"]
    if change == "timing":
        result["clean"]["started_ns"] = result["clean"]["finished_ns"] + 1
    elif change == "fraction":
        result["clean"]["queue_fraction"] = 0.5
    elif change == "step_id":
        result["clean"]["step_id"] = "baseline-0"
    elif change == "repeat_count":
        result["experiment"]["baseline_repeats"] = 10
    elif change == "checks":
        result["checks"]["stable_baseline"] = False
    else:
        evidence["spans"][0]["attributes"]["agentwatch.run_id"] = "another-run"
    with pytest.raises(ValidationError):
        RunIngest.model_validate(evidence)


def test_rejects_nonfinite_extra_evidence(evidence):
    data = deepcopy(evidence)
    data["spans"][0]["attributes"]["invalid"] = float("nan")
    with pytest.raises(ValidationError):
        RunIngest.model_validate(data)
