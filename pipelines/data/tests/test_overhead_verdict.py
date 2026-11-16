"""Overhead verdicts (instrumentation-overhead-acceptance-v1) from saved overhead commands.

Every attempt document here is a synthetic fixture shaped like the `overhead` and
`sampling-overhead` outputs; none is recorded evidence.
"""

import json
import math
from pathlib import Path

import pytest
from agentwatch_data.optimization_manifest import load_optimization_contract
from agentwatch_data.overhead_verdict import (
    AttemptError,
    combined_verdict,
    evaluate,
    main,
    read_attempt,
    relative_increase,
)
from optimization_fixtures import CONTRACT_PATH

CONTRACT = load_optimization_contract(CONTRACT_PATH)
# Baseline walls with about 1% spread: stable under the 15% CV gate.
STABLE = [1.0 + 0.01 * ((i % 5) - 2) for i in range(30)]


def attempt_document(baseline, treated, protocol="agent-overhead-v1", **changes) -> dict:
    """A synthetic attempt whose repetitions alternate baseline, treated, baseline, ..."""
    names = (
        ("uninstrumented", "instrumented")
        if protocol == "agent-overhead-v1"
        else ("unsampled", "sampled")
    )
    measurements = []
    for pair, (base, test) in enumerate(zip(baseline, treated, strict=True)):
        measurements.append({"index": 2 * pair, "condition": names[0], "wall_seconds": base})
        measurements.append({"index": 2 * pair + 1, "condition": names[1], "wall_seconds": test})
    document = {
        "protocol": {"version": protocol},
        "task_id": "fixture-task",
        "host": {"system": "Linux", "machine": "x86_64"},
        "sampling_interval_seconds": 0.25,
        "measurements": measurements,
    }
    if protocol != "agent-overhead-v1":
        document |= {"status": "completed", "error": None, "model": "fixture-model"}
    return document | changes


def write(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def scaled(factor: float) -> list[float]:
    return [wall * factor for wall in STABLE]


def only_attempt(report: dict) -> dict:
    (group,) = report["groups"]
    (attempt,) = group["attempts"]
    return attempt


def test_relative_increase_is_the_ratio_of_condition_medians_minus_one():
    assert relative_increase([(1.0, 1.1), (2.0, 2.0), (3.0, 3.3)]) == pytest.approx(0.0)
    assert relative_increase([(1.0, 1.1), (1.0, 1.1), (1.0, 1.1)]) == pytest.approx(0.1)


def test_small_increase_is_within_budget(tmp_path):
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(1.01)))
    attempt = only_attempt(evaluate([path], CONTRACT))
    assert attempt["verdict"] == "within_budget"
    assert attempt["relative_increase"] == pytest.approx(0.01)
    # Every pair has the same ratio, so the interval collapses onto the estimate.
    assert attempt["interval"] == pytest.approx([0.01, 0.01])
    assert attempt["pairs"] == 30


def test_large_increase_is_over_budget(tmp_path):
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(1.2)))
    attempt = only_attempt(evaluate([path], CONTRACT))
    assert attempt["verdict"] == "over_budget"
    assert attempt["interval"][0] > CONTRACT.max_median_instrumentation_overhead


def test_interval_spanning_the_budget_is_inconclusive(tmp_path):
    # Half the instrumented repetitions add nothing and half add 10%: the median increase
    # sits near the 5% budget and resampling moves it to either side.
    treated = [wall * (1.0 if i % 2 else 1.1) for i, wall in enumerate(STABLE)]
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, treated))
    attempt = only_attempt(evaluate([path], CONTRACT))
    low, high = attempt["interval"]
    assert low <= CONTRACT.max_median_instrumentation_overhead < high
    assert attempt["verdict"] == "inconclusive"


def test_unstable_baseline_is_inconclusive_whatever_the_increase(tmp_path):
    unstable = [1.0 if i % 2 else 1.6 for i in range(30)]
    path = write(tmp_path / "a/overhead.json", attempt_document(unstable, unstable))
    attempt = only_attempt(evaluate([path], CONTRACT))
    assert attempt["baseline_cv"] > CONTRACT.max_baseline_cv
    assert attempt["verdict"] == "inconclusive"
    assert "coefficient of variation" in attempt["reason"]


def test_a_faster_instrumented_condition_is_reported_as_measured(tmp_path):
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(0.97)))
    attempt = only_attempt(evaluate([path], CONTRACT))
    assert attempt["relative_increase"] == pytest.approx(-0.03)
    assert attempt["verdict"] == "within_budget"


def test_budget_and_bootstrap_settings_come_from_the_contract(tmp_path):
    text = CONTRACT_PATH.read_text(encoding="utf-8").replace(
        "max_median_instrumentation_overhead = 0.05", "max_median_instrumentation_overhead = 0.3"
    )
    loose = load_optimization_contract(write_text(tmp_path / "contract.toml", text))
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(1.2)))
    report = evaluate([path], loose)
    assert only_attempt(report)["verdict"] == "within_budget"
    assert report["budget"]["max_median_instrumentation_overhead"] == 0.3
    assert report["contract"]["sha256"] == loose.sha256[:8]
    assert report["contract"]["status"] == "proposed"


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_the_interval_is_reproducible_from_the_recorded_seed(tmp_path):
    treated = [wall * (1.0 if i % 3 else 1.08) for i, wall in enumerate(STABLE)]
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, treated))
    first, again = evaluate([path], CONTRACT, seed=7), evaluate([path], CONTRACT, seed=7)
    assert first["bootstrap"]["seed"] == 7
    assert only_attempt(first)["interval"] == only_attempt(again)["interval"]


def test_a_failed_attempt_is_kept_but_not_evaluated(tmp_path):
    document = attempt_document(
        STABLE[:5], scaled(1.01)[:5], protocol="resource-sampling-overhead-v1"
    )
    document |= {"status": "failed", "error": "TimeoutError: model server did not answer"}
    path = write(tmp_path / "a/sampling-overhead.json", document)
    attempt = only_attempt(evaluate([path], CONTRACT))
    assert attempt["status"] == "failed"
    assert attempt["verdict"] == "not_evaluated"
    assert "TimeoutError" in attempt["reason"]


def test_an_attempt_without_per_repetition_walls_is_not_evaluated(tmp_path):
    document = attempt_document(STABLE, scaled(1.01))
    del document["measurements"]
    path = write(tmp_path / "a/overhead.json", document)
    attempt = only_attempt(evaluate([path], CONTRACT))
    assert attempt["verdict"] == "not_evaluated"
    assert "per-repetition" in attempt["reason"].lower()
    assert attempt["relative_increase"] is None and attempt["interval"] is None


def test_an_unpaired_final_repetition_is_counted_but_not_compared(tmp_path):
    document = attempt_document(STABLE, scaled(1.01), protocol="resource-sampling-overhead-v1")
    document["measurements"].append({"index": 60, "condition": "unsampled", "wall_seconds": 9.0})
    path = write(tmp_path / "a/sampling-overhead.json", document)
    attempt = only_attempt(evaluate([path], CONTRACT))
    assert attempt["pairs"] == 30 and attempt["unpaired_repetitions"] == 1
    assert attempt["relative_increase"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        (["within_budget", "within_budget"], "within_budget"),
        (["over_budget", "over_budget", "over_budget"], "over_budget"),
        (["within_budget"], "inconclusive"),
        (["within_budget", "over_budget"], "inconclusive"),
        (["within_budget", "inconclusive"], "inconclusive"),
        (["within_budget", "within_budget", "not_evaluated"], "inconclusive"),
        ([], "inconclusive"),
    ],
)
def test_combined_verdict_needs_two_completed_attempts_that_agree(verdicts, expected):
    attempts = [{"status": "completed", "verdict": verdict} for verdict in verdicts]
    assert combined_verdict(attempts) == expected


def test_failed_attempts_do_not_block_agreement_of_completed_ones():
    attempts = [
        {"status": "completed", "verdict": "within_budget"},
        {"status": "failed", "verdict": "not_evaluated"},
        {"status": "completed", "verdict": "within_budget"},
    ]
    assert combined_verdict(attempts) == "within_budget"


def test_attempts_with_the_same_settings_are_combined(tmp_path):
    paths = [
        write(tmp_path / f"{name}/overhead.json", attempt_document(STABLE, scaled(1.01)))
        for name in ("first", "second")
    ]
    (group,) = evaluate(paths, CONTRACT)["groups"]
    assert len(group["attempts"]) == 2
    assert group["verdict"] == "within_budget"
    assert group["settings"]["protocol"] == "agent-overhead-v1"


def test_attempts_with_different_settings_are_never_combined(tmp_path):
    first = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(1.01)))
    other = attempt_document(STABLE, scaled(1.01), sampling_interval_seconds=0.5)
    second = write(tmp_path / "b/overhead.json", other)
    groups = evaluate([first, second], CONTRACT)["groups"]
    assert len(groups) == 2
    assert [group["verdict"] for group in groups] == ["inconclusive", "inconclusive"]


def test_the_report_names_sources_without_host_paths(tmp_path):
    path = write(tmp_path / "attempt-1/overhead.json", attempt_document(STABLE, STABLE))
    report = evaluate([path], CONTRACT)
    assert only_attempt(report)["source"] == "attempt-1/overhead.json"
    assert len(only_attempt(report)["source_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(protocol={"version": "unknown-v9"}), "protocol"),
        (lambda d: d["measurements"][1].update(condition="uninstrumented"), "alternate"),
        (lambda d: d["measurements"][2].update(index=7), "index"),
        (lambda d: d["measurements"][0].update(wall_seconds=math.inf), "wall_seconds"),
        (lambda d: d["measurements"][0].update(wall_seconds=0.0), "wall_seconds"),
        (lambda d: d["measurements"][0].update(wall_seconds=True), "wall_seconds"),
        (lambda d: d.update(measurements={"index": 0}), "measurements"),
        (lambda d: d.update(status="finished"), "status"),
    ],
)
def test_malformed_attempts_are_rejected(tmp_path, change, message):
    document = attempt_document(STABLE, STABLE)
    change(document)
    path = tmp_path / "a/overhead.json"
    path.parent.mkdir()
    # allow_nan keeps the infinite value so the reader, not the writer, rejects it.
    path.write_text(json.dumps(document, allow_nan=True), encoding="utf-8")
    with pytest.raises(AttemptError, match=message):
        read_attempt(path)


def test_unreadable_attempts_are_rejected(tmp_path):
    path = tmp_path / "overhead.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AttemptError, match="overhead.json"):
        read_attempt(path)
    with pytest.raises(AttemptError, match="cannot be read"):
        read_attempt(tmp_path / "missing.json")


def test_main_writes_the_report_and_prints_each_verdict(tmp_path, capsys):
    paths = [
        write(tmp_path / f"{name}/overhead.json", attempt_document(STABLE, scaled(1.01)))
        for name in ("first", "second")
    ]
    output = tmp_path / "report"
    code = main([*map(str, paths), "--contract", str(CONTRACT_PATH), "--output", str(output)])
    assert code == 0
    report = json.loads((output / "overhead-verdict.json").read_text(encoding="utf-8"))
    assert report["groups"][0]["verdict"] == "within_budget"
    printed = capsys.readouterr().out
    assert "within_budget" in printed and "first/overhead.json" in printed


def test_main_exits_1_on_a_malformed_attempt(tmp_path, capsys):
    path = write(tmp_path / "overhead.json", {"protocol": {"version": "unknown-v9"}})
    code = main([str(path), "--contract", str(CONTRACT_PATH), "--output", str(tmp_path / "r")])
    assert code == 1
    assert "overhead.json" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()
