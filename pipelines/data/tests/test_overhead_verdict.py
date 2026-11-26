"""Overhead verdicts (instrumentation-overhead-acceptance-v1) from saved overhead commands.

Every attempt document here is a synthetic fixture shaped like the `overhead` and
`sampling-overhead` outputs; none is recorded evidence.
"""

import itertools
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
HOST = "test-host"
ATTEMPT_SECONDS = itertools.count()
# Baseline walls with about 1% spread: stable under the 15% CV gate.
STABLE = [1.0 + 0.01 * ((i % 5) - 2) for i in range(30)]


def attempt_document(baseline, treated, protocol="agent-overhead-v1", **changes) -> dict:
    """A synthetic attempt whose repetitions alternate baseline, treated, baseline, ..."""
    names = {
        "agent-overhead-v1": ("uninstrumented", "instrumented"),
        "admission-overhead-v1": ("admission_off", "admission_on"),
    }.get(protocol, ("unsampled", "sampled"))
    measurements = []
    for pair, (base, test) in enumerate(zip(baseline, treated, strict=True)):
        measurements.append({"index": 2 * pair, "condition": names[0], "wall_seconds": base})
        measurements.append({"index": 2 * pair + 1, "condition": names[1], "wall_seconds": test})
    document = {
        # Each real attempt records its own time, so no two attempt files are identical.
        "recorded_at": f"2026-11-16T00:00:{next(ATTEMPT_SECONDS):02d}+00:00",
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
    attempt = only_attempt(evaluate([path], CONTRACT, HOST))
    assert attempt["verdict"] == "within_budget"
    assert attempt["relative_increase"] == pytest.approx(0.01)
    # Every pair has the same ratio, so the interval collapses onto the estimate.
    assert attempt["interval"] == pytest.approx([0.01, 0.01])
    assert attempt["pairs"] == 30


def test_large_increase_is_over_budget(tmp_path):
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(1.2)))
    attempt = only_attempt(evaluate([path], CONTRACT, HOST))
    assert attempt["verdict"] == "over_budget"
    assert attempt["interval"][0] > CONTRACT.max_median_instrumentation_overhead


def test_interval_spanning_the_budget_is_inconclusive(tmp_path):
    # Half the instrumented repetitions add nothing and half add 10%: the median increase
    # sits near the 5% budget and resampling moves it to either side.
    treated = [wall * (1.0 if i % 2 else 1.1) for i, wall in enumerate(STABLE)]
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, treated))
    attempt = only_attempt(evaluate([path], CONTRACT, HOST))
    low, high = attempt["interval"]
    assert low <= CONTRACT.max_median_instrumentation_overhead < high
    assert attempt["verdict"] == "inconclusive"


def test_unstable_baseline_is_inconclusive_whatever_the_increase(tmp_path):
    unstable = [1.0 if i % 2 else 1.6 for i in range(30)]
    path = write(tmp_path / "a/overhead.json", attempt_document(unstable, unstable))
    attempt = only_attempt(evaluate([path], CONTRACT, HOST))
    assert attempt["baseline_cv"] > CONTRACT.max_baseline_cv
    assert attempt["verdict"] == "inconclusive"
    assert "coefficient of variation" in attempt["reason"]


def test_a_faster_instrumented_condition_is_reported_as_measured(tmp_path):
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(0.97)))
    attempt = only_attempt(evaluate([path], CONTRACT, HOST))
    assert attempt["relative_increase"] == pytest.approx(-0.03)
    assert attempt["verdict"] == "within_budget"


def test_budget_and_bootstrap_settings_come_from_the_contract(tmp_path):
    text = CONTRACT_PATH.read_text(encoding="utf-8").replace(
        "max_median_instrumentation_overhead = 0.05", "max_median_instrumentation_overhead = 0.3"
    )
    loose = load_optimization_contract(write_text(tmp_path / "contract.toml", text))
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(1.2)))
    report = evaluate([path], loose, HOST)
    assert only_attempt(report)["verdict"] == "within_budget"
    assert report["budget"]["max_median_instrumentation_overhead"] == 0.3
    assert report["contract"]["sha256"] == loose.sha256[:8]
    assert report["contract"]["status"] == loose.status


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_the_interval_is_reproducible_from_the_recorded_seed(tmp_path):
    treated = [wall * (1.0 if i % 3 else 1.08) for i, wall in enumerate(STABLE)]
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, treated))
    first, again = (
        evaluate([path], CONTRACT, HOST, seed=7),
        evaluate([path], CONTRACT, HOST, seed=7),
    )
    assert first["bootstrap"]["seed"] == 7
    assert only_attempt(first)["interval"] == only_attempt(again)["interval"]


def test_a_failed_attempt_is_kept_but_not_evaluated(tmp_path):
    document = attempt_document(
        STABLE[:5], scaled(1.01)[:5], protocol="resource-sampling-overhead-v1"
    )
    document |= {"status": "failed", "error": "TimeoutError: model server did not answer"}
    path = write(tmp_path / "a/sampling-overhead.json", document)
    attempt = only_attempt(evaluate([path], CONTRACT, HOST))
    assert attempt["status"] == "failed"
    assert attempt["verdict"] == "not_evaluated"
    assert "TimeoutError" in attempt["reason"]


def test_an_attempt_without_per_repetition_walls_is_not_evaluated(tmp_path):
    document = attempt_document(STABLE, scaled(1.01))
    del document["measurements"]
    path = write(tmp_path / "a/overhead.json", document)
    attempt = only_attempt(evaluate([path], CONTRACT, HOST))
    assert attempt["verdict"] == "not_evaluated"
    assert "per-repetition" in attempt["reason"].lower()
    assert attempt["relative_increase"] is None and attempt["interval"] is None


def test_an_unpaired_final_repetition_is_counted_but_not_compared(tmp_path):
    document = attempt_document(STABLE, scaled(1.01), protocol="resource-sampling-overhead-v1")
    document["measurements"].append({"index": 60, "condition": "unsampled", "wall_seconds": 9.0})
    path = write(tmp_path / "a/sampling-overhead.json", document)
    attempt = only_attempt(evaluate([path], CONTRACT, HOST))
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
        (["not_evaluated", "not_evaluated"], "not_evaluated"),
        ([], "not_evaluated"),
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
    (group,) = evaluate(paths, CONTRACT, HOST)["groups"]
    assert len(group["attempts"]) == 2
    assert group["verdict"] == "within_budget"
    assert group["settings"]["protocol"] == "agent-overhead-v1"


def test_attempts_with_different_settings_are_never_combined(tmp_path):
    first = write(tmp_path / "a/overhead.json", attempt_document(STABLE, scaled(1.01)))
    other = attempt_document(STABLE, scaled(1.01), sampling_interval_seconds=0.5)
    second = write(tmp_path / "b/overhead.json", other)
    groups = evaluate([first, second], CONTRACT, HOST)["groups"]
    assert len(groups) == 2
    assert [group["verdict"] for group in groups] == ["inconclusive", "inconclusive"]


def test_the_report_names_sources_without_host_paths(tmp_path):
    path = write(tmp_path / "attempt-1/overhead.json", attempt_document(STABLE, STABLE))
    report = evaluate([path], CONTRACT, HOST)
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
    code = main(
        [
            *map(str, paths),
            "--contract",
            str(CONTRACT_PATH),
            "--host-label",
            HOST,
            "--output",
            str(output),
        ]
    )
    assert code == 0
    report = json.loads((output / "overhead-verdict.json").read_text(encoding="utf-8"))
    assert report["groups"][0]["verdict"] == "within_budget"
    printed = capsys.readouterr().out
    assert "within_budget" in printed and "first/overhead.json" in printed


def test_main_exits_1_on_a_malformed_attempt(tmp_path, capsys):
    path = write(tmp_path / "overhead.json", {"protocol": {"version": "unknown-v9"}})
    code = main(
        [str(path), "--contract", str(CONTRACT_PATH), "--host-label", HOST]
        + ["--output", str(tmp_path / "r")]
    )
    assert code == 1
    assert "overhead.json" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


# --- Review regressions (#93) ----------------------------------------------------------------

CPU_METRICS = {"cpu.psi.some_avg10.v1": {"measured": 10}}
GPU_METRICS = CPU_METRICS | {"gpu.nvml.utilization.gpu.v1": {"measured": 10}}


def sampled(counts: dict, **changes) -> dict:
    document = attempt_document(
        STABLE, scaled(1.01), protocol="resource-sampling-overhead-v1", **changes
    )
    return document | {"sampling": {"availability_counts": counts}}


def test_attempts_with_different_probe_sets_are_never_combined(tmp_path):
    gpu = write(tmp_path / "gpu/sampling-overhead.json", sampled(GPU_METRICS))
    cpu = write(tmp_path / "cpu/sampling-overhead.json", sampled(CPU_METRICS))
    groups = evaluate([gpu, cpu], CONTRACT, HOST)["groups"]
    assert len(groups) == 2
    assert [len(g["settings"]["probe_metrics"]) for g in groups] == [2, 1]
    assert {g["verdict"] for g in groups} == {"inconclusive"}


def test_the_host_label_is_recorded_and_validated(tmp_path):
    path = write(tmp_path / "a/overhead.json", attempt_document(STABLE, STABLE))
    (group,) = evaluate([path], CONTRACT, "benoy-rtx5070-wsl2")["groups"]
    assert group["settings"]["host_label"] == "benoy-rtx5070-wsl2"
    for bad in ("", "/Users/alice", "has space", "x" * 65):
        with pytest.raises(AttemptError, match="host label"):
            evaluate([path], CONTRACT, bad)


def test_the_same_file_cannot_count_as_two_attempts(tmp_path):
    document = attempt_document(STABLE, scaled(1.01))
    path = write(tmp_path / "a/overhead.json", document)
    copy = write(tmp_path / "b/overhead.json", document)
    with pytest.raises(AttemptError, match="same file as attempt 1"):
        evaluate([path, path], CONTRACT, HOST)
    # Identical content under another name is the same evidence too.
    with pytest.raises(AttemptError, match="same file"):
        evaluate([path, copy], CONTRACT, HOST)


def test_an_attempt_with_fewer_than_the_declared_pairs_is_inconclusive(tmp_path):
    trimmed = attempt_document(STABLE[:3], scaled(1.2)[:3])
    attempt = only_attempt(evaluate([write(tmp_path / "a/overhead.json", trimmed)], CONTRACT, HOST))
    assert attempt["pairs"] == 3
    assert attempt["verdict"] == "inconclusive"
    assert "fewer than the protocol's 30" in attempt["reason"]


def test_a_group_with_nothing_evaluated_is_not_evaluated(tmp_path):
    documents = [attempt_document(STABLE, STABLE), attempt_document(STABLE, scaled(1.01))]
    paths = []
    for name, document in zip(("a", "b"), documents, strict=True):
        del document["measurements"]
        paths.append(write(tmp_path / f"{name}/overhead.json", document))
    (group,) = evaluate(paths, CONTRACT, HOST)["groups"]
    assert group["verdict"] == "not_evaluated"


@pytest.mark.parametrize("bad", [math.nan, math.inf, {"nested": 1}, [0.25], True])
def test_settings_must_be_strings_or_finite_numbers(tmp_path, bad):
    document = attempt_document(STABLE, STABLE, sampling_interval_seconds=bad)
    path = tmp_path / "a/overhead.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document, allow_nan=True), encoding="utf-8")
    with pytest.raises(AttemptError, match="sampling_interval_seconds"):
        read_attempt(path)


def test_main_exits_1_instead_of_crashing_on_a_non_finite_setting(tmp_path, capsys):
    document = attempt_document(STABLE, STABLE, sampling_interval_seconds=math.nan)
    path = tmp_path / "a/overhead.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document, allow_nan=True), encoding="utf-8")
    code = main([str(path), "--contract", str(CONTRACT_PATH), "--host-label", HOST])
    assert code == 1
    assert "sampling_interval_seconds" in capsys.readouterr().err


def test_fields_an_attempt_does_not_record_are_left_out(tmp_path):
    path = write(tmp_path / "a/sampling-overhead.json", sampled(CPU_METRICS))
    (group,) = evaluate([path], CONTRACT, HOST)["groups"]
    assert "runner_version" not in group["settings"]
    assert group["settings"]["model"] == "fixture-model"


def test_attempts_are_numbered_so_same_named_files_stay_distinct(tmp_path):
    first = write(tmp_path / "x/run/overhead.json", attempt_document(STABLE, scaled(1.01)))
    second = write(tmp_path / "y/run/overhead.json", attempt_document(STABLE, scaled(1.02)))
    (group,) = evaluate([first, second], CONTRACT, HOST)["groups"]
    assert [(a["attempt"], a["source"]) for a in group["attempts"]] == [
        (1, "run/overhead.json"),
        (2, "run/overhead.json"),
    ]


def test_an_over_budget_verdict_still_exits_0_with_the_report_written(tmp_path):
    paths = [
        write(tmp_path / f"{name}/overhead.json", attempt_document(STABLE, scaled(factor)))
        for name, factor in (("a", 1.2), ("b", 1.25))
    ]
    output = tmp_path / "report"
    args = [*map(str, paths), "--contract", str(CONTRACT_PATH), "--host-label", HOST]
    assert main([*args, "--output", str(output)]) == 0
    report = json.loads((output / "overhead-verdict.json").read_text(encoding="utf-8"))
    assert report["groups"][0]["verdict"] == "over_budget"


def test_admission_measurements_are_judged_against_the_admission_budget(tmp_path):
    # A copy of the contract whose admission budget alone is 20%: +10% with admission passes it,
    # while the same walls measured as sampling overhead exceed the 5% instrumentation budget.
    contract_path = tmp_path / "contract.toml"
    contract_path.write_text(
        CONTRACT_PATH.read_text().replace(
            "max_median_admission_overhead = 0.05", "max_median_admission_overhead = 0.2"
        )
    )
    contract = load_optimization_contract(contract_path)
    treated = [wall * 1.10 for wall in STABLE]
    for protocol, expected in (
        ("admission-overhead-v1", "within_budget"),
        ("resource-sampling-overhead-v1", "over_budget"),
    ):
        paths = [
            write(
                tmp_path / protocol / f"attempt-{n}.json",
                attempt_document(STABLE, treated, protocol),
            )
            for n in (1, 2)
        ]
        report = evaluate(paths, contract, HOST)
        assert report["budget"]["max_median_admission_overhead"] == 0.2
        (group,) = report["groups"]
        assert group["settings"]["protocol"] == protocol
        assert group["verdict"] == expected, group


def test_admission_conditions_must_alternate_off_then_on(tmp_path):
    document = attempt_document(STABLE, STABLE, "admission-overhead-v1")
    document["measurements"][0]["condition"] = "admission_on"
    with pytest.raises(AttemptError, match="alternate admission_off, admission_on"):
        read_attempt(write(tmp_path / "a" / "admission-overhead.json", document))


def admission_attempt(limits, enforced, poll_seconds=1.0, service="vllm") -> dict:
    return attempt_document(
        STABLE,
        STABLE,
        "admission-overhead-v1",
        policy={
            "version": 3,
            "limits": limits,
            "inference_service": service,
            "enforced": enforced,
            "not_enforced_reason": None if enforced else "no limit",
        },
        policy_poll_seconds=poll_seconds,
    )


def test_admission_attempts_under_different_admission_settings_are_never_combined(tmp_path):
    # One enforced limit polled every second, one unlimited baseline polled every 5 s, and one
    # matching the first: only the first and third measure the same admission path.
    documents = [
        admission_attempt({"inference_requests": 2}, True),
        admission_attempt({}, False, poll_seconds=5.0),
        admission_attempt({"inference_requests": 2}, True),
    ]
    paths = [write(tmp_path / f"a{n}" / "admission.json", doc) for n, doc in enumerate(documents)]
    groups = evaluate(paths, CONTRACT, HOST)["groups"]
    assert sorted(len(group["attempts"]) for group in groups) == [1, 2]
    enforced = next(group for group in groups if len(group["attempts"]) == 2)
    assert {
        key: enforced["settings"][key]
        for key in enforced["settings"]
        if "admission" in key or "poll" in key
    } == {
        "admission_limits": '{"inference_requests":2}',
        "admission_inference_service": "vllm",
        "admission_inference_enforced": "enforced",
        "policy_poll_seconds": 1.0,
    }
    # Each group says which budget it was judged against.
    assert {group["budget"] for group in groups} == {CONTRACT.max_median_admission_overhead}


def test_each_group_records_its_protocols_budget(tmp_path):
    paths = [
        write(
            tmp_path / "s" / "sampling.json",
            attempt_document(STABLE, STABLE, "resource-sampling-overhead-v1"),
        ),
        write(tmp_path / "a" / "admission.json", admission_attempt({}, False)),
    ]
    budgets = {
        group["settings"]["protocol"]: group["budget"]
        for group in evaluate(paths, CONTRACT, HOST)["groups"]
    }
    assert budgets == {
        "resource-sampling-overhead-v1": CONTRACT.max_median_instrumentation_overhead,
        "admission-overhead-v1": CONTRACT.max_median_admission_overhead,
    }


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("vllm", "policy must be an object"),
        (
            {"limits": {"inference_requests": "2"}, "inference_service": "vllm", "enforced": True},
            "integer limits",
        ),
        (
            {"limits": {}, "inference_service": "vllm", "enforced": None},
            "enforced must be true or false",
        ),
    ],
)
def test_malformed_admission_settings_are_rejected(tmp_path, policy, message):
    document = admission_attempt({}, False) | {"policy": policy}
    with pytest.raises(AttemptError, match=message):
        read_attempt(write(tmp_path / "a" / "admission.json", document))
