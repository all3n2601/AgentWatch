import re

import pytest
from agentwatch_data.labels import (
    LabelingError,
    excess_fraction,
    label_rows,
    label_run,
    load_config,
)
from agentwatch_data.schema import COLUMNS

# Durations from a real `make demo-coding` run on macOS (2026-10-05).
BASELINE_SECONDS = [0.368, 0.374, 0.368, 0.369, 0.368]


def run_rows(run_id="run-1", kind="coding", clean=0.362, contended=2.868):
    rows = [
        {"run_id": run_id, "kind": kind, "condition": "baseline", "duration_seconds": seconds}
        for seconds in BASELINE_SECONDS
    ]
    rows.append({"run_id": run_id, "kind": kind, "condition": "clean", "duration_seconds": clean})
    rows.append(
        {"run_id": run_id, "kind": kind, "condition": "contended", "duration_seconds": contended}
    )
    return rows


@pytest.fixture
def config():
    return load_config()


def test_packaged_rules_are_frozen_and_versioned(config):
    assert (config.stdev_multiplier, config.min_seconds) == (3.0, 0.01)
    assert config.contention == {"cpu": "cpu_sandbox", "coding": "cpu_sandbox"}
    assert config.label_version == f"labels-v1+{config.sha256[:8]}"


def test_real_run_is_labeled_from_experiment_design(config):
    labeled = label_run(run_rows(), config)
    assert [row["label_resource"] for row in labeled] == ["none"] * 6 + ["cpu_sandbox"]
    assert [row["label_basis"] for row in labeled] == ["baseline_reference"] * 5 + [
        "clean_within_noise",
        "contention_exceeded_threshold",
    ]
    contended = labeled[-1]
    assert contended["baseline_mean_seconds"] == pytest.approx(0.3694)
    assert contended["noise_threshold_seconds"] == pytest.approx(0.01)
    assert contended["label_excess_fraction"] == pytest.approx((2.868 - 0.3694) / 2.868)
    assert contended["label_version"] == config.label_version


def test_contention_within_noise_is_labeled_none(config):
    contended = label_run(run_rows(contended=0.375), config)[-1]
    assert contended["label_resource"] == "none"
    assert contended["label_basis"] == "contention_below_threshold"


def test_slow_clean_control_is_flagged_but_stays_none(config):
    clean = label_run(run_rows(clean=0.9), config)[5]
    assert clean["label_resource"] == "none"
    assert clean["label_basis"] == "clean_exceeded_noise"


def test_noisy_baseline_widens_the_threshold(config):
    rows = run_rows(contended=0.6)
    for row, seconds in zip(rows, [0.2, 0.5, 0.3, 0.45, 0.25]):
        row["duration_seconds"] = seconds
    contended = label_run(rows, config)[-1]
    assert contended["noise_threshold_seconds"] > 0.3
    assert contended["label_resource"] == "none"


def test_excess_fraction_is_clamped():
    assert excess_fraction(0.3, 0.4) == 0.0
    assert excess_fraction(2.0, 0.5) == 0.75
    with pytest.raises(LabelingError):
        excess_fraction(0.0, 0.4)


def test_rows_from_several_runs_keep_order_and_own_baselines(config):
    first, second = run_rows("a"), run_rows("b", kind="cpu", contended=0.37)
    mixed = [row for pair in zip(first, second) for row in pair]
    labeled = label_rows(mixed, config)
    assert [row["run_id"] for row in labeled] == [row["run_id"] for row in mixed]
    by_run = {row["run_id"]: row for row in labeled if row["condition"] == "contended"}
    assert by_run["a"]["label_resource"] == "cpu_sandbox"
    assert by_run["b"]["label_resource"] == "none"


def test_input_rows_are_not_modified(config):
    rows = run_rows()
    label_run(rows, config)
    assert "label_resource" not in rows[-1]


def test_labeled_columns_belong_to_the_schema(config):
    added = set(label_run(run_rows(), config)[0]) - set(run_rows()[0])
    assert added <= {column.name for column in COLUMNS}


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (run_rows()[4:], "at least two baseline steps"),
        (run_rows() + [run_rows()[0] | {"condition": "mixed"}], "conditions ['mixed']"),
        (run_rows(kind="llm"), "No contended resource"),
        (run_rows("a") + run_rows("b"), "exactly one run"),
    ],
)
def test_unlabelable_evidence_fails_loudly(config, rows, message):
    with pytest.raises(LabelingError, match=re.escape(message)):
        label_run(rows, config)


def test_changed_rules_change_the_label_version(tmp_path, config):
    path = tmp_path / "labels.toml"
    path.write_text(
        'version = "labels-v1"\n[noise]\nstdev_multiplier = 2.0\nmin_seconds = 0.01\n'
        '[contention]\ncpu = "cpu_sandbox"\n'
    )
    retuned = load_config(path)
    assert retuned.version == config.version
    assert retuned.label_version != config.label_version


@pytest.mark.parametrize(
    ("noise", "contention"),
    [
        ("stdev_multiplier = 0\nmin_seconds = 0.01", 'cpu = "cpu_sandbox"'),
        ("stdev_multiplier = 3\nmin_seconds = -1", 'cpu = "cpu_sandbox"'),
        ("stdev_multiplier = 3\nmin_seconds = 0.01", 'cpu = "gpu"'),
        ("stdev_multiplier = 3\nmin_seconds = 0.01", 'cpu = "none"'),
    ],
)
def test_invalid_rules_are_rejected(tmp_path, noise, contention):
    path = tmp_path / "labels.toml"
    path.write_text(f'version = "x"\n[noise]\n{noise}\n[contention]\n{contention}\n')
    with pytest.raises(LabelingError):
        load_config(path)
