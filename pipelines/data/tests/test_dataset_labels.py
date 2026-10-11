import re

import pytest
from agentwatch_data import labels
from agentwatch_data.labels import (
    Contention,
    LabelingError,
    excess_fraction,
    label_rows,
    label_run,
    load_config,
)
from agentwatch_data.schema import COLUMNS

# Durations from a real `make demo-coding` run on macOS (2026-10-05).
BASELINE_SECONDS = [0.368, 0.374, 0.368, 0.369, 0.368]

VALID_RULES = """
version = "labels-test"
[noise]
stdev_multiplier = {multiplier}
min_seconds = {floor}
[contention.cpu]
resource = "{resource}"
injection_method = "{method}"
"""


def run_rows(run_id="run-1", kind="coding", clean=0.362, contended=2.868, failed=None):
    def row(step_id, condition, seconds):
        return {
            "run_id": run_id,
            "kind": kind,
            "step_id": step_id,
            "condition": condition,
            "duration_seconds": seconds,
            "failed_checks": failed,
        }

    rows = [row(f"baseline-{i}", "baseline", s) for i, s in enumerate(BASELINE_SECONDS)]
    return rows + [row("clean", "clean", clean), row("contended", "contended", contended)]


def write_rules(tmp_path, text=None, **values):
    values = {"multiplier": "3.0", "floor": "0.01", "resource": "cpu_sandbox"} | values
    values.setdefault("method", "worker_queue_blocker")
    path = tmp_path / "labels.toml"
    path.write_text(text if text is not None else VALID_RULES.format(**values))
    return path


@pytest.fixture
def config():
    return load_config()


def test_packaged_rules_are_frozen_and_versioned(config):
    assert (config.stdev_multiplier, config.min_seconds) == (3.0, 0.01)
    blocker = Contention("cpu_sandbox", "worker_queue_blocker")
    assert config.contention == {"cpu": blocker, "coding": blocker}
    assert config.label_version == f"labels-v1.1+{config.fingerprint[:8]}"


def test_real_run_is_labeled_from_experiment_design(config):
    labeled = label_run(run_rows(), config)
    assert [row["label_resource"] for row in labeled] == ["none"] * 6 + ["cpu_sandbox"]
    assert [row["label_basis"] for row in labeled] == ["baseline_reference"] * 5 + [
        "clean_within_noise",
        "contention_exceeded_threshold",
    ]
    assert [row["injection_method"] for row in labeled] == ["none"] * 6 + ["worker_queue_blocker"]
    contended = labeled[-1]
    assert contended["baseline_mean_seconds"] == pytest.approx(0.3694)
    assert contended["noise_threshold_seconds"] == pytest.approx(0.01)
    assert contended["label_excess_fraction"] == pytest.approx((2.868 - 0.3694) / 2.868)
    assert contended["label_version"] == config.label_version


def test_contention_within_noise_is_labeled_none(config):
    contended = label_run(run_rows(contended=0.375), config)[-1]
    assert contended["label_resource"] == "none"
    assert contended["label_basis"] == "contention_below_threshold"
    assert contended["injection_method"] == "worker_queue_blocker"


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


def test_identical_baselines_use_the_floor(config):
    rows = run_rows()
    for row in rows[:5]:
        row["duration_seconds"] = 0.4
    assert label_run(rows, config)[-1]["noise_threshold_seconds"] == 0.01


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


# Evidence that cannot be labeled


def duplicated_step(rows):
    return rows + [rows[0] | {"duration_seconds": 0.369}]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (run_rows()[4:], "at least two baseline steps"),
        (run_rows() + [run_rows()[0] | {"condition": "mixed", "step_id": "m"}], "['mixed']"),
        (run_rows(kind="llm"), "No contended resource"),
        (run_rows("a") + run_rows("b"), "exactly one run"),
        (duplicated_step(run_rows()), "repeats step IDs"),
        (run_rows()[:-1] + [run_rows(kind="cpu")[-1]], "mixes experiment kinds"),
        (run_rows(failed="same_workload_output"), "not attributable"),
        (run_rows(failed="queue_increased,same_workload_output"), "not attributable"),
    ],
)
def test_unlabelable_evidence_fails_loudly(config, rows, message):
    with pytest.raises(LabelingError, match=re.escape(message)):
        label_run(rows, config)


def test_duplicated_baseline_would_have_shifted_the_threshold():
    """Why duplicate steps are refused: a repeated baseline narrows the stdev."""
    import statistics

    assert statistics.stdev(BASELINE_SECONDS + [0.368]) < statistics.stdev(BASELINE_SECONDS)


@pytest.mark.parametrize("failed", ["contention_detected", "stable_baseline,queue_increased"])
def test_outcome_checks_are_recorded_but_do_not_block_labels(config, failed):
    """Dropping runs when the rule disagreed would select data by the rule's answer."""
    assert label_run(run_rows(failed=failed), config)[-1]["label_resource"] == "cpu_sandbox"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("index", [0, 5, 6], ids=["baseline", "clean", "contended"])
def test_nonfinite_durations_fail_instead_of_labeling(config, index, value):
    rows = run_rows()
    rows[index] = rows[index] | {"duration_seconds": value}
    with pytest.raises(LabelingError, match="finite positive"):
        label_run(rows, config)


@pytest.mark.parametrize("value", ["0.4", None, True])
def test_nonnumeric_durations_fail(config, value):
    rows = run_rows()
    rows[-1] = rows[-1] | {"duration_seconds": value}
    with pytest.raises(LabelingError, match="finite positive"):
        label_run(rows, config)


@pytest.mark.parametrize(
    ("duration", "baseline_mean"),
    [
        (float("nan"), 0.4),
        (float("inf"), 0.4),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (1.0, 0.0),
        (True, 0.4),
    ],
)
def test_excess_fraction_rejects_invalid_inputs(duration, baseline_mean):
    with pytest.raises(LabelingError):
        excess_fraction(duration, baseline_mean)


# Rule file validation


@pytest.mark.parametrize(
    "values",
    [
        {"multiplier": "0"},
        {"floor": "-1"},
        {"multiplier": "true"},
        {"floor": '"0.01"'},
        {"resource": "gpu"},
        {"resource": "none"},
        {"method": "none"},
        {"method": " "},
    ],
)
def test_invalid_rules_are_rejected(tmp_path, values):
    with pytest.raises(LabelingError):
        load_config(write_rules(tmp_path, **values))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
@pytest.mark.parametrize(("setting", "key"), [("multiplier", "stdev"), ("floor", "min_seconds")])
def test_nonfinite_noise_settings_are_rejected(tmp_path, setting, key, value):
    with pytest.raises(LabelingError, match=key):
        load_config(write_rules(tmp_path, **{setting: value}))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "[noise]\nstdev_multiplier = 3\nmin_seconds = 0.01\n",
            "missing ['contention', 'version']",
        ),
        (VALID_RULES.replace("stdev_multiplier", "stdev_multipler"), "unknown ['stdev_multipler']"),
        (VALID_RULES + "\nthreshold = 2\n", "unknown"),
        ('version = "x"\nnoise = 3\ncontention = {}\n', "noise must be a table"),
        (
            'version = "x"\n[noise]\nstdev_multiplier = 3\nmin_seconds = 0.01\n[contention]\n',
            "at least one",
        ),
        ("version = = 1", "not valid TOML"),
    ],
)
def test_malformed_rule_files_give_clear_errors(tmp_path, text, message):
    if "{multiplier}" in text:
        text = text.format(multiplier=3, floor=0.01, resource="cpu_sandbox", method="m")
    with pytest.raises(LabelingError, match=re.escape(message)):
        load_config(write_rules(tmp_path, text=text))


# Label version fingerprint


def test_changed_rule_values_change_the_label_version(tmp_path):
    base = load_config(write_rules(tmp_path))
    retuned = load_config(write_rules(tmp_path, multiplier="2.0"))
    assert retuned.version == base.version
    assert retuned.label_version != base.label_version


def test_comments_formatting_and_line_endings_do_not_change_the_version(tmp_path):
    path = write_rules(tmp_path)
    base = load_config(path).label_version
    text = path.read_text()
    variants = [
        "# a new comment\n" + text,
        text.replace("\n", "\r\n"),
        text.replace("3.0", "3").replace(" = ", "   =   "),
    ]
    for variant in variants:
        path.write_bytes(variant.encode())
        assert load_config(path).label_version == base


def test_labeling_code_changes_the_version(tmp_path, monkeypatch):
    path = write_rules(tmp_path)
    base = load_config(path).label_version
    monkeypatch.setattr(labels, "logic_source", lambda: "def label_run(): ...  # changed rule")
    assert load_config(path).label_version != base


def test_packaged_version_matches_on_crlf_checkouts(tmp_path, monkeypatch):
    original = labels.logic_source()
    monkeypatch.setattr(
        labels.Path, "read_text", lambda self, encoding=None: original.replace("\n", "\r\n")
    )
    assert labels.logic_source() == original


def test_invalidating_check_on_any_row_blocks_the_run(config):
    rows = run_rows()
    rows[-1] = rows[-1] | {"failed_checks": "same_workload_output"}
    with pytest.raises(LabelingError, match="not attributable"):
        label_run(rows, config)
