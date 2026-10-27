import pytest
from agentwatch_model.rules import DEFAULT_RULES, RulesError, load_rules
from rule_fixtures import row


@pytest.fixture(scope="module")
def rules():
    return load_rules(DEFAULT_RULES)


@pytest.mark.parametrize(
    ("queue_fraction", "expected"),
    [(0.6, "cpu_sandbox"), (0.5, "cpu_sandbox"), (0.49, "none"), (0.0, "none")],
)
def test_queue_fraction_at_or_above_the_threshold_names_cpu(rules, queue_fraction, expected):
    assert rules.predict(row(queue_fraction=queue_fraction)) == expected


def test_cpu_pressure_counts_only_when_it_was_readable(rules):
    assert rules.predict(row(psi=25.0)) == "cpu_sandbox"
    assert rules.predict(row(psi=19.9)) == "none"
    unreadable = row(psi=25.0) | {"cpu_pressure_available": False}
    assert rules.predict(unreadable) == "none"


def test_throttling_is_measured_as_a_share_of_the_step_duration(rules):
    assert rules.predict(row(throttled_usec=200_000, duration_seconds=1.0)) == "cpu_sandbox"
    assert rules.predict(row(throttled_usec=50_000, duration_seconds=1.0)) == "none"
    assert rules.predict(row(throttled_usec=200_000, duration_seconds=4.0)) == "none"
    unmeasured = row(throttled_usec=200_000) | {"cgroup_available": False}
    assert rules.predict(unmeasured) == "none"


def test_missing_readings_never_fire_a_rule(rules):
    missing = row() | {"queue_fraction": None, "duration_seconds": None}
    assert rules.predict(missing) == "none"


def test_rules_cover_only_classes_with_a_measured_signal(rules):
    assert rules.covered_classes == ("none", "cpu_sandbox")
    predictions = {rules.predict(row(queue_fraction=q, psi=p)) for q in (0, 1) for p in (0, 99)}
    assert predictions <= {"none", "cpu_sandbox"}


def test_version_includes_a_hash_of_the_parsed_rules(rules, tmp_path):
    assert rules.version.startswith("rules-v1+") and len(rules.version) == len("rules-v1+") + 8
    edited = tmp_path / "rules.toml"
    edited.write_text(DEFAULT_RULES.read_text().replace("0.5", "0.6", 1))
    assert load_rules(edited).version != rules.version
    commented = tmp_path / "commented.toml"
    commented.write_text("# a comment only\r\n" + DEFAULT_RULES.read_text())
    assert load_rules(commented).version == rules.version


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("min_queue_fraction = 0.5", "min_queue_fraction = nan", "finite"),
        ("min_queue_fraction = 0.5", "min_queue_fraction = 1.5", "between 0 and 1"),
        ("min_cpu_pressure_avg10 = 20.0", "min_cpu_pressure_avg10 = -1.0", "between 0 and 100"),
        ("min_queue_fraction = 0.5", "min_queue_fraction = true", "number"),
        ('version = "rules-v1"', 'version = "rules"', "version"),
    ],
)
def test_invalid_rules_are_rejected(tmp_path, old, new, message):
    bad = tmp_path / "rules.toml"
    bad.write_text(DEFAULT_RULES.read_text().replace(old, new))
    with pytest.raises(RulesError, match=message):
        load_rules(bad)
