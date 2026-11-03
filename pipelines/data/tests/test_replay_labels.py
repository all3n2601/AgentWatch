import pytest
from agentwatch_data.labels import LabelingError
from agentwatch_data.replay_labels import load_replay_config

VALID_RULES = """
version = "labels-v2"
[noise]
stdev_multiplier = {multiplier}
min_seconds = 0.01
[condition_resources]
cpu = "cpu_sandbox"
inference = "inference"
retrieval = "retrieval"
"""


def write_rules(tmp_path, text=None, multiplier="3.0"):
    path = tmp_path / "labels-v2.toml"
    path.write_text(text if text is not None else VALID_RULES.format(multiplier=multiplier))
    return path


@pytest.fixture
def config():
    return load_replay_config()


def test_packaged_replay_rules_load(config):
    assert (config.version, config.stdev_multiplier, config.min_seconds) == ("labels-v2", 3.0, 0.01)
    assert config.condition_resources == {
        "cpu": "cpu_sandbox",
        "inference": "inference",
        "retrieval": "retrieval",
    }
    assert config.label_version.startswith("labels-v2+")


# The full label_version of the packaged replay rules and labeling code, pinned so no
# change to either reaches a release unnoticed. When this fails: if the change can alter
# the label of any row the pipeline produces, or whether such a run is labeled at all,
# bump `version` in labels-v2.toml, which also needs an evaluation-contract change.
# Otherwise (a comment or refactor) keep the version, and say why in the PR. Either way,
# update the pin in the same PR.
PINNED_REPLAY_LABEL_VERSION = "labels-v2+cd725a68"


def test_replay_label_version_is_pinned(config):
    assert config.label_version == PINNED_REPLAY_LABEL_VERSION, (
        "replay_labels.py or labels-v2.toml changed; see the comment above "
        "PINNED_REPLAY_LABEL_VERSION"
    )


def test_formatting_does_not_change_fingerprint(config, tmp_path):
    text = "# extra comment\n" + VALID_RULES.format(multiplier="3")
    for variant in (text, text.replace("\n", "\r\n")):
        assert load_replay_config(write_rules(tmp_path, variant)).fingerprint == (
            load_replay_config(write_rules(tmp_path)).fingerprint
        )


def test_changed_values_change_fingerprint(tmp_path):
    base = load_replay_config(write_rules(tmp_path)).fingerprint
    assert load_replay_config(write_rules(tmp_path, multiplier="2.5")).fingerprint != base


@pytest.mark.parametrize(
    "text",
    [
        VALID_RULES.format(multiplier="3.0") + "extra = 1\n",
        'version = "labels-v2"\n[noise]\nstdev_multiplier = 3.0\nmin_seconds = 0.01\n',
        VALID_RULES.format(multiplier="nan"),
        VALID_RULES.format(multiplier="3.0") + 'gpu = "cpu_sandbox"\n',
        VALID_RULES.format(multiplier="3.0").replace('"retrieval"\n', '"none"\n', 1),
        VALID_RULES.format(multiplier="3.0").replace('cpu = "cpu_sandbox"', 'cpu = "gpu"'),
        "not toml [",
    ],
)
def test_invalid_rules_are_rejected(tmp_path, text):
    with pytest.raises(LabelingError):
        load_replay_config(write_rules(tmp_path, text))


def test_missing_rules_file_is_rejected(tmp_path):
    with pytest.raises(LabelingError):
        load_replay_config(tmp_path / "absent.toml")
