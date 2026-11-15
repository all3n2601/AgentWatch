"""The CI verdict and Slack message, from synthetic registration records (no MLflow, no network)."""

import json

import pytest
from agentwatch_model.ci_report import (
    CANDIDATE_PACKAGE,
    CHAMPION_PACKAGE,
    ReportError,
    main,
    message,
    stage_upload,
    verdict,
)

IDENTITY = "ab12cd34ef56" + "0" * 52
RUN_URL = "https://github.com/example/repo/actions/runs/1"


def registration(**model):
    """A synthetic registration.json; eligible unless the overrides say otherwise."""
    base = {
        "identity_sha256": IDENTITY,
        "training_version": "training-v3+71c4dd58",
        "calibration_version": "calibration-v1+08119e78",
        "macro_f1": 0.79,
        "failed_gates": [],
        "bias_test_outcome": "pass",
        "blocks_promotion": False,
        "eligible": True,
        "ineligible_because": [],
    }
    return {"model_name": "agentwatch-attribution", "model": base | model}


def blocked(reasons=("it blocks promotion (contract gates not evaluated)",), **model):
    return registration(eligible=False, ineligible_because=list(reasons), **model)


def test_eligible_model_goes_to_both_packages():
    decision = verdict(registration())

    assert decision["packages"] == [CANDIDATE_PACKAGE, CHAMPION_PACKAGE]
    assert decision["package_version"] == "m-ab12cd34ef56"


def test_ineligible_model_goes_only_to_the_candidate_package():
    decision = verdict(blocked(blocks_promotion=True, bias_test_outcome="fail"))

    assert decision["packages"] == [CANDIDATE_PACKAGE]
    assert decision["eligible"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"identity_sha256": "short"},
        {"identity_sha256": "AB12CD34EF56" + "0" * 52},
        {"eligible": "yes"},
        {"eligible": True, "ineligible_because": ["x"]},
        {"eligible": False, "ineligible_because": []},
        {"macro_f1": 1},
        {"failed_gates": "none"},
        {"blocks_promotion": None},
    ],
)
def test_malformed_model_record_is_refused(override):
    with pytest.raises(ReportError):
        verdict(registration(**override))


def test_record_without_a_model_is_refused():
    with pytest.raises(ReportError, match="'model' must be an object"):
        verdict({"model_name": "agentwatch-attribution"})


def lay_out_outputs(root, *, classifier=True):
    for folder, names in (
        ("candidates", ("classifier.txt", "regressor.txt", "report.json")),
        ("calibration", ("calibration.json", "report.json")),
    ):
        (root / folder).mkdir(parents=True)
        for name in names:
            if name == "classifier.txt" and not classifier:
                continue
            (root / folder / name).write_text(f"{folder}/{name}")
    (root / "registry").mkdir()
    record = root / "registry" / "registration.json"
    record.write_text(json.dumps(registration()))
    return record


def test_stage_copies_model_files_by_folder(tmp_path):
    record = lay_out_outputs(tmp_path / "outputs")

    stage_upload(tmp_path / "outputs", record, tmp_path / "upload")

    assert (tmp_path / "upload" / "candidates" / "classifier.txt").read_text() == (
        "candidates/classifier.txt"
    )
    assert (tmp_path / "upload" / "calibration" / "report.json").read_text() == (
        "calibration/report.json"
    )
    assert not (tmp_path / "upload" / "candidates" / "preprocessing.json").exists()
    assert (tmp_path / "upload" / "registration.json").is_file()


def test_stage_refuses_a_model_without_its_classifier(tmp_path):
    record = lay_out_outputs(tmp_path / "outputs", classifier=False)

    with pytest.raises(ReportError, match="classifier.txt is missing"):
        stage_upload(tmp_path / "outputs", record, tmp_path / "upload")


def test_message_reports_an_eligible_model_that_was_pushed():
    text = message(verdict(registration()), "pushed", RUN_URL)

    assert text.startswith(":white_check_mark:")
    assert "attribution-candidate, attribution-champion as m-ab12cd34ef56" in text
    assert "Failed gates: none" in text
    assert RUN_URL in text


def test_message_lists_why_a_model_is_not_eligible():
    decision = verdict(
        blocked(
            ("failed gate(s): macro_f1", "the bias check fails on test"),
            failed_gates=["macro_f1"],
            bias_test_outcome="fail",
        )
    )

    text = message(decision, "pushed", RUN_URL)

    assert text.startswith(":warning:")
    assert "Failed gates: macro_f1" in text
    assert "Bias check on test: fail" in text
    assert "• Not eligible: the bias check fails on test" in text
    assert "attribution-champion" not in text


def test_message_says_when_the_push_failed_or_was_skipped():
    decision = verdict(registration())

    assert message(decision, "failed", RUN_URL).startswith(":x:")
    assert "did not complete" in message(decision, "failed", RUN_URL)
    assert "not pushed" in message(decision, "skipped", RUN_URL)


def test_message_reports_missing_bias_result_as_unavailable():
    text = message(verdict(registration(bias_test_outcome=None)), "skipped", RUN_URL)

    assert "Bias check on test: unavailable" in text


def test_message_escapes_slack_markup():
    text = message(verdict(blocked(("<!channel> & more",))), "skipped", RUN_URL)

    assert "&lt;!channel&gt; &amp; more" in text
    assert "<!channel>" not in text


def test_message_rejects_an_unknown_push_status():
    with pytest.raises(ReportError, match="push status"):
        message(verdict(registration()), "maybe", RUN_URL)


def test_cli_stages_then_writes_the_slack_payload(tmp_path, capsys):
    lay_out_outputs(tmp_path / "outputs")
    output = tmp_path / "ci"

    assert main(["stage", "--root", str(tmp_path / "outputs"), "--output", str(output)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "package_version=m-ab12cd34ef56",
        "eligible=true",
        "packages=attribution-candidate attribution-champion",
    ]

    arguments = ["message", "--output", str(output), "--push-status", "pushed"]
    assert main([*arguments, "--run-url", RUN_URL]) == 0
    payload = json.loads((output / "slack.json").read_text())
    assert payload["text"].startswith(":white_check_mark:")


def test_cli_stage_fails_without_a_registration_record(tmp_path, capsys):
    assert main(["stage", "--root", str(tmp_path), "--output", str(tmp_path / "ci")]) != 0
    assert "registration record" in capsys.readouterr().err


def test_cli_message_refuses_a_verdict_that_lost_its_fields(tmp_path, capsys):
    (tmp_path / "verdict.json").write_text(json.dumps({"eligible": True}))

    code = main(
        ["message", "--output", str(tmp_path), "--push-status", "pushed", "--run-url", RUN_URL]
    )

    assert code != 0
    assert "verdict.json" in capsys.readouterr().err
