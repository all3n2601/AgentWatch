"""CI verdict for a model pipeline run: what to push to GCP Artifact Registry and what to tell Slack.

The workflow (.github/workflows/model.yaml) runs the model pipeline image, which registers the
evaluated model in MLflow and writes registration.json. This module reads that record and
decides, without any network access:

- `stage` writes verdict.json and a folder holding the model files for upload. Every evaluated
  model goes to the candidate package; only an eligible one (registry.toml: no failed gate, no
  failed bias check on test, not blocking promotion) also goes to the champion package.
- `message` turns the verdict and the push outcome into the Slack payload, so a failed gate or
  bias check is reported rather than a quiet channel standing in for a clean run.

The package version is the model's identity hash, so a rerun of the same model names the same
version and the workflow skips an upload that already exists. MLflow's database does not outlive
a CI run, so the registry's champion-versus-previous comparison is not applied across runs; the
champion package holds every eligible model and a rollback is choosing an earlier version.
"""

import argparse
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from agentwatch_data.files import json_text, write_atomic

from agentwatch_model.register_model import CALIBRATION_FILES, CANDIDATE_FILES, RECORD
from agentwatch_model.rule_baseline import EXIT_ERROR

CANDIDATE_PACKAGE = "attribution-candidate"
CHAMPION_PACKAGE = "attribution-champion"
IDENTITY_DIGITS = 12
VERDICT = "verdict.json"
SLACK_PAYLOAD = "slack.json"
UPLOAD_FOLDER = "upload"
PUSH_STATUSES = ("pushed", "skipped", "failed")
# Slack reads <...> as links and mentions and & as an entity.
SLACK_ESCAPES = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})


class ReportError(ValueError):
    """The registration record or a model file is missing or does not have the expected shape."""


def read_object(path: Path, what: str) -> dict:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"{what} {path} could not be read: {error}") from error
    if not isinstance(document, dict):
        raise ReportError(f"{what} {path} is not a JSON object")
    return document


def require(document: Mapping, key: str, kind: type, where: str):
    value = document.get(key)
    if type(value) is not kind:
        raise ReportError(f"{where}: {key!r} must be {kind.__name__}, got {value!r}")
    return value


def verdict(registration: Mapping) -> dict:
    """The push decision for one registered model, from registration.json's contents."""
    model = registration.get("model")
    if not isinstance(model, dict):
        raise ReportError(f"{RECORD}: 'model' must be an object")
    identity = require(model, "identity_sha256", str, f"{RECORD} model")
    if len(identity) < IDENTITY_DIGITS or any(c not in "0123456789abcdef" for c in identity):
        raise ReportError(f"{RECORD} model: 'identity_sha256' is not a lowercase hex digest")
    eligible = require(model, "eligible", bool, f"{RECORD} model")
    reasons = model.get("ineligible_because")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise ReportError(f"{RECORD} model: 'ineligible_because' must be a list of strings")
    if eligible == bool(reasons):
        raise ReportError(f"{RECORD} model: eligible={eligible} contradicts its reasons {reasons}")
    macro_f1 = model.get("macro_f1")
    if type(macro_f1) is not float:
        raise ReportError(f"{RECORD} model: 'macro_f1' must be a float, got {macro_f1!r}")
    failed_gates = model.get("failed_gates")
    if not isinstance(failed_gates, list) or not all(isinstance(g, str) for g in failed_gates):
        raise ReportError(f"{RECORD} model: 'failed_gates' must be a list of strings")
    return {
        "model_name": require(registration, "model_name", str, RECORD),
        "package_version": f"m-{identity[:IDENTITY_DIGITS]}",
        "identity_sha256": identity,
        "training_version": require(model, "training_version", str, f"{RECORD} model"),
        "calibration_version": require(model, "calibration_version", str, f"{RECORD} model"),
        "macro_f1": macro_f1,
        "failed_gates": failed_gates,
        "bias_test_outcome": model.get("bias_test_outcome"),
        "blocks_promotion": require(model, "blocks_promotion", bool, f"{RECORD} model"),
        "eligible": eligible,
        "ineligible_because": reasons,
        "packages": [CANDIDATE_PACKAGE, *([CHAMPION_PACKAGE] if eligible else [])],
    }


def stage_upload(root: Path, registration_path: Path, destination: Path) -> None:
    """Copy the model files, calibration, reports and the registration record into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registration_path, destination / RECORD)
    for folder, names in (("candidates", CANDIDATE_FILES), ("calibration", CALIBRATION_FILES)):
        (destination / folder).mkdir(exist_ok=True)
        for name in names:
            source = root / folder / name
            if source.is_file():
                shutil.copy2(source, destination / folder / name)
    # A model without its classifier is not a registrable model, whatever the record says.
    if not (destination / "candidates" / "classifier.txt").is_file():
        raise ReportError(f"{root / 'candidates' / 'classifier.txt'} is missing; nothing to upload")


def slack_text(value: object) -> str:
    return str(value).translate(SLACK_ESCAPES)


def message(decision: Mapping, push_status: str, run_url: str) -> str:
    """The Slack text: outcome first, then the gates, the bias result and where the model went."""
    if push_status not in PUSH_STATUSES:
        raise ReportError(f"push status must be one of {PUSH_STATUSES}, got {push_status!r}")
    if push_status == "failed":
        heading = ":x: AgentWatch model pipeline finished, but the registry push failed"
    elif decision["eligible"]:
        heading = ":white_check_mark: AgentWatch model pipeline finished: model is eligible"
    else:
        heading = ":warning: AgentWatch model pipeline finished: model is not eligible"
    where = {
        "pushed": f"pushed to {', '.join(decision['packages'])} as {decision['package_version']}",
        "skipped": "not pushed (GCP registry not configured, or this run is not on main)",
        "failed": f"push to {', '.join(decision['packages'])} did not complete",
    }[push_status]
    lines = [
        heading,
        (
            f"Model: {decision['training_version']} with {decision['calibration_version']}, "
            f"test macro F1 {decision['macro_f1']:.4f}"
        ),
        f"Failed gates: {', '.join(decision['failed_gates']) or 'none'}",
        f"Bias check on test: {decision['bias_test_outcome'] or 'unavailable'}",
        f"Registry: {where}",
        *(f"• Not eligible: {reason}" for reason in decision["ineligible_because"]),
        f"Run: {run_url}",
    ]
    return "\n".join(slack_text(line) for line in lines)


def stage(root: Path, output: Path) -> dict:
    registration_path = root / "registry" / RECORD
    decision = verdict(read_object(registration_path, "registration record"))
    stage_upload(root, registration_path, output / UPLOAD_FOLDER)
    write_atomic(output / VERDICT, json_text(decision))
    return decision


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide the registry push and Slack message.")
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage", help="write the verdict and the upload folder")
    staging.add_argument("--root", type=Path, required=True, help="model pipeline outputs folder")
    staging.add_argument("--output", type=Path, required=True)
    notify = commands.add_parser("message", help="write the Slack payload from the verdict")
    notify.add_argument("--output", type=Path, required=True, help="folder holding verdict.json")
    notify.add_argument("--push-status", choices=PUSH_STATUSES, required=True)
    notify.add_argument("--run-url", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "stage":
            decision = stage(args.root, args.output)
            # key=value lines, appended to $GITHUB_OUTPUT by the workflow
            print(f"package_version={decision['package_version']}")
            print(f"eligible={str(decision['eligible']).lower()}")
            print(f"packages={' '.join(decision['packages'])}")
        else:
            decision = read_object(args.output / VERDICT, "verdict")
            text = message(verdict_from_file(decision), args.push_status, args.run_url)
            write_atomic(args.output / SLACK_PAYLOAD, json_text({"text": text}))
    except ReportError as error:
        print(f"CI report error: {error}", file=sys.stderr)
        return EXIT_ERROR
    return 0


def verdict_from_file(decision: dict) -> dict:
    """A verdict.json read back must still have the shape message() relies on."""
    for key, kind in (
        ("packages", list),
        ("package_version", str),
        ("eligible", bool),
        ("training_version", str),
        ("calibration_version", str),
        ("macro_f1", float),
        ("failed_gates", list),
        ("ineligible_because", list),
    ):
        require(decision, key, kind, VERDICT)
    return decision


if __name__ == "__main__":
    sys.exit(main())
