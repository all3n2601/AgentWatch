"""Push an evaluated model to the model registry, keeping the champion unless the new one qualifies.

Until the team's GCP registry and its credentials exist, MLflow's model registry on the local
tracking database stands in for it (registry.toml, registry-v1). A push logs the model files, the
serving contract, the calibration and the reports as one MLflow run, registers them as a new
version tagged with its identity and gate outcomes, points `candidate` at it, and moves `champion`
only under the declared rule. `--rollback` points `champion` back at the version it replaced.
"""

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.files import json_text, write_atomic

from agentwatch_model.candidates import HASH_DIGITS
from agentwatch_model.metrics import is_number
from agentwatch_model.rule_baseline import EXIT_ERROR
from agentwatch_model.tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    SQLITE_PREFIX,
    experiment_id,
    tracking_errors,
)

PACKAGE = Path(__file__).parent
DEFAULT_REGISTRY = PACKAGE / "registry.toml"
DEFAULT_CANDIDATES = Path(".data/candidates")
DEFAULT_CALIBRATED = Path(".data/calibration")
DEFAULT_OUTPUT = Path(".data/registry")
VERSION = "registry-v1"
EXPERIMENT = "agentwatch-registry"
CANDIDATE = "candidate"
CHAMPION = "champion"
PREVIOUS = "previous_champion"
RECORD = "registration.json"
ARTIFACT_PATH = "model"
CANDIDATE_FILES = ("classifier.txt", "regressor.txt", "preprocessing.json", "report.json")
CALIBRATION_FILES = ("calibration.json", "report.json")


class RegistryError(ValueError):
    """A model or registry state that cannot be pushed or rolled back as asked."""


@dataclass(frozen=True)
class RegistryConfig:
    version: str
    model_name: str
    max_macro_f1_drop: float


def load_registry_config(path: Path = DEFAULT_REGISTRY) -> RegistryConfig:
    parsed = tomllib.loads(path.read_text())
    if parsed.get("version") != VERSION:
        raise RegistryError(f"{path.name}: version must be {VERSION!r}")
    name, drop = parsed.get("model_name"), parsed.get("max_macro_f1_drop")
    if not isinstance(name, str) or not name:
        raise RegistryError("model_name must be a nonempty string")
    if not is_number(drop) or not 0 <= drop <= 1:
        raise RegistryError("max_macro_f1_drop must be a number in [0, 1]")
    return RegistryConfig(f"{VERSION}+{canonical_sha256(parsed)[:HASH_DIGITS]}", name, float(drop))


def read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"Cannot read the {what} {path}: {error}") from error


def failed_gates(gates: dict) -> list[str]:
    outcome = {name: g.get("outcome") if isinstance(g, dict) else g for name, g in gates.items()}
    return sorted(name for name, value in outcome.items() if value == "fail")


def bias_outcome(bias_report: Path | None, training_version: str) -> str | None:
    """The bias check's test outcome for this training version, or None without a report."""
    if bias_report is None:
        return None
    report = read_json(bias_report, "bias report")
    for model in report.get("models", {}).values():
        if model.get("training_version") == training_version:
            return model["test"]["outcome"]
    raise RegistryError(f"{bias_report} has no model with training version {training_version}")


def evaluate(candidates: Path, calibrated: Path, bias_report: Path | None) -> dict:
    """The model's identity, gate outcomes and eligibility, after every consistency check."""
    report = read_json(candidates / "report.json", "candidates report")
    if report.get("status") != "evaluated" or report.get("scored_split") != "test":
        raise RegistryError(f"{candidates / 'report.json'} is not an evaluated test report")
    calibration = read_json(calibrated / "calibration.json", "calibration")
    if calibration.get("fitted") is not True:
        raise RegistryError(f"{calibrated / 'calibration.json'} is not fitted")
    training_version = report["model"]["version"]
    if calibration.get("training_version") != training_version:
        raise RegistryError(
            f"calibration.json training_version {calibration.get('training_version')!r} is not "
            f"the candidates' {training_version!r}"
        )
    failed = failed_gates(report.get("gates", {})) + failed_gates(calibration.get("gates", {}))
    bias = bias_outcome(bias_report, training_version)
    reasons = []
    if failed:
        reasons.append(f"failed gate(s): {', '.join(failed)}")
    if bias == "fail":
        reasons.append("the bias check fails on test")
    return {
        "training_version": training_version,
        "calibration_version": calibration["version"],
        "macro_f1": report["comparison"]["macro_f1_all_classes"]["model"],
        "blocks_promotion": bool(calibration.get("blocks_promotion", True)),
        "failed_gates": failed,
        "bias_test_outcome": bias,
        "eligible": not reasons,
        "ineligible_because": reasons,
    }


def client(tracking_uri: str):
    import mlflow
    from mlflow import MlflowClient

    if tracking_uri.startswith(SQLITE_PREFIX):
        Path(tracking_uri.removeprefix(SQLITE_PREFIX)).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow, MlflowClient()


def champion_version(registry, name: str):
    """The model version the `champion` alias points at, or None while it is unset."""
    number = registry.get_registered_model(name).aliases.get(CHAMPION)
    return registry.get_model_version(name, number) if number is not None else None


def register(
    candidates: Path = DEFAULT_CANDIDATES,
    calibrated: Path = DEFAULT_CALIBRATED,
    bias_report: Path | None = None,
    *,
    output: Path = DEFAULT_OUTPUT,
    registry_path: Path = DEFAULT_REGISTRY,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict:
    """Register a new version, move `candidate`, and move `champion` only under the rule."""
    config = load_registry_config(registry_path)
    model = evaluate(candidates, calibrated, bias_report)
    mlflow, registry = client(tracking_uri)
    tags = {
        "training_version": model["training_version"],
        "calibration_version": model["calibration_version"],
        "macro_f1": repr(model["macro_f1"]),
        "blocks_promotion": str(model["blocks_promotion"]).lower(),
        "eligible": str(model["eligible"]).lower(),
        "bias_test_outcome": str(model["bias_test_outcome"]),
        "registry_version": config.version,
    }
    with mlflow.start_run(experiment_id=experiment_id(artifact_root, EXPERIMENT)) as run:
        mlflow.set_tags({"task": "M05-registry", **tags})
        for folder, names in ((candidates, CANDIDATE_FILES), (calibrated, CALIBRATION_FILES)):
            for name in names:
                if (folder / name).is_file():
                    subfolder = "calibration" if folder == calibrated else "candidates"
                    mlflow.log_artifact(str(folder / name), f"{ARTIFACT_PATH}/{subfolder}")
        source = f"{run.info.artifact_uri}/{ARTIFACT_PATH}"
    names = {m.name for m in registry.search_registered_models()}
    if config.model_name not in names:
        registry.create_registered_model(config.model_name)
    version = registry.create_model_version(
        config.model_name, source, run_id=run.info.run_id, tags=tags
    )
    number = int(version.version)
    registry.set_registered_model_alias(config.model_name, CANDIDATE, str(number))
    current = champion_version(registry, config.model_name)
    champion = int(current.version) if current else None
    reason = "; ".join(model["ineligible_because"])
    if model["eligible"]:
        best = float(current.tags["macro_f1"]) if current else None
        if best is None or model["macro_f1"] >= best - config.max_macro_f1_drop:
            if champion is not None:
                registry.set_registered_model_tag(config.model_name, PREVIOUS, str(champion))
            registry.set_registered_model_alias(config.model_name, CHAMPION, str(number))
            champion = number
            reason = "eligible and at least as good as the champion"
        else:
            reason = f"macro F1 {model['macro_f1']:.4f} is below the champion's {best:.4f}"
    outcome = {
        "registry": config.version,
        "model_name": config.model_name,
        "version": number,
        "run_id": run.info.run_id,
        "source": source,
        "champion": champion,
        "eligible": model["eligible"],
        "reason": reason,
        "model": model,
        "tags": tags,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(output / RECORD, json_text(outcome))
    return outcome


def rollback(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict:
    """Point `champion` back at the version it replaced."""
    config = load_registry_config(registry_path)
    _, registry = client(tracking_uri)
    model = registry.get_registered_model(config.model_name)
    previous = model.tags.get(PREVIOUS)
    current = model.aliases.get(CHAMPION)
    if not previous or current is None:
        raise RegistryError("There is no previous champion to roll back to")
    registry.set_registered_model_alias(config.model_name, CHAMPION, previous)
    registry.set_registered_model_tag(config.model_name, PREVIOUS, str(current))
    return {"champion": int(previous), "replaced": int(current)}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push an evaluated model to the model registry.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--calibrated", type=Path, default=DEFAULT_CALIBRATED)
    parser.add_argument("--bias-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--rollback", action="store_true", help="Restore the previous champion")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common = {
        "registry_path": args.registry,
        "tracking_uri": args.tracking_uri,
        "artifact_root": args.artifact_root,
    }
    try:
        if args.rollback:
            result = rollback(**common)
        else:
            result = register(
                args.candidates, args.calibrated, args.bias_report, output=args.output, **common
            )
    except (RegistryError, tomllib.TOMLDecodeError, *tracking_errors()) as error:
        print(f"Registry error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps({k: result[k] for k in result if k not in ("model", "tags")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
