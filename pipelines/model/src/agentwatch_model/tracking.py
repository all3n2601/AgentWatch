"""Experiment tracking in MLflow: every attempt is logged, including blocked and failed ones.

The default backend is a local SQLite database under .data/, with artifacts beside it, so
tracking works offline and nothing is uploaded.
"""

from pathlib import Path

DEFAULT_TRACKING_URI = "sqlite:///.data/mlflow/mlflow.db"
DEFAULT_ARTIFACT_ROOT = Path(".data/mlflow/artifacts")
EXPERIMENT = "agentwatch-attribution"
SQLITE_PREFIX = "sqlite:///"
MAX_TAG_LENGTH = 5000  # MLflow's tag value limit
UNAVAILABLE = "unavailable"  # logged for a value that was not recorded, never "None"


def tracking_errors() -> tuple[type[BaseException], ...]:
    """Errors a tracking backend can raise; a failure here must not discard the report."""
    errors: list[type[BaseException]] = [OSError, ImportError]
    try:
        from mlflow.exceptions import MlflowException
        from sqlalchemy.exc import SQLAlchemyError

        errors += [MlflowException, SQLAlchemyError]
    except ImportError:
        pass
    return tuple(errors)


def text(value: object) -> str:
    return UNAVAILABLE if value is None else str(value)


def run_params(report: dict) -> dict[str, str]:
    """Lineage and settings: every version and hyperparameter that shaped the attempt."""
    model = report.get("model") or {}
    params = {
        "training_version": text(model.get("version")),
        "rules_version": report["rules"]["version"],
        "contract_version": report["contract"]["version"],
        "contract_hash": report["contract"]["hash"],
        "dataset_version": text(report.get("dataset_version")),
        "label_version": text(model.get("label_version")),
        "scored_split": text(report.get("scored_split")),
        "split_sha256": text((report.get("split") or {}).get("sha256")),
        "class_order": text(model.get("class_order")),
        "classes_absent_from_train": text(model.get("classes_absent_from_train")),
        "validation_rows_dropped": text(model.get("validation_rows_dropped")),
        "early_stopping": text(model.get("early_stopping")),
        "lightgbm_version": text(model.get("lightgbm_version")),
        "numpy_version": text(model.get("numpy_version")),
    }
    for name, value in (model.get("best_iterations") or {}).items():
        params[f"best_iteration.{name}"] = text(value)
    settings = model.get("settings") or {}
    for name in ("num_boost_round", "early_stopping_rounds"):
        params[name] = text(settings.get(name))
    for booster in ("classifier", "regressor"):
        for name, value in (settings.get(booster) or {}).items():
            params[f"{booster}.{name}"] = text(value)
    return params


def run_metrics(report: dict) -> dict[str, float]:
    """Numeric results only; unmeasured values are left out rather than logged as zero."""
    metrics = report.get("metrics") or {}
    candidates = {}
    for source in ("model", "rules"):
        macro = (metrics.get(source) or {}).get("macro_f1") or {}
        candidates[f"{source}_macro_f1_all_classes"] = macro.get("all_classes")
        candidates[f"{source}_macro_f1_defined_classes"] = macro.get("defined_classes")
    regression = metrics.get("regression") or {}
    candidates["model_mae"] = regression.get("mae")
    for name, value in (regression.get("constant_references") or {}).items():
        candidates[f"reference_{name}"] = value
    return {name: float(value) for name, value in candidates.items() if value is not None}


def experiment_id(artifact_root: Path) -> str:
    import mlflow
    from mlflow.exceptions import MlflowException

    existing = mlflow.get_experiment_by_name(EXPERIMENT)
    if existing:
        return existing.experiment_id
    try:
        return mlflow.create_experiment(
            EXPERIMENT, artifact_location=artifact_root.resolve().as_uri()
        )
    except MlflowException:
        # Another attempt created it first; use that one.
        created = mlflow.get_experiment_by_name(EXPERIMENT)
        if created is None:
            raise
        return created.experiment_id


def log_run(
    report: dict, output: Path, *, tracking_uri: str, artifact_root: Path, task: str
) -> str:
    """Log one attempt with its report and artifacts; return the MLflow run id."""
    import mlflow

    if tracking_uri.startswith(SQLITE_PREFIX):
        Path(tracking_uri.removeprefix(SQLITE_PREFIX)).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(experiment_id=experiment_id(artifact_root)) as active:
        mlflow.set_tags(
            {
                "task": task,
                "status": report["status"],
                "evaluation": str(report["evaluation"]),
                "reason": str(report.get("reason", ""))[:MAX_TAG_LENGTH],
            }
        )
        mlflow.log_params(run_params(report))
        metrics = run_metrics(report)
        if metrics:
            mlflow.log_metrics(metrics)
        # The attempt's own files only: log_artifacts would recurse into an evaluated run's
        # exploratory/ subdirectory and file an exploratory model under an evaluated run.
        for path in sorted(output.iterdir()):
            if path.is_file():
                mlflow.log_artifact(str(path))
        return active.info.run_id
