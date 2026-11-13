from pathlib import Path

from agentwatch_model.tracking import EXPERIMENT, log_run

REPORT = {
    "status": "ok",
    "evaluation": "held-out",
    "rules": {"version": "rules-v1"},
    "contract": {"version": "c1", "hash": "abc"},
}


def _log(tmp_path: Path, **kwargs) -> str:
    import mlflow

    run_id = log_run(
        REPORT,
        tmp_path,
        artifacts=(),
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_root=tmp_path / "artifacts",
        task="test",
        **kwargs,
    )
    return mlflow.get_run(run_id).info.experiment_id


def test_log_run_uses_the_named_experiment(tmp_path):
    import mlflow

    named = _log(tmp_path, experiment="a06-held-out-scoring")
    assert mlflow.get_experiment_by_name("a06-held-out-scoring").experiment_id == named
    default = _log(tmp_path)
    assert mlflow.get_experiment_by_name(EXPERIMENT).experiment_id == default
    assert named != default
