import mlflow
from mlflow.tracking import fluent

from src.experiments import tracking


def test_project_tracking_reuses_experiment_and_keeps_runs(tmp_path, monkeypatch):
    uri = "sqlite:///" + str(tmp_path / "mlflow.db")
    monkeypatch.setattr(tracking, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tracking, "PROJECT_TRACKING_URI", uri)
    monkeypatch.setattr(fluent, "_active_experiment_id", None)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    previous_uri = mlflow.get_tracking_uri()
    try:
        experiment = tracking.set_project_experiment("svm-specimen")
        with mlflow.start_run(run_name="first") as run:
            mlflow.log_metric("critical_errors", 2)
            mlflow.log_text("verified", "result.txt")

        same_experiment = tracking.set_project_experiment("svm-specimen")
        other_experiment = tracking.set_project_experiment("class2-training-bodies-group-cv")
        client = mlflow.MlflowClient()
        assert mlflow.get_tracking_uri() == uri
        assert same_experiment.experiment_id == experiment.experiment_id
        assert other_experiment.experiment_id != experiment.experiment_id
        assert client.get_run(run.info.run_id).data.metrics == {"critical_errors": 2.0}
        assert (tmp_path / f"mlruns/svm-specimen/{run.info.run_id}/artifacts/result.txt").read_text() == "verified"
    finally:
        mlflow.set_tracking_uri(previous_uri)
