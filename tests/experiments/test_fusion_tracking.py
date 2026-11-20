import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import fluent

from src.experiments import tracking
from src.experiments.model_registry import create_model
from src.experiments.specimen_fusion import run_model


def test_registry_fusion_fit_logs_metrics_and_prediction_artifacts(tmp_path, monkeypatch):
    uri = "sqlite:///" + str(tmp_path / "mlflow.db")
    monkeypatch.setattr(tracking, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tracking, "PROJECT_TRACKING_URI", uri)
    monkeypatch.setattr(fluent, "_active_experiment_id", None)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    previous_uri = mlflow.get_tracking_uri()
    y = np.tile([0, 1, 2, 3], 4)
    X = np.eye(4)[y]
    cohort = pd.DataFrame({
        'instance_id': [f'measurement{i}' for i in range(4)],
        'specimen_id': ['holdout_body'] * 4,
        'source_dataset': ['source'] * 4,
        'label': y[:4],
    })

    try:
        experiment = tracking.set_project_experiment('fused-synthetic')
        with mlflow.start_run() as parent:
            predictions, rows = run_model(
                'logistic_regression', create_model('logistic_regression'), (X, y),
                {'cv': X[:4]}, {'cv': cohort}, tmp_path, 'fold_0', parent.info.run_id,
            )
        np.testing.assert_array_equal(predictions['cv'], y[:4])
        assert len(rows) == 1 and rows[0]['critical_errors'] == 0
        run = mlflow.MlflowClient().get_run(rows[0]['run_id'])
        assert run.info.experiment_id == experiment.experiment_id
        assert run.data.tags['mlflow.parentRunId'] == parent.info.run_id
        assert run.data.metrics['cv_critical_errors'] == 0
        assert run.data.metrics['cv_critical_2_support'] == 1
        artifact = tmp_path / f'mlruns/fused-synthetic/{run.info.run_id}/artifacts/cv/predictions.csv'
        assert pd.read_csv(artifact).prediction.tolist() == y[:4].tolist()
    finally:
        mlflow.set_tracking_uri(previous_uri)
