"""Use the project's local MLflow backend and artifact directory."""

import mlflow
from mlflow.exceptions import MlflowException

from src.paths import PROJECT_ROOT

PROJECT_TRACKING_URI = "sqlite:///" + str(PROJECT_ROOT / "mlflow.db")


def set_project_experiment(name):
    """Select an existing experiment, or create it with a local artifact path."""
    mlflow.set_tracking_uri(PROJECT_TRACKING_URI)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(name) is None:
        try:
            client.create_experiment(name, artifact_location=(PROJECT_ROOT / "mlruns" / name).as_uri())
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
    return mlflow.set_experiment(name)
