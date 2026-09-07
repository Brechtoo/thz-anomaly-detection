from pathlib import Path

import joblib

def create_experiment_dir(base_dir, experiment_name):
    experiment_dir = Path(base_dir) / experiment_name

    models_dir = experiment_dir / "models"

    experiment_dir.mkdir(parents=True, exist_ok=True)

    models_dir.mkdir(parents=True, exist_ok=True)

    return experiment_dir, models_dir


def save_model(model, model_name, models_dir):

    path = models_dir / f"{model_name}.joblib"

    joblib.dump(model, path)


def save_results(results_df, experiment_dir):
    path = experiment_dir / "validation_results.csv"

    results_df.to_csv(path, index=False)
