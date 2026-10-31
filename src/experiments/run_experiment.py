import mlflow
import mlflow.sklearn

import hashlib
import subprocess

import pandas as pd

from src.data.dataset import prepare_dataset
from src.data.splitting import (
    create_splits,
    RANDOM_STATE,
    TEMP_SIZE,
    TEST_SIZE_OF_TEMP,
)

from src.evaluation.metrics import evaluate_predictions, calculate_error_counts
from src.evaluation.reporting import save_evaluation_report

from src.models.hgb import create_hgb, train_hgb

from src.experiments.saving import create_experiment_dir, save_model, save_results

from src.paths import FEATURE_TABLE_ROOT, EXPERIMENTS_ROOT

# ============================================================
# settings
# ============================================================

EXPERIMENT_NAME = "hgb_baseline_reproducibility"

RESULTS_DIR = EXPERIMENTS_ROOT

def get_git_info():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()

    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip()

    return commit, bool(status)

def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def run_experiment():

    mlflow.set_experiment("thz-baselines")
    mlflow.sklearn.autolog(
        log_post_training_metrics=False
    )

    # ============================================================
    # experimentenordner erstellen
    # ============================================================

    experiment_dir, models_dir = create_experiment_dir(RESULTS_DIR, EXPERIMENT_NAME)

    # ============================================================
    # daten laden
    # ============================================================

    df = pd.read_csv(FEATURE_TABLE_ROOT)

    X, y = prepare_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)

    git_commit, git_dirty = get_git_info()
    data_hash = calculate_file_hash(FEATURE_TABLE_ROOT)

    split_info = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],

        "split_random_state": RANDOM_STATE,
        "split_temp_size": TEMP_SIZE,
        "split_test_size_of_temp": TEST_SIZE_OF_TEMP,
        "split_method": "stratified",
    }

    # ============================================================
    # modelle (aktuell nur hgb)
    # ============================================================

    models = {
        "hgb": {
            "model": create_hgb(),
            "train_function": train_hgb,
        },
    }

    # ============================================================
    # results
    # ============================================================

    all_results = []

    # ============================================================
    # training und validierung
    # ============================================================

    for model_name, config in models.items():

        with mlflow.start_run(run_name=model_name):

            mlflow.set_tags({
                "git_commit": git_commit,
                "git_dirty": str(git_dirty),
                "data_sha256": data_hash,
                "data_file": str(FEATURE_TABLE_ROOT),
                "evaluation_split": "validation",
            })

            mlflow.log_params(split_info)

            mlflow.log_dict(
                {
                    "feature_columns": X_train.columns.tolist(),
                    "split_sizes": split_info,
                },
                "metadata/dataset.json",
            )

            print(f"train: {model_name} ...")

            model = config["model"]
            train_function = config["train_function"]

            # Training
            if train_function is None:
                model.fit(X_train, y_train)
            else:
                model = train_function(model, X_train, y_train)

            # Modell wie bisher speichern
            save_model(model, model_name, models_dir)

            # Vorhersagen
            y_pred = model.predict(X_val)

            # Evaluation
            results = evaluate_predictions(y_val, y_pred)
            errors = calculate_error_counts(y_val, y_pred)

            save_evaluation_report(
                y_val, y_pred, experiment_dir, model_name
            )

            # Eigene Metriken in MLflow
            mlflow.log_metrics({
                "val_accuracy": results["accuracy"],
                "val_balanced_accuracy": results["balanced_accuracy"],
                "val_macro_f1": results["macro_f1"],
                "val_weighted_f1": results["weighted_f1"],
                "val_critical_errors": errors["critical_errors"],
                "val_critical_error_rate": errors["critical_error_rate"],
                "val_false_alarms": errors["false_alarms"],
                "val_false_alarm_rate": errors["false_alarm_rate"],
                "val_red_to_yellow": errors["red_to_yellow"],
                "val_red_to_yellow_rate": errors["red_to_yellow_rate"],
                "val_green_to_yellow": errors["green_to_yellow"],
                "val_green_to_yellow_rate": errors["green_to_yellow_rate"],
            })

            # Bestehende Ergebnisübersicht beibehalten
            all_results.append({
                "model": model_name,
                **{
                    key: value
                    for key, value in results.items()
                    if key not in ("confusion_matrix", "classification_report")
                },
                **errors,
            })

    # ============================================================
    # save
    # ============================================================

    results_df = pd.DataFrame(all_results)

    save_results(results_df, experiment_dir)

    print(f"ergebnisse gespeichert unter: {experiment_dir}")

    return results_df


if __name__ == "__main__":
    run_experiment()