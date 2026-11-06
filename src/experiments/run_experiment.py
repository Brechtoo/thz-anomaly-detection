import mlflow
import mlflow.sklearn

import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from threadpoolctl import threadpool_info

from src.data.dataset import prepare_dataset
from src.data.splitting import (
    RANDOM_STATE,
    TEMP_SIZE,
    TEST_SIZE_OF_TEMP,
)
from src.data.split_manifest import restore_splits
from src.experiments.provenance import capture_git_state, check_locked_environment

from src.evaluation.metrics import (
    ERROR_METRICS_VERSION, evaluate_predictions, calculate_error_counts,
)
from src.evaluation.reporting import save_evaluation_report

from src.models.hgb import create_hgb, train_hgb

from src.experiments.saving import create_experiment_dir, save_model, save_results

from src.paths import FEATURE_TABLE_ROOT, EXPERIMENTS_ROOT, PROJECT_ROOT

# ============================================================
# settings
# ============================================================

EXPERIMENT_NAME = "hgb_baseline_reproducibility"

RESULTS_DIR = EXPERIMENTS_ROOT
SPLIT_MANIFEST_DIR = PROJECT_ROOT / "configs/hgb_baseline/splits"
DEPENDENCY_LOCK = PROJECT_ROOT / "requirements-lock.txt"

def get_git_info():
    state = capture_git_state(PROJECT_ROOT)
    return state["git_commit"], state["git_dirty"]

def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def run_hgb_baseline():

    # Validate all inputs before creating output files or MLflow runs.
    if os.environ.get("MLFLOW_RUN_ID") or mlflow.active_run() is not None:
        raise ValueError("Baseline requires a new MLflow run; clear MLFLOW_RUN_ID and end active runs")
    git_state = capture_git_state(PROJECT_ROOT)
    check_locked_environment(DEPENDENCY_LOCK)
    manifest = json.loads((SPLIT_MANIFEST_DIR / "manifest.json").read_text())
    data_hash = calculate_file_hash(FEATURE_TABLE_ROOT)
    if data_hash != manifest["feature_table_sha256"]:
        raise ValueError("Feature-table hash differs from the frozen HGB baseline")
    df = pd.read_csv(FEATURE_TABLE_ROOT)
    X, y = prepare_dataset(df)
    if X.columns.tolist() != manifest["feature_columns"]:
        raise ValueError("Feature columns differ from the frozen HGB baseline")
    X_train, X_val, X_test, y_train, y_val, y_test = restore_splits(
        X, y, df["instance_id"], SPLIT_MANIFEST_DIR
    )
    if [len(X_train), len(X_val), len(X_test)] != [
        manifest["sizes"][name] for name in ("train", "validation", "test")
    ]:
        raise ValueError("Split sizes differ from the frozen HGB baseline")

    # Always use the local store, never an inherited remote tracking endpoint.
    mlflow.set_tracking_uri("sqlite:///" + str(PROJECT_ROOT / "mlflow.db"))
    mlflow.set_experiment("thz-baselines")
    mlflow.sklearn.autolog(
        log_post_training_metrics=False
    )

    # ============================================================
    # experimentenordner erstellen
    # ============================================================

    run_name = (
        EXPERIMENT_NAME + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid4().hex[:8]
    )
    # Reserve a new directory; refuse collisions rather than reuse old outputs.
    (RESULTS_DIR / run_name).mkdir(parents=True, exist_ok=False)
    experiment_dir, models_dir = create_experiment_dir(RESULTS_DIR, run_name)
    metadata_dir = experiment_dir / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "git_state.json").write_text(json.dumps(git_state, indent=2))
    (metadata_dir / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "threadpools": threadpool_info(),
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED"
        )},
        "dependency_lock_sha256": calculate_file_hash(DEPENDENCY_LOCK),
    }, indent=2))
    shutil.copytree(SPLIT_MANIFEST_DIR, metadata_dir / "splits")
    shutil.copy2(DEPENDENCY_LOCK, metadata_dir / DEPENDENCY_LOCK.name)
    # Archive actual source bytes even when HEAD has uncommitted code changes.
    for source in (PROJECT_ROOT / "src").rglob("*.py"):
        target = metadata_dir / "source" / source.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # ============================================================
    # metadaten der gespeicherten splits
    # ============================================================

    split_info = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],

        "split_random_state": RANDOM_STATE,
        "split_temp_size": TEMP_SIZE,
        "split_test_size_of_temp": TEST_SIZE_OF_TEMP,
        "split_method": "stratified",
        "split_source": "saved_instance_ids",
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

        with mlflow.start_run(run_name=run_name):

            mlflow.set_tags({
                "git_commit": git_state["git_commit"],
                "git_dirty": str(git_state["git_dirty"]),
                "data_sha256": data_hash,
                "data_file": str(FEATURE_TABLE_ROOT),
                "evaluation_split": "validation",
                "error_metrics_version": ERROR_METRICS_VERSION,
                "split_manifest_sha256": calculate_file_hash(SPLIT_MANIFEST_DIR / "manifest.json"),
                "dependency_lock_sha256": calculate_file_hash(DEPENDENCY_LOCK),
                "split_source_run_id": manifest["source_run_id"],
            })

            mlflow.log_artifacts(str(metadata_dir), artifact_path="metadata")
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
                **{f"val_{key}": value for key, value in errors.items()},
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
                "error_metrics_version": ERROR_METRICS_VERSION,
            })

    # ============================================================
    # save
    # ============================================================

    results_df = pd.DataFrame(all_results)

    save_results(results_df, experiment_dir)

    print(f"ergebnisse gespeichert unter: {experiment_dir}")

    return results_df


def run_experiment(*, baseline=False, enabled=None, config_path=None):
    """Run the separate tsfresh comparison, or explicitly replay frozen HGB."""
    if baseline:
        return run_hgb_baseline()
    from src.experiments.model_comparison import CONFIG_PATH, run_model_comparison
    return run_model_comparison(config_path or CONFIG_PATH, enabled=enabled)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="tsfresh validation model comparison")
    parser.add_argument("--baseline", action="store_true", help="Replay frozen HGB instead")
    args = parser.parse_args()
    run_experiment(baseline=args.baseline)
