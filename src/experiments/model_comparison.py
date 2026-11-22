"""Vergleich mehrerer Modelle mit tsfresh-Features auf einem festen Validierungssplit.

Ergebnisse werden lokal und in MLflow gespeichert. Scheitert eine Phase,
bleiben bereits ermittelte Kennzahlen für das jeweilige Modell verfügbar.
"""

import importlib.metadata
import json
import os
import platform
import shutil
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

from src.data.dataset import prepare_dataset
from src.data.split_manifest import restore_splits, sha256_file
from src.evaluation.metrics import (
    ERROR_METRICS_VERSION,
    calculate_error_counts,
    evaluate_predictions,
)
from src.evaluation.reporting import save_evaluation_report
from src.experiments.model_registry import build_models
from src.experiments.model_serialization import trusted_types_for_model
from src.experiments.provenance import capture_git_state, snapshot_source_tree
from src.experiments.saving import (
    create_experiment_dir,
    save_model,
    save_results,
)
from src.paths import EXPERIMENTS_ROOT, PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs/tsfresh_comparison.json"


def load_comparison_data(config):
    """Prüft die Eingabeprovenienz und gibt Train- und Validierungsdaten zurück."""
    path = PROJECT_ROOT / config["feature_table"]

    if sha256_file(path) != config["feature_table_sha256"]:
        raise ValueError("tsfresh feature-table checksum mismatch")

    split_dir = PROJECT_ROOT / config["split_dir"]

    if sha256_file(split_dir / "manifest.json") != config["split_manifest_sha256"]:
        raise ValueError("Split manifest checksum mismatch")

    metadata_path = path.with_suffix(".json")

    if sha256_file(metadata_path) != config["feature_metadata_sha256"]:
        raise ValueError("tsfresh metadata checksum mismatch")

    metadata = json.loads(metadata_path.read_text())

    if (
        metadata["output_sha256"] != config["feature_table_sha256"]
        or metadata["split_manifest_sha256"] != config["split_manifest_sha256"]
    ):
        raise ValueError(
            "tsfresh extraction provenance differs from comparison inputs"
        )

    df = pd.read_csv(path)

    X, y = prepare_dataset(df)

    if (
        X.columns.tolist() != config["feature_columns"]
        or metadata["selected_features"] != X.columns.tolist()
    ):
        raise ValueError("tsfresh feature columns mismatch")

    # Rekonstruieren nach Messungs-ID prüft alle Split-Mitgliedschaften.
    # In den Modellvergleich gehen ausschließlich Training und Validierung ein.
    splits = restore_splits(X, y, df.instance_id, split_dir)
    train, val, _, y_train, y_val, _ = splits

    for frame in (train, val):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(
                "Expected finite, already imputed tsfresh features"
            )

    return train, val, y_train, y_val


def persist_comparison_row(row, rows, directory):
    """Lokale Ergebnisse unabhängig vom MLflow-Status sichern."""
    pd.DataFrame([row]).to_csv(directory / f"{row['model']}_results.csv", index=False)
    save_results(pd.DataFrame(rows), directory)


def run_model_comparison(config_path=CONFIG_PATH, enabled=None):
    """Trainiert die aktivierten Modelle und gibt ihre Validierungsergebnisse zurück."""
    # Jeder Modellvergleich benötigt eigene MLflow-Runs für eine eindeutige Provenienz.
    if os.environ.get("MLFLOW_RUN_ID") or mlflow.active_run() is not None:
        raise ValueError("Comparison requires fresh MLflow runs")

    config = json.loads(Path(config_path).read_text())

    print("Prüfe tsfresh-Hashes und gespeicherte Splits ...", flush=True)

    train, val, y_train, y_val = load_comparison_data(config)

    models = build_models(enabled, config["seed"])

    git_state = capture_git_state(PROJECT_ROOT)

    run_name = (
        "tsfresh_comparison_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:8]
    )

    # Bei einer Namenskollision abbrechen, damit frühere Ergebnisse erhalten bleiben.
    directory = EXPERIMENTS_ROOT / run_name
    directory.mkdir(parents=True, exist_ok=False)

    _, models_dir = create_experiment_dir(EXPERIMENTS_ROOT, run_name)

    metadata = directory / "metadata"
    metadata.mkdir()

    (metadata / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    (metadata / "git_state.json").write_text(
        json.dumps(git_state, indent=2)
    )

    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
    }

    (metadata / "environment.json").write_text(
        json.dumps({
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "threadpools": threadpool_info(),
            "thread_limit": 1,

            "thread_environment": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "PYTHONHASHSEED",
                )
            },
        }, indent=2)
    )

    shutil.copytree(
        PROJECT_ROOT / config["split_dir"],
        metadata / "splits"
    )

    shutil.copy2(
        (PROJECT_ROOT / config["feature_table"]).with_suffix(".json"),
        metadata / "feature_extraction.json"
    )

    # Quellbaum sichern; lokale Tracking-Stores nicht rekursiv mitkopieren.
    snapshot_source_tree(PROJECT_ROOT / "src", metadata / "source/src")

    mlflow.set_tracking_uri(
        "sqlite:///" + str(PROJECT_ROOT / "mlflow.db")
    )

    mlflow.set_experiment("thz-tsfresh-comparison")

    # Explizites Logging hält lokale und im Tracking gespeicherte Phasenstatus synchron.
    mlflow.sklearn.autolog(disable=True)

    rows = []

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"[{i}/{len(models)}] Trainiere {name} ...", flush=True)

        row = {
            "model": name, "run_id": None, "status": "RUNNING",
            "training_status": "NOT_STARTED", "evaluation_status": "NOT_STARTED",
            "local_save_status": "NOT_STARTED", "mlflow_status": "NOT_STARTED",
            "error_metrics_version": ERROR_METRICS_VERSION,
        }
        rows.append(row)

        run = None
        stage = "training"

        try:
            params = {
                k: str(v)
                for k, v in model.get_params(deep=True).items()
            }

            (metadata / f"{name}_parameters.json").write_text(
                json.dumps(params, indent=2)
            )

            started = time.perf_counter()

            # Begrenze die Threadzahl und halte Konvergenzwarnungen als Diagnostik fest.
            with (
                warnings.catch_warnings(record=True) as caught,
                threadpool_limits(limits=1),
            ):

                warnings.simplefilter(
                    "always",
                    ConvergenceWarning
                )

                model.fit(train, y_train)

                duration = time.perf_counter() - started

                row["training_seconds"] = duration
                row["training_status"] = "FINISHED"
                stage = "evaluation"

                prediction_started = time.perf_counter()

                predicted = model.predict(val)

                prediction_duration = (
                    time.perf_counter() - prediction_started
                )
                row["prediction_seconds"] = prediction_duration

            # Die Varianzmaske stammt aus dem Fit auf dem Trainingssplit.
            retained = train.columns[
                model.named_steps["constants"].get_support()
            ].tolist()

            metrics = evaluate_predictions(y_val, predicted)

            # Nur skalare Metriken behalten.
            # Komplexere Berichte werden separat gespeichert.
            metrics = {
                k: float(v)
                for k, v in metrics.items()
                if k not in (
                    "confusion_matrix",
                    "classification_report"
                )
            }

            metrics.update(
                calculate_error_counts(y_val, predicted)
            )

            row.update({"prediction_seconds": prediction_duration,
                        "n_features": len(retained), **metrics})
            row["evaluation_status"] = "FINISHED"
            stage = "local_save"

            # Kennzahlen vor dem Speichern des Modells und dem Tracking sichern.
            persist_comparison_row(row, rows, directory)

            save_model(model, name, models_dir)

            save_evaluation_report(y_val, predicted, directory, name)
            (metadata / f"{name}_preprocessing.json").write_text(json.dumps(
                {
                    "retained_features": retained,
                    "removed_constants": [
                        c for c in train if c not in retained
                    ],
                    "warnings": [
                        str(w.message) for w in caught
                    ],
                },
                indent=2
            ))

            row["local_save_status"] = "FINISHED"
            persist_comparison_row(row, rows, directory)
            stage = "mlflow"

            # Tracking beginnt erst nach dem lokalen Speichern von Modell und Berichten.
            with mlflow.start_run(run_name=f"{run_name}_{name}") as run:
                row["run_id"] = run.info.run_id
                mlflow.set_tags({
                    "evaluation_split": "validation",
                    "error_metrics_version": ERROR_METRICS_VERSION,
                    "data_sha256": config["feature_table_sha256"],
                    "git_commit": git_state["git_commit"],
                    "git_dirty": str(git_state["git_dirty"]),
                    **{key: row[key] for key in ("training_status", "evaluation_status", "local_save_status")},
                })

                mlflow.log_params(params)
                mlflow.log_metrics({
                    "training_seconds": duration,
                    "prediction_seconds": prediction_duration,
                    **{
                        f"val_{k}": v
                        for k, v in metrics.items()
                    },
                })

                mlflow.log_artifacts(
                    str(metadata),
                    artifact_path="metadata"
                )

                mlflow.log_artifacts(
                    str(directory / "evaluations" / name),
                    artifact_path="evaluation"
                )

                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    serialization_format="skops",
                    skops_trusted_types=trusted_types_for_model(name)
                )

                row["mlflow_status"] = "FINISHED"
                row["status"] = "FINISHED"

                mlflow.set_tags({
                    key: row[key]
                    for key in (
                        "training_status",
                        "evaluation_status",
                        "local_save_status",
                        "mlflow_status",
                    )
                })

                # Erst den endgültigen Status lokal sichern, dann genau diese
                # CSV hochladen. So stimmen Run-Artefakt und Vergleichstabelle überein.
                persist_comparison_row(row, rows, directory)
                mlflow.log_artifact(str(directory / f"{name}_results.csv"))

            print(f"{name}: {duration:.1f}s, kritische Fehler: {metrics['critical_errors']}; "
                  f"gespeichert in {directory}", flush=True)

        except Exception as exc:
            # Nur die fehlgeschlagene Phase ändern. Vorhandene Metriken,
            # Zeiten und Status früherer Phasen bleiben in derselben Zeile.
            row[f"{stage}_status"] = "FAILED"
            row[f"{stage}_error_type"] = type(exc).__name__
            row[f"{stage}_error"] = str(exc)
            row["status"] = "PARTIAL" if row["evaluation_status"] == "FINISHED" else "FAILED"
            error_path = directory / f"{name}_{stage}_error.txt"
            error_path.write_text(traceback.format_exc())

            # Den Fehler nach Ende des Run-Kontexts über die gespeicherte Run-ID anhängen.
            if run is not None:
                try:
                    client = mlflow.MlflowClient()
                    client.set_tag(run.info.run_id, f"{stage}_status", "FAILED")
                    client.log_artifact(run.info.run_id, str(error_path), "errors")
                except Exception as logging_error:
                    print(f"Fehlerdetails nur lokal gespeichert: {logging_error}", flush=True)

            print(f"{name}: Fehler in {stage}: {exc}. Fahre mit nächstem Modell fort.", flush=True)
        finally:
            persist_comparison_row(row, rows, directory)

    return pd.DataFrame(rows)
