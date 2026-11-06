
"""
Vergleich verschiedener ML-Modelle mit tsfresh-Features.

- Verwendet die bestehenden Train-/Val-/Test-Splits.
- Trainiert mehrere Modelle nacheinander.
- Evaluiert ausschließlich auf Validierungsdaten.
- Speichert Ergebnisse und Modelle in MLflow.
- Dokumentiert Fehler, ohne das gesamte Experiment abzubrechen.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import importlib.metadata  # Installierte Pakete und Versionen
import json  # JSON-Dateien lesen und schreiben
import os  # Umgebungsvariablen
import platform  # Informationen zur Systemumgebung
import shutil  # Dateien und Verzeichnisse kopieren
import time  # Laufzeiten messen
import traceback  # Vollständige Fehlermeldungen speichern
import warnings  # Python-Warnungen erfassen
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

# Projektinterne Funktionen
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
from src.experiments.provenance import capture_git_state
from src.experiments.saving import (
    create_experiment_dir,
    save_model,
    save_results,
)
from src.paths import EXPERIMENTS_ROOT, PROJECT_ROOT

# Konfigurationsdatei enthält unter anderem:
# Feature-Pfad, erwartete Hashes, Split-Pfad und Zufallsstartwert.
CONFIG_PATH = PROJECT_ROOT / "configs/tsfresh_comparison.json"


# ============================================================
# 2. DATEN LADEN UND ÜBERPRÜFEN
# ============================================================

def load_comparison_data(config):
    """
    Lädt die tsfresh-Feature-Tabelle und rekonstruiert
    die ursprünglichen Trainings- und Validierungsdaten.

    Rückgabe:
        X_train, X_val, y_train, y_val
    """

    # Vollständigen Pfad der Feature-Tabelle bestimmen.
    path = PROJECT_ROOT / config["feature_table"]

    # Prüfen, ob die Feature-Tabelle exakt der erwarteten
    # Datei entspricht.
    if sha256_file(path) != config["feature_table_sha256"]:
        raise ValueError("tsfresh feature-table checksum mismatch")

    # Verzeichnis mit gespeicherten Datensplits.
    split_dir = PROJECT_ROOT / config["split_dir"]

    # Integrität des Split-Manifests prüfen.
    if sha256_file(split_dir / "manifest.json") != config["split_manifest_sha256"]:
        raise ValueError("Split manifest checksum mismatch")

    # Pfad der zur Feature-Tabelle gehörenden JSON-Metadaten.
    metadata_path = path.with_suffix(".json")

    # Prüfen, ob diese Metadatendatei verändert wurde.
    if sha256_file(metadata_path) != config["feature_metadata_sha256"]:
        raise ValueError("tsfresh metadata checksum mismatch")

    # JSON einlesen und in ein Python-Dictionary umwandeln.
    metadata = json.loads(metadata_path.read_text())

    # Sicherstellen, dass die Metadaten dieselben Eingaben
    # referenzieren wie die Experimentkonfiguration.
    if (
        metadata["output_sha256"] != config["feature_table_sha256"]
        or metadata["split_manifest_sha256"] != config["split_manifest_sha256"]
    ):
        raise ValueError(
            "tsfresh extraction provenance differs from comparison inputs"
        )

    # Gesamte Feature-Tabelle laden.
    df = pd.read_csv(path)

    # Metadaten/Zielvariable von den Eingabefeatures trennen.
    X, y = prepare_dataset(df)

    # Feature-Namen und deren Reihenfolge prüfen.
    if (
        X.columns.tolist() != config["feature_columns"]
        or metadata["selected_features"] != X.columns.tolist()
    ):
        raise ValueError("tsfresh feature columns mismatch")

    # Ursprüngliche Splits anhand gespeicherter IDs rekonstruieren.
    #
    # Rückgabe:
    # X_train, X_val, X_test, y_train, y_val, y_test
    splits = restore_splits(X, y, df.instance_id, split_dir)

    # Nur Trainings- und Validierungsdaten übernehmen.
    # "_" bedeutet, dass die Testdaten nicht weiterverwendet werden.
    train, val, _, y_train, y_val, _ = splits

    # Überprüfen, ob alle Features gültige Zahlen enthalten.
    # NaN oder unendliche Werte würden einen Fehler auslösen.
    for frame in (train, val):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(
                "Expected finite, already imputed tsfresh features"
            )

    # Testdaten wurden lediglich bei der Split-Rekonstruktion geprüft.
    return train, val, y_train, y_val


# ============================================================
# 3. MODELLVERGLEICH INITIALISIEREN
# ============================================================

def persist_comparison_row(row, rows, directory):
    """Lokale Ergebnisse unabhängig vom MLflow-Status sichern."""
    pd.DataFrame([row]).to_csv(directory / f"{row['model']}_results.csv", index=False)
    save_results(pd.DataFrame(rows), directory)


def run_model_comparison(config_path=CONFIG_PATH, enabled=None):
    """
    Führt den Modellvergleich durch.

    config_path: Pfad zur Experimentkonfiguration.
    enabled: Optionales Dictionary zur Modellauswahl.

    Gibt einen DataFrame mit allen Ergebnissen zurück.
    """

    # Verhindern, dass versehentlich ein bestehender
    # MLflow-Run wiederverwendet wird.
    if os.environ.get("MLFLOW_RUN_ID") or mlflow.active_run() is not None:
        raise ValueError("Comparison requires fresh MLflow runs")

    # Experimentkonfiguration laden.
    config = json.loads(Path(config_path).read_text())

    print("Prüfe tsfresh-Hashes und gespeicherte Splits ...", flush=True)

    # Daten laden und validieren.
    train, val, y_train, y_val = load_comparison_data(config)

    # Dictionary der aktivierten Modell-Pipelines erstellen.
    models = build_models(enabled, config["seed"])

    # Git-Zustand (Commit und ungespeicherte Änderungen) erfassen.
    git_state = capture_git_state(PROJECT_ROOT)

    # Eindeutigen Namen aus UTC-Zeitstempel und UUID erzeugen.
    run_name = (
        "tsfresh_comparison_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:8]
    )

    # Neuen Experimentordner bestimmen und erstellen
    directory = EXPERIMENTS_ROOT / run_name
    directory.mkdir(parents=True, exist_ok=False)

    # Projektfunktion zur Einrichtung der Ergebnisordner.
    _, models_dir = create_experiment_dir(EXPERIMENTS_ROOT, run_name)

    # Unterordner für Reproduzierbarkeitsinformationen.
    metadata = directory / "metadata"
    metadata.mkdir()

    # Verwendete Experimentkonfiguration speichern.
    (metadata / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    # Git-Zustand speichern.
    (metadata / "git_state.json").write_text(
        json.dumps(git_state, indent=2)
    )

    # Installierte Pakete und ihre Versionen erfassen.
    #
    # d.metadata["Name"] -> Paketname
    # d.version          -> installierte Version
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
    }

    # Systemumgebung dokumentieren.
    (metadata / "environment.json").write_text(
        json.dumps({
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "threadpools": threadpool_info(),
            "thread_limit": 1,

            # Relevante Umgebungsvariablen für numerische Bibliotheken.
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

    # Ursprüngliche Split-Dateien archivieren.
    shutil.copytree(
        PROJECT_ROOT / config["split_dir"],
        metadata / "splits"
    )

    # Metadaten der tsfresh-Extraktion archivieren.
    shutil.copy2(
        (PROJECT_ROOT / config["feature_table"]).with_suffix(".json"),
        metadata / "feature_extraction.json"
    )

    # Alle Python-Quelldateien archivieren.
    # rglob durchsucht auch Unterverzeichnisse.
    for source in (PROJECT_ROOT / "src").rglob("*.py"):

        # Die ursprüngliche Verzeichnisstruktur beibehalten.
        target = metadata / "source" / source.relative_to(PROJECT_ROOT)

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # ========================================================
    # 4. MLFLOW EINRICHTEN
    # ========================================================

    # Lokale MLflow-Datenbank verwenden.
    mlflow.set_tracking_uri(
        "sqlite:///" + str(PROJECT_ROOT / "mlflow.db")
    )

    # Eigenes Experiment für den Modellvergleich auswählen.
    mlflow.set_experiment("thz-tsfresh-comparison")

    # Automatisches Logging ausschalten.
    # Parameter, Metriken und Artefakte werden manuell gespeichert.
    mlflow.sklearn.autolog(disable=True)

    # Liste für die Ergebnistabellen aller Modelle.
    rows = []

    # ========================================================
    # 5. TRAININGSSCHLEIFE
    # ========================================================

    # Jedes aktivierte Modell nacheinander durchlaufen.
    # enumerate(..., 1) erzeugt die fortlaufenden Nummern.
    for i, (name, model) in enumerate(models.items(), 1):
        print(f"[{i}/{len(models)}] Trainiere {name} ...", flush=True)

        # Status und Messwerte werden fortlaufend ergänzt, niemals durch
        # einen späteren Speicherfehler durch eine neue Fehlerzeile ersetzt.
        row = {
            "model": name, "run_id": None, "status": "RUNNING",
            "training_status": "NOT_STARTED", "evaluation_status": "NOT_STARTED",
            "local_save_status": "NOT_STARTED", "mlflow_status": "NOT_STARTED",
            "error_metrics_version": ERROR_METRICS_VERSION,
        }
        rows.append(row)

        # Hier wird später das MLflow-Run-Objekt gespeichert.
        # None bedeutet zunächst: Es wurde noch kein Run gestartet.
        run = None
        stage = "training"

        # try fängt Fehler während des Modell-Runs ab.
        # Damit kann nach einem Fehler das nächste Modell starten.
        try:
            # Alle Parameter der Pipeline und ihrer Modelle abfragen.
            #
            # deep=True:
            # Auch Parameter der verschachtelten Pipeline-Schritte.
            params = {
                k: str(v)
                for k, v in model.get_params(deep=True).items()
            }

            # Parameter zusätzlich lokal als JSON speichern.
            (metadata / f"{name}_parameters.json").write_text(
                json.dumps(params, indent=2)
            )

            # Trainingszeitmessung starten.
            started = time.perf_counter()

            # Warnungen während des Trainings aufzeichnen.
            # Numerische Bibliotheken auf einen Thread begrenzen.
            with (
                warnings.catch_warnings(record=True) as caught,
                threadpool_limits(limits=1),
            ):

                # Konvergenzwarnungen ausdrücklich erfassen.
                warnings.simplefilter(
                    "always",
                    ConvergenceWarning
                )

                # Pipeline und Klassifikationsmodell trainieren.
                model.fit(train, y_train)

                # Verstrichene Trainingsdauer berechnen.
                duration = time.perf_counter() - started

                row["training_seconds"] = duration
                row["training_status"] = "FINISHED"
                stage = "evaluation"

                # Separate Zeitmessung für die Vorhersage starten.
                prediction_started = time.perf_counter()

                # Validierungsdaten klassifizieren.
                predicted = model.predict(val)

                # Vorhersagedauer berechnen.
                prediction_duration = (
                    time.perf_counter() - prediction_started
                )
                # Diese Messung bleibt auch bei späteren Evaluationsfehlern erhalten.
                row["prediction_seconds"] = prediction_duration

            # ====================================================
            # 6. PREPROCESSING DOKUMENTIEREN
            # ====================================================

            # VarianceThreshold hat beim Training ermittelt,
            # welche Features keine konstante Spalte darstellen.
            #
            # get_support() liefert eine boolesche Maske.
            # Damit werden die behaltenen Spalten ausgewählt.
            retained = train.columns[
                model.named_steps["constants"].get_support()
            ].tolist()

            # ====================================================
            # 7. MODELL EVALUIEREN
            # ====================================================

            # Klassische Evaluationsmetriken berechnen.
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

            # Projektspezifische Fehlerzahlen ergänzen.
            metrics.update(
                calculate_error_counts(y_val, predicted)
            )

            # Alle wichtigen Informationen dieses Modells in einer Ergebniszeile
            # zusammenfassen. **metrics übernimmt alle Einträge des Dictionarys.
            row.update({"prediction_seconds": prediction_duration,
                        "n_features": len(retained), **metrics})
            row["evaluation_status"] = "FINISHED"
            stage = "local_save"

            # Gesamtergebnisse sofort speichern. Bereits berechnete Metriken
            # bleiben auch bei nachfolgenden Speicherfehlern erhalten.
            persist_comparison_row(row, rows, directory)

            # Trainierte Pipeline speichern.
            save_model(model, name, models_dir)

            # Konfusionsmatrix und Classification Report speichern.
            save_evaluation_report(y_val, predicted, directory, name)
            # Dokumentieren:
            # - Behaltene Features
            # - Entfernte konstante Features
            # - Aufgetretene Warnungen
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

            # Eigenen MLflow-Run für dieses Modell starten. Training und
            # Evaluation sind jetzt unabhängig vom Tracking lokal gesichert.
            with mlflow.start_run(run_name=f"{run_name}_{name}") as run:
                row["run_id"] = run.info.run_id
                # Allgemeine Informationen als MLflow-Tags speichern.
                mlflow.set_tags({
                    "evaluation_split": "validation",
                    "error_metrics_version": ERROR_METRICS_VERSION,
                    "data_sha256": config["feature_table_sha256"],
                    "git_commit": git_state["git_commit"],
                    "git_dirty": str(git_state["git_dirty"]),
                    **{key: row[key] for key in ("training_status", "evaluation_status", "local_save_status")},
                })

                # Parameter in MLflow speichern.
                mlflow.log_params(params)
                # Metriken in MLflow speichern.
                #
                # Der Sternchen-Ausdruck **{...} fügt mehrere
                # Dictionary-Einträge hinzu.
                #
                # f"val_{k}" erzeugt z. B. "val_macro_f1".
                mlflow.log_metrics({
                    "training_seconds": duration,
                    "prediction_seconds": prediction_duration,
                    **{
                        f"val_{k}": v
                        for k, v in metrics.items()
                    },
                })

                # Experimentmetadaten in MLflow speichern.
                mlflow.log_artifacts(
                    str(metadata),
                    artifact_path="metadata"
                )

                # Evaluationsberichte hochladen.
                mlflow.log_artifacts(
                    str(directory / "evaluations" / name),
                    artifact_path="evaluation"
                )

                # Modell in MLflow speichern.
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    serialization_format="skops",
                    skops_trusted_types=trusted_types_for_model(name)
                )

                # Erst nach erfolgreichem Logging Status aktualisieren.
                row["mlflow_status"] = "FINISHED"
                row["status"] = "FINISHED"

                # Abschließenden Status in MLflow dokumentieren.
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

            # Abschlussmeldung ausgeben.
            print(f"{name}: {duration:.1f}s, kritische Fehler: {metrics['critical_errors']}; "
                  f"gespeichert in {directory}", flush=True)

        # ========================================================
        # 8. FEHLERBEHANDLUNG
        # ========================================================
        except Exception as exc:
            # Nur die fehlgeschlagene Phase ändern. Vorhandene Metriken,
            # Zeiten und Status früherer Phasen bleiben in derselben Zeile.
            row[f"{stage}_status"] = "FAILED"
            row[f"{stage}_error_type"] = type(exc).__name__
            row[f"{stage}_error"] = str(exc)
            row["status"] = "PARTIAL" if row["evaluation_status"] == "FINISHED" else "FAILED"
            error_path = directory / f"{name}_{stage}_error.txt"
            error_path.write_text(traceback.format_exc())

            # Falls ein MLflow-Run existiert: Versuchen, den Fehlerbericht
            # dort hochzuladen. MlflowClient erlaubt direkten Zugriff auf
            # bereits existierende Runs anhand ihrer ID.
            if run is not None:
                try:
                    client = mlflow.MlflowClient()
                    client.set_tag(run.info.run_id, f"{stage}_status", "FAILED")
                    client.log_artifact(run.info.run_id, str(error_path), "errors")
                except Exception as logging_error:
                    # Falls selbst das Hochladen fehlschlägt, nur ausgeben.
                    print(f"Fehlerdetails nur lokal gespeichert: {logging_error}", flush=True)

            # Danach geht die for-Schleife automatisch weiter.
            print(f"{name}: Fehler in {stage}: {exc}. Fahre mit nächstem Modell fort.", flush=True)
        finally:
            # Einzelne Ergebnisdatei und gemeinsame Ergebnisübersicht
            # aktualisieren, ohne eine bereits erfasste Zeile zu entfernen.
            persist_comparison_row(row, rows, directory)

    return pd.DataFrame(rows)
