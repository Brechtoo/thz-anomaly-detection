
"""
Extraktion und Selektion von tsfresh-Features für THz-Messungen.

Ablauf:
1. Datensatz laden und überprüfen
2. Gespeicherte Trainings-/Validierungs-/Test-Splits rekonstruieren
3. Messdateien laden und tsfresh-Features extrahieren
4. Fehlende Feature-Werte ersetzen (Imputation)
5. Statistisch relevante Features anhand der Trainingsdaten auswählen
6. Ergebnisse und Metadaten speichern

Ausführung:
    python -m src.features.extract_tsfresh_features --preset efficient
"""


# ============================================================
# 1. IMPORTS
# ============================================================

# Standardbibliothek
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Datenverarbeitung
import numpy as np
import pandas as pd

# Feature-Extraktion und -Selektion
import tsfresh
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import (
    EfficientFCParameters,
    MinimalFCParameters,
)
from tsfresh.utilities.dataframe_functions import (
    get_range_values_per_column,
    impute_dataframe_range,
)

# Eigene Projektfunktionen
from src.data.split_manifest import restore_splits, sha256_file
from src.features.loading import load_measurement
from src.paths import COMBINED_DATASET_ROOT, PROJECT_ROOT, RESULTS_ROOT


# ============================================================
# 2. FEATURE-EXTRAKTION EINER EINZELNEN MESSUNG
# ============================================================

def extract_measurement(task):
    """
    Lädt ein THz-Signal und berechnet seine tsfresh-Features.

    Parameter:
        task: Tupel (Dateipfad, Feature-Preset)

    Rückgabe:
        Dictionary {Feature-Name: Feature-Wert}
    """

    # Tupel in seine zwei Bestandteile zerlegen.
    path, preset = task

    # Messdatei laden.
    # Erwartet einen DataFrame mit den Spalten x und y.
    signal = load_measurement(path)

    # Prüfen, ob das Signal leer ist oder ungültige Werte enthält.
    if signal.empty or not np.isfinite(signal.to_numpy()).all():
        raise ValueError(f"Empty or non-finite signal: {path}")

    # tsfresh benötigt eine ID, um zusammengehörige
    # Zeitreihenwerte einer Messung zuzuordnen.
    #
    # Da hier nur EINE Messung verarbeitet wird,
    # erhalten alle Zeitpunkte dieselbe ID (0).
    signal = signal.assign(measurement=0)

    # Feature-Preset auswählen:
    # minimal   -> wenige, einfache Features
    # efficient -> deutlich umfangreichere Feature-Sammlung
    if preset == "efficient": settings = EfficientFCParameters()
    else: settings = MinimalFCParameters()

    # tsfresh berechnet aus der Zeitreihe statistische Merkmale.
    features = extract_features(
        signal,
        column_id="measurement",  # Gruppierung nach Messungs-ID
        column_sort="x",          # Zeitachse
        column_value="y",         # Signalamplitude
        default_fc_parameters=settings,
        n_jobs=0,                 # Keine zusätzliche Parallelisierung
        disable_progressbar=True,
    )

    # Es gibt genau eine Messung, also eine Feature-Zeile.
    # Diese als Dictionary zurückgeben.
    return features.iloc[0].to_dict()


# ============================================================
# 3. IMPUTATION UND FEATURE-SELEKTION
# ============================================================

def select_training_features(features, labels, train_index, *, n_jobs=0):
    """
    Ersetzt ungültige Feature-Werte und selektiert Features.

    Wichtig:
        Die Imputationsstatistiken und die Feature-Selektion
        werden ausschließlich anhand der Trainingsdaten bestimmt,
        um Data Leakage zu verhindern.

    Parameter:
        features: DataFrame aller extrahierten Features
        labels: Zielklassen (0, 1, 2, 3)
        train_index: Zeilenindizes der Trainingsmessungen
        n_jobs: Anzahl paralleler Jobs für die Selektion

    Rückgabe:
        DataFrame mit den selektierten Features aller Messungen.
    """

    # Kopie erstellen, damit die ursprüngliche Tabelle
    # nicht verändert wird.
    features = features.copy()

    # Nur Trainingszeilen zur Berechnung der Ersatzwerte verwenden.
    #
    # Die Funktion bestimmt pro Feature unter anderem
    # endliche Minimal-, Maximal- und Medianwerte.
    ranges = get_range_values_per_column(
        features.loc[train_index].copy()
    )

    # Fehlende oder unendliche Werte ersetzen.
    # Dabei werden die aus dem Training bestimmten Werte
    # auf den GESAMTEN Datensatz angewendet.
    impute_dataframe_range(features, *ranges)

    # Statistische Feature-Selektion:
    # Es werden nur Trainingsfeatures und Trainingslabels verwendet.
    selected = select_features(
        features.loc[train_index],
        labels.loc[train_index],
        ml_task="classification",  # Klassifikationsproblem
        multiclass=True,           # Vier Klassen
        n_significant=1,           # Relevanz für mindestens eine Klasse
        fdr_level=0.05,            # Signifikanzkontrolle
        n_jobs=n_jobs,
    )

    # selected enthält nur Trainingszeilen.
    # Wir übernehmen deshalb die ausgewählten SPALTEN
    # aus dem vollständigen DataFrame.
    #
    # Ergebnis: Alle Messungen, aber nur relevante Features.
    return features.loc[:, selected.columns]


# ============================================================
# 4. DATENSATZ LADEN UND ÜBERPRÜFEN
# ============================================================

def build_table(input_path, output_path, split_dir, *,
                preset="efficient", n_jobs=4):
    """
    Führt die vollständige Feature-Extraktion und -Selektion aus.

    Parameter:
        input_path: Pfad zur combined_dataset.csv
        output_path: Speicherort der Feature-Tabelle
        split_dir: Verzeichnis mit gespeicherten Split-Manifests
        preset: minimal oder efficient
        n_jobs: Anzahl paralleler Prozesse
    """

    # Pfadangaben in Path-Objekte umwandeln.
    input_path, output_path, split_dir = map(
        Path, (input_path, output_path, split_dir)
    )

    # Zur CSV wird eine gleichnamige JSON-Metadatendatei erstellt.
    metadata_path = output_path.with_suffix(".json")

    # Verhindern, dass bestehende Ergebnisse überschrieben werden.
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    # combined_dataset.csv laden
    dataset = pd.read_csv(input_path)

    # Diese Spalten müssen vorhanden sein.
    required = {"file_path", "instance_id", "label"}

    # Prüfen, ob die benötigten Spalten existieren
    # und der Datensatz nicht leer ist.
    if not required <= set(dataset) or dataset.empty:
        raise ValueError(
            "Nonempty dataset with file_path, instance_id and label required"
        )

    # Prüfen, ob ausschließlich die vier erwarteten Labels vorkommen.
    if not dataset.label.isin([0, 1, 2, 3]).all():
        raise ValueError("Expected original labels 0/1/2/3")


    # ========================================================
    # 5. TRAININGSSPLIT UND MESSDATEIEN VORBEREITEN
    # ========================================================

    # Die ursprünglich gespeicherte Train/Val/Test-Zuordnung
    # anhand der instance_ids wiederherstellen.
    #
    # restore_splits überprüft dabei auch:
    # - IDs und Labels
    # - Vollständigkeit und Überschneidungen der Splits
    # - Checksummen der gespeicherten Split-Dateien
    #
    # Wir benötigen hier nur den Trainingssplit.
    train, *_ = restore_splits(
        dataset[["instance_id"]],
        dataset.label,
        dataset.instance_id,
        split_dir,
    )

    # Dateipfade der THz-Messungen bestimmen.
    #
    # Absolute Pfade bleiben unverändert.
    # Relative Pfade werden relativ zur Eingabe-CSV aufgelöst.
    paths = [
        Path(p) if Path(p).is_absolute()
        else input_path.parent / p
        for p in dataset.file_path
    ]

    # Vor der aufwendigen Extraktion prüfen,
    # ob sämtliche Messdateien existieren.
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    # SHA-256-Hash der Eingabe (combined_dataset.csv) speichern.
    # Damit können wir später eine Veränderung erkennen.
    input_hash = sha256_file(input_path)


    # ========================================================
    # 6. FEATURE-EXTRAKTION ALLER MESSUNGEN
    # ========================================================

    # In dieser Liste werden die Feature-Dictionaries gesammelt.
    rows = []

    # Mehrere Messungen parallel verarbeiten.
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:

        # Für jede Messdatei ein Tupel (Pfad, Preset) erstellen.
        tasks = ((str(path), preset) for path in paths)

        # executor.map führt extract_measurement für alle
        # Tupel parallel aus.
        #
        # chunksize=32 bedeutet, dass Aufgaben in Blöcken
        # von 32 Messungen an Worker-Prozesse verteilt werden.
        #
        # Die zurückgegebenen Ergebnisse behalten die
        # ursprüngliche Reihenfolge der Eingaben.
        for i, row in enumerate(
            executor.map(extract_measurement, tasks, chunksize=32),
            start=1,
        ):

            rows.append(row)

            # Fortschritt nach jeweils 1000 Messungen anzeigen.
            if i % 1000 == 0:
                print(
                    f"Extracted {i}/{len(dataset)} measurements",
                    flush=True,
                )

    # Aus den Dictionaries eine Feature-Tabelle erstellen.
    #
    # Zeilen: Messungen
    # Spalten: tsfresh-Features
    features = pd.DataFrame(rows, index=dataset.index)


    # ========================================================
    # 7. FEATURE-SELEKTION UND ERGEBNIS SPEICHERN
    # ========================================================

    # Imputation und statistische Feature-Selektion ausführen.
    selected = select_training_features(
        features,
        dataset.label,
        train.index,
        n_jobs=n_jobs,
    )

    # Abbrechen, falls kein Feature als relevant ausgewählt wurde.
    if selected.shape[1] == 0:
        raise ValueError(
            "No significant features selected; no output written"
        )

    # Prüfen, ob sich die Eingabe-CSV während
    # der Feature-Extraktion verändert hat.
    if sha256_file(input_path) != input_hash:
        raise ValueError("Input CSV changed during extraction")

    # Metadatenspalten auswählen, sofern sie existieren.
    metadata_columns = [
        c for c in (
            "file_path", "label", "x", "y",
            "source_dataset", "instance_id"
        )
        if c in dataset
    ]

    # Ursprüngliche Metadaten und selektierte Features verbinden.
    #
    # pd.concat(axis=1) fügt DataFrames spaltenweise zusammen.
    table = pd.concat(
        [dataset[metadata_columns], selected],
        axis=1,
    )

    # Ausgabeordner erstellen, falls noch nicht vorhanden.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Feature-Tabelle speichern.
    # mode="x" verhindert das Überschreiben bestehender Dateien.
    table.to_csv(output_path, index=False, mode="x")


    # ========================================================
    # 8. METADATEN SPEICHERN
    # ========================================================

    # Dokumentieren, mit welchen Daten und Einstellungen
    # die Feature-Tabelle erstellt wurde.
    metadata = {
        "input_csv": str(input_path.resolve()),
        "input_sha256": input_hash,
        "tsfresh_version": tsfresh.__version__,
        "preset": preset,
        "split_dir": str(split_dir.resolve()),
        "split_manifest_sha256": sha256_file(
            split_dir / "manifest.json"
        ),
        "selection": (
            "training only; multiclass one-vs-rest; "
            "n_significant=1; FDR=0.05"
        ),
        "imputation": (
            "training finite max/min/median; "
            "all-nonfinite columns use zero"
        ),
        "rows": len(table),
        "training_rows": len(train),
        "extracted_features": features.shape[1],
        "selected_features": list(selected.columns),
        "output_sha256": sha256_file(output_path),
    }

    # Dictionary als lesbare JSON-Datei speichern.
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(
        f"Saved {len(table)} rows, "
        f"{selected.shape[1]} selected features to {output_path}"
    )

    return table


# ============================================================
# 9. PROGRAMMSTART UND TERMINALARGUMENTE
# ============================================================

def main():
    """
    Liest Terminalargumente und startet die Feature-Extraktion.
    """

    # Parser zur Verarbeitung von Terminalargumenten erstellen.
    parser = argparse.ArgumentParser(description=__doc__)

    # Eingabe-CSV
    parser.add_argument(
        "--input",
        type=Path,
        default=COMBINED_DATASET_ROOT,
    )

    # Ausgabe-CSV
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "feature_tables" / "tsfresh_features.csv",
    )

    # Verzeichnis der gespeicherten Splits
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "configs/hgb_baseline/splits",
    )

    # Auswahl der tsfresh-Feature-Sammlung
    parser.add_argument(
        "--preset",
        choices=("minimal", "efficient"),
        default="efficient",
    )

    # Anzahl paralleler Prozesse
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
    )

    # Terminalargumente einlesen.
    args = parser.parse_args()

    # Mindestens ein Worker-Prozess ist erforderlich.
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")

    # Hauptfunktion mit den eingelesenen Argumenten aufrufen.
    build_table(
        args.input,
        args.output,
        args.split_dir,
        preset=args.preset,
        n_jobs=args.n_jobs,
    )

if __name__ == "__main__":
    main()
