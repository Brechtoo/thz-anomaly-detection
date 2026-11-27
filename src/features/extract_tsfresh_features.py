"""Zeitreihenmerkmale mit tsfresh berechnen und anhand der eingefrorenen Trainingszeilen auswählen."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
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

from src.data.reference_measurements import require_no_references
from src.data.split_manifest import restore_splits, sha256_file
from src.features.loading import load_measurement
from src.paths import COMBINED_DATASET_ROOT, PROJECT_ROOT, RESULTS_ROOT


def extract_measurement(task):
    """Eine Messdatei bildet genau eine Zeitreihe. Parallelisiert wird über Dateien."""

    path, preset = task

    signal = load_measurement(path)

    if signal.empty or not np.isfinite(signal.to_numpy()).all():
        raise ValueError(f"Empty or non-finite signal: {path}")

    signal = signal.assign(measurement=0)

    settings = EfficientFCParameters() if preset == "efficient" else MinimalFCParameters()

    features = extract_features(
        signal,
        column_id="measurement",  # Gruppierung nach Messungs-ID
        column_sort="x",          # Zeitachse
        column_value="y",         # Signalamplitude
        default_fc_parameters=settings,
        n_jobs=0,                 # Keine zusätzliche Parallelisierung
        disable_progressbar=True,
    )

    return features.iloc[0].to_dict()


def select_training_features(features, labels, train_index, *, n_jobs=0, return_ranges=False):
    """Ersatzwerte und relevante Spalten im Training bestimmen und auf alle Zeilen anwenden."""

    features = features.copy()

    # Derselbe Trainingsindex bestimmt Ersatzwerte und überwachte Merkmalsauswahl.
    ranges = get_range_values_per_column(
        features.loc[train_index].copy()
    )

    impute_dataframe_range(features, *ranges)

    selected = select_features(
        features.loc[train_index],
        labels.loc[train_index],
        ml_task="classification",  # Klassifikationsproblem
        multiclass=True,           # Vier Klassen
        n_significant=1,           # Relevanz für mindestens eine Klasse
        fdr_level=0.05,            # Signifikanzkontrolle
        n_jobs=n_jobs,
    )

    # Die übrigen Messungen behalten ihre Zeilen und erhalten dasselbe Spaltenschema.
    result = features.loc[:, selected.columns]
    return (result, ranges) if return_ranges else result


def build_table(input_path, output_path, split_dir, *,
                preset="efficient", n_jobs=4):
    """Die Messreihenfolge aus dem Katalog beibehalten und die Feature-Tabelle mit ihrem Manifest speichern."""

    input_path, output_path, split_dir = map(
        Path, (input_path, output_path, split_dir)
    )

    metadata_path = output_path.with_suffix(".json")

    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    dataset = pd.read_csv(input_path)
    require_no_references(dataset)

    required = {"file_path", "instance_id", "label"}

    if not required <= set(dataset) or dataset.empty:
        raise ValueError(
            "Nonempty dataset with file_path, instance_id and label required"
        )

    if not dataset.label.isin([0, 1, 2, 3]).all():
        raise ValueError("Expected original labels 0/1/2/3")

    train, *_ = restore_splits(
        dataset[["instance_id"]],
        dataset.label,
        dataset.instance_id,
        split_dir,
    )

    paths = [
        Path(p) if Path(p).is_absolute()
        else input_path.parent / p
        for p in dataset.file_path
    ]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    input_hash = sha256_file(input_path)

    rows = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:

        tasks = ((str(path), preset) for path in paths)

        # map liefert Ergebnisse in Katalogreihenfolge, auch bei unterschiedlich langen Messungen.
        for i, row in enumerate(
            executor.map(extract_measurement, tasks, chunksize=32),
            start=1,
        ):

            rows.append(row)

            if i % 1000 == 0:
                print(
                    f"Extracted {i}/{len(dataset)} measurements",
                    flush=True,
                )

    features = pd.DataFrame(rows, index=dataset.index)

    selected = select_training_features(
        features,
        dataset.label,
        train.index,
        n_jobs=n_jobs,
    )

    if selected.shape[1] == 0:
        raise ValueError(
            "No significant features selected; no output written"
        )

    if sha256_file(input_path) != input_hash:
        raise ValueError("Input CSV changed during extraction")

    metadata_columns = [
        c for c in (
            "file_path", "label", "x", "y",
            "source_dataset", "instance_id"
        )
        if c in dataset
    ]

    table = pd.concat(
        [dataset[metadata_columns], selected],
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(output_path, index=False, mode="x")

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

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(
        f"Saved {len(table)} rows, "
        f"{selected.shape[1]} selected features to {output_path}"
    )

    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        type=Path,
        default=COMBINED_DATASET_ROOT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "feature_tables" / "tsfresh_features.csv",
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "configs/hgb_baseline/splits",
    )

    parser.add_argument(
        "--preset",
        choices=("minimal", "efficient"),
        default="efficient",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")

    build_table(
        args.input,
        args.output,
        args.split_dir,
        preset=args.preset,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
