from pathlib import Path
import pandas as pd


# ============================================================
# gelabelte messreihen laden, standardisieren und zu einer csv zusammenführen
# ============================================================

DATASETS = [
    {
        "name": "messreihe_1",
        "path": Path("/results/measurements/data1_labeled.csv"),
        "drop_columns": ["label_found"],
    },
    {
        "name": "messreihe_2",
        "path": Path("/results/measurements/data2_labeled.csv"),
        "drop_columns": ["label_found"],
    },
    {
        "name": "messreihe_3",
        "path": Path("/results/measurements/data3_labeled.csv"),
        "drop_columns": ["label_found"],
    },
]

OUTPUT_PATH = Path("/results/measurements/combined_dataset.csv")

COORD_COLUMNS = ["x", "y"]


# ============================================================
# Datensätze laden und vorbereiten
# ============================================================

all_dfs = []

for dataset in DATASETS:
    name = dataset["name"]
    path = dataset["path"]
    drop_columns = dataset["drop_columns"]

    print(f"Lade {name}...")

    df = pd.read_csv(path)

    # Prüfen, ob Koordinatenspalten vorhanden sind
    missing_coord_cols = [col for col in COORD_COLUMNS if col not in df.columns]
    if missing_coord_cols:
        raise ValueError(
            f"Im Datensatz '{name}' fehlen Koordinatenspalten: {missing_coord_cols}"
        )

    # Gewünschte Spalten entfernen
    existing_drop_cols = [col for col in drop_columns if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # Datensatz-Name speichern, damit gleiche Koordinaten unterscheidbar bleiben
    df["source_dataset"] = name

    # Eindeutige Instanz-ID erzeugen
    df["instance_id"] = (
        df["source_dataset"].astype(str)
        + "_x"
        + df["x"].astype(str)
        + "_y"
        + df["y"].astype(str)
    )

    all_dfs.append(df)

# zusammenführen
combined_df = pd.concat(all_dfs, ignore_index=True)


# ============================================================
# Kontrolle auf Koordinaten-Überschneidungen
# ============================================================

# Gleiche Koordinaten über verschiedene Datensätze hinweg
coord_counts = (
    combined_df
    .groupby(COORD_COLUMNS)["source_dataset"]
    .nunique()
    .reset_index(name="num_datasets")
)

overlapping_coords = coord_counts[coord_counts["num_datasets"] > 1]

print()
print("Gesamtanzahl Zeilen:", len(combined_df))
print("Anzahl Koordinaten, die in mehreren Datensätzen vorkommen:", len(overlapping_coords))

if len(overlapping_coords) > 0:
    print()
    print("Beispiele für überlappende Koordinaten:")
    print(overlapping_coords.head(20))

# result: 20971 (innerhalb des selben datensatzes: 0)

# ============================================================
# Speichern
# ============================================================

combined_df.to_csv(OUTPUT_PATH, index=False)

print()
print(f"Kombinierter Datensatz gespeichert unter:")
print(OUTPUT_PATH)