from pathlib import Path
import pandas as pd


# ============================================================
# gelabelte messreihen laden, standardisieren und zu einer csv zusammenführen
# ============================================================

DATASETS = [
    {
        "name": "messreihe_1",
        "path": Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/measurements/data1_labeled.csv"),
        "drop_columns": ["label_found"],
    },
    {
        "name": "messreihe_2",
        "path": Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/measurements/data2_labeled.csv"),
        "drop_columns": ["label_found"],
    },
    {
        "name": "messreihe_3",
        "path": Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/measurements/data3_labeled.csv"),
        "drop_columns": ["label_found"],
    },
]

OUTPUT_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/measurements/combined_dataset.csv")

COORD_COLUMNS = ["x", "y"]


# ============================================================
# datensätze laden und vorbereiten
# ============================================================

all_dfs = []

for dataset in DATASETS:
    name = dataset["name"]
    path = dataset["path"]
    drop_columns = dataset["drop_columns"]

    print(f"lade {name}...")

    df = pd.read_csv(path)

    # koordinatenspalten prüfen
    missing_coord_cols = [col for col in COORD_COLUMNS if col not in df.columns]
    if missing_coord_cols:
        raise ValueError(
            f"im datensatz '{name}' fehlen koordinatenspalten: {missing_coord_cols}"
        )

    existing_drop_cols = [col for col in drop_columns if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # merken aus welchem datensatz
    df["source_dataset"] = name

    # eindeutige id pro messung
    df["instance_id"] = (df["source_dataset"].astype(str) + "_x" + df["x"].astype(str) + "_y" + df["y"].astype(str))

    all_dfs.append(df)

# zusammenführen
combined_df = pd.concat(all_dfs, ignore_index=True)

# ============================================================
# überprüfung gleicher koordinaten
# ============================================================

coord_counts = (combined_df.groupby(COORD_COLUMNS)["source_dataset"].nunique().reset_index(name="num_datasets"))

overlapping_coords = coord_counts[coord_counts["num_datasets"] > 1]

print("anzahl überlappender koordinaten: ", len(overlapping_coords))

# result: 20971 (innerhalb des selben datensatzes: 0)

# ============================================================
# save
# ============================================================

combined_df.to_csv(OUTPUT_PATH, index=False)

print(f"kombinierter daten satz gespeichert unter: {OUTPUT_PATH}")