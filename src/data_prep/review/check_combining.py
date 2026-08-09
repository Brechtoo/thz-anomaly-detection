from pathlib import Path
import pandas as pd


# ============================================================
# test, ob zusammenführen und droppen der spalte label_found geklappt hat
# ============================================================

# zur bestimmung der herkunft der daten
DATASET_PATHS = {
    "messreihe_1": Path("/results/measurements/data1_labeled.csv"),
    "messreihe_2": Path("/results/measurements/data2_labeled.csv"),
    "messreihe_3": Path("/results/measurements/data3_labeled.csv"),
}

COMBINED_PATH = Path("/results/measurements/combined_dataset.csv")

DROPPED_COLUMN = "label_found" # überprüfung ob gedroppt wurde

COORD_COLUMNS = ["x", "y"]


# ============================================================
# Daten laden
# ============================================================

original_dfs = {}
# für jedes objekt in DATASET_PATHS an stelle name lies path
for name, path in DATASET_PATHS.items():
    original_dfs[name] = pd.read_csv(path)

combined_df = pd.read_csv(COMBINED_PATH)


# ============================================================
# 1. Zeilenanzahl prüfen
# ============================================================

original_total_rows = sum(len(df) for df in original_dfs.values())
combined_total_rows = len(combined_df)

print("=== Zeilenanzahl ===")
print("Summe Originaldaten:", original_total_rows)
print("Kombinierter Datensatz:", combined_total_rows)

if original_total_rows == combined_total_rows:
    print("✅ Zeilenanzahl passt.")
else:
    print("❌ Zeilenanzahl passt NICHT.")

# result: ✅ Zeilenanzahl passt.


# ============================================================
# 2. Entfernte Spalte prüfen
# ============================================================

print("\n=== Entfernte Spalte ===")

if DROPPED_COLUMN not in combined_df.columns:
    print(f"✅ Spalte '{DROPPED_COLUMN}' wurde entfernt.")
else:
    print(f"❌ Spalte '{DROPPED_COLUMN}' ist noch vorhanden.")

# result: ✅ Spalte 'label_found' wurde entfernt.


# ============================================================
# 3. Source-Datensatz prüfen
# ============================================================

print("\n=== Zeilen pro source_dataset ===")

print(combined_df["source_dataset"].value_counts())

for name, original_df in original_dfs.items():
    expected = len(original_df)
    actual = (combined_df["source_dataset"] == name).sum()

    if expected == actual:
        print(f"✅ {name}: {actual} Zeilen passen.")
    else:
        print(f"❌ {name}: erwartet {expected}, gefunden {actual}")

# result: ✅ messreihe_1: 21304 Zeilen passen.
# ✅ messreihe_2: 21422 Zeilen passen.
# ✅ messreihe_3: 10466 Zeilen passen.


# ============================================================
# 4. Eindeutigkeit der instance_id prüfen
# ============================================================

print("\n=== instance_id Eindeutigkeit ===")

num_rows = len(combined_df)
num_unique_ids = combined_df["instance_id"].nunique()

print("Zeilen:", num_rows)
print("Eindeutige instance_ids:", num_unique_ids)

if num_rows == num_unique_ids:
    print("✅ Jede instance_id ist eindeutig.")
else:
    print("❌ Es gibt doppelte instance_ids.")

    duplicates = combined_df[
        combined_df.duplicated(subset=["instance_id"], keep=False)
    ]

    print(duplicates[["source_dataset", "x", "y", "instance_id"]].head(20))

# result: ✅ Jede instance_id ist eindeutig.


# ============================================================
# 5. Koordinaten-Überschneidungen zwischen Datensätzen prüfen
# ============================================================

print("\n=== Koordinaten-Überschneidungen ===")

coord_overlap = (
    combined_df
    .groupby(COORD_COLUMNS)["source_dataset"]
    .nunique()
    .reset_index(name="num_datasets")
)

overlapping_coords = coord_overlap[coord_overlap["num_datasets"] > 1]

print(
    "Koordinaten, die in mehreren Datensätzen vorkommen:",
    len(overlapping_coords)
)

if len(overlapping_coords) > 0:
    print("\nBeispiele:")
    print(overlapping_coords.head(20))

# result: Koordinaten, die in mehreren Datensätzen vorkommen: 20971


# ============================================================
# 6. Doppelte Koordinaten innerhalb desselben Datensatzes prüfen
# ============================================================

print("\n=== Doppelte Koordinaten innerhalb eines Datensatzes ===")

duplicates_within_source = combined_df[
    combined_df.duplicated(subset=["source_dataset", "x", "y"], keep=False)
]

print("Anzahl betroffener Zeilen:", len(duplicates_within_source))

if len(duplicates_within_source) > 0:
    print(duplicates_within_source[["source_dataset", "x", "y", "instance_id"]].head(20))
else:
    print("✅ Keine doppelten Koordinaten innerhalb einzelner Datensätze.")

# result: ✅ Keine doppelten Koordinaten innerhalb einzelner Datensätze.


# ============================================================
# 7. Labelverteilung je Datensatz: absolut und prozentual
# ============================================================

if "label" in combined_df.columns:
    print("\n=== Labelverteilung je Datensatz: absolut ===")

    label_counts = pd.crosstab(
        combined_df["source_dataset"],
        combined_df["label"],
        dropna=False
    )

    print(label_counts)

    print("\n=== Labelverteilung je Datensatz: prozentual ===")

    label_percentages = pd.crosstab(
        combined_df["source_dataset"],
        combined_df["label"],
        normalize="index",
        dropna=False
    ) * 100

    label_percentages = label_percentages.round(2)

    print(label_percentages)
