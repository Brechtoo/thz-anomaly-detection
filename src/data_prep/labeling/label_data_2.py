from pathlib import Path
import re
import pandas as pd

from src.paths import *


# ============================================================
# Pfade
# ============================================================

MEASUREMENT_ROOT = Path(HOHLRAUM_TESTKÖRPER_2_ROOT)

# Falls LABELS_1_ROOT direkt die Datei ist:
LABEL_FILE = Path(LABELS_2_ROOT)

# Falls LABELS_1_ROOT ein Ordner ist, stattdessen z. B.:
# LABEL_FILE = Path(LABELS_1_ROOT) / "labels.txt"

OUTPUT_CSV = Path(RESULTS_ROOT) / "data2_labeled.csv"


# ============================================================
# Labeldatei einlesen
#
# Aufbau der Labeldatei:
# # X Y HohlrmMark
# -79 -39 0
# -79 -32 0
#
# Bedeutung:
# X           -> x-Koordinate
# Y           -> y-Koordinate
# HohlrmMark  -> Label / Zielvariable
# ============================================================

labels = pd.read_csv(
    LABEL_FILE,
    sep=r"\s+",
    comment="#",
    names=["x", "y", "label"]
)

# Spaltenreihenfolge explizit setzen
labels = labels[["x", "y", "label"]]

# Datentypen sauber setzen
labels["x"] = labels["x"].astype(int)
labels["y"] = labels["y"].astype(int)
labels["label"] = labels["label"].astype(int)

# Optional: Labeltabelle sortieren
labels = labels.sort_values(["x", "y"]).reset_index(drop=True)

# Dictionary für schnellen Zugriff:
# (x, y) -> label
label_lookup = {
    (row.x, row.y): row.label
    for row in labels.itertuples(index=False)
}


# ============================================================
# Koordinaten aus Dateinamen extrahieren
#
# Beispiel-Dateiname:
# 2026-06-09T16-03-34.779323-Hohlrmtestk3_tr-[1]-[-79.0,-39.0,4.55]-[1.0,0.0,0.0,0.0]-delta[0.017mm-0.0deg]-avg20.txt
#
# Gesucht wird:
# [-79.0,-39.0,4.55]
#
# Ergebnis:
# x = -79
# y = -39
# z = 4.55
# ============================================================

coord_pattern = re.compile(
    r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]"
)


def extract_coordinates(filename: str):
    """
    Extrahiert x, y, z aus dem Dateinamen.

    x und y werden zu int gerundet, damit sie zur Labeldatei passen.
    z bleibt float.
    """
    match = coord_pattern.search(filename)

    if match is None:
        return None, None, None

    x = float(match.group(1))
    y = float(match.group(2))
    z = float(match.group(3))

    return int(round(x)), int(round(y)), z


# ============================================================
# Messdateien sammeln und Manifest bauen
# ============================================================

rows = []

for file_path in MEASUREMENT_ROOT.rglob("*.txt"):
    filename = file_path.name

    x, y, z = extract_coordinates(filename)

    if x is None or y is None:
        label = None
        label_found = False
    else:
        label = label_lookup.get((x, y))
        label_found = label is not None

    rows.append({
        "file_path": str(file_path),
        "filename": filename,
        "x": x,
        "y": y,
        "z": z,
        "label": label,
        "label_found": label_found,
    })


# ============================================================
# DataFrame erstellen, Spalten sortieren und speichern
# ============================================================

df = pd.DataFrame(rows)

df = df[
    [
        "file_path",
        "filename",
        "x",
        "y",
        "z",
        "label",
        "label_found",
    ]
]

df = df.sort_values(["x", "y"]).reset_index(drop=True)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# Ausgabe
# ============================================================

print(f"CSV gespeichert unter: {OUTPUT_CSV}")
print(f"Anzahl Messungen: {len(df)}")
print(f"Labels gefunden: {df['label_found'].sum()}")
print(f"Labels nicht gefunden: {(~df['label_found']).sum()}")
