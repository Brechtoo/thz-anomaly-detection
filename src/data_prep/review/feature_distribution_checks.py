from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# untersuchung der werteverteilung aller features in einer datei
# ============================================================

FEATURE_CSV_PATH = Path("/results/feature_tables/feature_baseline.csv")

# ============================================================
# Daten laden
# ============================================================

df = pd.read_csv(FEATURE_CSV_PATH)

LABEL_COL = "label"

cols_to_ignore = [
    LABEL_COL,
    "relative_path",
    "file_path",
    "source_dataset",
    "y",
    "x",
    "instance_id"
]

feature_cols = [c for c in df.columns if c not in cols_to_ignore]

print(f"Anzahl Features: {len(feature_cols)}")
print(f"Anzahl Messungen: {len(df)}")
print()


# ============================================================
# Zusammenfassung pro Feature
# ============================================================

summary_rows = []

for col in feature_cols:
    s = df[col]

    value_counts = s.value_counts(dropna=False) # wie oft jeder unterschiedliche wert vorkommt, NaN werden auch gezählt
    most_common_value = value_counts.index[0] # erster wert der häufigkeitstabelle
    most_common_count = value_counts.iloc[0] # anzahl wie oft der häufigste wert vorkommt
    most_common_percent = most_common_count / len(df) * 100 # prozentuale anteil der häufigsten wertes

    n_unique = s.nunique(dropna=False) # wie viele unterschiedliche werte
    n_missing = s.isna().sum() # wie viele werte fehlen
    missing_percent = n_missing / len(df) * 100 # prozentualer anteil fehlender werte

    if pd.api.types.is_numeric_dtype(s): # stastische größen für numerische werte
        std = s.std()
        min_val = s.min()
        max_val = s.max()
    else:
        std = np.nan
        min_val = np.nan
        max_val = np.nan

    summary_rows.append({
        "feature": col,
        "n_unique": n_unique,
        "most_common_value": most_common_value,
        "most_common_count": most_common_count,
        "most_common_percent": round(most_common_percent, 4),
        "missing_count": n_missing,
        "missing_percent": round(missing_percent, 4),
        "std": std,
        "min": min_val,
        "max": max_val,
        "almost_constant_99_percent": most_common_percent >= 99,
        "constant": n_unique == 1,
    })


summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by=["almost_constant_99_percent", "most_common_percent"],
    ascending=[False, False]
)

print("Feature-Zusammenfassung:")
print(summary_df.to_string(index=False))
print()

# ============================================================
# Features vorschlagen, die man löschen könnte
# ============================================================

drop_candidates = summary_df[
    (summary_df["constant"] == True) |
    (summary_df["almost_constant_99_percent"] == True)
]

print()
print("Mögliche Drop-Kandidaten:")
if len(drop_candidates) == 0:
    print("Keine offensichtlich konstanten oder fast konstanten Features gefunden.")
else:
    print(drop_candidates[[
        "feature",
        "n_unique",
        "most_common_value",
        "most_common_percent",
        "constant",
        "almost_constant_99_percent",
    ]].to_string(index=False))
