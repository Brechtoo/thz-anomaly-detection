from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from src.paths import FEATURE_BASELINE_ROOT

from sklearn.utils.class_weight import compute_sample_weight

# ============================================================
# settings
# ============================================================

DATA_PATH = FEATURE_BASELINE_ROOT

MODEL_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results"
                  "/hgb/hgb_baseline/hgb_baseline.joblib")

model = joblib.load(MODEL_PATH)

OUTPUT_DIR = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results"
                  "/analysis/hgb_baseline_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "hgb_baseline_permutation_importance.txt"

TARGET_COL = "label"

cols_to_drop = [
    "label",
    "file_path",
    "x",
    "y",
    "source_dataset",
    "instance_id",
]

RANDOM_STATE = 42


# ============================================================
# daten laden
# ============================================================

df = pd.read_csv(DATA_PATH)

y = df[TARGET_COL]

X = df.drop(
    columns=[col for col in cols_to_drop if col in df.columns]
)

X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")


# ============================================================
# train / validation / test split
#
# 64 % training
# 16 % validation
# 20 % test
# ============================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_val,
)


# ============================================================
# fehlende werte behandeln
# ============================================================

# median ausschließlich aus trainingsdaten
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_val = X_val.fillna(train_medians)
X_test = X_test.fillna(train_medians)


# ============================================================
# modell vorbereiten
# ============================================================

# Untrainierte Kopie mit exakt denselben Hyperparametern erzeugen
model = clone(model)

sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

# Auf neuem Trainingsset trainieren
# X_train bleibt DataFrame -> Feature-Namen werden mitgespeichert
model.fit(X_train, y_train, sample_weight=sample_weight)


# ============================================================
# überprüfen
# ============================================================

print(f"Training:   {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")

print(f"\nAnzahl Features: {X_val.shape[1]}")

print("\nFeatures:")
print(X_val.columns.tolist())


# ============================================================
# permutation importance
# ============================================================

permutation_result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="f1_macro",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ============================================================
# results
# ============================================================

importance_df = pd.DataFrame({
    "feature": X_val.columns,
    "importance_mean": permutation_result.importances_mean,
    "importance_std": permutation_result.importances_std,
})

importance_df = importance_df.sort_values(
    by="importance_mean",
    ascending=False,
)


# ============================================================
# save and write
# ============================================================

with open(OUTPUT_PATH, "w") as file:
    file.write("Permutation Importance - HGB Baseline\n")
    file.write("Scoring: Macro-F1\n")
    file.write("n_repeats: 10\n")
    file.write("Split: 64 % Train / 16 % Validation / 20 % Test\n\n")

    file.write(importance_df.to_string(index=False))

print("\nPermutation Importance:")
print(importance_df.to_string(index=False))

print(f"\nErgebnisse gespeichert unter:\n{OUTPUT_PATH}")