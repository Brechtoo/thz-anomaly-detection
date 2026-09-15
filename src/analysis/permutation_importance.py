from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.data.dataset import prepare_dataset
from src.data.splitting import create_splits
from src.paths import FEATURE_TABLE_ROOT, EXTENDED_EXPERIMENT_DIR, PERMUTATION_IMPORTANCE_RESULTS

# ============================================================
# settings
# ============================================================

EXPERIMENT_DIR = EXTENDED_EXPERIMENT_DIR

MODEL_DIR = EXPERIMENT_DIR / "models"

OUTPUT_DIR = PERMUTATION_IMPORTANCE_RESULTS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True,)

MODELS = {
    "rf": MODEL_DIR / "random_forest.joblib",
    "hgb": MODEL_DIR / "hgb.joblib",
}

SCORING = "f1_macro"
N_REPEATS = 10
RANDOM_STATE = 42

BASELINE_FEATURES = [
    "y_max",
    "y_min",
    "y_peak_to_peak",
    "y_mean",
    "y_median",
    "y_std",
    "y_abs_mean",
    "y_energy",
    "y_abs_area",
    "x_at_y_max",
    "x_at_y_min",
    "first_third_energy",
    "middle_third_energy",
    "last_third_energy",
    "first_third_std",
    "middle_third_std",
    "last_third_std",
]


# ============================================================
# permutation importance
# ============================================================

def calculate_permutation_importance(model_name, model_path, X_train, X_val, y_train, y_val):

    print(f"\npermutation importance für {model_name.upper()}...")

    # --------------------------------------------------------
    # model laden und zurücksetzen
    # --------------------------------------------------------

    model = joblib.load(model_path)

    model = clone(model)

    # --------------------------------------------------------
    # missing values
    # --------------------------------------------------------

    train_medians = X_train.median()

    X_train_clean = X_train.fillna(train_medians)

    X_val_clean = X_val.fillna(train_medians)

    # --------------------------------------------------------
    # class weighting
    # --------------------------------------------------------

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train,)

    # --------------------------------------------------------
    # train model
    # --------------------------------------------------------

    if isinstance(model, Pipeline):

        final_step_name = model.steps[-1][0]

        model.fit(X_train_clean, y_train, **{f"{final_step_name}__sample_weight": sample_weight})

    else:

        model.fit(X_train_clean, y_train, sample_weight=sample_weight)

    # --------------------------------------------------------
    # permutation importance
    # --------------------------------------------------------

    permutation_result = permutation_importance(model, X_val_clean, y_val, scoring=SCORING, n_repeats=N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1)

    # --------------------------------------------------------
    # result
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": X_val_clean.columns,
        "importance_mean":
            permutation_result.importances_mean,
        "importance_std":
            permutation_result.importances_std,
    })

    importance_df = importance_df.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return importance_df


# ============================================================
# save
# ============================================================

def save_results(model_name, importance_df):

    txt_path = (OUTPUT_DIR / f"{model_name}_permutation_importance.txt")

    csv_path = (OUTPUT_DIR / f"{model_name}_permutation_importance.csv")

    importance_df.to_csv(csv_path, index=False)

    with open(txt_path, "w") as file:

        file.write(f"Permutation Importance - {model_name.upper()}\n")

        file.write(f"Scoring: {SCORING}\n")

        file.write(f"n_repeats: {N_REPEATS}\n")

        file.write("Split: 64 % Train / "
            "16 % Validation / "
            "20 % Test\n\n"
        )

        file.write(importance_df.to_string(index=False))

    print(f"ergebnisse gespeichert unter: {txt_path} und {csv_path}")


# ============================================================
# main
# ============================================================

def main():

    # --------------------------------------------------------
    # dataset
    # --------------------------------------------------------

    df = pd.read_csv(FEATURE_TABLE_ROOT)

    X, y = prepare_dataset(df)

    X = X[BASELINE_FEATURES]

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)

    # --------------------------------------------------------
    # safety cleanup
    # --------------------------------------------------------

    X_train = X_train.replace([np.inf, -np.inf], np.nan)

    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    valid_columns = (~X_train.isna().all())

    X_train = X_train.loc[:, valid_columns]

    X_val = X_val[X_train.columns]

    print(f"trainings set: {len(X_train)}")

    print(f"validation set: {len(X_val)}")

    print(f"features: {X_train.shape[1]}")

    # --------------------------------------------------------
    # models
    # --------------------------------------------------------

    for model_name, model_path in MODELS.items():

        importance_df = (calculate_permutation_importance(model_name=model_name, model_path=model_path, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val))

        save_results(model_name, importance_df)

        print("top 10: " + importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()