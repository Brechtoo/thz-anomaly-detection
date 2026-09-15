from pathlib import Path

import pandas as pd

from src.paths import FEATURE_EXTENSION_BLOCK1_ROOT, ISOLATION_FOREST_RESULTS
from src.unsupervised.processing.prepare_data import load_feature_data, prepare_feature_data
from src.unsupervised.processing.evaluation import evaluate_anomaly_scores
from src.unsupervised.models.isolation_forest import train_isolation_forest, calculate_anomaly_scores


# ============================================================
# settings
# ============================================================

DATA_PATH = FEATURE_EXTENSION_BLOCK1_ROOT
OUTPUT_DIR = ISOLATION_FOREST_RESULTS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"

EXCLUDE_COLUMNS = [
    "label",
    "file_path",
    "x",
    "y",
    "source_dataset",
    "instance_id",
]

RANDOM_STATE = 42

TOP_K_VALUES = [20, 50, 100, 500]


# ============================================================
# results
# ============================================================

def build_results(df, y_test, scores, if_labels):

    results = pd.DataFrame({
        "label": y_test.to_numpy(),
        "if_score": scores,
        "if_label": if_labels,
    }, index=y_test.index)

    if "file_path" in df.columns:
        results["file_path"] = df.loc[y_test.index, "file_path"]

    return results.reset_index(drop=True)


# ============================================================
# main
# ============================================================

def main():
    df, X, y = load_feature_data(DATA_PATH, TARGET_COL, EXCLUDE_COLUMNS, return_df=True)

    X_train, X_test, y_train, y_test = prepare_feature_data(X, y, random_state=RANDOM_STATE, scale=False)

    model = train_isolation_forest(X_train, random_state=RANDOM_STATE)

    scores, if_labels = calculate_anomaly_scores(model, X_test)

    results = build_results(df, y_test, scores, if_labels)

    results.to_csv(OUTPUT_DIR / "if_scores.csv", index=False)

    score_distribution = results.groupby("label")["if_score"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()

    score_distribution.to_csv(OUTPUT_DIR / "score_distribution.csv", index=False)

    metrics = evaluate_anomaly_scores(y_test, scores, TOP_K_VALUES)

    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)


if __name__ == "__main__":
    main()