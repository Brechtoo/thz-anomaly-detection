from pathlib import Path

import joblib
import pandas as pd

from src.data.dataset import prepare_dataset
from src.data.splitting import create_splits

from src.evaluation.metrics import (
    evaluate_predictions,
    calculate_error_counts,
)
from src.evaluation.thresholding import predict_with_thresholds

from src.paths import FEATURE_TABLE_ROOT

EXPERIMENT_DIR = Path("/results/experiments/extended_features")

OPTIMIZATION_DIR = EXPERIMENT_DIR / "optimization"

MODELS = [
    "hgb",
    "mlp",
]

GREEN_THRESHOLDS = [
    round(0.97 + i * 0.005, 3)
    for i in range(6)
]

RED_THRESHOLDS = [
    round(0.20 + i * 0.05, 2)
    for i in range(11)
]


def compare_optimized_thresholds():

    df = pd.read_csv(FEATURE_TABLE_ROOT)

    X, y = prepare_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)

    y_val_rgy = y_val.replace({0: 2,})

    results = []

    for model_name in MODELS:

        model_path = (OPTIMIZATION_DIR / f"{model_name}_optimized.joblib")

        model = joblib.load(model_path)

        for green_threshold in GREEN_THRESHOLDS:
            for red_threshold in RED_THRESHOLDS:

                y_pred = predict_with_thresholds(model, X_val, red_threshold=red_threshold, green_threshold=green_threshold)

                metrics = evaluate_predictions(y_val_rgy, y_pred)

                errors = calculate_error_counts(y_val, y_pred)

                results.append({
                    "model": model_name,
                    "red_threshold": red_threshold,
                    "green_threshold": green_threshold,

                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],

                    "critical_errors": errors["critical_errors"],
                    "critical_error_rate": errors["critical_error_rate"],

                    "false_alarms": errors["false_alarms"],
                    "false_alarm_rate": errors["false_alarm_rate"],

                    "red_to_yellow_rate": errors["red_to_yellow_rate"],
                    "green_to_yellow_rate": errors["green_to_yellow_rate"],
                })

    results_df = pd.DataFrame(results)

    OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(OPTIMIZATION_DIR / "optimized_rgy_threshold_comparison.csv", index=False)

    return results_df


if __name__ == "__main__":
    compare_optimized_thresholds()