from pathlib import Path

import joblib
import pandas as pd

from src.data.dataset import prepare_dataset
from src.data.splitting import create_splits

from src.evaluation.metrics import (
    evaluate_predictions,
    calculate_error_counts,
)
from src.evaluation.reporting import save_evaluation_report
from src.evaluation.thresholding import predict_with_thresholds

from src.paths import FEATURE_TABLE_ROOT


EXPERIMENT_DIR = Path("/results/experiments/extended_features")

OPTIMIZATION_DIR = EXPERIMENT_DIR / "optimization"
FINAL_EVALUATION_DIR = EXPERIMENT_DIR / "final_evaluation"

FINAL_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


CONFIGURATIONS = [
    {
        "name": "hgb_main",
        "red_threshold": 0.30,
        "green_threshold": 0.99,
        "role": "main",
    },
    {
        "name": "hgb_comparison",
        "red_threshold": 0.30,
        "green_threshold": 0.985,
        "role": "comparison",
    },
]


def final_evaluation():

    df = pd.read_csv(FEATURE_TABLE_ROOT)

    X, y = prepare_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)

    model = joblib.load(OPTIMIZATION_DIR / "hgb_optimized.joblib")

    # Für rot/gelb/grün-evaluation:
    # 0 und 2 werden beide als gelb behandelt
    y_test_rgy = y_test.replace({0: 2,})

    results = []

    for config in CONFIGURATIONS:

        y_pred = predict_with_thresholds(model, X_test, red_threshold=config["red_threshold"], green_threshold=config["green_threshold"],)

        metrics = evaluate_predictions(y_test_rgy, y_pred)

        errors = calculate_error_counts(y_test, y_pred)

        results.append({
            "name": config["name"],
            "role": config["role"],
            "red_threshold": config["red_threshold"],
            "green_threshold": config["green_threshold"],

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

        save_evaluation_report(y_test_rgy, y_pred, FINAL_EVALUATION_DIR, config["name"])

    results_df = pd.DataFrame(results)

    results_df.to_csv(FINAL_EVALUATION_DIR / "results.csv", index=False)

    pd.DataFrame(CONFIGURATIONS).to_csv(FINAL_EVALUATION_DIR / "configurations.csv", index=False)

    print(results_df.to_string(index=False))

if __name__ == "__main__":
    final_evaluation()