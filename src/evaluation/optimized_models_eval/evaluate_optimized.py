
import joblib
import pandas as pd

from src.data.dataset import prepare_dataset
from src.data.splitting import create_splits
from src.evaluation.metrics import (
    ERROR_METRICS_VERSION,
    calculate_error_counts,
    evaluate_predictions,
)
from src.evaluation.reporting import save_evaluation_report
from src.paths import EXTENDED_EXPERIMENT_DIR, FEATURE_TABLE_ROOT

EXPERIMENT_DIR = EXTENDED_EXPERIMENT_DIR

OPTIMIZATION_DIR = EXPERIMENT_DIR / "optimization"


def evaluate_optimized_models():

    df = pd.read_csv(FEATURE_TABLE_ROOT)

    X, y = prepare_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test, = create_splits(X, y)

    models = {
        "hgb_optimized": OPTIMIZATION_DIR / "hgb_optimized.joblib",
        "mlp_optimized": OPTIMIZATION_DIR / "mlp_optimized.joblib",
    }

    all_results = []

    for model_name, model_path in models.items():

        model = joblib.load(model_path)

        y_pred = model.predict(X_val)

        results = evaluate_predictions(y_val, y_pred)

        errors = calculate_error_counts(y_val, y_pred)

        save_evaluation_report(y_val, y_pred, OPTIMIZATION_DIR, model_name)

        all_results.append({
            "model": model_name,

            "accuracy": results["accuracy"],
            "balanced_accuracy": results["balanced_accuracy"],
            "macro_f1": results["macro_f1"],
            "weighted_f1": results["weighted_f1"],

            **errors,
            "error_metrics_version": ERROR_METRICS_VERSION,
        })

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(OPTIMIZATION_DIR / "optimized_validation_results.csv", index=False)

    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    evaluate_optimized_models()