from pathlib import Path

import pandas as pd

from src.data.dataset import prepare_dataset
from src.data.splitting import create_splits

from src.evaluation.metrics import evaluate_predictions, calculate_error_counts
from src.evaluation.reporting import save_evaluation_report

from src.models.rf import create_random_forest
from src.models.hgb import create_hgb, train_hgb
from src.models.lr import create_logistic_regression, create_lr_grid_search
from src.models.mlp import create_mlp

from src.experiments.saving import create_experiment_dir, save_model, save_results

from src.paths import FEATURE_TABLE_ROOT, EXPERIMENTS_ROOT

# ============================================================
# settings
# ============================================================

EXPERIMENT_NAME = "extended_features"

RESULTS_DIR = EXPERIMENTS_ROOT


def run_experiment():

    # ============================================================
    # experimentenordner erstellen
    # ============================================================

    experiment_dir, models_dir = create_experiment_dir(RESULTS_DIR, EXPERIMENT_NAME)

    # ============================================================
    # daten laden
    # ============================================================

    df = pd.read_csv(FEATURE_TABLE_ROOT)

    X, y = prepare_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)

    # ============================================================
    # modelle
    # ============================================================

    models = {
        "random_forest": {
            "model": create_random_forest(),
            "train_function": None,
        },

        "hgb": {
            "model": create_hgb(),
            "train_function": train_hgb,
        },

        "logistic_regression": {
            "model": create_lr_grid_search(
                create_logistic_regression()
            ),
            "train_function": None,
        },

        "mlp": {
            "model": create_mlp(),
            "train_function": None,
        },
    }

    # ============================================================
    # results
    # ============================================================

    all_results = []

    # ============================================================
    # training und validierung
    # ============================================================

    for model_name, config in models.items():

        print(f"train: {model_name} ...")

        model = config["model"]
        train_function = config["train_function"]

        if train_function is None:

            model.fit(X_train, y_train)

        else:

            model = train_function(model, X_train, y_train)

        # --------------------------------------------------------
        # save model
        # --------------------------------------------------------

        save_model(model, model_name, models_dir)

        # --------------------------------------------------------
        # pred
        # --------------------------------------------------------

        y_pred = model.predict(X_val)

        # --------------------------------------------------------
        # evaluate
        # --------------------------------------------------------

        results = evaluate_predictions(y_val, y_pred)

        errors = calculate_error_counts(y_val, y_pred)

        save_evaluation_report(y_val, y_pred, experiment_dir, model_name)

        # --------------------------------------------------------
        # ergebnisse sammeln
        # --------------------------------------------------------

        all_results.append({
            "model": model_name,
            "accuracy": results["accuracy"],
            "balanced_accuracy": results["balanced_accuracy"],
            "macro_f1": results["macro_f1"],
            "weighted_f1": results["weighted_f1"],

            "critical_errors": errors["critical_errors"],
            "critical_error_rate": errors["critical_error_rate"],

            "false_alarms": errors["false_alarms"],
            "false_alarm_rate": errors["false_alarm_rate"],

            "red_to_yellow": errors["red_to_yellow"],
            "red_to_yellow_rate": errors["red_to_yellow_rate"],

            "green_to_yellow": errors["green_to_yellow"],
            "green_to_yellow_rate": errors["green_to_yellow_rate"],
        })

    # ============================================================
    # save
    # ============================================================

    results_df = pd.DataFrame(all_results)

    save_results(results_df, experiment_dir)

    print(f"ergebnisse gespeichert unter: {experiment_dir}")

    return results_df


if __name__ == "__main__":
    run_experiment()