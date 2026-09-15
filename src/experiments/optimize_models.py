from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset import prepare_dataset
from src.data.splitting import create_splits
from src.paths import FEATURE_TABLE_ROOT, EXTENDED_EXPERIMENT_DIR

EXPERIMENT_DIR = EXTENDED_EXPERIMENT_DIR

OUTPUT_DIR = EXPERIMENT_DIR / "optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def optimize_models():

    df = pd.read_csv(FEATURE_TABLE_ROOT)

    X, y = prepare_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ============================================================
    # hgb
    # ============================================================

    hgb = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    hgb_params = {
        "hgb__learning_rate": [0.05, 0.07, 0.09],
        "hgb__max_iter": [400, 500, 600],
        "hgb__max_leaf_nodes": [31, 63, 95],
        "hgb__min_samples_leaf": [20, 30, 40],
        "hgb__l2_regularization": [2.0, 5.0, 10.0],
    }

    hgb_search = RandomizedSearchCV(estimator=hgb, param_distributions=hgb_params, n_iter=30, scoring="f1_macro", cv=cv, n_jobs=-1, random_state=RANDOM_STATE)

    print("optimize hgb ...")

    hgb_search.fit(X_train, y_train)

    # ============================================================
    # mlp
    # ============================================================

    mlp = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "mlp",
            MLPClassifier(
                random_state=RANDOM_STATE,
                max_iter=1000,
                early_stopping=True,
            ),
        ),
    ])

    mlp_params = {
        "mlp__hidden_layer_sizes": [
            (128, 64),
            (256, 128),
            (256, 128, 64),
            (512, 256),
        ],
        "mlp__alpha": [
            0.0001,
            0.00025,
            0.0005,
            0.001,
        ],
        "mlp__learning_rate_init": [
            0.001,
            0.002,
            0.003,
            0.005,
        ],
    }

    mlp_search = RandomizedSearchCV(estimator=mlp, param_distributions=mlp_params, n_iter=30, scoring="f1_macro", cv=cv, n_jobs=-1, random_state=RANDOM_STATE)

    print("optimize mlp ...")

    mlp_search.fit(X_train, y_train)

    # ============================================================
    # modelle speichern
    # ============================================================

    joblib.dump(hgb_search.best_estimator_, OUTPUT_DIR / "hgb_optimized.joblib")

    joblib.dump(mlp_search.best_estimator_, OUTPUT_DIR / "mlp_optimized.joblib")

    # ============================================================
    # ergebnisse speichern
    # ============================================================

    results = pd.DataFrame([
        {
            "model": "hgb",
            "best_cv_macro_f1": hgb_search.best_score_,
            "best_params": str(hgb_search.best_params_),
        },
        {
            "model": "mlp",
            "best_cv_macro_f1": mlp_search.best_score_,
            "best_params": str(mlp_search.best_params_),
        },
    ])

    results.to_csv(OUTPUT_DIR / "optimization_results.csv", index=False)

    return results


if __name__ == "__main__":
    optimize_models()