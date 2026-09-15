
import pandas as pd

from src.paths import PCA_RESULTS, FEATURE_EXTENSION_BLOCK1_ROOT
from src.unsupervised.processing.prepare_data import load_feature_data, prepare_feature_data
from src.unsupervised.processing.evaluation import evaluate_anomaly_scores
from src.unsupervised.models.pca import train_pca, calculate_reconstruction_scores


# ============================================================
# settings
# ============================================================

DATA_PATH = FEATURE_EXTENSION_BLOCK1_ROOT
OUTPUT_DIR = PCA_RESULTS

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

N_COMPONENTS = 12

TOP_K_VALUES = [20, 50, 100, 500]


def save_scores(y_test, scores):
    results = pd.DataFrame({
        "label": y_test.to_numpy(),
        "score": scores
    })

    results.to_csv(OUTPUT_DIR / "pca_scores.csv", index=False)

    return results


# ============================================================
# save
# ============================================================

def save_score_distribution(results):
    distribution = results.groupby("label")["score"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()

    distribution.to_csv(OUTPUT_DIR / "score_distribution.csv", index=False)


def save_results(metrics):
    results = pd.DataFrame([metrics])

    results.to_csv(OUTPUT_DIR / "metrics.csv", index=False)


def save_pca_info(model):
    info = pd.DataFrame({
        "n_components": [model.n_components_],
        "explained_variance": [model.explained_variance_ratio_.sum()]
    })

    info.to_csv(OUTPUT_DIR / "pca_info.csv", index=False)


# ============================================================
# main
# ============================================================

def main():

    # ============================================================
    # load and prepare data
    # ============================================================


    X, y = load_feature_data(DATA_PATH, TARGET_COL, EXCLUDE_COLUMNS)

    X_train, X_test, y_train, y_test = prepare_feature_data(X, y, random_state=RANDOM_STATE)

    # --------------------------------------------------------
    # train
    # --------------------------------------------------------

    model = train_pca(X_train, n_components=N_COMPONENTS)

    save_pca_info(model)

    # --------------------------------------------------------
    # scores
    # --------------------------------------------------------

    scores = calculate_reconstruction_scores(model, X_test)

    # --------------------------------------------------------
    # save scores
    # --------------------------------------------------------

    results = save_scores(y_test, scores)

    save_score_distribution(results)

    # --------------------------------------------------------
    # eval
    # --------------------------------------------------------

    metrics = evaluate_anomaly_scores(y_test, scores, TOP_K_VALUES)

    # --------------------------------------------------------
    # save results
    # --------------------------------------------------------

    save_results(metrics)


if __name__ == "__main__":
    main()