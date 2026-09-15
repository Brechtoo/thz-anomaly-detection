from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.paths import FEATURE_EXTENSION_BLOCK1_ROOT, EXTERNAL_SCAN_ROOT, EXTERNAL_TEST_RESULTS

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.extract_all_features import extract_all_features
from src.unsupervised.processing.prepare_data import load_feature_data
from src.unsupervised.models.pca import train_pca, calculate_reconstruction_scores as pca_scores
from src.unsupervised.models.autoencoder import train_autoencoder, calculate_reconstruction_scores as autoencoder_scores
from src.unsupervised.models.isolation_forest import train_isolation_forest, calculate_anomaly_scores


# ============================================================
# settings
# ============================================================

DATA_PATH = FEATURE_EXTENSION_BLOCK1_ROOT
DATA_DIR = EXTERNAL_SCAN_ROOT
OUTPUT_DIR = EXTERNAL_TEST_RESULTS / "unsupervised"

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


def extract_coordinates(file_path):
    match = re.search(r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]", file_path.name)
    return float(match.group(1)), float(match.group(2))


# ============================================================
# data
# ============================================================

X, y = load_feature_data(DATA_PATH, TARGET_COL, EXCLUDE_COLUMNS)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

files = list(DATA_DIR.glob("*.txt"))

X_external = pd.DataFrame([extract_all_features(file_path) for file_path in files])
X_external = X_external[X.columns]

coordinates = [extract_coordinates(file_path) for file_path in files]


# ============================================================
# preprocessing
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_external_imputed = imputer.transform(X_external)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
X_external_scaled = scaler.transform(X_external_imputed)


# ============================================================
# train
# ============================================================

pca = train_pca(X_train_scaled, n_components=N_COMPONENTS)
autoencoder = train_autoencoder(X_train_scaled, RANDOM_STATE)
isolation_forest = train_isolation_forest(X_train_imputed, random_state=RANDOM_STATE)


# ============================================================
# scores
# ============================================================

models = {
    "pca": (
        pca_scores(pca, X_test_scaled),
        pca_scores(pca, X_external_scaled),
    ),
    "autoencoder": (
        autoencoder_scores(autoencoder, X_test_scaled),
        autoencoder_scores(autoencoder, X_external_scaled),
    ),
    "isolation_forest": (
        calculate_anomaly_scores(isolation_forest, X_test_imputed)[0],
        calculate_anomaly_scores(isolation_forest, X_external_imputed)[0],
    ),
}


# ============================================================
# sum
# ============================================================

for name, (reference_scores, external_scores) in models.items():

    model_output = OUTPUT_DIR / name
    model_output.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "dataset": ["reference_test", "external"],
        "count": [len(reference_scores), len(external_scores)],
        "mean": [np.mean(reference_scores), np.mean(external_scores)],
        "median": [np.median(reference_scores), np.median(external_scores)],
        "std": [np.std(reference_scores), np.std(external_scores)],
        "min": [np.min(reference_scores), np.min(external_scores)],
        "max": [np.max(reference_scores), np.max(external_scores)],
    })

    summary.to_csv(model_output / "summary.csv", index=False)

    plt.scatter(
        [coord[0] for coord in coordinates],
        [coord[1] for coord in coordinates],
        c=external_scores,
        s=8,
    )

    plt.xlabel("Xpos [mm]")
    plt.ylabel("Ypos [mm]")
    plt.colorbar(label="Anomalie-Score")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(model_output / "anomaly_map.png", dpi=300)
    plt.close()