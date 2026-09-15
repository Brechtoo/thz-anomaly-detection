from pathlib import Path
import re

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.features.extract_all_features import extract_all_features
from src.paths import LR_MODEL_PATH, RF_MODEL_PATH, HGB_MODEL_PATH, MLP_MODEL_PATH, EXTERNAL_SCAN_ROOT, EXTERNAL_TEST_RESULTS

DATA_DIR = EXTERNAL_SCAN_ROOT
OUTPUT_DIR = EXTERNAL_TEST_RESULTS

MODELS = {
    "lr": LR_MODEL_PATH,
    "rf": RF_MODEL_PATH,
    "hgb": HGB_MODEL_PATH,
    "mlp": MLP_MODEL_PATH
}


def extract_coordinates(file_path):
    match = re.search(r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]", file_path.name)
    return float(match.group(1)), float(match.group(2))


def evaluate_model(name, model, X, coordinates):
    predictions = model.predict(X)
    proba = model.predict_proba(X)

    results = pd.DataFrame({
        "x": [coord[0] for coord in coordinates],
        "y": [coord[1] for coord in coordinates],
        "pred": predictions
    })

    for i, label in enumerate(model.classes_):
        results[f"p{label}"] = proba[:, i]

    results["ampel"] = results["pred"].map({
        0: "gelb",
        1: "rot",
        2: "gelb",
        3: "grün",
    })

    results["p_rot"] = results["p1"]
    results["p_gelb"] = results["p0"] + results["p2"]
    results["p_gruen"] = results["p3"]

    model_output = OUTPUT_DIR / name
    model_output.mkdir(parents=True, exist_ok=True)

    colors = results["ampel"].map({
        "rot": "red",
        "gelb": "gold",
        "grün": "green",
    })

    plt.scatter(results["x"], results["y"], c=colors, s=8)
    plt.xlabel("Xpos [mm]")
    plt.ylabel("Ypos [mm]")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(model_output / "prediction_map.png", dpi=300)
    plt.close()

    summary = pd.DataFrame({
        "ampel": ["rot", "gelb", "grün"],
        "count": [
            int((results["ampel"] == "rot").sum()),
            int((results["ampel"] == "gelb").sum()),
            int((results["ampel"] == "grün").sum()),
        ],
        "share": [
            float((results["ampel"] == "rot").mean()),
            float((results["ampel"] == "gelb").mean()),
            float((results["ampel"] == "grün").mean()),
        ],
        "mean_probability": [
            float(results["p_rot"].mean()),
            float(results["p_gelb"].mean()),
            float(results["p_gruen"].mean()),
        ]
    })

    summary.to_csv(model_output / "summary.csv", index=False)


def main():
    files = list(DATA_DIR.glob("*.txt"))

    X = pd.DataFrame([extract_all_features(file_path) for file_path in files])
    coordinates = [extract_coordinates(file_path) for file_path in files]

    for name, model_path in MODELS.items():
        model = joblib.load(model_path)
        evaluate_model(name, model, X, coordinates)


if __name__ == "__main__":
    main()