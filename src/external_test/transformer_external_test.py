from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.paths import TRANSFORMER_RESULTS, FEATURE_EXTENSION_BLOCK1_ROOT, EXTERNAL_SCAN_ROOT, EXTERNAL_TEST_RESULTS

from torch.utils.data import DataLoader, TensorDataset

from src.data.splitting import create_splits
from src.transformer.load_signal import prepare_signals
from src.models.transformer import TransformerModel


# ============================================================
# settings
# ============================================================

MODEL_PATH = TRANSFORMER_RESULTS / "transformer_class1_weight2.pt"
METADATA_PATH = FEATURE_EXTENSION_BLOCK1_ROOT
DATA_DIR = EXTERNAL_SCAN_ROOT
OUTPUT_DIR = EXTERNAL_TEST_RESULTS / "transformer"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def extract_coordinates(file_path):
    match = re.search(r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]", file_path.name)
    return float(match.group(1)), float(match.group(2))


# ============================================================
# bekannte skalierung
# ============================================================

df = pd.read_csv(METADATA_PATH)

train_df, _, _, _, _, _ = create_splits(df, df["label"])

X_train = prepare_signals(train_df)

mean = X_train.mean()
std = X_train.std()

del X_train

if DEVICE == "mps":
    torch.mps.empty_cache()


# ============================================================
# new data
# ============================================================

files = list(DATA_DIR.glob("*.txt"))

external_df = pd.DataFrame({
    "file_path": files
})

coordinates = [extract_coordinates(file_path) for file_path in files]

X = prepare_signals(external_df)

X = (X - mean) / std


# ============================================================
# model
# ============================================================

dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


model = TransformerModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# preds
# ============================================================

probabilities = []

with torch.no_grad():
    for (X_batch,) in loader:
        X_batch = X_batch.to(DEVICE)

        proba_batch = torch.softmax(model(X_batch), dim=1)

        probabilities.append(proba_batch.cpu().numpy())

proba = np.concatenate(probabilities)

predictions = np.argmax(proba, axis=1)


# ============================================================
# results
# ============================================================

results = pd.DataFrame({
    "x": [coord[0] for coord in coordinates],
    "y": [coord[1] for coord in coordinates],
    "pred": predictions,
    "p0": proba[:, 0],
    "p1": proba[:, 1],
    "p2": proba[:, 2],
    "p3": proba[:, 3],
})

results["ampel"] = results["pred"].map({
    0: "gelb",
    1: "rot",
    2: "gelb",
    3: "grün",
})

results["p_rot"] = results["p1"]
results["p_gelb"] = results["p0"] + results["p2"]
results["p_gruen"] = results["p3"]


# ============================================================
# plot
# ============================================================

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

plt.savefig(
    OUTPUT_DIR / "prediction_map.png",
    dpi=300
)

plt.close()


# ============================================================
# sum
# ============================================================

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

summary.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)