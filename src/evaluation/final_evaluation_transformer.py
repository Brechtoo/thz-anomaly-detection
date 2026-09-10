from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader

from src.data.splitting import create_splits
from src.evaluation.metrics import (
    evaluate_predictions,
    calculate_error_counts,
)
from src.evaluation.thresholding import threshold_predict

from src.models.transformer import (
    TransformerModel,
    prepare_signals,
)

from src.paths import FEATURE_TABLE_ROOT


# ============================================================
# settings
# ============================================================

BATCH_SIZE = 128

RED_THRESHOLD = 0.45
GREEN_THRESHOLD = 0.70

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/transformer/transformer_unweighted.pt")

OUTPUT_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/transformer/transformer_final_results.csv")

# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # load and split
    # --------------------------------------------------------

    df = pd.read_csv(FEATURE_TABLE_ROOT)

    train_df, _, test_df, _, _, y_test = create_splits(df, df["label"])

    # --------------------------------------------------------
    # rohe signale
    # --------------------------------------------------------

    X_train = prepare_signals(train_df)
    X_test = prepare_signals(test_df)

    y_test = y_test.to_numpy()

    # --------------------------------------------------------
    # skalierung
    # --------------------------------------------------------

    mean = X_train.mean()
    std = X_train.std()

    X_test = (X_test - mean) / std

    # --------------------------------------------------------
    # test loader
    # --------------------------------------------------------

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # ungewichtetes modell
    # --------------------------------------------------------

    model = TransformerModel().to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.eval()

    # --------------------------------------------------------
    # wahrscheinlichkeiten
    # --------------------------------------------------------

    probabilities = []

    with torch.no_grad():

        for X_batch, _ in test_loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)

            proba = torch.softmax(logits, dim=1)

            probabilities.extend(proba.cpu().numpy())

    probabilities = np.array(probabilities)

    # --------------------------------------------------------
    # threshold logik
    # --------------------------------------------------------

    y_pred = threshold_predict(
        probabilities,
        classes=np.array([0, 1, 2, 3]),
        red_threshold=RED_THRESHOLD,
        green_threshold=GREEN_THRESHOLD,
    )

    y_test_rgy = pd.Series(y_test).replace({0: 2})

    # --------------------------------------------------------
    # evaluation
    # --------------------------------------------------------

    metrics = evaluate_predictions(y_test_rgy, y_pred)

    errors = calculate_error_counts(y_test, y_pred)

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    results = {
        "model": "transformer_unweighted",
        "red_threshold": RED_THRESHOLD,
        "green_threshold": GREEN_THRESHOLD,
        **metrics,
        **errors,
    }

    results_df = pd.DataFrame([results])

    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"finale evaluation gespeichert unter: {OUTPUT_PATH}")