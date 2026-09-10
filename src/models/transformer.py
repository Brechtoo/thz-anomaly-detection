import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.splitting import create_splits

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader

from src.features.loading import load_measurement
from src.paths import FEATURE_TABLE_ROOT


# ============================================================
# settings
# ============================================================

RANDOM_STATE = 42

N_POINTS = 256
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3

METADATA_PATH = Path(FEATURE_TABLE_ROOT)

OUTPUT_DIR = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/transformer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# messungen vorbereiten
# ============================================================

def load_signal(file_path):

    df = load_measurement(file_path)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    surface_mask = (x >= 980) & (x <= 1005)

    surface_x = x[surface_mask]
    surface_y = y[surface_mask]

    surface_idx = np.argmax(np.abs(surface_y))
    t_surface = surface_x[surface_idx]

    x = x - t_surface

    mask = (x >= 0) & (x <= 80)

    x = x[mask]
    y = y[mask]

    # interpolieren
    x_new = np.linspace(0, 80, N_POINTS)

    y_new = np.interp(x_new, x, y)

    return y_new.astype(np.float32)


def prepare_signals(df):
    signals = []

    for i, file_path in enumerate(df["file_path"]):

        signal = load_signal(file_path)
        signals.append(signal)

        if i % 1000 == 0:
            print(f"{i} messungen geladen...")

    return np.array(signals)


# ============================================================
# transformer
# ============================================================

class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Linear(1, 64)
        self.position = nn.Parameter(torch.zeros(1, N_POINTS, 64))

        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

        self.output = nn.Linear(64, 4)

    def forward(self, x):

        # (batch, 256)
        x = x.unsqueeze(-1)

        # (batch, 256, 64)
        x = self.embedding(x)

        x = x + self.position

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.output(x)


# ============================================================
# training
# ============================================================

def train_model(model, train_loader, val_loader, loss_function, optimizer):

    best_val_f1 = 0
    best_model = None

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        # training
        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_function(predictions, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # validation
        y_val_true, y_val_pred, _ = predict(model, val_loader)

        val_f1 = f1_score(y_val_true, y_val_pred, average="macro")

        critical_errors = np.sum((y_val_true == 1) & (y_val_pred == 3))

        false_alarms = np.sum((y_val_true == 3) & (y_val_pred == 1))

        green_to_yellow = np.sum((y_val_true == 3) & np.isin(y_val_pred, [0, 2]))

        print(
            f"epoch {epoch + 1:02d} | "
            f"loss: {total_loss / len(train_loader):.4f} | "
            f"val F1: {val_f1:.4f} | "
            f"critical: {critical_errors} | "
            f"false alarms: {false_alarms} | "
            f"green→yellow: {green_to_yellow}"
        )

        # bestes modell merken
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_model = copy.deepcopy(model.state_dict())

    # bestes modell wiederherstellen
    model.load_state_dict(best_model)

    print(f"\nbest validation macro-F1: {best_val_f1:.4f}")

    return model


# ============================================================
# preds
# ============================================================

def predict(model, loader):

    model.eval()

    y_true = []
    y_pred = []
    probabilities = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)

            proba = torch.softmax(logits, dim=1)

            pred = torch.argmax(proba, dim=1)

            y_true.extend(y_batch.numpy())

            y_pred.extend(pred.cpu().numpy())

            probabilities.extend(proba.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(probabilities)


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # --------------------------------------------------------
    # load and split
    # --------------------------------------------------------

    df = pd.read_csv(METADATA_PATH)

    train_df, val_df, test_df, y_train, y_val, y_test = create_splits(df, df["label"])

    # --------------------------------------------------------
    # signale verarbeiten
    # --------------------------------------------------------

    X_train = prepare_signals(train_df)
    X_val = prepare_signals(val_df)
    X_test = prepare_signals(test_df)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    # --------------------------------------------------------
    # skalierung
    # --------------------------------------------------------

    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # --------------------------------------------------------
    # tensor dataset
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.long,
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    test_dataset = TensorDataset(
        torch.tensor(
            X_test,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_test,
            dtype=torch.long,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # klasssengewichtung
    # --------------------------------------------------------

    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0],dtype=torch.float32).to(DEVICE)

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = TransformerModel().to(DEVICE)

    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------------
    # train
    # --------------------------------------------------------

    model = train_model(model, train_loader, val_loader, loss_function, optimizer)

    # --------------------------------------------------------
    # test
    # --------------------------------------------------------

    y_true, y_pred, proba = predict(model, val_loader)

    print(f"accuracy: {accuracy_score(y_true, y_pred)}")

    print(f"balanced Accuracy: {balanced_accuracy_score(y_true, y_pred)}")

    print(f"macro-F1: {f1_score(y_true, y_pred, average='macro')}")

    print(f"confusion Matrix: {confusion_matrix(y_true, y_pred)}")

    print(f"\nClassification Report: {classification_report(y_true, y_pred, digits=4)}")

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    results = val_df.copy()

    results["y_pred"] = y_pred

    for label in range(4):
        results[f"proba_{label}"] = (proba[:, label])

    results.to_csv(OUTPUT_DIR / "transformer_class1_weighted_validation_predictions.csv", index=False)

    torch.save(model.state_dict(), OUTPUT_DIR / "transformer_class1_weighted.pt")