import numpy as np

from torch.utils.data import DataLoader, TensorDataset
import torch

from src.features.loading import load_measurement

from pathlib import Path

from src.paths import FEATURE_BASELINE_ROOT, TRANSFORMER_RESULTS

# ============================================================
# setings
# ============================================================

RANDOM_STATE = 42

N_POINTS = 256
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3

N_CLASSES = 4

METADATA_PATH = Path(FEATURE_BASELINE_ROOT)

OUTPUT_DIR = TRANSFORMER_RESULTS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = ("mps" if torch.backends.mps.is_available() else "cpu")

WEIGHT_CONFIGS = {
    "no_weights": None,
    "class1_weight2": [1.0, 2.0, 1.0, 1.0],
    "balanced": "balanced",
}

# ============================================================
# load functions
# ============================================================


def load_signal(file_path):

    df = load_measurement(file_path)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    surface_mask = ((x >= 980) & (x <= 1005))

    surface_x = x[surface_mask]
    surface_y = y[surface_mask]

    surface_idx = np.argmax(np.abs(surface_y))

    t_surface = surface_x[surface_idx]

    x = x - t_surface

    mask = ((x >= 0) & (x <= 80))

    x = x[mask]
    y = y[mask]

    x_new = np.linspace(0, 80, N_POINTS)

    y_new = np.interp(x_new, x, y)

    return y_new.astype(np.float32)


def prepare_signals(df):

    signals = []

    for i, file_path in enumerate(df["file_path"]):

        signals.append(load_signal(file_path))

        if (i + 1) % 2000 == 0: print(f"{i + 1}/{len(df)} signale geladen")

    return np.array(signals)


def standardize_signals(X_train, X_val):

    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / std

    X_val = (X_val - mean) / std

    return X_train, X_val


def create_data_loaders(X_train, X_val, y_train, y_val, batch_size=BATCH_SIZE):

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
        torch.tensor(
            X_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_val,
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

    return train_loader, val_loader