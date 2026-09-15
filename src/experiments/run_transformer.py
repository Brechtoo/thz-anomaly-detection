import numpy as np
import pandas as pd
import torch

from src.data.splitting import create_splits

from src.transformer.load_signal import prepare_signals, standardize_signals, create_data_loaders

from src.experiments.transformer_experiments import run_configuration

from pathlib import Path

from src.paths import FEATURE_BASELINE_ROOT, TRANSFORMER_RESULTS

BATCH_SIZE = 128

METADATA_PATH = Path(FEATURE_BASELINE_ROOT)

OUTPUT_DIR = TRANSFORMER_RESULTS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_CONFIGS = {
    "no_weights": None,
    "class1_weight2": [1.0, 2.0, 1.0, 1.0],
    "balanced": "balanced",
}


def main():

    np.random.seed(42)

    torch.manual_seed(42)

    # --------------------------------------------------------
    # data load and split
    # --------------------------------------------------------

    df = pd.read_csv(METADATA_PATH)

    train_df, val_df, test_df, y_train, y_val, y_test = create_splits(df, df["label"])

    print("load signals ...")

    X_train = prepare_signals(train_df)

    X_val = prepare_signals(val_df)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    X_train, X_val = standardize_signals(X_train, X_val)

    train_loader, val_loader = (create_data_loaders(X_train, X_val, y_train, y_val, batch_size=BATCH_SIZE))

    # --------------------------------------------------------
    # weight configs
    # --------------------------------------------------------

    all_results = []

    for name, weight_config in (WEIGHT_CONFIGS.items()):

        result = run_configuration(
            name=name,
            weight_config=weight_config,
            train_loader=train_loader,
            val_loader=val_loader,
            y_train=y_train,
            val_df=val_df,
        )

        all_results.append(result)

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    comparison = pd.DataFrame(all_results)

    comparison.to_csv(OUTPUT_DIR / "transformer_weight_comparison.csv", index=False)

    print(f"vergleich gespeichert unter {OUTPUT_DIR / 'transformer_weight_comparison.csv'}")

if __name__ == "__main__":
    main()